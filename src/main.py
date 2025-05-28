import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Optional, Dict

# Absolute imports from 'src'
from .config.env_manager import load_environment, get_env_var
from .data.ercot_api.client import ERCOTClient
from .data.ercot_api.queries import ERCOTQueries
from .data.weather_api.weather import WeatherAPIClient
from .storage.pgvector_storage import PgVectorStorage
from .services.embedding_service import EmbeddingService # Keep this if still directly used in main
from .services.sentence_builder import create_semantic_sentence # Corrected import path

LOCATIONS = {
    "houston_temp_c": "Houston,TX,USA", 
    "austin_temp_c": "Austin,TX,USA",
    "dallas_temp_c": "Dallas,TX,USA"
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# create_semantic_sentence function is now in src.services.sentence_builder.py

async def fetch_daily_ercot_metric(ercot_queries: ERCOTQueries, date_to_fetch: str) -> Optional[float]:
    """Fetches and processes ERCOT data to get a single daily generation metric."""
    logger.info(f"Fetching ERCOT generation metric for {date_to_fetch}...")
    try:
        gen_summary_response = ercot_queries.get_agg_gen_summary(
            delivery_date_from_override=date_to_fetch, 
            delivery_date_to_override=date_to_fetch
        )
        
        records = gen_summary_response.get('data', [])
        if not records:
            logger.info(f"No ERCOT generation records found for {date_to_fetch}.")
            return None
        
        total_generation_for_day = 0.0
        generation_field_found = False

        # Prioritize known total fields, then attempt to sum interval fields if multiple records exist.
        possible_total_fields = ['TOTAL_GEN', 'totalActualGenerationMW', 'totalMW', 'sumActualGeneration']
        possible_interval_fields = ['actualGeneration', 'MWH_Output', 'Value']

        if len(records) == 1: # Likely a single summary record
            record = records[0]
            for field_name in possible_total_fields:
                if field_name in record and record[field_name] is not None:
                    try:
                        total_generation_for_day = float(record[field_name])
                        generation_field_found = True
                        break 
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert field '{field_name}' value '{record[field_name]}' to float for {date_to_fetch}.")
        
        if not generation_field_found and records: # If not found in a single record or multiple records exist
            logger.info(f"Attempting to sum interval generation fields for {date_to_fetch}.")
            for record in records:
                for field_name in possible_interval_fields:
                    if field_name in record and record[field_name] is not None:
                        try:
                            total_generation_for_day += float(record[field_name])
                            generation_field_found = True 
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert interval field '{field_name}' value '{record[field_name]}' to float for {date_to_fetch}.")
                        break # Assume one relevant interval field per record
        
        if not generation_field_found:
            logger.warning(f"Could not identify or sum a generation metric from ERCOT data for {date_to_fetch}. Records sample: {json.dumps(records[:1])}")
            return None

        logger.info(f"Processed ERCOT generation metric for {date_to_fetch}: {total_generation_for_day}")
        return float(total_generation_for_day)

    except Exception as e:
        logger.error(f"Error fetching/processing ERCOT daily metric for {date_to_fetch}: {e}", exc_info=True)
        return None

async def fetch_daily_weather_metrics(weather_client: WeatherAPIClient, date_to_fetch: str) -> Optional[Dict[str, float]]:
    """Fetches and processes weather data to get key daily temperature metrics."""
    logger.info(f"Fetching weather metrics for {date_to_fetch}...")
    try:
        weather_df = weather_client.get_historical_weather(date_str=date_to_fetch, locations=LOCATIONS)
        if weather_df.empty:
            logger.info(f"No weather data found for {date_to_fetch}.")
            return None

        # Assuming the first row of the DataFrame is the daily summary.
        first_row = weather_df.iloc[0]
        metrics = {
            "avg_temp_c": float(first_row.get('avg_temperature_c', np.nan)),
            "houston_temp_c": float(first_row.get('houston_temp_c', np.nan)),
            "austin_temp_c": float(first_row.get('austin_temp_c', np.nan)),
            "dallas_temp_c": float(first_row.get('dallas_temp_c', np.nan)),
        }
        # Filter out NaN values if any field failed to convert or was missing
        metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        if not metrics: # If all metrics ended up as NaN
             logger.warning(f"All weather metrics were NaN for {date_to_fetch}.")
             return None

        logger.info(f"Processed weather metrics for {date_to_fetch}: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error fetching/processing weather daily metrics for {date_to_fetch}: {e}", exc_info=True)
        return None

async def process_and_embed_daily_summary(
    date_to_process: str, 
    ercot_queries: ERCOTQueries, 
    weather_client: Optional[WeatherAPIClient], 
    pg_storage: PgVectorStorage, 
    embedding_service: Optional[EmbeddingService]
):
    """Orchestrates fetching, processing, sentence creation, embedding, and storage for a single day."""
    logger.info(f"Processing daily summary for embedding on {date_to_process}...")

    ercot_metric = await fetch_daily_ercot_metric(ercot_queries, date_to_process)
    
    weather_metrics = None
    if weather_client:
        weather_metrics = await fetch_daily_weather_metrics(weather_client, date_to_process)
    else:
        logger.info(f"Weather client not available, skipping weather metrics for {date_to_process}.")

    if ercot_metric is None:
        logger.warning(f"Could not retrieve ERCOT metric for {date_to_process}. Skipping embedding.")
        return
    
    # If weather client was available but metrics couldn't be fetched, log it.
    # create_semantic_sentence is expected to handle None weather_metrics.
    if weather_client and weather_metrics is None:
        logger.warning(f"Could not retrieve Weather metrics for {date_to_process} (weather client was available).")

    semantic_sentence = create_semantic_sentence(date_to_process, ercot_metric, weather_metrics)

    if not semantic_sentence:
        logger.warning(f"Semantic sentence could not be created for {date_to_process}. Skipping embedding.")
        return

    logger.info(f"Generated semantic sentence for {date_to_process}: \"{semantic_sentence}\"")

    if not embedding_service:
        logger.warning(f"Embedding service not available. Cannot generate embedding for {date_to_process}.")
        return
        
    try:
        embedding_list = embedding_service.generate_embedding(semantic_sentence)
        if embedding_list is None:
            logger.error(f"Embedding generation failed for {date_to_process}, received None.")
            return
        
        embedding_array = np.array(embedding_list).astype(np.float32)
        vector_id = f"daily_summary_{date_to_process}" 
        
        store_success = pg_storage.store_embedding(vector_id=vector_id, embedding=embedding_array)
        if store_success:
            logger.info(f"Successfully stored combined daily embedding for {date_to_process} with ID {vector_id}.")
        else:
            logger.error(f"Failed to store combined daily embedding for {date_to_process}.")
    except Exception as e:
        logger.error(f"Error generating or storing embedding for {date_to_process}: {e}", exc_info=True)

# Removed fetch_and_store_ercot_data function

async def fetch_and_store_weather_data(weather_client: WeatherAPIClient, pg_storage: PgVectorStorage, date_to_fetch: str):
    """Fetches historical weather data and stores the raw DataFrame."""
    logger.info(f"Fetching and storing raw weather data for {date_to_fetch}...")
    try:
        weather_df = weather_client.get_historical_weather(date_str=date_to_fetch, locations=LOCATIONS)
        
        if weather_df.empty:
            logger.info(f"No weather data fetched for {date_to_fetch}.")
            return
            
        logger.info(f"Fetched {len(weather_df)} weather records for {date_to_fetch}.")
        
        if 'time' in weather_df.columns:
             weather_df.rename(columns={'time': 'timestamp'}, inplace=True)
        
        # Ensure only relevant columns are selected for insertion
        table_columns = ['timestamp', 'houston_temp_c', 'austin_temp_c', 'dallas_temp_c', 'avg_temperature_c', 'avg_temperature_f']
        df_to_insert = weather_df[[col for col in table_columns if col in weather_df.columns]]

        if not df_to_insert.empty:
            pg_storage.insert_dataframe_to_table(df_to_insert, 'historical_weather_data')
            logger.info(f"Stored {len(df_to_insert)} weather records for {date_to_fetch} in 'historical_weather_data'.")
        else:
            logger.info(f"No relevant columns for weather data insertion for {date_to_fetch}.")
            
    except Exception as e:
        logger.error(f"Error during weather data processing for {date_to_fetch}: {e}", exc_info=True)

async def main_ingestion_pipeline():
    """Main function to orchestrate the data ingestion pipeline."""
    logger.info("Initializing data ingestion pipeline...")
    
    load_environment()

    days_to_process_for_embeddings = int(get_env_var("DAYS_TO_PROCESS_FOR_EMBEDDINGS", "1"))
    days_to_store_raw_weather = int(get_env_var("DAYS_TO_STORE_RAW_WEATHER", "1")) 

    ercot_client = ERCOTClient()
    ercot_queries = ERCOTQueries(client=ercot_client)
    
    weather_client = None
    weather_api_key = get_env_var("WEATHER_API_KEY")
    if weather_api_key:
        weather_client = WeatherAPIClient(api_key=weather_api_key)
    else:
        logger.warning("WEATHER_API_KEY not found. Weather data tasks will be skipped.")

    pg_vector_storage = PgVectorStorage(lazy_init=False, app_environment=get_env_var("APP_ENVIRONMENT", "prod"))
    
    embedding_service = None
    openai_api_key = get_env_var("OPENAI_API_KEY")
    if openai_api_key:
        try:
            embedding_service = EmbeddingService(api_key=openai_api_key)
            logger.info("Embedding service initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}", exc_info=True)
    else:
        logger.warning("OPENAI_API_KEY not found. Embedding tasks will be skipped.")
    
    logger.info("--- Starting Data Ingestion Tasks ---")
    today = datetime.today()

    # Process and Embed Combined Daily Summaries
    logger.info(f"--- Starting Combined Daily Summary Embedding for the last {days_to_process_for_embeddings} day(s) ---")
    embedding_tasks = []
    for i in range(1, days_to_process_for_embeddings + 1):
        date_to_process = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        embedding_tasks.append(
            process_and_embed_daily_summary(
                date_to_process, 
                ercot_queries, 
                weather_client,
                pg_vector_storage, 
                embedding_service
            )
        )
    if embedding_tasks:
        await asyncio.gather(*embedding_tasks)
    logger.info(f"Combined daily summary embedding tasks completed for the last {days_to_process_for_embeddings} day(s).")

    # Store Raw Weather Data
    if weather_client:
        logger.info(f"--- Starting Raw Weather DataFrame Storage for the last {days_to_store_raw_weather} day(s) ---")
        weather_storage_tasks = []
        for i in range(1, days_to_store_raw_weather + 1):
            date_for_weather_storage = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            weather_storage_tasks.append(
                fetch_and_store_weather_data(weather_client, pg_vector_storage, date_for_weather_storage)
            )
        if weather_storage_tasks:
            await asyncio.gather(*weather_storage_tasks)
        logger.info(f"Raw weather DataFrame storage tasks completed for the last {days_to_store_raw_weather} day(s).")
    else:
        logger.info("Skipping raw weather DataFrame storage as weather client is not initialized.")
    
    # Graceful shutdown
    if hasattr(ercot_client, 'auth_manager') and hasattr(ercot_client.auth_manager, 'shutdown'):
        ercot_client.auth_manager.shutdown()
        logger.info("ERCOT Auth Manager shutdown called.")
    
    pg_vector_storage.close_db_connection()
    logger.info("Database connections closed.")
    logger.info("Data ingestion pipeline finished.")

if __name__ == "__main__":
    logger.info("Starting main.py script...")
    try:
        asyncio.run(main_ingestion_pipeline())
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled error in main: {e}", exc_info=True)
    finally:
        logger.info("Main script finished.")
