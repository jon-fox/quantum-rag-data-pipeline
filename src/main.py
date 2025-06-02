import logging
import asyncio
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Optional, Dict


from src.config.env_manager import load_environment, get_env_var
from src.data.ercot_api.client import ERCOTClient
from src.data.ercot_api.queries import ERCOTQueries
from src.data.weather_api.weather import WeatherAPIClient
from src.storage.pgvector_storage import PgVectorStorage
from src.services.embedding_service import EmbeddingService

from src.services.sentence_builder import process_and_embed_daily_summary 
from src.config.constants import LOCATIONS # Import LOCATIONS from constants.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# create_semantic_sentence function is now in src.services.sentence_builder.py
# process_and_embed_daily_summary is now in src.services.sentence_builder.py

async def fetch_daily_ercot_metric(ercot_queries: ERCOTQueries, date_to_start_fetch: str, date_to_end_fetch: str) -> Optional[float]:
    """Fetches and processes ERCOT data to get a single daily generation metric."""
    logger.info(f"Starting fetch_daily_ercot_metric for date: {date_to_start_fetch} to {date_to_end_fetch}")
    
    try:
        logger.info(f"Calling ercot_queries.get_agg_gen_summary with date range: {date_to_start_fetch} to {date_to_end_fetch}")
        gen_summary_response = ercot_queries.get_agg_gen_summary(
            delivery_date_from_override=date_to_start_fetch, 
            delivery_date_to_override=date_to_end_fetch
        )
        
        logger.info(f"API response received. Type: {type(gen_summary_response)}")
        if isinstance(gen_summary_response, dict):
            logger.info(f"Response keys: {list(gen_summary_response.keys())}")
        

        # Extract data and field mappings
        records = gen_summary_response.get('data', [])
        fields = gen_summary_response.get('fields', [])
        
        logger.info(f"Found {len(records)} records in response data")
        
        if not records:
            logger.warning(f"No ERCOT generation records found for {date_to_start_fetch} to {date_to_end_fetch}")
            return None
        
        # Create field name to index mapping from the fields metadata
        field_mapping = {}
        for i, field in enumerate(fields):
            field_mapping[field['name']] = i
        
        logger.info(f"Available fields: {list(field_mapping.keys())}")
        
        # Look for generation field - "sumGenTelemMW" is the total generation field
        generation_field_index = field_mapping.get('sumGenTelemMW')
        if generation_field_index is None:
            logger.error(f"Could not find 'sumGenTelemMW' field in ERCOT data. Available fields: {list(field_mapping.keys())}")
            return None
        
        logger.info(f"Found generation field 'sumGenTelemMW' at index {generation_field_index}")
        
        # Sum all generation values across time intervals
        total_generation_for_day = 0.0
        valid_records = 0
        
        for idx, record in enumerate(records):
            if len(record) > generation_field_index:
                try:
                    generation_value = float(record[generation_field_index])
                    total_generation_for_day += generation_value
                    valid_records += 1
                    logger.debug(f"Record {idx + 1}: Added {generation_value} MW to total")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Record {idx + 1}: Could not convert generation value '{record[generation_field_index]}' to float. Error: {e}")
            else:
                logger.warning(f"Record {idx + 1}: Not enough fields ({len(record)} vs expected {generation_field_index + 1})")
        
        if valid_records == 0:
            logger.error(f"No valid generation records found for {date_to_start_fetch} to {date_to_end_fetch}")
            return None

        logger.info(f"Successfully processed ERCOT generation metric for {date_to_start_fetch} to {date_to_end_fetch}: {total_generation_for_day} MW from {valid_records} records")
        return float(total_generation_for_day)

    except Exception as e:
        logger.error(f"Exception in fetch_daily_ercot_metric for {date_to_start_fetch} to {date_to_end_fetch}: {type(e).__name__}: {e}")
        logger.error(f"Exception details:", exc_info=True)
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

    pg_vector_storage = PgVectorStorage(app_environment=get_env_var("APP_ENVIRONMENT", "prod"))
    
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
        target_date = today - timedelta(days=i)
        previous_day = target_date - timedelta(days=1)
        embedding_tasks.append(
            process_and_embed_daily_summary( 
                date_to_start_process=previous_day.strftime('%Y-%m-%d'),
                date_to_end_process=target_date.strftime('%Y-%m-%d'), 
                ercot_queries=ercot_queries,
                weather_client=weather_client,
                pg_storage=pg_vector_storage, 
                embedding_service=embedding_service,
                fetch_ercot_metric_func=fetch_daily_ercot_metric, # Pass local function
                fetch_weather_metrics_func=fetch_daily_weather_metrics # Pass local function
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
