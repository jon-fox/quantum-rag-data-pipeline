import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import json
import numpy as np

# Absolute imports from 'src'
from .config.env_manager import load_environment, get_env_var
from .data.ercot_api.client import ERCOTClient
from .data.ercot_api.queries import ERCOTQueries
from .data.weather_api.weather import WeatherAPIClient
from .storage.pgvector_storage import PgVectorStorage
from .services.embedding_service import EmbeddingService

LOCATIONS = {
    "houston_temp_c": "Houston,TX,USA", 
    "austin_temp_c": "Austin,TX,USA",
    "dallas_temp_c": "Dallas,TX,USA"
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_and_store_ercot_data(ercot_queries: ERCOTQueries, pg_storage: PgVectorStorage, date_to_fetch: str, embedding_service=None):
    """Fetches ERCOT data for a specific date, generates embeddings, and stores it."""
    logger.info(f"Starting ERCOT data fetch and store process for date: {date_to_fetch}...")
    try:
        # Fetch data for the specific single day by setting both from and to the same date.
        gen_summary = ercot_queries.get_agg_gen_summary(
            delivery_date_from_override=date_to_fetch, 
            delivery_date_to_override=date_to_fetch
        )
        records = gen_summary.get('data', []) 
        logger.info(f"Fetched {len(records)} generation summary records for {date_to_fetch}.")
        
        if not records:
            logger.info("No generation summary records to process.")
            return

        for record in records:
            # Ensure 'id_field_or_hash' and 'timestamp_field' are actual keys in your record or use appropriate fallbacks
            record_id_key = record.get('id_field_or_hash', str(hash(json.dumps(record)))) # Fallback for ID
            document_id = f"ercot_gen_{record_id_key}"
            document_content_for_embedding = json.dumps(record) 
            
            if embedding_service:
                embedding = embedding_service.generate_embedding(document_content_for_embedding)
            else:
                logger.warning(f"Embedding service not available for record {document_id}. Using placeholder embedding.")
                embedding = np.random.rand(1536).astype(np.float32)
            
            # Metadata is generated but not stored with PgVectorStorage alone after DualStorage removal.
            # If metadata storage is needed, PgVectorStorage or another mechanism would need to handle it.
            metadata = {
                "document_id": document_id,
                "doc_type": "ercot_generation_summary",
                "source": "ERCOT API",
                "data_timestamp": record.get('timestamp_field', datetime.now().isoformat()), # Fallback for timestamp
                "raw_data_snippet": str(record)[:200] 
            }
            logger.debug(f"Generated metadata for {document_id} (currently not stored): {metadata}")

            store_success = pg_storage.store_embedding(vector_id=document_id, embedding=embedding)
            if store_success:
                logger.info(f"Successfully stored ERCOT gen embedding: {document_id}")
            else:
                logger.error(f"Failed to store ERCOT gen embedding: {document_id}")

        # TODO: Add similar processing for other ERCOT endpoints (load summary, ancillary services)
        logger.info("ERCOT data fetch and store process completed.")
    except Exception as e:
        logger.error(f"Error during ERCOT data processing: {e}", exc_info=True)

async def fetch_and_store_weather_data(weather_client: WeatherAPIClient, pg_storage: PgVectorStorage, date_to_fetch: str):
    """Fetches historical weather data for a specific date and stores it in PostgreSQL."""
    logger.info(f"Starting weather data fetch and store process for date: {date_to_fetch}...")
    try:
        # DataFrame columns: time, houston_temp_c, austin_temp_c, dallas_temp_c, avg_temperature_c, avg_temperature_f
        weather_df = weather_client.get_historical_weather(date_str=date_to_fetch, locations=LOCATIONS)
        
        if not weather_df.empty:
            logger.info(f"Fetched {len(weather_df)} weather records for {date_to_fetch}.")
            
            if 'time' in weather_df.columns:
                 weather_df.rename(columns={'time': 'timestamp'}, inplace=True) # Match table schema
            
            table_columns = ['timestamp', 'houston_temp_c', 'austin_temp_c', 'dallas_temp_c', 'avg_temperature_c', 'avg_temperature_f']
            df_to_insert = weather_df[[col for col in table_columns if col in weather_df.columns]]

            pg_storage.insert_dataframe_to_table(df_to_insert, 'historical_weather_data')
            logger.info(f"Weather data for {len(df_to_insert)} records processed and stored in 'historical_weather_data'.")
        else:
            logger.info(f"No weather data fetched for {date_to_fetch}.")
            
    except Exception as e:
        logger.error(f"Error during weather data processing: {e}", exc_info=True)

async def main_ingestion_pipeline(days_to_fetch_ercot=1, days_to_fetch_weather=1):
    """Main function to orchestrate the data ingestion pipeline."""
    logger.info("Initializing data ingestion pipeline...")
    
    load_environment()

    ercot_client = ERCOTClient()
    # Initialize ERCOTQueries without global date overrides, so methods use their defaults or passed overrides
    ercot_queries = ERCOTQueries(client=ercot_client)
    
    weather_api_key = get_env_var("WEATHER_API_KEY")
    weather_client = None
    if not weather_api_key:
        logger.warning("WEATHER_API_KEY not found. Weather data ingestion will be skipped.")
    else:
        weather_client = WeatherAPIClient(api_key=weather_api_key)

    # PgVectorStorage loads DB params from env by default. lazy_init=False ensures schema is checked/created.
    pg_vector_storage = PgVectorStorage(lazy_init=False, app_environment=get_env_var("APP_ENVIRONMENT", "prod"))
    
    # Initialize Embedding Service
    openai_api_key = get_env_var("OPENAI_API_KEY")
    embedding_service = None
    if openai_api_key:
        try:
            embedding_service = EmbeddingService(api_key=openai_api_key)
            logger.info("Embedding service initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}", exc_info=True)
            embedding_service = None # Ensure it's None if initialization fails
    else:
        logger.warning("OPENAI_API_KEY not found. Embedding-dependent tasks will use placeholder embeddings or may fail.")
    
    logger.info("--- Starting Data Ingestion Tasks ---")

    today = datetime.today()

    # Fetch ERCOT data for the configured number of past days
    for i in range(1, days_to_fetch_ercot + 1):
        date_for_ercot = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        logger.info(f"Requesting ERCOT data for: {date_for_ercot}")
        await fetch_and_store_ercot_data(ercot_queries, pg_vector_storage, date_for_ercot, embedding_service)
    logger.info(f"ERCOT data processing tasks called for the last {days_to_fetch_ercot} day(s).")

    if weather_client:
        # Fetch Weather data for the configured number of past days
        for i in range(1, days_to_fetch_weather + 1):
            date_for_weather = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            logger.info(f"Requesting Weather data for: {date_for_weather}")
            await fetch_and_store_weather_data(weather_client, pg_vector_storage, date_for_weather)
        logger.info(f"Weather data processing tasks called for the last {days_to_fetch_weather} day(s).")
    else:
        logger.info("Skipping weather data ingestion as client is not initialized.")

    logger.info("Data ingestion pipeline tasks initiated.")
    
    # Graceful shutdown
    if hasattr(ercot_client, 'auth_manager') and hasattr(ercot_client.auth_manager, 'shutdown'):
        ercot_client.auth_manager.shutdown()
        logger.info("ERCOT Auth Manager shutdown called.")
    
    pg_vector_storage.close_db_connection()
    # Removed: dual_storage.close_connections()
    logger.info("Database connections closed.")

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
