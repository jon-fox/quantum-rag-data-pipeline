import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta

# Absolute imports from 'src' (assuming 'src' is effectively a root for imports
# or the parent of 'src' is in PYTHONPATH when running)
from src.config.env_manager import load_environment, get_env_var
from src.data.ercot_api.client import ERCOTClient
from src.data.ercot_api.queries import ERCOTQueries
from src.data.weather_api.weather import WeatherAPIClient
from src.storage.pgvector_storage import PgVectorStorage
from src.storage.dual_storage import DualStorage
# Assuming you will have an embedding model utility
# from src.services.embedding_service import EmbeddingService 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_and_store_ercot_data(ercot_queries: ERCOTQueries, dual_storage: DualStorage):
    """Fetches various ERCOT data and stores it."""
    logger.info("Starting ERCOT data fetch and store process...")
    try:
        # Example: Get 2-Day Aggregated Generation Summary
        # Dates can be parameterized or made dynamic
        # Default dates in ERCOTQueries are typically yesterday to today.
        gen_summary = ercot_queries.get_agg_gen_summary()
        # The actual data is usually in a sub-key like 'data' or 'docs'
        # Check the ERCOT API response structure for the correct key.
        # Example: records = gen_summary.get('publicReportList', {}).get('list', [])
        records = gen_summary.get('data', []) # Placeholder, adjust based on actual API response
        logger.info(f"Fetched {len(records)} generation summary records.")
        
        # Conceptual: Process and store gen_summary
        # for record in records:
        #     document_id = f"ercot_gen_{record.get('id_field_or_hash')}" 
        #     document_content_for_embedding = json.dumps(record) # Or a more structured text representation
        #     # embedding = embedding_service.generate_embedding(document_content_for_embedding)
        #     embedding = np.random.rand(1536).astype(np.float32) # Placeholder embedding
        #     metadata = {
        #         "document_id": document_id,
        #         "doc_type": "ercot_generation_summary",
        #         "source": "ERCOT API",
        #         "data_timestamp": record.get('timestamp_field'), 
        #         "raw_data_snippet": str(record)[:200] # Store a snippet or full if needed
        #     }
        #     store_result = dual_storage.store_document_and_embedding(metadata, embedding, vector_id=document_id)
        #     if store_result["overall_success"]:
        #         logger.info(f"Successfully stored ERCOT gen record: {document_id}")
        #     else:
        #         logger.error(f"Failed to store ERCOT gen record: {document_id}. Error: {store_result}")

        # Add similar processing for other ERCOT endpoints like load summary, ancillary services
        logger.info("ERCOT data fetch and store process completed (conceptual).")
    except Exception as e:
        logger.error(f"Error during ERCOT data processing: {e}", exc_info=True)

async def fetch_and_store_weather_data(weather_client: WeatherAPIClient, pg_storage: PgVectorStorage):
    """Fetches historical weather data and stores it in PostgreSQL (non-vector table)."""
    logger.info("Starting weather data fetch and store process...")
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        locations = {
            # Names used in DataFrame columns should match table columns if inserting directly
            "houston_temp_c": "Houston,TX,USA", 
            "austin_temp_c": "Austin,TX,USA",
            "dallas_temp_c": "Dallas,TX,USA"
        }
        # get_historical_weather returns a DataFrame with columns: time, houston_temp_c, austin_temp_c, dallas_temp_c, avg_temperature_c, avg_temperature_f
        weather_df = weather_client.get_historical_weather(date_str=yesterday, locations=locations)
        
        if not weather_df.empty:
            logger.info(f"Fetched {len(weather_df)} weather records for {yesterday}.")
            # Storing weather_df in 'historical_weather_data' table
            # This requires a method in PgVectorStorage or direct DB interaction to insert a pandas DataFrame.
            # The `create_weather_table.py` script sets up the table with columns:
            # timestamp, houston_temp_c, austin_temp_c, dallas_temp_c, avg_temperature_c, avg_temperature_f
            
            # Conceptual: Convert DataFrame to list of tuples for insertion
            # Ensure DataFrame columns match table structure or are selected/renamed.
            # Example columns needed for historical_weather_data: 
            # timestamp, houston_temp_c, austin_temp_c, dallas_temp_c, avg_temperature_c, avg_temperature_f
            
            # Rename 'time' to 'timestamp' to match table schema if needed
            if 'time' in weather_df.columns:
                 weather_df.rename(columns={'time': 'timestamp'}, inplace=True)
            
            # Select only columns that exist in the table
            table_columns = ['timestamp', 'houston_temp_c', 'austin_temp_c', 'dallas_temp_c', 'avg_temperature_c', 'avg_temperature_f']
            df_to_insert = weather_df[[col for col in table_columns if col in weather_df.columns]]

            # pg_storage.insert_dataframe_to_table(df_to_insert, 'historical_weather_data') # Conceptual
            logger.info("Weather data processed (actual storage implementation for DataFrame pending in PgVectorStorage).")
        else:
            logger.info(f"No weather data fetched for {yesterday}.")
            
    except Exception as e:
        logger.error(f"Error during weather data processing: {e}", exc_info=True)

async def main_ingestion_pipeline():
    """Main function to orchestrate the data ingestion pipeline."""
    logger.info("Initializing data ingestion pipeline...")
    
    load_environment()
    
    ercot_client = ERCOTClient()
    ercot_queries = ERCOTQueries(client=ercot_client)
    
    weather_api_key = get_env_var("WEATHER_API_KEY")
    weather_client = None
    if not weather_api_key:
        logger.warning("WEATHER_API_KEY not found. Weather data ingestion will be skipped.")
    else:
        weather_client = WeatherAPIClient(api_key=weather_api_key)

    # PgVectorStorage will load DB params from env by default.
    # lazy_init=False ensures schema (vector extension, table) is checked/created.
    pg_vector_storage = PgVectorStorage(lazy_init=False, app_environment=get_env_var("APP_ENVIRONMENT", "dev"))
    
    # DualStorage will initialize its own PgVectorStorage if not provided,
    # or use the one provided. Here, we let it create its own to simplify, 
    # ensuring it also gets app_environment if needed for SSM config.
    # It also initializes DynamoDBStorage internally.
    # Ensure DYNAMODB_TABLE env var is set for DynamoDBStorage in DualStorage.
    # And APP_ENVIRONMENT for PgVectorStorage if it loads config from SSM.
    dual_storage = DualStorage(app_environment=get_env_var("APP_ENVIRONMENT", "dev"))

    # Conceptual: Initialize Embedding Service
    # openai_api_key = get_env_var("OPENAI_API_KEY")
    # if openai_api_key:
    #     # embedding_service = EmbeddingService(api_key=openai_api_key)
    #     logger.info("Embedding service initialized (conceptual).")
    # else:
    #     logger.warning("OPENAI_API_KEY not found. Embedding-dependent tasks will fail or use placeholders.")
        # embedding_service = None 

    logger.info("--- Starting Data Ingestion Tasks ---")
    
    # Run ERCOT data processing (conceptual, needs embedding service and data transformation)
    # await fetch_and_store_ercot_data(ercot_queries, dual_storage) # Pass embedding_service if used
    logger.info("Conceptual ERCOT data processing call - full implementation for transformation, embedding, and storage needed.")

    if weather_client:
        # await fetch_and_store_weather_data(weather_client, pg_vector_storage)
        logger.info("Conceptual Weather data processing call - full implementation for DataFrame to DB table insertion needed.")
    else:
        logger.info("Skipping weather data ingestion as client is not initialized.")

    logger.info("Data ingestion pipeline tasks initiated (conceptual calls).")
    
    # Graceful shutdown of resources if they have explicit close/shutdown methods
    if hasattr(ercot_client, 'auth_manager') and hasattr(ercot_client.auth_manager, 'shutdown'):
        ercot_client.auth_manager.shutdown()
        logger.info("ERCOT Auth Manager shutdown called.")
    
    pg_vector_storage.close_connection() # Close connection managed by the instance we created
    dual_storage.close_connections() # Closes its internally managed PgVectorStorage connection
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
