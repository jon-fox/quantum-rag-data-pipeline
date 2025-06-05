import logging
import asyncio
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Optional, Dict, Any


from src.config.env_manager import load_environment, get_env_var
from src.data.ercot_api.client import ERCOTClient
from src.data.ercot_api.queries import ERCOTQueries
from src.data.weather_api.meteostat_weather import get_ercot_avg_temperature
from src.storage.pgvector_storage import PgVectorStorage
from src.services.embedding_service import EmbeddingService

from src.services.sentence_builder import process_and_embed_daily_summary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ENABLE_ERCOT_LOGGING = False

async def fetch_daily_weather_metrics(date_to_fetch: str) -> Optional[Dict[str, float]]:
    """Fetches and processes weather data to get key daily temperature metrics using meteostat."""
    logger.info(f"Fetching weather metrics for {date_to_fetch}...")
    try:
        avg_temp = get_ercot_avg_temperature(date_to_fetch)
        if avg_temp is None:
            logger.info(f"No weather data found for {date_to_fetch}.")
            return None

        metrics = {
            "avg_temp_c": avg_temp,
        }
        
        logger.info(f"Processed weather metrics for {date_to_fetch}: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error fetching/processing weather daily metrics for {date_to_fetch}: {e}", exc_info=True)
        return None



async def fetch_all_ercot_data(ercot_queries: ERCOTQueries, date_from: str, date_to: str, enable_logging: bool = False) -> Dict[str, Any]:
    """Fetches data from all ERCOT endpoints and extracts relevant metrics."""
    logger.info(f"Fetching all ERCOT data for {date_from} to {date_to}")
    
    all_data = {}
    
    def extract_field_values(data_dict: Dict, field_configs: list) -> Dict[str, float]:
        """Extract values for specified fields from ERCOT API response with configurable aggregation.
        
        Args:
            data_dict: ERCOT API response dictionary
            field_configs: List of tuples (field_name, aggregation_method) where 
                          aggregation_method is 'average', 'max', or 'sum'
        """
        records = data_dict.get('data', [])
        fields = data_dict.get('fields', [])
        
        if not records or not fields:
            return {}
        
        # Create field name to index mapping
        field_mapping = {field['name']: i for i, field in enumerate(fields)}
        
        extracted_values = {}
        for field_name, aggregation_method in field_configs:
            if field_name in field_mapping:
                field_index = field_mapping[field_name]
                values = []
                for record in records:
                    if len(record) > field_index:
                        try:
                            value = float(record[field_index])
                            values.append(value)
                        except (ValueError, TypeError):
                            continue
                
                if values:
                    if aggregation_method == 'average':
                        extracted_values[field_name] = sum(values) / len(values)
                    elif aggregation_method == 'max':
                        extracted_values[field_name] = max(values)
                    elif aggregation_method == 'sum':
                        extracted_values[field_name] = sum(values)
                    else:
                        extracted_values[field_name] = sum(values)  # Default to sum
                else:
                    extracted_values[field_name] = 0.0
        
        return extracted_values
    
    try:
        # Generation Summary - use averages for basepoint and sustainable limits
        gen_data = ercot_queries.get_agg_gen_summary(
            delivery_date_from_override=date_from, 
            delivery_date_to_override=date_to
        )
        gen_values = extract_field_values(gen_data, [
            ('sumBasePointNonIRR', 'average'),
            ('sumHASLNonIRR', 'average'),
            ('sumLASLNonIRR', 'average'),
            ('sumBasePointWGR', 'sum'),
            ('sumBasePointPVGR', 'sum'),
            ('sumBasePointREMRES', 'sum')
        ])
        all_data['generation_summary'] = {
            'raw_data': gen_data,
            'metrics': gen_values
        }
        if enable_logging:
            logger.info(f"Generation Summary data structure: {json.dumps(gen_data, indent=2, default=str)[:500]}...")
            logger.info(f"Generation metrics: {gen_values}")
        
        # Load Summary - use averages for load and telemetry generation
        load_data = ercot_queries.get_aggregated_load_summary(
            delivery_date_from_override=date_from,
            delivery_date_to_override=date_to
        )
        load_values = extract_field_values(load_data, [
            ('aggLoadSummary', 'average'),
            ('sumTelemGenMW', 'average')
        ])
        all_data['load_summary'] = {
            'raw_data': load_data,
            'metrics': load_values
        }
        if enable_logging:
            logger.info(f"Load Summary data structure: {json.dumps(load_data, indent=2, default=str)[:500]}...")
            logger.info(f"Load metrics: {load_values}")
        
        # Output Schedule - use averages for all output schedule metrics
        output_data = ercot_queries.get_agg_output_summary(
            delivery_date_from_override=date_from,
            delivery_date_to_override=date_to
        )
        
        output_values = extract_field_values(output_data, [
            ('sumOutputSched', 'average'),
            ('sumLSLOutputSched', 'average'),
            ('sumHSLOutputSched', 'average')
        ])
        all_data['output_schedule'] = {
            'raw_data': output_data,
            'metrics': output_values
        }
        if enable_logging:
            logger.info(f"Output Schedule data structure: {json.dumps(output_data, indent=2, default=str)[:500]}...")
            logger.info(f"Output metrics: {output_values}")
        
        # DSR Loads - use averages for DSR load and generation
        dsr_data = ercot_queries.get_aggregated_dsr_loads(
            delivery_date_from_override=date_from,
            delivery_date_to_override=date_to
        )

        dsr_values = extract_field_values(dsr_data, [
            ('sumTelemDSRLoad', 'average'),
            ('sumTelemDSRGen', 'average')
        ])
        all_data['dsr_loads'] = {
            'raw_data': dsr_data,
            'metrics': dsr_values
        }
        if enable_logging:
            logger.info(f"DSR Loads data structure: {json.dumps(dsr_data, indent=2, default=str)[:500]}...")
            logger.info(f"DSR metrics: {dsr_values}")
        
        # Key Ancillary Services
        for service in ["ecrss"]:
            try:
                service_data = ercot_queries.get_ancillary_service_offers(
                    service_type=service,
                    delivery_date_from_override=date_from,
                    delivery_date_to_override=date_to
                )
                # ECRSS - use maximum for MWOffered, average for price
                service_values = extract_field_values(service_data, [
                    ('MWOffered', 'max'),
                    ('ECRSSOfferPrice', 'average')
                ])
                all_data[f'ancillary_{service}'] = {
                    'raw_data': service_data,
                    'metrics': service_values
                }
                if enable_logging:
                    logger.info(f"Ancillary Service {service} data structure: {json.dumps(service_data, indent=2, default=str)[:500]}...")
                    logger.info(f"Ancillary {service} metrics: {service_values}")
            except Exception as e:
                logger.warning(f"Failed to fetch {service} data: {e}")
        
        # DAM Settlement Point Prices - HubAvg
        try:
            dam_data = ercot_queries.get_dam_settlement_point_prices(
                settlement_point="HB_HUBAVG",
                delivery_date_from_override=date_from,
                delivery_date_to_override=date_to
            )
            
            # Extract settlement point prices and compute average
            dam_values = extract_field_values(dam_data, [
                ('settlementPointPrice', 'average')
            ])
            
            avg_price = round(dam_values.get('settlementPointPrice', 0), 2) if dam_values.get('settlementPointPrice') else None
            all_data['dam_hubavg_price'] = {'avg_price': avg_price}
            
            if enable_logging:
                logger.info(f"DAM HubAvg Price data structure: {json.dumps(dam_data, indent=2, default=str)[:500]}...")
                logger.info(f"DAM HubAvg Price: average price {avg_price}")
                
        except Exception as e:
            logger.warning(f"Failed to fetch DAM HubAvg price data: {e}")
            all_data['dam_hubavg_price'] = {'avg_price': None}
        
        if enable_logging:
            logger.info(f"Complete ERCOT data summary:")
            for key, value in all_data.items():
                if isinstance(value, dict) and 'raw_data' in value:
                    raw_data = value['raw_data']
                    record_count = len(raw_data.get('data', []))
                    field_count = len(raw_data.get('fields', []))
                    metrics_count = len(value.get('metrics', {}))
                    logger.info(f"  {key}: {record_count} records, {field_count} fields, {metrics_count} metrics extracted")
                
        logger.info(f"Successfully fetched {len(all_data)} ERCOT datasets")
        return all_data
        
    except Exception as e:
        logger.error(f"Error fetching ERCOT data: {e}")
        return all_data

async def fetch_ercot_metrics_for_embedding(ercot_queries: ERCOTQueries, date_from: str, date_to: str) -> Optional[Dict[str, Any]]:
    """Wrapper function to fetch ERCOT data for embedding pipeline."""
    return await fetch_all_ercot_data(ercot_queries, date_from, date_to, enable_logging=False)

async def main_ingestion_pipeline(date_from: Optional[str] = None, date_to: Optional[str] = None):
    """Main function to orchestrate the data ingestion pipeline.
    
    Args:
        date_from: Start date for ERCOT data fetching (YYYY-MM-DD). Defaults to yesterday.
        date_to: End date for ERCOT data fetching (YYYY-MM-DD). Defaults to today.
    """
    logger.info("Initializing data ingestion pipeline...")
    
    # Set default dates if not provided
    today = datetime.today()
    if date_from is None:
        date_from = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    if date_to is None:
        date_to = today.strftime('%Y-%m-%d')
        
    logger.info(f"Using date range: {date_from} to {date_to}")
    
    load_environment()

    ercot_client = ERCOTClient()
    ercot_queries = ERCOTQueries(client=ercot_client, delivery_date_from=date_from, delivery_date_to=date_to)
    
    # Weather is now handled by meteostat, no API key needed
    logger.info("Using meteostat for weather data (no API key required)")

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

    # Process and Embed Combined Daily Summaries
    logger.info(f"--- Starting Combined Daily Summary Embedding for date range: {date_from} to {date_to} ---")
    embedding_tasks = []
    
    # Parse the date range from the passed parameters
    start_date_obj = datetime.strptime(date_from, '%Y-%m-%d')
    end_date_obj = datetime.strptime(date_to, '%Y-%m-%d')
    
    # Process each day in the specified range
    current_date = start_date_obj
    while current_date <= end_date_obj:
        previous_day = current_date - timedelta(days=1)
        embedding_tasks.append(
            process_and_embed_daily_summary( 
                date_to_start_process=previous_day.strftime('%Y-%m-%d'),
                date_to_end_process=current_date.strftime('%Y-%m-%d'), 
                ercot_queries=ercot_queries,
                pg_storage=pg_vector_storage,
                embedding_service=embedding_service,
                fetch_ercot_metric_func=fetch_ercot_metrics_for_embedding, # Pass the wrapper function
                fetch_weather_metrics_func=fetch_daily_weather_metrics # Pass local function
            )
        )
        current_date += timedelta(days=1)
    
    if embedding_tasks:
        await asyncio.gather(*embedding_tasks)
    logger.info(f"Combined daily summary embedding tasks completed for date range: {date_from} to {date_to}.")

    # Graceful shutdown
    if hasattr(ercot_client, 'auth_manager') and hasattr(ercot_client.auth_manager, 'shutdown'):
        ercot_client.auth_manager.shutdown()
        logger.info("ERCOT Auth Manager shutdown called.")
    
    pg_vector_storage.close_db_connection()
    logger.info("Database connections closed.")
    logger.info("Data ingestion pipeline finished.")

if __name__ == "__main__":
    ENABLE_ERCOT_LOGGING = True
    
    if ENABLE_ERCOT_LOGGING:
        logger.info("ERCOT data structure logging is ENABLED")
    
    # Configuration for date range processing
    # Set your desired date range here (YYYY-MM-DD format)
    START_DATE = "2025-05-01"  # Modify this to your desired start date
    END_DATE = "2025-06-01"    # Modify this to your desired end date
    
    # Parse start and end dates
    start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
    
    # Validate date range
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        exit(1)
    
    logger.info(f"Starting main.py script with date range: {START_DATE} to {END_DATE}")
    logger.info("Processing in 2-day increments with 1-day overlap")
    
    # Process date range in 2-day increments with 1-day overlap
    current_start = start_date
    increment_count = 1
    
    try:
        while current_start < end_date:
            # Calculate end date for current increment (2 days from start, but not beyond final end date)
            current_end = min(current_start + timedelta(days=1), end_date)
            
            # Format dates for processing
            date_from = current_start.strftime('%Y-%m-%d')
            date_to = current_end.strftime('%Y-%m-%d')
            
            logger.info(f"--- Processing Increment {increment_count}: {date_from} to {date_to} ---")
            
            # Run the main ingestion pipeline for this increment
            asyncio.run(main_ingestion_pipeline(date_from, date_to))
            
            logger.info(f"--- Completed Increment {increment_count}: {date_from} to {date_to} ---")
            
            # Move to next increment (1-day overlap: last day becomes first day of next increment)
            current_start = current_end
            increment_count += 1
            
            # Break if we've reached the end date
            if current_end >= end_date:
                break
                
        logger.info(f"Completed processing all increments. Total increments processed: {increment_count - 1}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled error in main: {e}", exc_info=True)
    finally:
        logger.info("Main script finished.")
