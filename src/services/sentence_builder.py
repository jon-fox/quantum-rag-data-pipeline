import logging
from typing import Optional, Dict, Callable, Awaitable, Any # Added Callable, Awaitable, Any
import numpy as np

# Corrected relative imports
from src.data.ercot_api.queries import ERCOTQueries
from src.data.weather_api.weather import WeatherAPIClient
from src.storage.pgvector_storage import PgVectorStorage
from src.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

def create_semantic_sentence(
    date_from: str,
    date_to: str,
    ercot_metrics: Optional[Dict[str, Any]],
    weather_metrics: Optional[Dict[str, float]]
) -> Optional[str]:
    """Constructs a semantic sentence from ERCOT and weather data for a given date."""

    if ercot_metrics is None and weather_metrics is None:
        logger.warning(f"Missing both ERCOT and weather data for {date_from} to {date_to}, cannot create semantic sentence.")
        return None
    
    ercot_part = ""
    if ercot_metrics:
        # Extract all metrics for comprehensive sentence
        gen_metrics = ercot_metrics.get('generation_summary', {}).get('metrics', {})
        load_metrics = ercot_metrics.get('load_summary', {}).get('metrics', {})
        output_metrics = ercot_metrics.get('output_schedule', {}).get('metrics', {})
        dsr_metrics = ercot_metrics.get('dsr_loads', {}).get('metrics', {})
        ancillary_metrics = ercot_metrics.get('ancillary_ecrss', {}).get('metrics', {})
        
        sentence_parts = []
        
        # Total Load & Generation section - now using daily averages
        if load_metrics and gen_metrics:
            agg_load = load_metrics.get('aggLoadSummary', 0)  # Already in MW, daily average
            telem_gen = load_metrics.get('sumTelemGenMW', 0)  # Already in MW, daily average
            sentence_parts.append(f"ERCOT reported a daily average aggregated system load of {agg_load:.0f} MW and telemetry generation of {telem_gen:.0f} MW")
        
        # Ancillary Services section - now using maximum values
        if ancillary_metrics:
            mw_offered = ancillary_metrics.get('MWOffered', 0)  # Already in MW, maximum value
            if mw_offered > 0:
                sentence_parts.append(f"ECRSS offers peaked at {mw_offered:.0f} MW with pricing data recorded")
        
        # Demand Side Resources section - now using daily averages
        if dsr_metrics:
            dsr_load = dsr_metrics.get('sumTelemDSRLoad', 0)  # Already in MW, daily average
            dsr_gen = dsr_metrics.get('sumTelemDSRGen', 0)  # Already in MW, daily average
            dsr_parts = []
            if dsr_load > 0:
                dsr_parts.append(f"DSR loads averaged {dsr_load:.0f} MW")
            if dsr_gen > 0:
                dsr_parts.append(f"DSR generation averaged {dsr_gen:.0f} MW")
            if dsr_parts:
                sentence_parts.append(", ".join(dsr_parts))
        
        # Forecast Accuracy section - now using daily averages
        if output_metrics and gen_metrics:
            output_sched = output_metrics.get('sumOutputSched', 0)  # Already in MW, daily average
            lsl_output = output_metrics.get('sumLSLOutputSched', 0)  # Already in MW, daily average
            hsl_output = output_metrics.get('sumHSLOutputSched', 0)  # Already in MW, daily average
            base_point = gen_metrics.get('sumBasePointNonIRR', 0)  # Already in MW, daily average
            hasl = gen_metrics.get('sumHASLNonIRR', 0)  # Already in MW, daily average
            lasl = gen_metrics.get('sumLASLNonIRR', 0)  # Already in MW, daily average
            
            forecast_parts = []
            if output_sched > 0:
                forecast_parts.append(f"Output schedules averaged {output_sched:.0f} MW")
            if lsl_output > 0 and hsl_output > 0:
                forecast_parts.append(f"with average lower and upper bounds at {lsl_output:.0f} MW and {hsl_output:.0f} MW respectively")
            if base_point > 0:
                forecast_parts.append(f"Basepoint generation averaged {base_point:.0f} MW")
            if hasl > 0 and lasl > 0:
                forecast_parts.append(f"with average high and low sustainable limits at {hasl:.0f} MW and {lasl:.0f} MW")
            
            if forecast_parts:
                sentence_parts.append(", ".join(forecast_parts))
        
        if sentence_parts:
            ercot_part = ". ".join(sentence_parts) + "."
    else:
        logger.warning(f"Missing ERCOT data for {date_from} to {date_to}. Sentence will not include ERCOT data.")

    weather_part = ""
    if weather_metrics:
        # Ensure all expected keys are present, provide defaults if not critical, or handle missing ones.
        # The format_temp helper handles None by returning "N/A".
        def format_temp(temp_value):
            if isinstance(temp_value, (int, float)) and not (isinstance(temp_value, float) and np.isnan(temp_value)):
                return f"{temp_value:.1f}°C"
            return "N/A"

        avg_temp_str = format_temp(weather_metrics.get('avg_temp_c'))
        houston_temp_str = format_temp(weather_metrics.get('houston_temp_c'))
        austin_temp_str = format_temp(weather_metrics.get('austin_temp_c'))
        dallas_temp_str = format_temp(weather_metrics.get('dallas_temp_c'))
        san_antonio_temp_str = format_temp(weather_metrics.get('san_antonio_temp_c'))
        fort_worth_temp_str = format_temp(weather_metrics.get('fort_worth_temp_c'))
        corpus_christi_temp_str = format_temp(weather_metrics.get('corpus_christi_temp_c'))

        weather_part = (
            f"On {date_from}, the average temperature across Texas was {avg_temp_str}. "
            f"Average temperatures across major Texas cities on the ERCOT grid were as follows — "
            f"Houston: {houston_temp_str}, Austin: {austin_temp_str}, "
            f"Dallas: {dallas_temp_str}, San Antonio: {san_antonio_temp_str}, "
            f"Fort Worth: {fort_worth_temp_str}, Corpus Christi: {corpus_christi_temp_str}."
        )
    else:
        logger.warning(f"Missing weather data for {date_from} to {date_to}. Sentence will not include weather data.")

    if not ercot_part and not weather_part:
        logger.error(f"Both ERCOT and weather parts are empty for {date_from} to {date_to} despite earlier checks. Cannot create sentence.")
        return None
        
    sentence_parts = [f"Between {date_from} to {date_to},"]
    if ercot_part:
        sentence_parts.append(ercot_part)
    if weather_part:
        sentence_parts.append(weather_part)
    
    return " ".join(sentence_parts)

async def process_and_embed_daily_summary(
    date_to_start_process: str, 
    date_to_end_process: str, 
    ercot_queries: ERCOTQueries, 
    weather_client: Optional[WeatherAPIClient], 
    pg_storage: PgVectorStorage, 
    embedding_service: Optional[EmbeddingService],
    fetch_ercot_metric_func: Callable[[ERCOTQueries, str, str], Awaitable[Optional[Dict[str, Any]]]],
    fetch_weather_metrics_func: Callable[[WeatherAPIClient, str], Awaitable[Optional[Dict[str, float]]]]
):
    """Orchestrates fetching, processing, sentence creation, embedding, and storage for a single day."""
    logger.info(f"Processing daily summary for embedding on {date_to_start_process} to {date_to_end_process} (via sentence_builder)...")

    ercot_metrics = await fetch_ercot_metric_func(ercot_queries, date_to_start_process, date_to_end_process)
    
    weather_metrics = None
    if weather_client:
        weather_metrics = await fetch_weather_metrics_func(weather_client, date_to_start_process)
    else:
        logger.info(f"Weather client not available, skipping weather metrics for {date_to_start_process} to {date_to_end_process}.")

    if ercot_metrics is None:
        logger.warning(f"Could not retrieve ERCOT metrics for {date_to_start_process} to {date_to_end_process}. Skipping embedding.")
        return
    
    if weather_client and weather_metrics is None:
        logger.warning(f"Could not retrieve Weather metrics for {date_to_start_process} to {date_to_end_process} (weather client was available).")

    semantic_sentence = create_semantic_sentence(date_to_start_process, date_to_end_process, ercot_metrics, weather_metrics)

    if not semantic_sentence:
        logger.warning(f"Semantic sentence could not be created for {date_to_start_process} to {date_to_end_process}. Skipping embedding.")
        return

    logger.info(f"Generated semantic sentence for {date_to_start_process} to {date_to_end_process}: \"{semantic_sentence}\"")

    if not embedding_service:
        logger.warning(f"Embedding service not available. Cannot generate embedding for {date_to_start_process} to {date_to_end_process}.")
        return
        
    try:
        embedding_list = embedding_service.generate_embedding(semantic_sentence)
        if embedding_list is None:
            logger.error(f"Embedding generation failed for {date_to_start_process} to {date_to_end_process}, received None.")
            return
        
        embedding_array = np.array(embedding_list).astype(np.float32)
        vector_id = f"daily_summary_{date_to_start_process}" 
        
        store_success = pg_storage.store_embedding(vector_id=vector_id, embedding=embedding_array, semantic_sentence=semantic_sentence)
        if store_success:
            logger.info(f"Successfully stored combined daily embedding for {date_to_start_process} to {date_to_end_process} with ID {vector_id}.")
        else:
            logger.error(f"Failed to store combined daily embedding for {date_to_start_process} to {date_to_end_process}.")
    except Exception as e:
        logger.error(f"Error generating or storing embedding for {date_to_start_process} to {date_to_end_process}: {e}", exc_info=True)
