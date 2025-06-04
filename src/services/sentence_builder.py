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
    """Constructs a structured data format from ERCOT and weather data for a given date."""

    if ercot_metrics is None and weather_metrics is None:
        logger.warning(f"Missing both ERCOT and weather data for {date_from} to {date_to}, cannot create semantic sentence.")
        return None
    
    # Extract metric dictionaries
    gen_metrics = ercot_metrics.get('generation_summary', {}).get('metrics', {}) if ercot_metrics else {}
    load_metrics = ercot_metrics.get('load_summary', {}).get('metrics', {}) if ercot_metrics else {}
    output_metrics = ercot_metrics.get('output_schedule', {}).get('metrics', {}) if ercot_metrics else {}
    dsr_metrics = ercot_metrics.get('dsr_loads', {}).get('metrics', {}) if ercot_metrics else {}
    ancillary_metrics = ercot_metrics.get('ancillary_ecrss', {}).get('metrics', {}) if ercot_metrics else {}
    
    # Helper function to format metrics with N/A for missing data
    def format_metric(value, unit="MW", precision=0):
        if value is None:
            return "N/A"
        if precision == 0:
            return f"{value:.0f} {unit}"
        else:
            return f"{value:.{precision}f} {unit}"
    
    # Calculate renewable averages if data is available
    wind_sum = gen_metrics.get('sumBasePointWGR')
    solar_sum = gen_metrics.get('sumBasePointPVGR') 
    remres_sum = gen_metrics.get('sumBasePointREMRES')
    gen_total = load_metrics.get('sumTelemGenMW')
    
    wind_avg = wind_sum / 96 if wind_sum is not None else None
    solar_avg = solar_sum / 96 if solar_sum is not None else None
    remres_avg = remres_sum / 96 if remres_sum is not None else None
    
    # Calculate renewable totals and percentage only if all renewable data is available
    if wind_avg is not None and solar_avg is not None and remres_avg is not None:
        renew_avg = wind_avg + solar_avg + remres_avg
        renew_pct = (renew_avg / gen_total) * 100 if gen_total and gen_total > 0 else None
    else:
        renew_avg = None
        renew_pct = None
    
    # Handle weather temperature with NaN check
    tx_temp = None
    if weather_metrics:
        temp_value = weather_metrics.get('avg_temp_c')
        if temp_value is not None and isinstance(temp_value, (int, float)) and not (isinstance(temp_value, float) and np.isnan(temp_value)):
            tx_temp = temp_value
    
    if not ercot_metrics:
        logger.warning(f"Missing ERCOT data for {date_from} to {date_to}. Using N/A for missing metrics.")
    if not weather_metrics:
        logger.warning(f"Missing weather data for {date_from} to {date_to}. Using N/A for temperature.")

    # Build structured output matching the specified format
    output_lines = [
        "ISO: ERCOT",
        f"Date_from: {date_from}",
        f"Date_to:   {date_to}",
        f"Avg system load: {format_metric(load_metrics.get('aggLoadSummary'))}",
        f"Telemetry generation: {format_metric(gen_total)}",
    ]
    
    # Add DAM HubAvg price
    price = ercot_metrics.get('dam_hubavg_price', {}).get('avg_price') if ercot_metrics else None
    price_str = f"{price:.2f} $/MWh" if price is not None else "N/A"
    output_lines.append(f"DAM HubAvg price: {price_str}")
    
    # Renewables line - format based on data availability
    if renew_avg is not None:
        renew_pct_str = f"{renew_pct:.0f}%" if renew_pct is not None else "N/A"
        output_lines.append(f"Renewables: {format_metric(renew_avg)} (wind {format_metric(wind_avg)} | solar {format_metric(solar_avg)} | other {format_metric(remres_avg)}) ({renew_pct_str})")
    else:
        output_lines.append("Renewables: N/A")
    
    # Continue with other metrics
    output_lines.extend([
        f"ECRSS max offer: {format_metric(ancillary_metrics.get('MWOffered'))}",
        f"DSR load: {format_metric(dsr_metrics.get('sumTelemDSRLoad'))}",
        f"SCED dispatchable: {format_metric(output_metrics.get('sumOutputSched'))} (headroom LSL {format_metric(output_metrics.get('sumLSLOutputSched'))} | HSL {format_metric(output_metrics.get('sumHSLOutputSched'))})",
        f"Base-point non-intermittent: {format_metric(gen_metrics.get('sumBasePointNonIRR'))} (SH {format_metric(gen_metrics.get('sumHASLNonIRR'))} | SL {format_metric(gen_metrics.get('sumLASLNonIRR'))})",
        f"Avg Texas temp: {format_metric(tx_temp, '°C', 1) if tx_temp is not None else 'N/A'}"
    ])
    
    return "\n".join(output_lines)

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
