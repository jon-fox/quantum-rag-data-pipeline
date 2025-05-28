import logging
from typing import Optional, Dict, Callable, Awaitable # Added Callable, Awaitable
import numpy as np

# Corrected relative imports
from src.data.ercot_api.queries import ERCOTQueries
from src.data.weather_api.weather import WeatherAPIClient
from src.storage.pgvector_storage import PgVectorStorage
from src.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

def create_semantic_sentence(
    date_str: str,
    ercot_metric: Optional[float],
    weather_metrics: Optional[Dict[str, float]]
) -> Optional[str]:
    """Constructs a semantic sentence from ERCOT and weather data for a given date."""

    if ercot_metric is None and weather_metrics is None:
        logger.warning(f"Missing both ERCOT and weather data for {date_str}, cannot create semantic sentence.")
        return None
    
    ercot_part = ""
    if ercot_metric is not None:
        ercot_part = f"ERCOT's aggregated generation summary indicated an output of {ercot_metric:.2f} MW."
    else:
        logger.warning(f"Missing ERCOT data for {date_str}. Sentence will not include ERCOT data.")

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

        weather_part = (
            f"Concurrent weather conditions across key Texan cities featured an average temperature of "
            f"{avg_temp_str}, with Houston at {houston_temp_str}, Austin at {austin_temp_str}, "
            f"and Dallas at {dallas_temp_str}."
        )
    else:
        logger.warning(f"Missing weather data for {date_str}. Sentence will not include weather data.")

    if not ercot_part and not weather_part:
        logger.error(f"Both ERCOT and weather parts are empty for {date_str} despite earlier checks. Cannot create sentence.")
        return None
        
    sentence_parts = [f"On {date_str},"]
    if ercot_part:
        sentence_parts.append(ercot_part)
    if weather_part:
        sentence_parts.append(weather_part)
    
    return " ".join(sentence_parts)

async def process_and_embed_daily_summary(
    date_to_process: str, 
    ercot_queries: ERCOTQueries, 
    weather_client: Optional[WeatherAPIClient], 
    pg_storage: PgVectorStorage, 
    embedding_service: Optional[EmbeddingService],
    fetch_ercot_metric_func: Callable[[ERCOTQueries, str], Awaitable[Optional[float]]],
    fetch_weather_metrics_func: Callable[[WeatherAPIClient, str], Awaitable[Optional[Dict[str, float]]]]
):
    """Orchestrates fetching, processing, sentence creation, embedding, and storage for a single day."""
    logger.info(f"Processing daily summary for embedding on {date_to_process} (via sentence_builder)...")

    ercot_metric = await fetch_ercot_metric_func(ercot_queries, date_to_process)
    
    weather_metrics = None
    if weather_client:
        weather_metrics = await fetch_weather_metrics_func(weather_client, date_to_process)
    else:
        logger.info(f"Weather client not available, skipping weather metrics for {date_to_process}.")

    if ercot_metric is None:
        logger.warning(f"Could not retrieve ERCOT metric for {date_to_process}. Skipping embedding.")
        return
    
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
