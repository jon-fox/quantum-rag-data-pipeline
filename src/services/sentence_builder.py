\
import logging
from typing import Optional, Dict

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
        def format_temp(temp_value):
            if isinstance(temp_value, (int, float)):
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
        # This case should ideally be caught by the first check, but as a safeguard:
        logger.error(f"Both ERCOT and weather parts are empty for {date_str} despite earlier checks. Cannot create sentence.")
        return None
        
    sentence_parts = [f"On {date_str},"]
    if ercot_part:
        sentence_parts.append(ercot_part)
    if weather_part:
        sentence_parts.append(weather_part)
    
    return " ".join(sentence_parts)
