from meteostat import Point, Daily
from datetime import datetime, date
from statistics import mean
from typing import Union, Optional

def get_ercot_avg_temperature(target_date: Union[datetime, date, str]) -> Optional[float]:
    """
    Fetch the average temperature for a specific date across major ERCOT cities.
    
    Args:
        target_date: The date to fetch temperature for. Can be datetime, date, or string (YYYY-MM-DD)
        
    Returns:
        Average temperature in Celsius, or None if no data available
    """
    # Convert input to datetime if needed
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # Major ERCOT cities
    cities = {
        "Houston": Point(29.76, -95.37),
        "Dallas": Point(32.78, -96.80),
        "Austin": Point(30.27, -97.74),
        "San Antonio": Point(29.42, -98.49),
        "Fort Worth": Point(32.75, -97.33),
        "Corpus Christi": Point(27.80, -97.40),
        "Abilene": Point(32.45, -99.74),
        "Waco": Point(31.55, -97.15)
    }
    
    # Convert date to datetime for meteostat API
    start_datetime = datetime.combine(target_date, datetime.min.time())
    end_datetime = datetime.combine(target_date, datetime.max.time())
    
    # Collect temperature data for the specific date
    temperatures = []
    
    for city, point in cities.items():
        try:
            data = Daily(point, start_datetime, end_datetime)
            data = data.fetch()
            
            for date_index, row in data.iterrows():
                tavg = row["tavg"]
                if tavg is not None and tavg == tavg:  # Not None and not NaN
                    temperatures.append(tavg)
                    break  # Only need one reading per city for the target date
        except Exception as e:
            print(f"Warning: Could not fetch data for {city}: {e}")
            continue
    
    if not temperatures:
        return None
    
    return round(mean(temperatures), 2)
