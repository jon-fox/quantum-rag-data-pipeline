import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Optional # Added

# Attempt to import PgVectorStorage
try:
    from src.storage.pgvector_storage import PgVectorStorage
except ImportError:
    PgVectorStorage = None # Allow script to run if PgVectorStorage is not found, but DB operations will fail
    logging.getLogger(__name__).warning("PgVectorStorage could not be imported. Database operations will be skipped.")


# Configure logging
logger = logging.getLogger(__name__)

class WeatherAPIClient:
    """
    A client to fetch historical weather data from WeatherAPI.com.
    """
    DEFAULT_BASE_URL = "http://api.weatherapi.com/v1/history.json"

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        """
        Initialize the WeatherAPIClient.

        Args:
            api_key (str): The API key for WeatherAPI.com.
            base_url (str, optional): The base URL for the history API. 
                                     Defaults to DEFAULT_BASE_URL.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.base_url = base_url

    def get_historical_weather(self, date_str: str, locations: dict, store_in_db: bool = False) -> pd.DataFrame:
        """
        Fetches historical weather data for a given date and multiple locations.

        Args:
            date_str (str): The date for which to fetch data (YYYY-MM-DD).
            locations (dict): A dictionary where keys are location names (e.g., "Downtown")
                              and values are city identifiers for the API (e.g., "Houston").
            store_in_db (bool, optional): If True, attempts to store the fetched data
                                          into the 'historical_weather_data' table. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the time, temperature for each location,
                          average temperature (Celsius and Fahrenheit).
                          Returns an empty DataFrame if no data is fetched.
        """
        if not locations:
            logger.warning("Locations dictionary is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        all_location_dfs = []

        for name, city_identifier in locations.items():
            params = {
                "key": self.api_key,
                "q": city_identifier,
                "dt": date_str
            }
            response = None # Initialize response to None
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
                
                hourly_data = response.json()["forecast"]["forecastday"][0]["hour"]
                df = pd.DataFrame({
                    "time": [entry["time"] for entry in hourly_data],
                    name: [entry["temp_c"] for entry in hourly_data]
                })
                all_location_dfs.append(df)
                logger.info(f"Successfully fetched weather data for {name} ({city_identifier}) on {date_str}")

            except requests.exceptions.HTTPError as http_err:
                error_message = f"HTTP error fetching data for {name} ({city_identifier}): {http_err}"
                if response is not None:
                    error_message += f" - Response: {response.text}"
                logger.error(error_message)
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Request error fetching data for {name} ({city_identifier}): {req_err}")
            except KeyError as key_err:
                logger.error(f"Key error parsing data for {name} ({city_identifier}): {key_err} - Likely unexpected API response structure.")
            except Exception as e:
                logger.error(f"An unexpected error occurred for {name} ({city_identifier}): {e}")

        if not all_location_dfs:
            logger.warning("No weather data was collected for any location.")
            return pd.DataFrame()

        # Combine DataFrames
        weather_df = all_location_dfs[0]
        for df in all_location_dfs[1:]:
            weather_df = weather_df.merge(df, on="time", how="outer")

        # Clean and compute averages
        # weather_df.dropna(inplace=True) # Consider if dropping all rows with any NaN is desired
        
        location_temp_columns = list(locations.keys())
        # Ensure only existing columns are used for mean calculation
        valid_temp_columns = [col for col in location_temp_columns if col in weather_df.columns]

        if not valid_temp_columns:
            logger.warning("No valid temperature columns found to calculate average. Returning raw merged data.")
            weather_df = weather_df.sort_values("time").reset_index(drop=True)
            return weather_df

        weather_df["avg_temperature_c"] = weather_df[valid_temp_columns].mean(axis=1)
        weather_df["avg_temperature_f"] = weather_df["avg_temperature_c"] * 9 / 5 + 32
        
        weather_df = weather_df.sort_values("time").reset_index(drop=True)
        
        # Select and reorder columns for the final output
        final_columns = ["time"] + valid_temp_columns + ["avg_temperature_c", "avg_temperature_f"]
        # Ensure all final columns exist before selecting
        final_columns_existing = [col for col in final_columns if col in weather_df.columns]
        
        result_df = weather_df[final_columns_existing].copy()

        if store_in_db and not result_df.empty and PgVectorStorage is not None:
            pg_storage_instance = None
            try:
                df_for_db = result_df.copy()
                if 'time' in df_for_db.columns:
                    df_for_db.rename(columns={'time': 'timestamp'}, inplace=True)
                else:
                    logger.warning("'time' column not found in DataFrame intended for DB. 'timestamp' column will be missing.")

                # Ensure APP_ENVIRONMENT is available, e.g., from .env file loaded by load_dotenv()
                app_env = os.getenv("APP_ENVIRONMENT", "prod")
                pg_storage_instance = PgVectorStorage(app_environment=app_env)
                
                # Define the expected DB columns based on create_weather_table.py
                db_schema_columns = ['timestamp', 'houston_temp_c', 'austin_temp_c', 'dallas_temp_c', 'avg_temperature_c', 'avg_temperature_f']
                # Filter DataFrame to only include columns that exist in the DB schema
                columns_to_store_in_db = [col for col in db_schema_columns if col in df_for_db.columns]
                df_to_actually_store = df_for_db[columns_to_store_in_db]

                if not df_to_actually_store.empty:
                    logger.info(f"Attempting to store {len(df_to_actually_store)} weather records into 'historical_weather_data' table.")
                    pg_storage_instance.insert_dataframe_to_table(
                        df_to_actually_store,
                        'historical_weather_data',
                        on_conflict_do_update=True 
                    )
                    logger.info("Successfully stored weather data in 'historical_weather_data' table.")
                else:
                    logger.info("DataFrame for DB storage is empty after filtering for schema columns. Skipping database storage.")

            except AttributeError as ae:
                if "insert_dataframe_to_table" in str(ae):
                    logger.error("PgVectorStorage does not have 'insert_dataframe_to_table' method. Please add it to the class.")
                else:
                    logger.error(f"Attribute error during database operation: {ae}", exc_info=True)
            except Exception as db_err:
                logger.error(f"Failed to store weather data in database: {db_err}", exc_info=True)
            finally:
                if pg_storage_instance and hasattr(pg_storage_instance, 'close_db_connection'):
                    pg_storage_instance.close_db_connection()
                    logger.info("PostgreSQL connection closed.")
        
        return result_df
