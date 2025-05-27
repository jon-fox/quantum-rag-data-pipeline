"""
Example script demonstrating how to use the WeatherAPIClient.
"""
import logging
import os
import sys
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.env_manager import load_environment
from src.data.weather_api.weather import WeatherAPIClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate WeatherAPIClient usage."""
    # Load environment variables (which should include WEATHER_API_KEY)
    load_environment()
    
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        logger.error("WEATHER_API_KEY not found in environment variables. Please set it in your .env file.")
        return

    # Initialize WeatherAPIClient
    weather_client = WeatherAPIClient(api_key=api_key)

    # Define locations and date for the weather data query
    # For the free tier of WeatherAPI, history.json allows dates up to 7 days in the past.
    # Let's fetch for 3 days ago.
    target_date_str = (datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    locations_to_query = {
        "HoustonTX": "Houston,TX",
        "AustinTX": "Austin,TX",
        "DallasTX": "Dallas,TX",
        # "NewYorkNY": "New York,NY" # Example of another location
    }
    
    logger.info(f"Attempting to fetch historical weather data for {target_date_str} for locations: {list(locations_to_query.keys())}")

    try:
        weather_data_df = weather_client.get_historical_weather(
            date_str=target_date_str,
            locations=locations_to_query
        )
        
        if not weather_data_df.empty:
            print("\n--- Historical Weather Data ---")
            print(f"Date: {target_date_str}")
            
            # Displaying a subset of columns for brevity if many locations
            display_columns = ["time", "avg_temperature_c", "avg_temperature_f"]
            
            # Add individual location columns if they exist and are not too many
            if len(locations_to_query) <= 3:
                 display_columns = ["time"] + list(locations_to_query.keys()) + ["avg_temperature_c", "avg_temperature_f"]
            
            # Filter out columns that might not exist if all data fetching failed for them
            existing_display_columns = [col for col in display_columns if col in weather_data_df.columns]

            print(weather_data_df[existing_display_columns].head())
            
            # You can save this DataFrame to a CSV or process it further:
            # output_filename = f"weather_data_{target_date_str.replace('-', '')}.csv"
            # weather_data_df.to_csv(output_filename, index=False)
            # logger.info(f"Weather data saved to {output_filename}")

        else:
            logger.info("No weather data was retrieved. This could be due to API errors for all locations or no locations provided.")
            
    except Exception as e:
        logger.error(f"An error occurred during the weather API example: {e}", exc_info=True)

if __name__ == "__main__":
    main()
