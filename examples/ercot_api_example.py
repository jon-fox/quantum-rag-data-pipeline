"""
Example script demonstrating how to use the ERCOT API client and queries
"""
import logging
import json
import os
import sys
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.env_manager import load_environment
from src.data.ercot_api.client import ERCOTClient
from src.data.ercot_api.queries import ERCOTQueries

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate ERCOT API usage"""
    # Load environment variables
    load_environment()
    
    # Initialize ERCOT API client
    client = ERCOTClient()
    
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    three_days_ago = today - timedelta(days=3)

    delivery_date_from_str = three_days_ago.strftime('%Y-%m-%d')
    delivery_date_to_str = yesterday.strftime('%Y-%m-%d')
    
    # Initialize ERCOT queries helper with delivery dates
    queries = ERCOTQueries(
        client=client,
        delivery_date_from=delivery_date_from_str,
        delivery_date_to=delivery_date_to_str
    )
    
    try:
        # Example 1: Get 2-Day Aggregated Load Summary for Houston region
        # The delivery dates are now set at the instance level
        logger.info(f"Fetching 2-Day Aggregated Load Summary for Houston from {delivery_date_from_str} to {delivery_date_to_str}...")
        load_houston = queries.get_aggregated_load_summary(
            region="Houston"
        )
        print("\n2-Day Aggregated Load Summary (Houston):")
        print(json.dumps(load_houston, indent=2))

        # Example 2: Get 2-Day Aggregated Generation Summary (overall)
        logger.info(f"Fetching 2-Day Aggregated Generation Summary from {delivery_date_from_str} to {delivery_date_to_str}...")
        gen_summary = queries.get_agg_gen_summary()
        print("\n2-Day Aggregated Generation Summary (Overall):")
        print(json.dumps(gen_summary, indent=2))

        # Example 3: Get 2-Day Ancillary Service Offers for REGUP
        ancillary_service_type = "REGUP"
        logger.info(f"Fetching 2-Day Ancillary Service Offers for {ancillary_service_type} from {delivery_date_from_str} to {delivery_date_to_str}...")
        ancillary_offers = queries.get_ancillary_service_offers(
            service_type=ancillary_service_type,
            # Optionally, you can add hour_ending_from and hour_ending_to
            # hour_ending_from=1,
            # hour_ending_to=24 
        )
        print(f"\n2-Day Ancillary Service Offers ({ancillary_service_type}):")
        print(json.dumps(ancillary_offers, indent=2))
        
    except Exception as e:
        logger.error(f"Error accessing ERCOT API: {str(e)}")

if __name__ == "__main__":
    main()
