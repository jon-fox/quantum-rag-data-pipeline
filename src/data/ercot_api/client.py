"""
ERCOT API client for accessing ERCOT energy data
"""
import logging
import requests
from src.data.ercot_api.auth import get_auth_manager
import os

# Configure logging
logger = logging.getLogger(__name__)

class ERCOTClient:
    """
    Client for interacting with the ERCOT API.
    Uses the ERCOTAuth manager for authentication.
    """
    
    # Base URL for ERCOT API
    BASE_URL = "https://api.ercot.com"  # Replace with actual ERCOT API base URL
    
    def __init__(self):
        """Initialize the ERCOT API client."""
        self.auth_manager = get_auth_manager()
    
    def get_data(self, endpoint, params=None):
        """
        Make an authenticated request to the ERCOT API.
        
        Args:
            endpoint (str): API endpoint (without base URL)
            params (dict, optional): Query parameters for the request
            
        Returns:
            dict: Response data from the API
        """
        # Get a valid authentication token
        token = self.auth_manager.get_auth_token()
        
        # Construct the full URL
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        # Make the request with authorization header
        try:
            response = requests.get(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Ocp-Apim-Subscription-Key": os.getenv("ERCOT_API_TOKEN"),  # ERCOT API requires this header
                    "Accept": "application/json"
                }
            )
            
            # Check for successful response
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"ERCOT API request failed: {str(e)}")
            raise

# Example usage:
# client = ERCOTClient()
# data = client.get_data("energy/prices", {"market": "dam", "date": "2023-05-18"})
