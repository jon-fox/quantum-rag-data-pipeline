"""
ERCOT API client for accessing ERCOT energy data
"""
import logging
import requests
import time
import random
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
    
    def get_data(self, endpoint, params=None, max_retries=3, base_delay=1.0):
        """
        Make an authenticated request to the ERCOT API with retry logic for rate limits.
        
        Args:
            endpoint (str): API endpoint (without base URL)
            params (dict, optional): Query parameters for the request
            max_retries (int): Maximum number of retry attempts for 429 errors
            base_delay (float): Base delay in seconds for exponential backoff
            
        Returns:
            dict: Response data from the API
        """
        # Get a valid authentication token
        token = self.auth_manager.get_auth_token()
        
        # Construct the full URL
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        # Make the request with authorization header
        for attempt in range(max_retries + 1):
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
                
                # Handle rate limit (429) with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff and jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit (429) on attempt {attempt + 1}. Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries")
                        response.raise_for_status()
                
                # Check for successful response
                response.raise_for_status()
                
                return response.json()
                
            except requests.RequestException as e:
                if attempt < max_retries and "429" in str(e):
                    # Additional retry for 429 errors caught as exceptions
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit exception on attempt {attempt + 1}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                
                logger.error(f"ERCOT API request failed: {str(e)}")
                raise

# Example usage:
# client = ERCOTClient()
# data = client.get_data("energy/prices", {"market": "dam", "date": "2023-05-18"})
