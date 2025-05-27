"""
Authentication module for the ERCOT API.
Handles token acquisition, refreshing, and management.
"""
import os
import time
import logging
import threading
import requests
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from src.config.env_manager import get_env_var

# Configure logging
logger = logging.getLogger(__name__)

class ERCOTAuth:
    """
    ERCOT API Authentication Manager
    
    Handles authentication to the ERCOT API, including token refresh logic.
    Tokens expire after 60 minutes, so this class refreshes them every 55 minutes.
    """
    
    # ERCOT B2C Authentication endpoint
    AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    
    # Client ID for the ERCOT Public API
    CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
    
    def __init__(self):
        """Initialize the ERCOT Auth manager."""
        self.token = None
        self.token_expires_at = None
        self.refresh_token = None
        self._refresh_timer = None
        self._refresh_lock = threading.RLock()
        
        # Credentials will be loaded from environment variables
        self.username = get_env_var("ERCOT_API_USERNAME")
        self.password = get_env_var("ERCOT_API_PASSWORD")
        self.scope = get_env_var("ERCOT_API_SCOPE", f"openid {self.CLIENT_ID} offline_access")
        
        if not self.username or not self.password:
            logger.error("ERCOT API credentials not found in environment variables")
        
    def get_auth_token(self):
        """
        Get a valid authentication token, refreshing if necessary.
        
        Returns:
            str: The current valid authentication token
        """
        with self._refresh_lock:
            # If token doesn't exist or is expired, get a new one
            if not self.token or self._is_token_expired():
                self._authenticate()
            
            return self.token
    
    def _authenticate(self):
        """
        Authenticate with the ERCOT API and get a new token.
        Sets up automatic refresh before expiration.
        """
        logger.info("Authenticating with ERCOT API")
        
        if not self.username or not self.password:
            raise ValueError("ERCOT API username or password not set")
        
        # Use the exact approach from the curl command example
        # For simplicity, create the URL directly as in the curl example
        auth_url = (f"{self.AUTH_URL}?username={quote_plus(str(self.username))}"
                   f"&password={quote_plus(str(self.password))}"
                   f"&grant_type=password"
                   f"&scope=openid+{self.CLIENT_ID}+offline_access"
                   f"&client_id={self.CLIENT_ID}"
                   f"&response_type=id_token")
        
        try:
            # Make authentication request exactly as shown in the curl example
            response = requests.post(
                auth_url,
                # headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            # Check if authentication was successful
            response.raise_for_status()
            
            # Parse response and store token
            token_data = response.json()
            self.token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            
            # Calculate token expiry time (setting to 55 minutes for safety buffer)
            # Convert expires_in to int, as it may be returned as a string from the API
            expires_in = int(token_data.get("expires_in", 3600))  # Default to 1 hour if not specified
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            # Schedule token refresh 5 minutes before expiration
            self._schedule_token_refresh(expires_in - 300)  # 5 minutes = 300 seconds
            
            logger.info(f"ERCOT API authentication successful, token valid until {self.token_expires_at}")
            
        except requests.RequestException as e:
            logger.error(f"ERCOT API authentication failed: {str(e)}")
            self.token = None
            self.token_expires_at = None
            self.refresh_token = None
            raise
    
    def _schedule_token_refresh(self, seconds_before_expiry):
        """
        Schedule token refresh before it expires.
        
        Args:
            seconds_before_expiry (int): Seconds before current token expires
        """
        # Cancel any existing refresh timer
        if self._refresh_timer:
            self._refresh_timer.cancel()
        
        # Set up new refresh timer
        self._refresh_timer = threading.Timer(seconds_before_expiry, self._authenticate)
        self._refresh_timer.daemon = True
        self._refresh_timer.start()
        
        logger.info(f"Token refresh scheduled in {seconds_before_expiry} seconds")
    
    def _is_token_expired(self):
        """
        Check if the current token is expired.
        
        Returns:
            bool: True if token is expired or about to expire, False otherwise
        """
        if not self.token_expires_at:
            return True
        
        # Consider token expired if less than 5 minutes remaining
        return datetime.now() + timedelta(minutes=5) >= self.token_expires_at
    
    def shutdown(self):
        """
        Shutdown the auth manager and cancel any pending refresh timers.
        Should be called during application shutdown.
        """
        if self._refresh_timer:
            self._refresh_timer.cancel()
            logger.info("ERCOT Auth manager shutdown")


# Singleton instance
_auth_manager = None

def get_auth_manager():
    """
    Get the ERCOT Auth manager singleton instance.
    
    Returns:
        ERCOTAuth: The authentication manager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = ERCOTAuth()
    return _auth_manager
