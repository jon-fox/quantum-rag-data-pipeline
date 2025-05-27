"""
Environment variable management for the application.
Centralizes loading and access to environment variables.
"""
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment():
    """
    Load environment variables from .env file and validate required variables.
    Should be called during application startup.
    """
    # Load environment variables from .env file
    load_dotenv()
    logger.info("Environment variables loaded")
    
    # Define required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "HUGGINGFACE_API_TOKEN",
        "ERCOT_API_USERNAME",
        "ERCOT_API_PASSWORD",
    ]
    
    # Check for missing environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
    else:
        logger.info("All required environment variables are set")

def get_env_var(name, default=None):
    """
    Get an environment variable or return a default value.
    
    Args:
        name (str): Name of the environment variable
        default: Default value to return if environment variable is not set
        
    Returns:
        The environment variable value or the default
    """
    value = os.getenv(name, default)
    if value is None:
        logger.warning(f"Environment variable {name} not set")
    return value
