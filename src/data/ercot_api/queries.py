"""
ERCOT API queries for common data endpoints
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .client import ERCOTClient

# Configure logging
logger = logging.getLogger(__name__)

class ERCOTQueries:
    """
    Provides structured access to common ERCOT API endpoints.
    """
    
    # Base path for the public reports API
    PUBLIC_REPORTS_BASE = "api/public-reports"
    
    def __init__(self, client: Optional[ERCOTClient] = None, delivery_date_from: Optional[str] = None, delivery_date_to: Optional[str] = None):
        """
        Initialize the ERCOT Queries helper.
        
        Args:
            client (ERCOTClient, optional): An existing ERCOT client instance.
                                           If not provided, a new one will be created.
            delivery_date_from (str, optional): Global default start date in format YYYY-MM-DD.
                                              Defaults to yesterday if not provided.
            delivery_date_to (str, optional): Global default end date in format YYYY-MM-DD.
                                            Defaults to today if not provided.
        """
        self.client = client or ERCOTClient()
        
        self.default_delivery_date_from = delivery_date_from or (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        self.default_delivery_date_to = delivery_date_to or datetime.today().strftime('%Y-%m-%d')
        logger.info(f"ERCOTQueries initialized with default date range: {self.default_delivery_date_from} to {self.default_delivery_date_to}")

    def get_aggregated_load_summary(
        self, 
        region: Optional[str] = None,
        page: int = 1,
        size: int = 100,
        delivery_date_from_override: Optional[str] = None,
        delivery_date_to_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Load Summary data.
        Uses override dates if provided, otherwise instance defaults.
        
        Args:
            region (str, optional): Region filter (Houston, North, South, West).
            page (int): Page number for pagination.
            size (int): Number of records per page.
            delivery_date_from_override (str, optional): Specific start date (YYYY-MM-DD).
            delivery_date_to_override (str, optional): Specific end date (YYYY-MM-DD).
            
        Returns:
            dict: The API response containing load summary data.
        """
        date_from = delivery_date_from_override or self.default_delivery_date_from
        date_to = delivery_date_to_override or self.default_delivery_date_to
        
        endpoint_suffix = "2d_agg_load_summary_houston" 
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-910-er/{endpoint_suffix}"
        
        sced_timestamp_from = f"{date_from}T00:00:00"
        sced_timestamp_to = f"{date_to}T00:00:00"
        
        params = {
            "SCEDTimestampFrom": sced_timestamp_from,
            "SCEDTimestampTo": sced_timestamp_to,
            "page": page,
            "size": size
        }
        
        logger.info(f"Fetching aggregated load summary from {date_from} to {date_to}")
        return self.client.get_data(endpoint, params)
    
    def get_agg_gen_summary(
        self, 
        region: Optional[str] = None,
        page: int = 1,
        size: int = 100,
        delivery_date_from_override: Optional[str] = None,
        delivery_date_to_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Generation Summary data.
        Uses override dates if provided, otherwise instance defaults.

        Args:
            region (str, optional): Region filter (Houston, North, South, West).
            page (int): Page number for pagination.
            size (int): Number of records per page.
            delivery_date_from_override (str, optional): Specific start date (YYYY-MM-DD).
            delivery_date_to_override (str, optional): Specific end date (YYYY-MM-DD).
            
        Returns:
            dict: The API response containing generation summary data.
        """
        date_from = delivery_date_from_override or self.default_delivery_date_from
        date_to = delivery_date_to_override or self.default_delivery_date_to

        endpoint_suffix = "2d_agg_gen_summary"
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-910-er/{endpoint_suffix}" 
        
        sced_timestamp_from = f"{date_from}T00:00:00"
        sced_timestamp_to = f"{date_to}T00:00:00"

        params = {
            "SCEDTimestampFrom": sced_timestamp_from,
            "SCEDTimestampTo": sced_timestamp_to,
            "page": page,
            "size": size
        }
        
        logger.info(f"Fetching aggregated generation summary from {date_from} to {date_to}")
        return self.client.get_data(endpoint, params)
    
    def get_ancillary_service_offers(
        self,
        service_type: str,
        hour_ending_from: Optional[int] = None,
        hour_ending_to: Optional[int] = None,
        page: int = 1,
        size: int = 100,
        delivery_date_from_override: Optional[str] = None,
        delivery_date_to_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Ancillary Service Offers.
        Uses override dates if provided, otherwise instance defaults.
        
        Args:
            service_type (str): Type of ancillary service. 
            hour_ending_from (int, optional): Starting hour ending.
            hour_ending_to (int, optional): Ending hour ending.
            page (int): Page number for pagination.
            size (int): Number of records per page.
            delivery_date_from_override (str, optional): Specific start date (YYYY-MM-DD).
            delivery_date_to_override (str, optional): Specific end date (YYYY-MM-DD).
            
        Returns:
            dict: The API response containing ancillary service offers data.
        """
        date_from = delivery_date_from_override or self.default_delivery_date_from
        date_to = delivery_date_to_override or self.default_delivery_date_to

        valid_types = ["ecrsm", "ecrss", "offns", "onns", "regdn", "regup", "rrsffr", "rrspfr", "rrsufr"]
        service_type_lower = service_type.lower()
        
        if service_type_lower not in valid_types:
            raise ValueError(f"Invalid service type. Must be one of: {', '.join(valid_types)}")
            
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-911-er/2d_agg_as_offers_{service_type_lower}"
        
        params = {
            "deliveryDateFrom": date_from,
            "deliveryDateTo": date_to,
            "page": page,
            "size": size
        }
        
        # Add optional hour ending parameters if provided
        if hour_ending_from is not None:
            params["hourEndingFrom"] = hour_ending_from
            
        if hour_ending_to is not None:
            params["hourEndingTo"] = hour_ending_to
        
        logger.info(f"Fetching {service_type} offers from {date_from} to {date_to}")
        return self.client.get_data(endpoint, params)
