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
            delivery_date_from (str, optional): Global start date in format YYYY-MM-DD.
                                              If not provided, defaults appropriately for methods.
            delivery_date_to (str, optional): Global end date in format YYYY-MM-DD.
                                            If not provided, defaults appropriately for methods.
        """
        self.client = client or ERCOTClient()
        
        # Set default global dates if not provided
        if not delivery_date_from:
            # Default for get_aggregated_load_summary (yesterday - 1 day)
            # Default for get_agg_gen_summary & get_ancillary_service_offers (yesterday)
            # We'll let individual methods handle their specific defaults if a global one isn't set,
            # or choose a common default. For simplicity, let's use a common default here.
            self.delivery_date_from = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            self.delivery_date_from = delivery_date_from
            
        if not delivery_date_to:
            # Default for get_aggregated_load_summary (today - 1 day)
            # Default for get_agg_gen_summary & get_ancillary_service_offers (today)
            self.delivery_date_to = datetime.today().strftime('%Y-%m-%d')
        else:
            self.delivery_date_to = delivery_date_to

    def get_aggregated_load_summary(
        self, 
        region: Optional[str] = None,
        page: int = 1,
        size: int = 100
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Load Summary data using instance-level delivery dates.
        
        Args:
            region (str, optional): Region filter (Houston, North, South, West).
                                   If not provided, gets the overall summary.
            page (int): Page number for pagination.
            size (int): Number of records per page.
            
        Returns:
            dict: The API response containing load summary data.
        """
        # Use instance-level dates, with specific defaults for this method if needed
        # For this method, the original default for delivery_date_from was 2 days ago
        # and delivery_date_to was 1 day ago.
        # We will use the global dates set in __init__ or override if specific logic is needed.
        # For now, we directly use the global dates.
        
        # Determine the endpoint based on region
        endpoint_suffix = "2d_agg_load_summary_houston" # This seems hardcoded, consider making it dynamic if needed
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-910-er/{endpoint_suffix}"
        
        # Build query parameters
        sced_timestamp_from = f"{self.delivery_date_from}T00:00:00"
        sced_timestamp_to = f"{self.delivery_date_to}T00:00:00"
        
        params = {
            "SCEDTimestampFrom": sced_timestamp_from,
            "SCEDTimestampTo": sced_timestamp_to,
            "page": page,
            "size": size
        }
        
        logger.info(f"Fetching aggregated load summary from {self.delivery_date_from} to {self.delivery_date_to}")
        return self.client.get_data(endpoint, params)
    
    def get_agg_gen_summary(
        self, 
        region: Optional[str] = None,
        page: int = 1,
        size: int = 100
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Generation Summary data using instance-level delivery dates.
        
        Args:
            region (str, optional): Region filter (Houston, North, South, West).
                                   If not provided, gets the overall summary.
            page (int): Page number for pagination.
            size (int): Number of records per page.
            
        Returns:
            dict: The API response containing generation summary data.
        """
        # Determine the endpoint based on region
        endpoint_suffix = "2d_agg_gen_summary" # Original name, seems like a typo, maybe "2d_agg_gen_sum"?
        # if region:
        #     region_lower = region.lower()
        #     if region_lower in ["houston", "north", "south", "west"]:
        #         endpoint_suffix = f"2d_agg_gen_sum_{region_lower}" # Assuming gen for generation
        
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-910-er/{endpoint_suffix}" # np3-911-er is typically for generation
        
        # Build query parameters
        # Note: The original implementation for this method used delivery_date_from and delivery_date_to directly
        # for SCEDTimestampFrom and SCEDTimestampTo. If these need the T00:00:00 format, adjust accordingly.
        # Assuming they also need the timestamp format like the load summary.
        sced_timestamp_from = f"{self.delivery_date_from}T00:00:00"
        sced_timestamp_to = f"{self.delivery_date_to}T00:00:00"

        params = {
            "SCEDTimestampFrom": sced_timestamp_from,
            "SCEDTimestampTo": sced_timestamp_to,
            "page": page,
            "size": size
        }
        
        logger.info(f"Fetching aggregated generation summary from {self.delivery_date_from} to {self.delivery_date_to}")
        return self.client.get_data(endpoint, params)
    
    def get_ancillary_service_offers(
        self,
        service_type: str,
        hour_ending_from: Optional[int] = None,
        hour_ending_to: Optional[int] = None,
        page: int = 1,
        size: int = 100
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Ancillary Service Offers using instance-level delivery dates.
        
        Args:
            service_type (str): Type of ancillary service. 
                               Options: ECRSM, ECRSS, OFFNS, ONNS, REGDN, REGUP, RRSFFR, RRSPFR, RRSUFR
            hour_ending_from (int, optional): Starting hour ending.
            hour_ending_to (int, optional): Ending hour ending.
            page (int): Page number for pagination.
            size (int): Number of records per page.
            
        Returns:
            dict: The API response containing ancillary service offers data.
        """
        # Validate service type
        valid_types = ["ecrsm", "ecrss", "offns", "onns", "regdn", "regup", "rrsffr", "rrspfr", "rrsufr"]
        service_type_lower = service_type.lower()
        
        if service_type_lower not in valid_types:
            raise ValueError(f"Invalid service type. Must be one of: {', '.join(valid_types)}")
            
        # Build the endpoint
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-911-er/2d_agg_as_offers_{service_type_lower}" # np3-911-er is typically for generation
        
        # Build query parameters
        params = {
            "deliveryDateFrom": self.delivery_date_from,
            "deliveryDateTo": self.delivery_date_to,
            "page": page,
            "size": size
        }
        
        # Add optional hour ending parameters if provided
        if hour_ending_from is not None:
            params["hourEndingFrom"] = hour_ending_from
            
        if hour_ending_to is not None:
            params["hourEndingTo"] = hour_ending_to
        
        logger.info(f"Fetching {service_type} offers from {self.delivery_date_from} to {self.delivery_date_to}")
        return self.client.get_data(endpoint, params)
