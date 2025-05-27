import boto3
import json
import logging
import os
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any
from botocore.exceptions import ClientError

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DynamoDBStorage:
    """Class for handling energy consumption data storage in DynamoDB"""
    
    def __init__(
        self,
        table_name: str = None,
        region: str = None,
        endpoint_url: str = None
    ):
        """
        Initialize DynamoDB storage handler
        
        Args:
            table_name: Override default table name from env var
            region: AWS region, defaults to env var or boto default
            endpoint_url: Custom endpoint for local DynamoDB testing
        """
        self.table_name = table_name or os.environ.get("DYNAMODB_TABLE", "energy_consumption_data")
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.endpoint_url = endpoint_url
        
        # Initialize DynamoDB resources
        self._init_dynamodb()
        
    def _init_dynamodb(self) -> None:
        """Initialize DynamoDB resource and table"""
        try:
            kwargs = {"region_name": self.region}
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url
                
            self.dynamodb = boto3.resource("dynamodb", **kwargs)
            self.table = self.dynamodb.Table(self.table_name)
            logger.info(f"DynamoDB initialized with table: {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to initialize DynamoDB: {e}")
            raise
    
    def store_energy_data(self, item: Dict[str, Any], condition_expression: str = None) -> Dict[str, Any]:
        """
        Store a single energy consumption data point in DynamoDB
        
        Args:
            item: Energy consumption data
            condition_expression: Optional DynamoDB condition expression
            
        Returns:
            Dict with result information
        """
        try:
            # Extract required fields with validation
            item_id = item.get("dataId")
            if not item_id:
                logger.error("Missing required field: dataId")
                return {"success": False, "error": "Missing required field: dataId"}
            
            # Prepare DynamoDB item with all possible fields
            db_item = {
                "item_id": item_id,
                "description": item.get("description", "Unknown"),
                "last_updated": self._get_timestamp(),
            }
            
            # Handle efficiency with proper validation
            efficiency_data = item.get("efficiency", {})
            if efficiency_data and isinstance(efficiency_data, dict):
                efficiency_value = efficiency_data.get("value")
                if efficiency_value:
                    try:
                        db_item["efficiency"] = Decimal(str(efficiency_value))
                        db_item["unit"] = efficiency_data.get("unit", "kWh")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid efficiency value for item {item_id}: {e}")
                        # Use 0 as fallback
                        db_item["efficiency"] = Decimal('0')
            elif isinstance(item.get("efficiency"), (int, float)):
                try:
                    db_item["efficiency"] = Decimal(str(item["efficiency"]))
                    db_item["unit"] = "kWh"  # Default unit
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid efficiency value for item {item_id}: {e}")
                    db_item["efficiency"] = Decimal('0')
            
            # Add all optional fields with type handling
            self._add_optional_field(db_item, item, "condition")
            self._add_optional_field(db_item, item, "conditionId", "condition_id")
            self._add_optional_field(db_item, item, "itemWebUrl", "item_url") 
            
            # Process nested fields
            if "image" in item and isinstance(item["image"], dict):
                self._add_optional_field(db_item, item["image"], "imageUrl", "image_url")
            
            if "seller" in item and isinstance(item["seller"], dict):
                self._add_optional_field(db_item, item["seller"], "username", "seller_username")
                self._add_optional_field(db_item, item["seller"], "feedbackScore", "feedback_score")
                
                # Handle feedback percentage as Decimal
                feedback_pct = item["seller"].get("feedbackPercentage")
                if feedback_pct:
                    try:
                        db_item["feedback_percent"] = Decimal(str(feedback_pct))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid feedback percentage for {item_id}")
            
            # Handle shipping costs
            shipping_options = item.get("shippingOptions", [])
            if shipping_options and isinstance(shipping_options, list) and shipping_options:
                shipping_cost = shipping_options[0].get("shippingCost", {}).get("value")
                if shipping_cost:
                    try:
                        db_item["shipping_cost"] = Decimal(str(shipping_cost))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid shipping cost for {item_id}")
            
            # Add more optional fields
            self._add_optional_field(db_item, item, "itemCreationDate", "creation_date")
            
            # Handle nested location fields
            if "itemLocation" in item and isinstance(item["itemLocation"], dict):
                self._add_optional_field(db_item, item["itemLocation"], "country", "location_country")
            
            self._add_optional_field(db_item, item, "listingMarketplaceId", "listing_marketplace")
            
            # Store the full JSON for future reference
            db_item["raw_json"] = json.dumps(item)
            
            # Prepare put_item arguments
            put_kwargs = {"Item": db_item}
            if condition_expression:
                put_kwargs["ConditionExpression"] = condition_expression
            
            # Store in DynamoDB with potential conditional write
            self.table.put_item(**put_kwargs)
            
            logger.info(f"Successfully stored item {item_id} in DynamoDB")
            return {"success": True, "item_id": item_id}
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "ConditionalCheckFailedException":
                logger.warning(f"Conditional check failed for item {item.get('itemId')}")
                return {
                    "success": False, 
                    "item_id": item.get("itemId"),
                    "error": "Item exists and condition check failed", 
                    "error_code": error_code
                }
            else:
                logger.error(f"DynamoDB error storing item {item.get('itemId')}: {error_code} - {str(e)}")
                return {
                    "success": False, 
                    "item_id": item.get("itemId"), 
                    "error": str(e), 
                    "error_code": error_code
                }
        except Exception as e:
            logger.error(f"Failed to store item {item.get('itemId', 'unknown')}: {str(e)}")
            return {"success": False, "item_id": item.get("itemId"), "error": str(e)}
    
    def batch_store_energy_data(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store multiple energy data items in batches
        
        Args:
            items: List of energy data items
            
        Returns:
            Dict with results summary
        """
        results = {
            "total": len(items),
            "succeeded": 0,
            "failed": 0,
            "failures": []
        }
        
        # Process in batches of 25 (DynamoDB batch write limit)
        batch_size = 25
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i+batch_size]
            self._process_batch(batch_items, results)
            
        logger.info(f"Batch operation complete: {results['succeeded']} succeeded, {results['failed']} failed")
        return results
    
    def _process_batch(self, items: List[Dict[str, Any]], results: Dict[str, Any]) -> None:
        """Process a batch of items with DynamoDB batch write"""
        if not items:
            return
            
        try:
            # Prepare batch write request
            batch_items = []
            for item in items:
                # For each item, try to store individually to capture specific errors
                result = self.store_listing(item)
                if result["success"]:
                    results["succeeded"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append({
                        "item_id": result.get("item_id", "unknown"),
                        "error": result.get("error", "Unknown error")
                    })
                    
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # If we have a general batch error, mark all as failed
            for item in items:
                results["failed"] += 1
                results["failures"].append({
                    "item_id": item.get("itemId", "unknown"),
                    "error": str(e)
                })
    
    def _add_optional_field(
        self, 
        db_item: Dict[str, Any], 
        source_item: Dict[str, Any],
        source_key: str, 
        target_key: str = None
    ) -> None:
        """Helper to add an optional field if it exists in source"""
        if not target_key:
            target_key = source_key
            
        if source_key in source_item and source_item[source_key] is not None:
            db_item[target_key] = source_item[source_key]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# Create a default instance for backward compatibility
default_storage = DynamoDBStorage()

def store_listing(item: dict) -> Dict[str, Any]:
    """
    Legacy function to store a listing using the default storage instance
    Maintained for backward compatibility
    """
    try:
        result = default_storage.store_listing(item)
        if result["success"]:
            print(f"Stored: {item['itemId']}")
        else:
            print(f"Failed to store {item.get('itemId', 'unknown')}: {result.get('error', 'Unknown error')}")
        return result
    except Exception as e:
        print(f"Failed to store {item.get('itemId', 'unknown')}: {e}")
        return {"success": False, "error": str(e)}

def batch_store_listings(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Store multiple listings using the default storage instance
    """
    return default_storage.batch_store_listings(items)
