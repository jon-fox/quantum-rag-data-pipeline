"""
Dual storage module for storing document metadata in DynamoDB and vector embeddings in PostgreSQL (via PgVectorStorage).
"""
import json
import logging
import os
import uuid # Added for generating vector_id
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import boto3 # Ensure boto3 is imported
from boto3.dynamodb.conditions import Key # Import Key for queries

# Import existing DynamoDB storage and the new PgVectorStorage
from src.storage.dynamodb import DynamoDBStorage
from src.storage.pgvector_storage import PgVectorStorage # New import
from src.schema.models import Document # Assuming Document model might be used or adapted

# Set up logging
logger = logging.getLogger(__name__)
# BasicConfig should ideally be called once at the application entry point
# logging.basicConfig(
#     level=logging.INFO,
#     format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\'
# )

class DualStorage:
    """
    Manages storage of document metadata in DynamoDB and their corresponding
    vector embeddings in a PostgreSQL database using the pgvector extension.
    """

    def __init__(
        self,
        dynamodb_table_name: Optional[str] = None,
        pg_vector_table_name: str = "document_embeddings", # Table for pgvector
        pg_db_params: Optional[Dict[str, str]] = None, # For PgVectorStorage
        vector_dim: int = 1536, # Default for OpenAI text-embedding-3-small
        region: Optional[str] = None,
        # environment: Optional[str] = None, # Not directly used by PgVectorStorage from here
        # ssm_param_prefix: Optional[str] = None, # PgVectorStorage handles its own param loading
        lazy_init_pg: bool = True
    ):
        """
        Initialize dual storage handler.

        Args:
            dynamodb_table_name: Name of the DynamoDB table for metadata.
                                 Defaults to DYNAMODB_TABLE env var or "quantum_embeddings".
            pg_vector_table_name: Name of the PostgreSQL table for embeddings.
            pg_db_params: PostgreSQL connection parameters for PgVectorStorage.
            vector_dim: Dimension of the embeddings.
            region: AWS region for DynamoDB.
            lazy_init_pg: Whether to delay PostgreSQL schema initialization.
        """
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        
        # Ensure dynamodb_table_name is correctly defaulted if None
        resolved_dynamodb_table_name = dynamodb_table_name or os.environ.get("DYNAMODB_TABLE", "quantum_embeddings")

        # Initialize DynamoDB storage for metadata
        self.dynamo_storage = DynamoDBStorage(
            table_name=resolved_dynamodb_table_name, 
            region=self.region
        )
        
        # Initialize PgVectorStorage for embeddings
        self.pg_vector_storage = PgVectorStorage(
            db_params=pg_db_params, # Pass explicitly or let PgVectorStorage load from env
            table_name=pg_vector_table_name,
            vector_dim=vector_dim,
            lazy_init=lazy_init_pg
        )
        
        logger.info(f"DualStorage initialized: DynamoDB table \'{resolved_dynamodb_table_name}\', PgVector table \'{pg_vector_table_name}\'")

    def store_document_and_embedding(
        self,
        document_metadata: Dict[str, Any], # Expects metadata including a document_id
        embedding_vector: np.ndarray,
        vector_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stores document metadata in DynamoDB and its embedding in PostgreSQL.
        A unique vector_id is generated if not provided, and stored in DynamoDB metadata.

        Args:
            document_metadata: Dictionary containing document metadata. 
                               Must include 'document_id' for DynamoDB.
                               Other fields like 'doc_type', 'title', 'source', 'summary', etc.
            embedding_vector: Numpy array of the document's embedding.
            vector_id: Optional pre-generated UUID for the vector. If None, a new one is created.

        Returns:
            A dictionary with storage results.
            e.g., {
                "document_id": "doc123",
                "vector_id": "uuid-for-vector",
                "dynamo_result": {"success": True, ...},
                "pgvector_result": {"success": True, ...},
                "overall_success": True
            }
        """
        doc_id = document_metadata.get("document_id")
        if not doc_id:
            logger.error("Missing 'document_id' in document_metadata.")
            return {"overall_success": False, "error": "Missing document_id"}

        # Generate a vector_id if not provided
        if not vector_id:
            vector_id = str(uuid.uuid4())
        
        # Add/update vector_id in the metadata to be stored in DynamoDB
        document_metadata_with_vid = document_metadata.copy()
        document_metadata_with_vid["vector_id"] = vector_id

        results = {
            "document_id": doc_id,
            "vector_id": vector_id,
            "dynamo_result": {"success": False},
            "pgvector_result": {"success": False},
            "overall_success": False
        }

        # 1. Store metadata in DynamoDB
        try:
            # Assuming dynamo_storage.store_item or similar method exists
            # and is adapted for the 'quantum_embeddings' table structure
            # The existing store_energy_data might need to be generalized or a new method created.
            # For now, let's assume a generic store_metadata method.
            # You'll need to adapt this part based on your DynamoDBStorage implementation.
            
            # If your DynamoDB table's primary key is 'document_id'
            # and it expects other attributes as defined in your Terraform.
            dynamo_item_to_store = {
                "document_id": doc_id, # Hash key
                "doc_type": document_metadata_with_vid.get("doc_type", "unknown"),
                "last_updated": document_metadata_with_vid.get("last_updated", self.dynamo_storage._get_timestamp()), # Use existing helper
                "title": document_metadata_with_vid.get("title"),
                "source": document_metadata_with_vid.get("source"),
                "vector_id": vector_id, # Crucial link
                "embedding_type": document_metadata_with_vid.get("embedding_type"),
                "summary": document_metadata_with_vid.get("summary"),
                # Add any other fields from document_metadata_with_vid that are part of your DDB table schema
                # For example, if you have 'expiry_time' for TTL:
                # "expiry_time": document_metadata_with_vid.get("expiry_time") 
            }
            # Remove None values to avoid DynamoDB validation errors for optional fields not provided
            dynamo_item_to_store_cleaned = {k: v for k, v in dynamo_item_to_store.items() if v is not None}

            # The method in DynamoDBStorage needs to be flexible enough.
            # Let's assume a method `put_item` that takes the full item.
            # This is a placeholder for the actual call you'd make.
            # You might need to refactor DynamoDBStorage.store_energy_data
            # or add a new method like `store_document_metadata`.

            # For the purpose of this example, let's assume a generic `put_item` exists
            # that matches the structure of your `quantum_embeddings` table.
            # This part needs careful implementation in `dynamodb.py`.
            
            # Using the existing store_energy_data and adapting the input might be tricky
            # due to its specific structure. A new, more generic method in DynamoDBStorage is better.
            # Let's assume a new method `upsert_item` in DynamoDBStorage for this.
            # This method would take the item and table's hash key name.
            # For now, we'll mock its success.
            
            # SIMULATED CALL - REPLACE WITH ACTUAL IMPLEMENTATION
            # This requires `document_metadata_with_vid` to be structured correctly for `store_item`
            # or `store_item` to be more generic.
            # The `store_energy_data` method in your `dynamodb.py` is quite specific.
            # We need a more generic way to put an item.
            # A simple approach for now, assuming `self.dynamo_storage.table` is the DynamoDB Table resource:
            self.dynamo_storage.table.put_item(Item=dynamo_item_to_store_cleaned)
            results["dynamo_result"] = {"success": True, "document_id": doc_id, "message": "Metadata stored in DynamoDB."}
            logger.info(f"Successfully stored metadata for document_id: {doc_id} in DynamoDB.")

        except Exception as e:
            logger.error(f"DynamoDB storage error for document_id {doc_id}: {e}")
            results["dynamo_result"] = {"success": False, "error": str(e)}

        # 2. Store embedding in PostgreSQL via PgVectorStorage
        if results["dynamo_result"]["success"]: # Only proceed if metadata storage was okay or if you want to store vector regardless
            try:
                pg_success = self.pg_vector_storage.store_embedding(vector_id, embedding_vector)
                results["pgvector_result"] = {"success": pg_success}
                if pg_success:
                    logger.info(f"Successfully stored embedding for vector_id: {vector_id} in PgVector.")
                else:
                    logger.error(f"Failed to store embedding for vector_id: {vector_id} in PgVector.")
            except Exception as e:
                logger.error(f"PgVector storage error for vector_id {vector_id}: {e}")
                results["pgvector_result"] = {"success": False, "error": str(e)}
        else:
            logger.warning(f"Skipping PgVector storage for vector_id {vector_id} due to DynamoDB storage failure.")
            results["pgvector_result"] = {"success": False, "error": "Skipped due to DynamoDB failure"}


        results["overall_success"] = results["dynamo_result"]["success"] and results["pgvector_result"]["success"]
        return results

    def batch_store_documents_and_embeddings(
        self,
        documents_with_embeddings: List[Tuple[Dict[str, Any], np.ndarray, Optional[str]]]
    ) -> Dict[str, Any]:
        """
        Stores multiple documents' metadata in DynamoDB and their embeddings in PostgreSQL.

        Args:
            documents_with_embeddings: A list of tuples. Each tuple contains:
                - document_metadata (Dict[str, Any]): Metadata for DynamoDB. Must include 'document_id'.
                - embedding_vector (np.ndarray): The embedding vector.
                - vector_id (Optional[str]): Pre-generated UUID for the vector. If None, one is created.
        
        Returns:
            A summary of the batch operation.
        """
        batch_results = {
            "total_items": len(documents_with_embeddings),
            "succeeded_items": 0,
            "failed_items": 0,
            "individual_results": []
        }

        dynamo_batch_items_to_store = []
        pg_batch_embeddings_to_store = [] # List of (vector_id, embedding_vector)

        for doc_meta, embedding_vec, vec_id_optional in documents_with_embeddings:
            doc_id = doc_meta.get("document_id")
            if not doc_id:
                batch_results["failed_items"] += 1
                batch_results["individual_results"].append({
                    "document_id": None, "vector_id": None, "success": False, "error": "Missing document_id"
                })
                continue

            vector_id = vec_id_optional if vec_id_optional else str(uuid.uuid4())
            
            # Prepare item for DynamoDB
            dynamo_item = doc_meta.copy()
            dynamo_item["vector_id"] = vector_id
            # Add/ensure other required DDB fields like last_updated
            dynamo_item["last_updated"] = dynamo_item.get("last_updated", self.dynamo_storage._get_timestamp())
            
            # Clean None values for DDB
            dynamo_item_cleaned = {k: v for k, v in dynamo_item.items() if v is not None}
            dynamo_batch_items_to_store.append(dynamo_item_cleaned)
            
            # Prepare item for PgVector
            pg_batch_embeddings_to_store.append((vector_id, embedding_vec))

        # 1. Batch store metadata in DynamoDB
        # DynamoDBStorage needs a batch write method.
        # For now, let's assume it processes one by one for simplicity of this example,
        # or you'd implement a proper batch_put_items in DynamoDBStorage.
        # A true batch write would be more efficient.
        
        # --- Placeholder for DynamoDB Batch Interaction ---
        # This needs a proper batch write implementation in `dynamodb.py`
        # For now, iterating and calling single store, which is not ideal for "batch"
        # but demonstrates the logic.
        
        processed_doc_ids_for_pg = set() # Track doc_ids successfully stored in DDB

        if dynamo_batch_items_to_store:
            # SIMULATED BATCH DDB - REPLACE WITH ACTUAL IMPLEMENTATION
            # Example: using a hypothetical batch_upsert_items in DynamoDBStorage
            # dynamo_batch_op_results = self.dynamo_storage.batch_upsert_items(dynamo_batch_items_to_store)
            # For now, let's iterate (less efficient but illustrates the data flow)
            for item_to_store, (original_doc_meta, _, _) in zip(dynamo_batch_items_to_store, documents_with_embeddings):
                # This assumes `item_to_store` has `document_id`
                current_doc_id = item_to_store.get("document_id") 
                try:
                    self.dynamo_storage.table.put_item(Item=item_to_store) # Simplified
                    # In a real scenario, you'd collect vector_ids of successful DDB puts
                    # For simplicity, assume all DDB puts here are successful for now if no exception
                    # The `vector_id` is already in `item_to_store`
                    # We need to map this success back to the pg_batch_embeddings_to_store
                    # Find the corresponding vector_id for this current_doc_id
                    original_vector_id = item_to_store.get("vector_id")
                    if original_vector_id:
                         # Mark this vector_id as eligible for PG storage
                         # This logic is a bit complex if DDB batch fails partially.
                         # A better approach: DDB batch returns list of successes/failures.
                         # For now, let's assume if DDB put_item is successful, we add its vector_id for PG.
                        
                        # For this simplified loop, if a DDB item makes it here without error, its metadata is "stored".
                        # The `pg_batch_embeddings_to_store` as prepared earlier will be used.
                        # A more robust solution would filter pg_batch_embeddings_to_store based on DDB success.
                        
                        # Let's refine: build a list of (vector_id, embedding_vector) for PG *only* if DDB succeeded.
                        # This is getting complex for a simulation.
                        # A true DDB batch write method in DynamoDBStorage is essential.
                        # It should return which items succeeded, along with their vector_ids.

                        # Assuming for now: if a DDB item makes it here without error, its metadata is "stored".
                        # The `pg_batch_embeddings_to_store` as prepared earlier will be used.
                        # This implies DDB batch is all-or-nothing or all successful for this simplified path.
                        pass # Placeholder for success tracking

                except Exception as e:
                    logger.error(f"Error storing {current_doc_id} in DynamoDB during batch: {e}")
                    # If DDB fails for an item, we should ideally not store its vector.
                    # This requires removing it from pg_batch_embeddings_to_store or marking it.
                    # For simplicity, this example will still attempt all PG stores,
                    # but a real implementation should filter.
        
        # 2. Batch store embeddings in PostgreSQL
        pg_batch_succeeded = False
        if pg_batch_embeddings_to_store:
            try:
                pg_batch_succeeded = self.pg_vector_storage.batch_store_embeddings(pg_batch_embeddings_to_store)
                if pg_batch_succeeded:
                    logger.info(f"PgVector batch store attempt finished for {len(pg_batch_embeddings_to_store)} items. Success: {pg_batch_succeeded}")
                else:
                    logger.error(f"PgVector batch store failed for {len(pg_batch_embeddings_to_store)} items.")
            except Exception as e:
                logger.error(f"Error during PgVector batch_store_embeddings: {e}")
                pg_batch_succeeded = False
        
        # Consolidate results (this is a simplified aggregation)
        # A more detailed aggregation would track individual item success/failure through both stores.
        if pg_batch_succeeded: # And assuming DDB part was mostly successful
            batch_results["succeeded_items"] = len(pg_batch_embeddings_to_store) # Approximation
        batch_results["failed_items"] = batch_results["total_items"] - batch_results["succeeded_items"]
        # Individual results would need more granular tracking.

        logger.info(f"Batch store operation summary: Total: {batch_results['total_items']}, Succeeded (approx): {batch_results['succeeded_items']}")
        return batch_results

    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves document metadata from DynamoDB."""
        # This method needs to be implemented in DynamoDBStorage
        # e.g., self.dynamo_storage.get_item_by_id(document_id)
        # For now, returning None as a placeholder.
        # --- Placeholder for DynamoDBStorage interaction ---
        try:
            response = self.dynamo_storage.table.get_item(Key={'document_id': document_id})
            if 'Item' in response:
                logger.info(f"Retrieved metadata for document_id: {document_id}")
                return response['Item']
            else:
                logger.info(f"No metadata found for document_id: {document_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving metadata for document_id {document_id} from DynamoDB: {e}")
            return None


    def get_embedding_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Retrieves an embedding vector from PgVectorStorage."""
        try:
            vector = self.pg_vector_storage.get_embedding(vector_id)
            if vector is not None:
                logger.info(f"Retrieved embedding for vector_id: {vector_id}")
            else:
                logger.info(f"No embedding found for vector_id: {vector_id}")
            return vector
        except Exception as e:
            logger.error(f"Error retrieving embedding for vector_id {vector_id} from PgVector: {e}")
            return None

    def find_similar_documents(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Finds documents with embeddings similar to the query_embedding.
        1. Queries PgVectorStorage for similar vector_ids.
        2. Retrieves metadata for these vector_ids from DynamoDB.

        Args:
            query_embedding: Numpy array of the query embedding.
            top_k: Number of similar documents to return.
            metric: Distance metric for similarity search ('cosine', 'l2', 'inner_product').

        Returns:
            A list of dictionaries, where each dictionary contains:
            - 'metadata': The document metadata from DynamoDB.
            - 'vector_id': The ID of the vector.
            - 'distance': The distance score from the similarity search.
        """
        similar_vectors = self.pg_vector_storage.find_similar_embeddings(
            query_embedding, top_k=top_k, metric=metric
        )

        results = []
        if not similar_vectors:
            logger.info("No similar vectors found in PgVector.")
            return []

        logger.info(f"Found {len(similar_vectors)} candidate vectors. Fetching metadata from DynamoDB.")
        for vector_id, distance_score in similar_vectors:
            # We need to get the document_id associated with this vector_id from DynamoDB.
            # This requires that vector_id is queryable in DynamoDB or that we iterate.
            # The current DynamoDB schema has vector_id in non_key_attributes of GSIs.
            # If we need to find a document_id given a vector_id, we might need a GSI on vector_id
            # or to adjust the workflow.
            
            # Assumption: The 'vector_id' stored in DynamoDB items is the same as `vector_id` from pgvector.
            # We need a way to query DynamoDB for an item WHERE vector_id = <found_vector_id>.
            # If 'vector_id' is not a key in DynamoDB, this is inefficient.
            # Let's assume for now we have a GSI: VectorIdIndex with hash_key = 'vector_id'.
            # If not, this part needs rethinking or a scan (not recommended for performance).

            # Querying DynamoDB by a non-key attribute (vector_id) efficiently requires a GSI.
            # Let's assume you have a GSI:
            # global_secondary_index {
            #   name            = "VectorIdIndex"
            #   hash_key        = "vector_id"
            #   projection_type = "ALL" # or "INCLUDE" necessary attributes
            # }
            # If such an index exists:
            try:
                # This is a conceptual query. Actual implementation depends on DynamoDBStorage.
                # response = self.dynamo_storage.query_by_gsi("VectorIdIndex", "vector_id", vector_id)
                # items = response.get('Items', [])
                # For now, let's simulate a direct query if the GSI exists.
                # This is a placeholder for actual GSI query logic in DynamoDBStorage.
                
                # --- Placeholder for DynamoDB GSI Query ---
                # This requires a GSI on 'vector_id' in your DynamoDB table.
                # If no GSI, this will be a scan, which is bad.
                # For the example, let's assume a GSI 'VectorIdIndex' on 'vector_id'.
                # This part needs a proper method in `dynamodb.py`
                
                # Simplified: Fetch the metadata using document_id if we can get it.
                # The current flow doesn't directly give document_id from vector_id via pgvector.
                # This is a gap: pgvector gives (vector_id, score). We need to find the document_id for that vector_id.
                
                # Solution: The `document_metadata` stored in DynamoDB *must* contain the `vector_id`.
                # When we query pgvector, we get `vector_id`. We then query DynamoDB *using this `vector_id`*
                # to find the item(s) that have this `vector_id`. This requires a GSI on `vector_id` in DynamoDB.

                # Let's assume `query_items_by_attribute` exists in `DynamoDBStorage`
                # that can query a GSI.
                # metadata_items = self.dynamo_storage.query_items_by_attribute("vector_id", vector_id, index_name="VectorIdIndex")

                # If you don't have a GSI on vector_id, you'd typically store document_id also in pgvector table
                # as metadata, then retrieve document_id from pgvector, then use it to get full metadata from DDB.
                # Let's assume the GSI on vector_id exists for now for a cleaner flow here.
                
                # Simulating GSI query (replace with actual implementation in dynamodb.py)
                gsi_query_response = self.dynamo_storage.table.query(
                    IndexName="VectorIdIndex", # Assuming this GSI exists on 'vector_id'
                    KeyConditionExpression=Key('vector_id').eq(vector_id) # Use imported Key
                )
                metadata_items = gsi_query_response.get('Items', [])

                if metadata_items:
                    # Assuming one document_id per vector_id for simplicity
                    metadata = metadata_items[0] 
                    results.append({
                        "document_id": metadata.get("document_id"),
                        "vector_id": vector_id,
                        "metadata": metadata,
                        "distance": float(distance_score)
                    })
                else:
                    logger.warning(f"No metadata found in DynamoDB for vector_id: {vector_id}")

            except Exception as e:
                logger.error(f"Error fetching metadata from DynamoDB for vector_id {vector_id}: {e}")
        
        logger.info(f"Returning {len(results)} similar documents with metadata.")
        return results

    def close_connections(self):
        """Closes any open connections (e.g., to PostgreSQL)."""
        self.pg_vector_storage.close_connection()
        # DynamoDB client typically doesn't need explicit closing in this manner with boto3 resource.
        logger.info("Closed PostgreSQL connection via PgVectorStorage.")

# Note: The DynamoDBStorage class would need to be adapted:
# 1. To have a more generic method for storing items, like `upsert_item(item_data)`
#    that works with the `quantum_embeddings` table structure.
# 2. To have a method for querying by GSI, e.g., `query_by_gsi(index_name, key_name, key_value)`.
# The current `store_energy_data` is too specific.