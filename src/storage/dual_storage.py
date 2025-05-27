"""
Dual storage module for storing document metadata in DynamoDB and vector embeddings in PostgreSQL (via PgVectorStorage).
"""
import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError # Import ClientError
from psycopg2 import sql # Import psycopg2.sql

# Assuming these are in the same directory or PYTHONPATH is set up
from .dynamodb import DynamoDBStorage
from .pgvector_storage import PgVectorStorage

logger = logging.getLogger(__name__)

class DualStorage:
    """
    Manages storage of document metadata in DynamoDB and their corresponding
    vector embeddings in a PostgreSQL database using the pgvector extension.
    """

    def __init__(
        self,
        dynamodb_table_name: Optional[str] = None,
        pg_vector_table_name: str = "document_embeddings",
        pg_db_params: Optional[Dict[str, str]] = None,
        vector_dim: int = 1536,
        region: Optional[str] = None,
        lazy_init_pg: bool = True,
        dynamodb_vector_id_gsi_name: str = "VectorIdIndex",
        app_environment: Optional[str] = None # Added app_environment
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
            dynamodb_vector_id_gsi_name: Name of the GSI on DynamoDB table for vector_id lookups.
            app_environment: The application environment (e.g., 'dev', 'prod') for PgVectorStorage SSM path.
        """
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        
        resolved_dynamodb_table_name = dynamodb_table_name or os.environ.get("DYNAMODB_TABLE", "quantum_embeddings")
        self.dynamodb_vector_id_gsi_name = dynamodb_vector_id_gsi_name

        self.dynamo_storage = DynamoDBStorage(
            table_name=resolved_dynamodb_table_name, 
            region=self.region
        )
        
        self.pg_vector_storage = PgVectorStorage(
            db_params=pg_db_params,
            table_name=pg_vector_table_name,
            vector_dim=vector_dim,
            lazy_init=lazy_init_pg,
            app_environment=app_environment # Pass app_environment
        )
        
        logger.info(f"DualStorage initialized: DynamoDB table '{resolved_dynamodb_table_name}', PgVector table '{pg_vector_table_name}', Vector ID GSI '{self.dynamodb_vector_id_gsi_name}'")

    def _prepare_dynamo_item(self, metadata: Dict[str, Any], vector_id: str) -> Dict[str, Any]:
        """Helper to prepare a consistent item for DynamoDB storage."""
        item = metadata.copy()
        item["vector_id"] = vector_id
        from datetime import datetime, timezone
        item["last_updated"] = item.get("last_updated", datetime.now(timezone.utc).isoformat())

        return {k: v for k, v in item.items() if v is not None}


    def store_document_and_embedding(
        self,
        document_metadata: Dict[str, Any],
        embedding_vector: np.ndarray,
        vector_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stores a single document's metadata in DynamoDB and its embedding in PostgreSQL.

        Args:
            document_metadata: Metadata for DynamoDB. Must include 'document_id'.
            embedding_vector: The embedding vector (numpy array).
            vector_id: Optional pre-generated UUID for the vector. If None, one is created.

        Returns:
            A dictionary with the results of the storage operations.
        """
        doc_id = document_metadata.get("document_id")
        if not doc_id:
            logger.error("Missing 'document_id' in document_metadata.")
            return {"overall_success": False, "error": "Missing document_id", "document_id": None, "vector_id": None}

        if not vector_id:
            vector_id = str(uuid.uuid4())
        
        results = {
            "document_id": doc_id,
            "vector_id": vector_id,
            "dynamo_result": {"success": False, "error": None},
            "pgvector_result": {"success": False, "error": None},
            "overall_success": False
        }

        # 1. Store metadata in DynamoDB
        try:
            dynamo_item_to_store = self._prepare_dynamo_item(document_metadata, vector_id)
            self.dynamo_storage.table.put_item(Item=dynamo_item_to_store)
            results["dynamo_result"] = {"success": True, "message": "Metadata stored in DynamoDB."}
            logger.info(f"Successfully stored metadata for document_id: {doc_id} (vector_id: {vector_id}) in DynamoDB.")

        except ClientError as ce: 
            logger.error(f"DynamoDB ClientError for document_id {doc_id} (vector_id: {vector_id}): {ce}")
            results["dynamo_result"] = {"success": False, "error": str(ce), "error_code": ce.response.get('Error', {}).get('Code')}
        except Exception as e:
            logger.error(f"DynamoDB storage error for document_id {doc_id} (vector_id: {vector_id}): {e}")
            results["dynamo_result"] = {"success": False, "error": str(e)}

        # 2. Store embedding in PostgreSQL via PgVectorStorage
        if results["dynamo_result"]["success"]:
            try:
                pg_success = self.pg_vector_storage.store_embedding(vector_id, embedding_vector)
                if pg_success:
                    results["pgvector_result"] = {"success": True, "message": "Embedding stored in PgVector."}
                    logger.info(f"Successfully stored embedding for vector_id: {vector_id} in PgVector.")
                else:
                    results["pgvector_result"] = {"success": False, "error": "PgVector store_embedding returned False."}
                    logger.error(f"Failed to store embedding for vector_id: {vector_id} in PgVector (method returned False).")
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
        Attempts to store all metadata first, then all corresponding successful embeddings.

        Args:
            documents_with_embeddings: A list of tuples. Each tuple contains:
                - document_metadata (Dict[str, Any]): Metadata for DynamoDB. Must include 'document_id'.
                - embedding_vector (np.ndarray): The embedding vector.
                - vector_id (Optional[str]): Pre-generated UUID for the vector. If None, one is created.
        
        Returns:
            A summary of the batch operation including individual results.
        """
        total_items = len(documents_with_embeddings)
        batch_operation_summary = {
            "total_items_processed": total_items,
            "dynamodb_successful_puts": 0,
            "dynamodb_failed_puts": 0,
            "pgvector_successful_stores": 0,
            "pgvector_failed_stores": 0,
            "overall_successful_documents": 0,
            "individual_results": []
        }

        items_for_dynamo_put = []
        successfully_stored_in_dynamo_for_pg = [] 

        for index, (doc_meta, embedding_vec, vec_id_optional) in enumerate(documents_with_embeddings):
            doc_id = doc_meta.get("document_id")
            item_result = {
                "original_index": index, "document_id": doc_id, "vector_id": None,
                "dynamodb_status": "pending", "dynamodb_error": None,
                "pgvector_status": "pending", "pgvector_error": None,
                "overall_success": False
            }

            if not doc_id:
                item_result.update({"dynamodb_status": "failed", "dynamodb_error": "Missing document_id", "pgvector_status": "skipped"})
                batch_operation_summary["dynamodb_failed_puts"] += 1
                batch_operation_summary["individual_results"].append(item_result)
                continue

            vector_id = vec_id_optional if vec_id_optional else str(uuid.uuid4())
            item_result["vector_id"] = vector_id
            
            dynamo_item_prepared = self._prepare_dynamo_item(doc_meta, vector_id)
            items_for_dynamo_put.append({
                "item_data": dynamo_item_prepared, 
                "original_index": index, 
                "item_result_ref": item_result 
            })
            batch_operation_summary["individual_results"].append(item_result)

        if items_for_dynamo_put:
            logger.info(f"Attempting to store {len(items_for_dynamo_put)} items in DynamoDB.")
            for ddb_payload in items_for_dynamo_put:
                item_to_store = ddb_payload["item_data"]
                current_doc_id = item_to_store.get("document_id")
                current_vector_id = item_to_store.get("vector_id")
                item_result_ref = ddb_payload["item_result_ref"]
                
                try:
                    self.dynamo_storage.table.put_item(Item=item_to_store)
                    item_result_ref["dynamodb_status"] = "success"
                    batch_operation_summary["dynamodb_successful_puts"] += 1
                    
                    _ , original_embedding_vec, _ = documents_with_embeddings[ddb_payload["original_index"]]
                    successfully_stored_in_dynamo_for_pg.append(
                        (current_vector_id, original_embedding_vec, current_doc_id, item_result_ref)
                    )
                except ClientError as ce:
                    logger.error(f"DynamoDB ClientError for document_id {current_doc_id} (vector_id: {current_vector_id}): {ce}")
                    item_result_ref.update({
                        "dynamodb_status": "failed", "dynamodb_error": str(ce), 
                        "pgvector_status": "skipped", "error_code": ce.response.get('Error', {}).get('Code')
                    })
                    batch_operation_summary["dynamodb_failed_puts"] += 1
                except Exception as e:
                    logger.error(f"DynamoDB put_item error for document_id {current_doc_id} (vector_id: {current_vector_id}): {e}")
                    item_result_ref.update({"dynamodb_status": "failed", "dynamodb_error": str(e), "pgvector_status": "skipped"})
                    batch_operation_summary["dynamodb_failed_puts"] += 1
        
        if successfully_stored_in_dynamo_for_pg:
            pg_payloads_to_store = [(item[0], item[1]) for item in successfully_stored_in_dynamo_for_pg] 
            logger.info(f"Attempting to store {len(pg_payloads_to_store)} embeddings in PgVector.")
            try:
                pg_batch_overall_success = self.pg_vector_storage.batch_store_embeddings(pg_payloads_to_store)

                if pg_batch_overall_success:
                    logger.info(f"PgVector batch_store_embeddings reported success for {len(pg_payloads_to_store)} items.")
                    for _, _, _, item_result_ref in successfully_stored_in_dynamo_for_pg:
                        item_result_ref["pgvector_status"] = "success"
                        batch_operation_summary["pgvector_successful_stores"] += 1
                        if item_result_ref["dynamodb_status"] == "success": 
                            item_result_ref["overall_success"] = True
                            batch_operation_summary["overall_successful_documents"] +=1
                else:
                    logger.error(f"PgVector batch_store_embeddings failed for a batch of {len(pg_payloads_to_store)} items (method returned False).")
                    for _, _, _, item_result_ref in successfully_stored_in_dynamo_for_pg:
                        item_result_ref["pgvector_status"] = "failed"
                        item_result_ref["pgvector_error"] = "PgVector batch store reported failure."
                        item_result_ref["overall_success"] = False 
            except Exception as e:
                logger.error(f"Error during PgVector batch_store_embeddings for {len(pg_payloads_to_store)} items: {e}")
                for _, _, _, item_result_ref in successfully_stored_in_dynamo_for_pg:
                    item_result_ref["pgvector_status"] = "failed"
                    item_result_ref["pgvector_error"] = str(e)
                    item_result_ref["overall_success"] = False
        else:
            logger.info("No items were eligible for PgVector batch store (either none successful in DDB or list was empty).")

        final_pg_failed_count = sum(1 for res in batch_operation_summary["individual_results"] if res["pgvector_status"] != "success")
        batch_operation_summary["pgvector_failed_stores"] = final_pg_failed_count
        
        logger.info(f"Batch store operation summary: {batch_operation_summary}")
        return batch_operation_summary

    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves document metadata from DynamoDB by its document_id.
        Assumes 'document_id' is the hash key of the table.
        """
        try:
            response = self.dynamo_storage.table.get_item(Key={'document_id': document_id})
            item = response.get('Item')
            if item:
                logger.info(f"Retrieved metadata for document_id: {document_id}")
                return item
            else:
                logger.info(f"No metadata found for document_id: {document_id}")
                return None
        except ClientError as ce:
            logger.error(f"DynamoDB ClientError getting metadata for document_id {document_id}: {ce}")
            return None
        except Exception as e:
            logger.error(f"Error getting metadata for document_id {document_id}: {e}")
            return None

    def get_embedding_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """
        Retrieves an embedding vector from PgVectorStorage by its vector_id.
        Uses a direct SQL query; ensure PgVectorStorage.execute_select_query is secure.
        """
        try:
            # Validate vector_id format to prevent basic SQL injection risks
            uuid.UUID(vector_id) 
        except ValueError:
            logger.error(f"Invalid vector_id format: {vector_id}")
            return None

        # Construct query using psycopg2.sql for safety
        query = sql.SQL("SELECT embedding FROM {table} WHERE vector_id = %s;").format(
            table=sql.Identifier(self.pg_vector_storage.table_name)
        )
        
        # Convert query to string and pass parameters separately if execute_select_query supports it
        # For now, assuming execute_select_query can take the sql object or a formatted string + params
        # This part needs to align with how execute_select_query in PgVectorStorage is implemented.
        # If it takes a raw string, we must be extremely careful or adapt it.
        # Let's assume it can take a query string and a tuple of params for %s substitution.
        # However, the current PgVectorStorage.execute_select_query takes a single query string.
        # This is a security risk if not handled well in execute_select_query.
        # For now, we will format it directly into the string, relying on the UUID validation above.
        # THIS IS NOT IDEAL FOR SECURITY.
        query_string = f"SELECT embedding FROM \"{self.pg_vector_storage.table_name}\" WHERE vector_id = '{vector_id}';"

        try:
            cols, rows, error = self.pg_vector_storage.execute_select_query(query_string)
            if error:
                logger.error(f"Error fetching embedding for {vector_id} from PgVector: {error}")
                return None
            if rows and rows[0] and rows[0][0]:
                embedding_data = rows[0][0]
                if isinstance(embedding_data, str):
                    import ast
                    embedding_data = ast.literal_eval(embedding_data) 
                return np.array(embedding_data, dtype=np.float32)
            else:
                logger.info(f"No embedding found for vector_id: {vector_id}")
                return None
        except Exception as e:
            logger.error(f"Error processing result from PgVector for {vector_id}: {e}")
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
        2. Retrieves metadata for these vector_ids from DynamoDB using the configured GSI.
        """
        similar_vectors = self.pg_vector_storage.find_similar_embeddings(
            query_embedding, top_k=top_k, metric=metric
        )

        results = []
        if not similar_vectors:
            logger.info("No similar vectors found in PgVector.")
            return []

        logger.info(f"Found {len(similar_vectors)} candidate vectors. Fetching metadata from DynamoDB using GSI '{self.dynamodb_vector_id_gsi_name}'.")
        for vector_id, distance_score in similar_vectors:
            try:
                gsi_query_response = self.dynamo_storage.table.query(
                    IndexName=self.dynamodb_vector_id_gsi_name, 
                    KeyConditionExpression=Key('vector_id').eq(vector_id)
                )
                metadata_items = gsi_query_response.get('Items', [])

                if metadata_items:
                    for metadata in metadata_items: 
                        results.append({
                            "document_id": metadata.get("document_id"),
                            "vector_id": vector_id, 
                            "metadata": metadata,
                            "distance": float(distance_score) 
                        })
                else:
                    logger.warning(f"No metadata found in DynamoDB for vector_id: {vector_id} using GSI '{self.dynamodb_vector_id_gsi_name}'.")

            except ClientError as ce: 
                error_code = ce.response.get('Error', {}).get('Code')
                if error_code == 'ResourceNotFoundException' or 'ValidationException' in str(ce):
                    logger.error(f"Error querying GSI '{self.dynamodb_vector_id_gsi_name}' for vector_id {vector_id}. The GSI may not exist or is misconfigured: {ce}")
                else:
                    logger.error(f"DynamoDB GSI query ClientError for vector_id {vector_id}: {ce}")
            except Exception as e:
                logger.error(f"Unexpected error fetching metadata from DynamoDB for vector_id {vector_id}: {e}")
        
        logger.info(f"Returning {len(results)} similar documents with metadata.")
        return results

    def close_connections(self):
        """Closes any open connections (e.g., to PostgreSQL).""" 
        if hasattr(self.pg_vector_storage, 'close_db_connection'):
            self.pg_vector_storage.close_db_connection()
        logger.info("DualStorage connections closed (if applicable).")

    def get_embedding_by_id(self, vector_id: str) -> Optional[np.ndarray]:
        """
        Retrieves an embedding vector from PgVectorStorage by its vector_id.
        This is a more robust implementation than the one in get_embedding_vector.
        It assumes PgVectorStorage.execute_select_query is safe and returns structured data.
        """
        if not self.pg_vector_storage or not hasattr(self.pg_vector_storage, 'execute_select_query'):
            logger.error("PgVectorStorage not available or does not have 'execute_select_query' method.")
            return None
        if not self.pg_vector_storage.table_name:
             logger.error("PgVectorStorage table_name is not configured.")
             return None

        try:
            uuid.UUID(vector_id) 
        except ValueError:
            logger.error(f"Invalid vector_id format: {vector_id}")
            return None

        # Using f-string for query construction after UUID validation.
        # This is safer than direct concatenation but parameterized queries are best.
        query = f"SELECT embedding FROM \"{self.pg_vector_storage.table_name}\" WHERE vector_id = '{vector_id}';"
        
        try:
            column_names, rows, error_msg = self.pg_vector_storage.execute_select_query(query)

            if error_msg:
                logger.error(f"Error retrieving embedding for vector_id {vector_id} from PgVectorStorage: {error_msg}")
                return None

            if rows and len(rows) > 0 and rows[0] and len(rows[0]) > 0:
                embedding_data = rows[0][0] 
                
                if isinstance(embedding_data, str):
                    import ast
                    try:
                        embedding_list = ast.literal_eval(embedding_data)
                    except (ValueError, SyntaxError) as e:
                        logger.error(f"Failed to parse embedding string for vector_id {vector_id}: {embedding_data}, error: {e}")
                        return None
                elif isinstance(embedding_data, list):
                    embedding_list = embedding_data
                else:
                    logger.error(f"Unexpected embedding data type for vector_id {vector_id}: {type(embedding_data)}")
                    return None
                
                return np.array(embedding_list, dtype=np.float32)
            else:
                logger.info(f"No embedding found for vector_id: {vector_id}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error in get_embedding_by_id for vector_id {vector_id}: {e}", exc_info=True)
            return None