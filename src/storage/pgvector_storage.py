"""
Storage module for managing vector embeddings in PostgreSQL with pgvector.
"""
import os
import logging
import psycopg2
import numpy as np
import pandas as pd # Add pandas import
from psycopg2 import sql
from psycopg2.extras import Json, execute_values # type: ignore
from typing import List, Tuple, Optional, Dict, Any

try:
    import boto3
except ImportError:
    boto3 = None # type: ignore
    logging.getLogger(__name__).warning("boto3 not found, AWS SSM Parameter Store functionality will be disabled.")

from dotenv import load_dotenv # Import dotenv
load_dotenv() # Load .env file from root or parent directories

logger = logging.getLogger(__name__)

class PgVectorStorage:
    """
    Manages storage/retrieval of vector embeddings in PostgreSQL with pgvector.
    """
    def __init__(
        self,
        db_params: Optional[Dict[str, str]] = None,
        table_name: str = "document_embeddings",
        vector_dim: int = 1536, # Default for OpenAI text-embedding-3-small
        lazy_init: bool = True,
        app_environment: Optional[str] = None # For SSM path
    ):
        """
        Initialize PgVectorStorage.

        Args:
            db_params: PostgreSQL connection parameters. Loads from SSM then ENV if None.
            table_name: Name of the embeddings table.
            vector_dim: Dimension of embeddings.
            lazy_init: If True, defer schema initialization.
            app_environment: App environment for SSM path (e.g., 'dev', 'prod').
        """
        self.table_name = table_name
        self.vector_dim = vector_dim
        self.pg_conn: Optional[psycopg2.extensions.connection] = None
        self.pg_params: Optional[Dict[str, str]] = db_params
        self.schema_initialized = False
        self.app_environment = app_environment or os.environ.get("APP_ENVIRONMENT", "prod")
        logger.info(f"PgVectorStorage: Initializing with app_environment='{self.app_environment}'") # Added log

        if not self.pg_params:
            logger.info("PgVectorStorage: db_params not provided, attempting to load from AWS SSM Parameter Store...")
            if boto3 and self._load_db_params_from_ssm(): # _load_db_params_from_ssm logs its own success/failure
                pass # Logging is handled inside the method
            else:
                # This block is reached if boto3 is None OR _load_db_params_from_ssm returns False
                logger.info("PgVectorStorage: Failed to load DB params from SSM or boto3 not available. Falling back to environment variables.")
                self._load_db_params_from_env() # _load_db_params_from_env logs the params it loads
        
        if self.pg_params:
            # Log successfully loaded parameters, masking password
            masked_params = {k: (v if k != 'password' else '********') for k, v in self.pg_params.items()}
            logger.info(f"PgVectorStorage: Resolved DB parameters: {masked_params}")
        else:
            logger.warning("PgVectorStorage: DB connection parameters NOT successfully loaded after all attempts (SSM, Env Vars). Connection attempts will likely fail.")

        if not lazy_init:
            self._init_schema()

    def _load_db_params_from_ssm(self) -> bool:
        """
        Loads PostgreSQL params from AWS SSM. Uses self.app_environment.
        Returns True on success, False otherwise.
        """
        if not boto3:
            logger.warning("PgVectorStorage: boto3 is not installed or importable. Cannot load DB params from SSM.") # Added PgVectorStorage prefix
            return False
        if not self.app_environment:
            logger.warning("PgVectorStorage: APP_ENVIRONMENT is not set. Cannot load DB params from SSM.") # Added PgVectorStorage prefix
            return False

        try:
            ssm_client = boto3.client('ssm', region_name=os.environ.get("AWS_REGION", "us-east-1"))
            base_path = f"/{self.app_environment}/quantum-rag/db"

            param_names_map = {
                "host": f"{base_path}/address",
                "port": f"{base_path}/port",
                "dbname": f"{base_path}/name",
                "user": f"{base_path}/username",
                "password": f"{base_path}/password" # SecureString
            }

            loaded_params: Dict[str, str] = {}
            for key, param_name in param_names_map.items():
                try:
                    response = ssm_client.get_parameter(
                        Name=param_name,
                        WithDecryption=(key == "password") # Only decrypt password
                    )
                    value = response.get('Parameter', {}).get('Value')
                    if value is None:
                        logger.error(f"PgVectorStorage: SSM parameter '{param_name}' is missing Value field in response.") # Added PgVectorStorage prefix
                        return False
                    loaded_params[key] = value
                except ssm_client.exceptions.ParameterNotFound:
                    logger.error(f"PgVectorStorage: SSM parameter '{param_name}' not found.") # Added PgVectorStorage prefix
                    return False
                except Exception as e:
                    logger.error(f"PgVectorStorage: Failed to fetch SSM parameter '{param_name}': {e}") # Added PgVectorStorage prefix
                    return False
            
            # Ensure all required parameters were loaded
            required_keys = ["host", "port", "dbname", "user", "password"]
            if not all(key in loaded_params for key in required_keys):
                logger.error(f"PgVectorStorage: One or more required DB parameters missing after SSM load attempt. Loaded: {list(loaded_params.keys())}") # Added PgVectorStorage prefix
                return False

            self.pg_params = loaded_params
            logger.info(f"PgVectorStorage: Successfully loaded PostgreSQL params from SSM: host={self.pg_params.get('host')}, dbname={self.pg_params.get('dbname')}") # Added PgVectorStorage prefix
            return True

        except Exception as e:
            logger.error(f"PgVectorStorage: Error initializing SSM client or loading parameters: {e}") # Added PgVectorStorage prefix
            return False

    def _load_db_params_from_env(self):
        """Loads PostgreSQL connection params from environment variables."""
        self.pg_params = {
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": os.environ.get("DB_PORT", "5432"),
            "dbname": os.environ.get("DB_NAME", "energy_data"),
            "user": os.environ.get("DB_USER", "energyadmin"),
            "password": os.environ.get("DB_PASSWORD", "") # Ensure DB_PASSWORD is set
        }
        # Log loaded env vars, masking password
        masked_env_params = {k: (v if k != 'password' else '********') for k, v in self.pg_params.items()}
        logger.info(f"PgVectorStorage: Loaded PostgreSQL params from environment: {masked_env_params}") # Enhanced log
        if not self.pg_params.get("password"):
            logger.warning("PgVectorStorage: DB_PASSWORD environment variable is not set or is empty.")


    def close_db_connection(self) -> None:
        """Closes the PostgreSQL connection if open."""
        if self.pg_conn and not self.pg_conn.closed:
            try:
                self.pg_conn.close()
                logger.info("PostgreSQL connection closed successfully.")
            except psycopg2.Error as e:
                logger.error(f"Error closing PostgreSQL connection: {e}")
            finally:
                self.pg_conn = None
        else:
            logger.info("PostgreSQL connection was not open or already closed.")

    def _get_connection(self, retry: bool = True) -> Optional[psycopg2.extensions.connection]:
        """
        Establishes or returns an existing PostgreSQL connection.

        Args:
            retry: Whether to retry connection on failure.
        Returns:
            psycopg2 connection or None on failure.
        """
        if self.pg_conn is None or self.pg_conn.closed:
            if not self.pg_params:
                logger.error("PostgreSQL connection parameters are not set.")
                return None
            
            # Ensure all parameters are strings, especially port if it was loaded as int elsewhere
            connect_params = {k: str(v) for k, v in self.pg_params.items()}

            try:
                self.pg_conn = psycopg2.connect(**connect_params) # type: ignore
                # self.pg_conn.autocommit = False # Use autocommit False for explicit transaction control
                logger.info(f"Successfully connected to PostgreSQL database '{connect_params.get('dbname')}' on {connect_params.get('host')}.")
            except psycopg2.Error as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                self.pg_conn = None
                if not retry:
                    raise
        return self.pg_conn

    def _init_schema(self):
        """
        Initializes DB schema: pgvector extension and embeddings table.
        """
        if self.schema_initialized:
            return True

        conn = self._get_connection(retry=False)
        if not conn:
            logger.warning("Skipping schema initialization as PostgreSQL connection is unavailable.")
            return False

        try:
            with conn.cursor() as cur:
                logger.info("Initializing pgvector schema...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                table_creation_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    vector_id UUID PRIMARY KEY,
                    embedding VECTOR({vector_dim}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """).format(
                    table=sql.Identifier(self.table_name),
                    vector_dim=sql.Literal(self.vector_dim)
                )
                cur.execute(table_creation_query)

                index_name = f"idx_{self.table_name}_embedding_hnsw"
                cur.execute(f"SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = '{index_name}' AND n.nspname = 'public');") # Assumes public schema
                row = cur.fetchone()
                index_exists = row[0] if row else False

                if not index_exists:
                    # Using HNSW index as an example. For cosine distance, use vector_cosine_ops.
                    # For L2 distance, use vector_l2_ops.
                    # m and ef_construction are HNSW parameters.
                    index_creation_query = sql.SQL("""
                    CREATE INDEX {index_name} ON {table} USING HNSW (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                    """).format(
                        index_name=sql.Identifier(index_name),
                        table=sql.Identifier(self.table_name)
                    )
                    # cur.execute(index_creation_query) # Uncomment to create HNSW index
                    # logger.info(f"Created HNSW index '{index_name}' on {self.table_name}.embedding")

                    ivfflat_index_name = f"idx_{self.table_name}_embedding_ivfflat"
                    cur.execute(f"SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = '{ivfflat_index_name}' AND n.nspname = 'public');")
                    row = cur.fetchone()
                    ivfflat_index_exists = row[0] if row else False
                    if not ivfflat_index_exists:
                        ivfflat_index_creation_query = sql.SQL("""
                        CREATE INDEX {index_name} ON {table} USING IVFFLAT (embedding vector_cosine_ops)
                        WITH (lists = 100);
                        """).format(
                            index_name=sql.Identifier(ivfflat_index_name),
                            table=sql.Identifier(self.table_name)
                        )
                        cur.execute(ivfflat_index_creation_query)
                        logger.info(f"Created IVFFlat index '{ivfflat_index_name}' on {self.table_name}.embedding")


            conn.commit()
            self.schema_initialized = True
            logger.info(f"pgvector schema initialized successfully for table '{self.table_name}'.")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error initializing pgvector schema: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close()
    
    def store_embedding(self, vector_id: str, embedding: np.ndarray) -> bool:
        """
        Stores a single embedding vector. Upserts on conflict.

        Args:
            vector_id: UUID string for the vector.
            embedding: Numpy array of the embedding.
        Returns:
            True on success, False otherwise.
        """
        if not self.schema_initialized:
            self._init_schema()
            if not self.schema_initialized: # Check again after attempting init
                logger.error("Cannot store embedding, schema not initialized and initialization failed.")
                return False

        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                # pgvector expects a list or string representation of the vector
                embedding_list = embedding.tolist()
                
                upsert_query = sql.SQL("""
                INSERT INTO {table} (vector_id, embedding, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (vector_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP;
                """).format(table=sql.Identifier(self.table_name))
                
                cur.execute(upsert_query, (vector_id, embedding_list))
            conn.commit()
            logger.info(f"Successfully stored/updated embedding for vector_id: {vector_id}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error storing embedding for vector_id {vector_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close()

    def batch_store_embeddings(self, embeddings_data: List[Tuple[str, np.ndarray]]) -> bool:
        """
        Stores multiple embeddings in a batch. Upserts on conflict.

        Args:
            embeddings_data: List of (vector_id, embedding_array) tuples.
        Returns:
            True on success, False otherwise.
        """
        if not embeddings_data:
            return True 
        
        if not self.schema_initialized:
            self._init_schema()
            if not self.schema_initialized:
                logger.error("Cannot batch store embeddings, schema not initialized and initialization failed.")
                return False

        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                # Prepare data for execute_values
                # (vector_id, embedding_list_as_string, created_at, updated_at)
                # pgvector can take string representation '[1,2,3]'
                data_to_insert = [
                    (vector_id, np.array(embedding).tolist()) for vector_id, embedding in embeddings_data
                ]

                upsert_query = sql.SQL("""
                INSERT INTO {table} (vector_id, embedding, updated_at)
                VALUES %s
                ON CONFLICT (vector_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP;
                """).format(table=sql.Identifier(self.table_name))

                # page_size is important for performance with large batches
                execute_values(cur, upsert_query, data_to_insert, template=None, page_size=100)
            conn.commit()
            logger.info(f"Successfully batch stored/updated {len(embeddings_data)} embeddings.")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error batch storing embeddings: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close()
        return True # Assuming success if no exceptions

    def insert_dataframe_to_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        on_conflict_do_update: bool = False,
        conflict_target_columns: Optional[List[str]] = None,
        conflict_update_columns: Optional[List[str]] = None
    ) -> bool:
        """
        Inserts a pandas DataFrame into the specified PostgreSQL table.

        Args:
            df: Pandas DataFrame to insert.
            table_name: Name of the target table.
            on_conflict_do_update: If True, performs an UPSERT.
                                   Requires conflict_target_columns.
            conflict_target_columns: List of column(s) for conflict detection (e.g., primary key).
                                     Required if on_conflict_do_update is True.
            conflict_update_columns: List of column(s) to update on conflict.
                                     If None and on_conflict_do_update is True, all columns except target are updated.
        Returns:
            True if insertion was successful, False otherwise.
        """
        if df.empty:
            logger.info(f"DataFrame is empty. No data to insert into table \'{table_name}\'.")
            return True # Or False, depending on desired behavior for empty df

        conn = self._get_connection()
        if not conn:
            logger.error(f"Cannot insert DataFrame into \'{table_name}\', no database connection.")
            return False

        try:
            with conn.cursor() as cur:
                # Prepare column names and data tuples
                columns = df.columns.tolist()
                # Convert NaT/NaN to None for SQL compatibility
                data_tuples = [tuple(x.item() if hasattr(x, 'item') else x for x in record) for record in df.where(pd.notnull(df), None).to_records(index=False)]

                # Base INSERT query
                insert_sql_template = "INSERT INTO {table} ({cols}) VALUES %s"

                if on_conflict_do_update:
                    if not conflict_target_columns:
                        logger.error("conflict_target_columns must be specified for ON CONFLICT DO UPDATE.")
                        conn.rollback() # Rollback before returning
                        return False
                    
                    target_cols_sql = ", ".join([sql.Identifier(col).strings[0] for col in conflict_target_columns])

                    if conflict_update_columns:
                        update_cols_sql = ", ".join(
                            [f"{sql.Identifier(col).strings[0]} = EXCLUDED.{sql.Identifier(col).strings[0]}" for col in conflict_update_columns]
                        )
                    else: # Update all columns not in conflict_target_columns
                        update_cols = [col for col in columns if col not in conflict_target_columns]
                        if not update_cols:
                            logger.warning(f"No columns to update for table \'{table_name}\' on conflict. All columns are part of conflict target.")
                            # Proceed with insert or treat as error? For now, proceed.
                            update_cols_sql = "" # This will effectively make it DO NOTHING if all cols are target
                        else:
                            update_cols_sql = ", ".join(
                                [f"{sql.Identifier(col).strings[0]} = EXCLUDED.{sql.Identifier(col).strings[0]}" for col in update_cols]
                            )
                    
                    if update_cols_sql: # Only add DO UPDATE if there's something to update
                        insert_sql_template += f" ON CONFLICT ({target_cols_sql}) DO UPDATE SET {update_cols_sql}"
                    else: # If no update_cols_sql, it means either all columns are conflict targets or no update columns specified
                          # In this case, ON CONFLICT DO NOTHING might be more appropriate if that's the intent.
                          # For now, if update_cols_sql is empty, it will be an INSERT ... ON CONFLICT DO NOTHING (implicitly)
                          # or an error if target_cols_sql is also empty (caught above).
                          # To be explicit for DO NOTHING:
                        insert_sql_template += f" ON CONFLICT ({target_cols_sql}) DO NOTHING"


                query = sql.SQL(insert_sql_template).format(
                    table=sql.Identifier(table_name),
                    cols=sql.SQL(', ').join(map(sql.Identifier, columns))
                )
                
                logger.info(f"Executing batch insert/upsert into \'{table_name}\' with {len(data_tuples)} rows. Sample query: {cur.mogrify(query, (data_tuples[0],)).decode('utf-8')[:500]}...")

                execute_values(cur, query, data_tuples, page_size=100)
            conn.commit()
            logger.info(f"Successfully inserted/updated {len(data_tuples)} rows into table \'{table_name}\'.")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error inserting DataFrame into table \'{table_name}\': {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred while inserting DataFrame into table \'{table_name}\': {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close() # Close connection after operation

    def delete_embedding_by_id(self, vector_id: str) -> bool:
        """
        Deletes an embedding vector by its ID.

        Args:
            vector_id: UUID string of the vector.
        Returns:
            True on success, False otherwise.
        """
        if not self.schema_initialized:
            logger.warning("Schema not initialized. Attempting to initialize now.")
            self._init_schema()
            if not self.schema_initialized:
                logger.error("Cannot delete embedding, schema not initialized and initialization failed.")
                return False

        conn = self._get_connection()
        if not conn:
            logger.error("Cannot delete embedding, no database connection.")
            return False

        try:
            with conn.cursor() as cur:
                query = sql.SQL("""
                    DELETE FROM {table}
                    WHERE vector_id = %s;
                """).format(table=sql.Identifier(self.table_name))

                cur.execute(query, (vector_id,))
            conn.commit()
            logger.info(f"Successfully deleted embedding for vector_id: {vector_id}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error deleting embedding for vector_id {vector_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close()
