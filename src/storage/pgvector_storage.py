"""
Storage module for data loading into PostgreSQL with pgvector support.
"""
import os
import logging
import psycopg2
import numpy as np
from psycopg2 import sql
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class PgVectorStorage:
    """Data loading class for PostgreSQL with pgvector support."""
    
    def __init__(self, table_name: str = "document_embeddings", app_environment: Optional[str] = None):
        """Initialize storage with database connection."""
        self.table_name = table_name
        self.app_environment = app_environment or os.environ.get("APP_ENVIRONMENT", "prod")
        self.pg_conn = None
        self.pg_params = self._load_db_params()
        
    def _load_db_params(self):
        """Load database parameters from AWS SSM or environment variables."""
        # Try AWS SSM first
        try:
            import boto3
            ssm_client = boto3.client('ssm', region_name=os.environ.get("AWS_REGION", "us-east-1"))
            base_path = f"/{self.app_environment}/quantum-rag/db"
            
            params = {}
            for key, param_name in {
                "host": f"{base_path}/address",
                "port": f"{base_path}/port", 
                "dbname": f"{base_path}/name",
                "user": f"{base_path}/username",
                "password": f"{base_path}/password"
            }.items():
                response = ssm_client.get_parameter(Name=param_name, WithDecryption=(key == "password"))
                params[key] = response['Parameter']['Value']
            
            logger.info(f"Loaded DB params from AWS SSM for {params.get('host')}")
            return params
            
        except Exception:
            # Fallback to environment variables
            params = {
                "host": os.environ.get("DB_HOST", "localhost"),
                "port": os.environ.get("DB_PORT", "5432"),
                "dbname": os.environ.get("DB_NAME", "energy_data"),
                "user": os.environ.get("DB_USER", "energyadmin"),
                "password": os.environ.get("DB_PASSWORD", "")
            }
            logger.info(f"Loaded DB params from environment for {params.get('host')}")
            return params if params.get("password") else None

    def _get_connection(self):
        """Get database connection."""
        if not self.pg_conn or self.pg_conn.closed:
            if not self.pg_params:
                logger.error("No database parameters available")
                return None
            try:
                # Use only the basic connection parameters that psycopg2 expects
                params = {
                    'host': self.pg_params['host'],
                    'port': int(self.pg_params['port']),
                    'dbname': self.pg_params['dbname'],
                    'user': self.pg_params['user'],
                    'password': self.pg_params['password']
                }
                self.pg_conn = psycopg2.connect(**params)
                logger.info(f"Connected to {self.pg_params.get('host')}")
            except psycopg2.Error as e:
                logger.error(f"Connection failed: {e}")
                return None
        return self.pg_conn

    def store_embedding(self, vector_id: str, embedding: np.ndarray, semantic_sentence: Optional[str] = None) -> bool:
        """Store vector embedding with optional semantic sentence."""
        conn = self._get_connection()
        if not conn:
            return False

        try:
            # Ensure table exists with correct schema
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        vector_id TEXT PRIMARY KEY,
                        embedding VECTOR(1536),
                        semantic_sentence TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """).format(table=sql.Identifier(self.table_name)))

                # Insert/update embedding with semantic sentence
                cur.execute(sql.SQL("""
                    INSERT INTO {table} (vector_id, embedding, semantic_sentence, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (vector_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        semantic_sentence = EXCLUDED.semantic_sentence,
                        updated_at = CURRENT_TIMESTAMP;
                """).format(table=sql.Identifier(self.table_name)), 
                (vector_id, embedding.tolist(), semantic_sentence))

            conn.commit()
            logger.info(f"Stored embedding for {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            print(f"Exception details: {type(e).__name__}: {e}")
            conn.rollback()
            return False

    def insert_dataframe_to_table(self, df, table_name: str) -> bool:
        """Insert DataFrame into table."""
        if df.empty:
            return True
            
        conn = self._get_connection()
        if not conn:
            return False

        try:
            import pandas as pd
            from psycopg2.extras import execute_values
            
            with conn.cursor() as cur:
                columns = df.columns.tolist()
                data = [tuple(x.item() if hasattr(x, 'item') else x for x in record) 
                       for record in df.where(pd.notnull(df), None).to_records(index=False)]

                query = sql.SQL("INSERT INTO {table} ({cols}) VALUES %s").format(
                    table=sql.Identifier(table_name),
                    cols=sql.SQL(', ').join(map(sql.Identifier, columns))
                )
                execute_values(cur, query, data, page_size=100)
                
            conn.commit()
            logger.info(f"Inserted {len(data)} rows into {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            conn.rollback()
            return False

    def close_db_connection(self):
        """Close database connection."""
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
            logger.info("Connection closed")
