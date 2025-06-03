import sys
import os
import logging
import psycopg2

# Adjust path to import from src
# This assumes the script is in 'quantum_work/scripts/' and 'src' is in 'quantum_work/'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.storage.pgvector_storage import PgVectorStorage
except ImportError as e:
    print(f"Error importing PgVectorStorage: {e}")
    print("Please ensure the script is run from the 'quantum_work/scripts' directory or the PYTHONPATH is set correctly.")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_historical_weather_table():
    """
    Creates the 'historical_weather_data' table in PostgreSQL using
    the connection handling from PgVectorStorage.
    """
    pg_storage_instance = None
    conn = None

    try:
        logger.info("Initializing PgVectorStorage to obtain database connection details...")
        # Instantiate PgVectorStorage. It will load DB parameters from environment variables
        # by default if no db_params are passed.
        # lazy_init=True is the default, so it won't try to create its own schema yet.
        pg_storage_instance = PgVectorStorage()

        logger.info("Attempting to get PostgreSQL connection via PgVectorStorage...")
        # Accessing the protected _get_connection method as per user request context.
        # This method establishes a new connection if one isn't active or returns an existing one.
        conn = pg_storage_instance._get_connection()

        if not conn:
            logger.error("Failed to get PostgreSQL connection from PgVectorStorage. Aborting.")
            return

        with conn.cursor() as cur:
            logger.info("Creating 'historical_weather_data' table if it does not exist...")
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS historical_weather_data (
                timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL PRIMARY KEY,
                houston_temp_c REAL,
                austin_temp_c REAL,
                dallas_temp_c REAL,
                san_antonio_temp_c REAL,
                fort_worth_temp_c REAL,
                corpus_christi_temp_c REAL,
                avg_temperature_c REAL,
                avg_temperature_f REAL
            );
            """
            cur.execute(create_table_sql)
            logger.info("'historical_weather_data' table created or already exists.")

            logger.info("Adding comments to 'historical_weather_data' table and columns...")
            comments = [
                "COMMENT ON TABLE historical_weather_data IS 'Stores historical hourly weather data for major Texas cities.';",
                "COMMENT ON COLUMN historical_weather_data.timestamp IS 'The specific date and time of the weather reading (UTC or a consistent timezone).';",
                "COMMENT ON COLUMN historical_weather_data.houston_temp_c IS 'Temperature in Houston, Texas, in Celsius.';",
                "COMMENT ON COLUMN historical_weather_data.austin_temp_c IS 'Temperature in Austin, Texas, in Celsius.';",
                "COMMENT ON COLUMN historical_weather_data.dallas_temp_c IS 'Temperature in Dallas, Texas, in Celsius.';",
                "COMMENT ON COLUMN historical_weather_data.san_antonio_temp_c IS 'Temperature in San Antonio, Texas, in Celsius.';",
                "COMMENT ON COLUMN historical_weather_data.fort_worth_temp_c IS 'Temperature in Fort Worth, Texas, in Celsius.';",
                "COMMENT ON COLUMN historical_weather_data.corpus_christi_temp_c IS 'Temperature in Corpus Christi, Texas, in Celsius.';",
                "COMMENT ON COLUMN historical_weather_data.avg_temperature_c IS 'Average temperature across the monitored cities, in Celsius.';",
                "COMMENT ON COLUMN historical_weather_data.avg_temperature_f IS 'Average temperature across the monitored cities, in Fahrenheit.';"
            ]
            for comment_sql in comments:
                cur.execute(comment_sql)
            logger.info("Comments added successfully to 'historical_weather_data'.")

        conn.commit()
        logger.info("Table creation and commenting process committed successfully.")

    except psycopg2.Error as e:
        logger.error(f"PostgreSQL Error occurred: {e}")
        if conn:
            try:
                conn.rollback()
                logger.info("Transaction rolled back due to error.")
            except psycopg2.Error as rb_e:
                logger.error(f"Error during rollback: {rb_e}")
    except ImportError:
        # Handled by the initial try-except for PgVectorStorage import
        pass
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if conn: # Should not happen if psycopg2.Error is caught, but for safety
            try:
                conn.rollback()
                logger.info("Transaction rolled back due to unexpected error.")
            except psycopg2.Error as rb_e:
                logger.error(f"Error during rollback on unexpected error: {rb_e}")
    finally:
        if conn and not conn.closed:
            conn.close()
            logger.info("PostgreSQL connection closed.")
        # Note: PgVectorStorage.close_connection() closes its own self.pg_conn.
        # Since we got 'conn' directly from _get_connection(), we manage it here.

if __name__ == "__main__":
    logger.info("Starting script to create historical weather table...")
    create_historical_weather_table()
    logger.info("Script finished.")
