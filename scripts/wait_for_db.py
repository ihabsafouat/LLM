"""
Script to check database connectivity and wait for it to be ready.
"""
import os
import time
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def wait_for_db(max_retries=30, retry_interval=2):
    """Wait for database to be ready."""
    db_url = os.getenv('POSTGRES_URI', 'postgresql://airflow:airflow@postgres/airflow')
    
    for i in range(max_retries):
        try:
            # Try to connect to the database
            conn = psycopg2.connect(db_url)
            conn.close()
            print("Database is ready!")
            return True
        except psycopg2.OperationalError as e:
            print(f"Database not ready yet... (attempt {i+1}/{max_retries})")
            if i < max_retries - 1:
                time.sleep(retry_interval)
            else:
                print("Could not connect to database after maximum retries")
                return False

if __name__ == "__main__":
    wait_for_db() 