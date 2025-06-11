"""
Script to initialize the validation database.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from src.data.validation.models import Base

# Load environment variables
load_dotenv()

def init_database():
    """Initialize the validation database."""
    # Get database URL from environment
    db_url = os.getenv('DB_URL')
    if not db_url:
        raise ValueError("Database URL not found in environment variables")
    
    # Create engine
    engine = create_engine(db_url)
    
    # Create database if it doesn't exist
    if not database_exists(engine.url):
        create_database(engine.url)
        print(f"Created database: {engine.url.database}")
    
    # Create tables
    Base.metadata.create_all(engine)
    print("Created database tables")
    
    print("Database initialization completed successfully")

if __name__ == "__main__":
    init_database() 