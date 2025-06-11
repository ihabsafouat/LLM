"""
Script to run the data pipeline using ELT approach.
"""
import os
import asyncio
import logging
from dotenv import load_dotenv
from src.data.pipeline.data_pipeline import DataPipeline

def main():
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Get MinIO credentials from environment
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    
    if not minio_access_key or not minio_secret_key:
        raise ValueError("MinIO credentials not found in environment variables")
    
    # Initialize and run pipeline
    pipeline = DataPipeline(
        output_dir="data/raw",
        minio_endpoint="localhost:9000",
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        batch_size=1000  # Process 1000 items at a time
    )
    
    # Define sources and their parameters
    sources = {
        'gutenberg': {'max_books': 100},
        'wikipedia': {'language': 'en'},
        'arxiv': {'max_papers': 100}
    }
    
    # Extract and load data
    raw_data_paths = asyncio.run(pipeline.extract_and_load_all(
        list(sources.keys()),
        **sources
    ))
    
    # Transform and load data in batches
    for source, raw_path in raw_data_paths.items():
        pipeline.transform_and_load(source, raw_path)
    
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    main() 