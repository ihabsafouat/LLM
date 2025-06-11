"""
Script to run the entire data pipeline.
"""
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from src.data.pipeline.data_pipeline import DataPipeline
from scripts.validate_data import validate_data

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'pipeline.log')),
        logging.StreamHandler()
    ]
)

async def main():
    try:
        # Initialize pipeline
        pipeline = DataPipeline(
            output_dir=Path('data'),
            minio_endpoint=os.getenv('MINIO_ENDPOINT'),
            minio_access_key=os.getenv('MINIO_ACCESS_KEY'),
            minio_secret_key=os.getenv('MINIO_SECRET_KEY'),
            batch_size=int(os.getenv('BATCH_SIZE', '1000'))
        )
        
        # Process data from all sources
        sources = ['gutenberg', 'wikipedia', 'arxiv']
        
        # Extract and load data
        logging.info("Starting data extraction and loading...")
        raw_data_paths = await pipeline.extract_and_load_all(
            sources,
            max_books=100,  # for Gutenberg
            language='en',  # for Wikipedia
            max_papers=100  # for arXiv
        )
        
        # Transform and load data in batches
        logging.info("Starting data transformation and loading...")
        for source, raw_path in raw_data_paths.items():
            pipeline.transform_and_load(source, raw_path)
        
        # Validate data
        logging.info("Starting data validation...")
        validate_data()
        
        # Cleanup
        logging.info("Cleaning up temporary files...")
        pipeline.cleanup()
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 