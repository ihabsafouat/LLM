"""
Main data pipeline for processing and storing data from multiple sources using ELT approach.
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator
from datetime import datetime
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
import lzma

from ..connectors.data_connectors import DataConnectorFactory
from ..lake.minio_connector import MinioConnector

# Load environment variables
load_dotenv()

class DataPipeline:
    def __init__(self, 
                 output_dir: str = None,
                 minio_endpoint: str = None,
                 minio_access_key: str = None,
                 minio_secret_key: str = None,
                 batch_size: int = None):
        """
        Initialize the data pipeline.
        
        Args:
            output_dir: Directory for temporary storage
            minio_endpoint: MinIO server endpoint
            minio_access_key: MinIO access key
            minio_secret_key: MinIO secret key
            batch_size: Size of data batches for processing
        """
        # Load configuration from environment variables
        self.output_dir = Path(output_dir or os.getenv('OUTPUT_DIR', 'data/raw'))
        self.batch_size = batch_size or int(os.getenv('BATCH_SIZE', '1000'))
        
        # Initialize MinIO connector
        self.minio = MinioConnector(
            endpoint=minio_endpoint or os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
            access_key=minio_access_key or os.getenv('MINIO_ACCESS_KEY'),
            secret_key=minio_secret_key or os.getenv('MINIO_SECRET_KEY'),
            secure=os.getenv('MINIO_SECURE', 'false').lower() == 'true'
        )
        
        # Initialize Spark session with configuration
        self.spark = SparkSession.builder \
            .appName("DataPipeline") \
            .config("spark.sql.warehouse.dir", os.getenv('SPARK_WAREHOUSE_DIR', str(self.output_dir / "spark-warehouse"))) \
            .config("spark.driver.memory", os.getenv('SPARK_DRIVER_MEMORY', '4g')) \
            .config("spark.executor.memory", os.getenv('SPARK_EXECUTOR_MEMORY', '2g')) \
            .config("spark.executor.cores", os.getenv('SPARK_EXECUTOR_CORES', '2')) \
            .getOrCreate()
            
        # Create MinIO buckets
        self.raw_bucket = "gpt-raw-data"
        self.processed_bucket = "gpt-processed-data"
        self.minio.create_bucket(self.raw_bucket)
        self.minio.create_bucket(self.processed_bucket)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.getenv('LOG_FILE', 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
        
    def batch_generator(self, data: List[Dict[str, Any]]) -> Generator[List[Dict[str, Any]], None, None]:
        """Generate batches of data."""
        for i in range(0, len(data), self.batch_size):
            yield data[i:i + self.batch_size]
            
    async def extract_and_load(self, source: str, **kwargs) -> str:
        """
        Extract data from source and load into raw storage.
        Returns the path to the raw data in MinIO.
        """
        logging.info(f"Extracting data from source: {source}")
        
        # Create connector for the source
        connector = DataConnectorFactory.create_connector(
            source=source,
            output_dir=str(self.output_dir / source),
            **kwargs
        )
        
        # Fetch data
        data = await connector.fetch()
        
        # Save to local file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_{timestamp}.json"
        connector.save(data, filename)
        
        # Upload to MinIO raw storage
        file_path = connector.output_dir / filename
        object_name = f"{source}/{filename}"
        self.minio.upload_file(
            bucket_name=self.raw_bucket,
            object_name=object_name,
            file_path=str(file_path)
        )
        
        return object_name
        
    async def extract_and_load_all(self, sources: List[str], **kwargs) -> Dict[str, str]:
        """Extract and load data from all sources."""
        tasks = []
        for source in sources:
            tasks.append(self.extract_and_load(source, **kwargs))
            
        results = await asyncio.gather(*tasks)
        return dict(zip(sources, results))
        
    def transform_batch(self, batch: List[Dict[str, Any]], source: str) -> pd.DataFrame:
        """Transform a batch of data."""
        # Add metadata
        timestamp = datetime.now()
        for item in batch:
            item['source'] = source
            item['timestamp'] = timestamp
            
        # Convert to DataFrame
        df = pd.DataFrame(batch)
        
        # Basic cleaning
        df['text'] = df['text'].str.strip()
        df['title'] = df['title'].str.strip()
        
        # Apply text length filters
        min_length = int(os.getenv('MIN_TEXT_LENGTH', '100'))
        max_length = int(os.getenv('MAX_TEXT_LENGTH', '10000'))
        df = df[df['text'].str.len().between(min_length, max_length)]
        
        # Remove HTML if configured
        if os.getenv('REMOVE_HTML', 'true').lower() == 'true':
            df['text'] = df['text'].str.replace(r'<[^>]+>', '', regex=True)
        
        # Remove special characters if configured
        if os.getenv('REMOVE_SPECIAL_CHARS', 'true').lower() == 'true':
            df['text'] = df['text'].str.replace(r'[^\w\s]', ' ', regex=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])
        
        # Apply word count filters
        min_words = int(os.getenv('MIN_WORDS_PER_DOCUMENT', '50'))
        max_words = int(os.getenv('MAX_WORDS_PER_DOCUMENT', '5000'))
        df['word_count'] = df['text'].str.split().str.len()
        df = df[df['word_count'].between(min_words, max_words)]
        
        # Check unique words ratio
        min_unique = int(os.getenv('MIN_UNIQUE_WORDS', '20'))
        df['unique_words'] = df['text'].str.split().apply(lambda x: len(set(x)))
        df = df[df['unique_words'] >= min_unique]
        
        # Remove rows with high duplicate content
        max_duplicate = float(os.getenv('MAX_DUPLICATE_RATIO', '0.1'))
        df['duplicate_ratio'] = 1 - (df['unique_words'] / df['word_count'])
        df = df[df['duplicate_ratio'] <= max_duplicate]
        
        # Drop temporary columns
        df = df.drop(['word_count', 'unique_words', 'duplicate_ratio'], axis=1)
        
        return df
        
    def compress_data(self, data: bytes, compression_level: int = 9) -> bytes:
        """
        Compress data using LZMA compression.
        
        Args:
            data: Data to compress
            compression_level: Compression level (1-9, 9 being highest)
            
        Returns:
            Compressed data
        """
        return lzma.compress(data, preset=compression_level)
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """
        Decompress LZMA compressed data.
        
        Args:
            compressed_data: Compressed data to decompress
            
        Returns:
            Decompressed data
        """
        return lzma.decompress(compressed_data)
    
    def compress_file(self, input_path: str, output_path: str, compression_level: int = 9):
        """
        Compress a file using LZMA compression.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            compression_level: Compression level (1-9, 9 being highest)
        """
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        compressed_data = self.compress_data(data, compression_level)
        with open(output_path, 'wb') as f_out:
            f_out.write(compressed_data)
    
    def decompress_file(self, input_path: str, output_path: str):
        """
        Decompress an LZMA compressed file.
        
        Args:
            input_path: Path to compressed file
            output_path: Path to output file
        """
        with open(input_path, 'rb') as f_in:
            compressed_data = f_in.read()
        data = self.decompress_data(compressed_data)
        with open(output_path, 'wb') as f_out:
            f_out.write(data)

    def process_batch(self, df: pd.DataFrame, source: str, batch_num: int):
        """Process a batch of data and save to MinIO."""
        # Convert to Spark DataFrame
        spark_df = self.spark.createDataFrame(df)
        
        # Write to parquet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(self.output_dir / "processed" / f"{source}_batch{batch_num}_{timestamp}.parquet")
        spark_df.write.parquet(output_path)
        
        # Compress the parquet file before uploading
        compressed_path = f"{output_path}.xz"
        self.compress_file(output_path, compressed_path)
        
        # Upload compressed file to MinIO
        self.minio.upload_file(
            bucket_name=self.processed_bucket,
            object_name=f"processed/{source}/batch{batch_num}_{timestamp}.parquet.xz",
            file_path=compressed_path
        )
        
        # Clean up local files
        os.remove(output_path)
        os.remove(compressed_path)
        
    def transform_and_load(self, source: str, raw_data_path: str):
        """Transform and load data in batches."""
        logging.info(f"Transforming data for source: {source}")
        
        # Download raw data from MinIO
        local_path = self.output_dir / "temp" / f"{source}_raw.json"
        local_path.parent.mkdir(exist_ok=True)
        self.minio.download_file(self.raw_bucket, raw_data_path, str(local_path))
        
        # If the file is compressed, decompress it first
        if str(local_path).endswith('.xz'):
            decompressed_path = str(local_path)[:-3]  # Remove .xz extension
            self.decompress_file(str(local_path), decompressed_path)
            local_path = Path(decompressed_path)
        
        # Load data
        with open(local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process in batches
        for i, batch in enumerate(tqdm(self.batch_generator(data), desc=f"Processing {source} batches")):
            # Transform batch
            df = self.transform_batch(batch, source)
            
            # Process and save batch
            self.process_batch(df, source, i)
            
        # Cleanup
        local_path.unlink()
        if str(local_path).endswith('.xz'):
            Path(decompressed_path).unlink()
        
    def cleanup(self):
        """Clean up temporary files."""
        for file in self.output_dir.glob("**/*.json"):
            file.unlink()
            
    def __del__(self):
        """Cleanup when the pipeline is destroyed."""
        self.cleanup()
        self.spark.stop()

async def main():
    # Initialize pipeline
    pipeline = DataPipeline(
        output_dir=None,
        minio_endpoint=None,
        minio_access_key=os.getenv("MINIO_ACCESS_KEY"),
        minio_secret_key=os.getenv("MINIO_SECRET_KEY"),
        batch_size=None
    )
    
    # Process data from all sources
    sources = ['gutenberg', 'wikipedia', 'arxiv']
    
    # Extract and load data
    raw_data_paths = await pipeline.extract_and_load_all(
        sources,
        max_books=100,  # for Gutenberg
        language='en',  # for Wikipedia
        max_papers=100  # for arXiv
    )
    
    # Transform and load data in batches
    for source, raw_path in raw_data_paths.items():
        pipeline.transform_and_load(source, raw_path)
        
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 