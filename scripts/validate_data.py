"""
Script to validate data quality using Great Expectations.
"""
import os
import logging
from pathlib import Path
import pandas as pd
import lzma
from dotenv import load_dotenv
from src.data.lake.minio_connector import MinioConnector
from src.data.validation.expectations import DataValidator
from src.data.validation.database import DatabaseManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', '/opt/airflow/logs/validation.log')),
        logging.StreamHandler()
    ]
)

def validate_data():
    """Validate data quality in the processed data."""
    # Initialize MinIO connector
    minio = MinioConnector(
        endpoint=os.getenv('MINIO_ENDPOINT', 'minio:9000'),  # Use Docker service name
        access_key=os.getenv('MINIO_ACCESS_KEY'),
        secret_key=os.getenv('MINIO_SECRET_KEY'),
        secure=os.getenv('MINIO_SECURE', 'false').lower() == 'true'
    )
    
    # Initialize validator and database manager
    validator = DataValidator()
    db = DatabaseManager()
    
    # Get list of processed files
    processed_files = minio.list_objects(
        bucket_name=os.getenv('PROCESSED_BUCKET', 'gpt-processed-data'),
        prefix='processed/'
    )
    
    # Create temp directory in Airflow's temp folder
    temp_dir = Path('/opt/airflow/temp')
    temp_dir.mkdir(exist_ok=True)
    
    # Validate each file
    for file in processed_files:
        logging.info(f"Validating file: {file}")
        
        # Download file
        local_path = temp_dir / file.split('/')[-1]
        minio.download_file(
            bucket_name=os.getenv('PROCESSED_BUCKET', 'gpt-processed-data'),
            object_name=file,
            file_path=str(local_path)
        )
        
        try:
            # Handle compressed files
            if str(local_path).endswith('.xz'):
                decompressed_path = str(local_path)[:-3]  # Remove .xz extension
                with lzma.open(local_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                local_path = Path(decompressed_path)
            
            # Load data
            df = pd.read_parquet(local_path)
            
            # Determine source from file path
            source = file.split('/')[1]  # processed/source/filename.parquet
            
            # Validate data
            validation_result = validator.validate_data(df, source)
            
            # Store validation run
            run = db.store_validation_run(
                source=source,
                success=validation_result['success'],
                statistics=validation_result['statistics'],
                data_docs_url=validation_result['data_docs_url'],
                environment=os.getenv('ENVIRONMENT', 'production')
            )
            
            # Store individual expectations
            for result in validation_result['results']:
                db.store_expectation(
                    run_id=run.id,
                    expectation_type=result['expectation_config']['expectation_type'],
                    success=result['success'],
                    kwargs=result['expectation_config']['kwargs'],
                    result=result['result']
                )
                
            # Store quality metrics
            if validation_result['statistics']:
                metrics = {
                    'total_expectations': validation_result['statistics'].get('total_expectations', 0),
                    'successful_expectations': validation_result['statistics'].get('successful_expectations', 0),
                    'unsuccessful_expectations': validation_result['statistics'].get('unsuccessful_expectations', 0)
                }
                
                for metric_name, value in metrics.items():
                    db.store_quality_metric(
                        source=source,
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=0,  # No threshold for these metrics
                        passed=True,  # These are informational metrics
                        metadata={'validation_run_id': run.id}
                    )
            
            # Log validation results
            if validation_result['success']:
                logging.info(f"Validation successful for {file}")
                logging.info(f"Validation statistics: {validation_result['statistics']}")
                logging.info(f"Data docs available at: {validation_result['data_docs_url']}")
            else:
                logging.error(f"Validation failed for {file}")
                logging.error(f"Validation statistics: {validation_result['statistics']}")
                logging.error(f"Validation results: {validation_result['results']}")
                
        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")
            raise
            
        finally:
            # Cleanup
            if local_path.exists():
                local_path.unlink()
            if str(local_path).endswith('.xz') and Path(decompressed_path).exists():
                Path(decompressed_path).unlink()
        
    logging.info("Data validation completed")

if __name__ == "__main__":
    validate_data() 