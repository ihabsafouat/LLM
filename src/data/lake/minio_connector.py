"""
MinIO connector for data lake operations.
"""
from minio import Minio
from minio.error import S3Error
import os
from typing import Optional, List
import logging

class MinioConnector:
    def __init__(self, 
                 endpoint: str = "localhost:9000",
                 access_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 secure: bool = False):
        """
        Initialize MinIO connector.
        
        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Whether to use HTTPS
        """
        self.endpoint = endpoint
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY")
        self.secure = secure
        
        if not self.access_key or not self.secret_key:
            raise ValueError("MinIO credentials not provided")
            
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
    def create_bucket(self, bucket_name: str) -> bool:
        """Create a new bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logging.info(f"Created bucket: {bucket_name}")
            return True
        except S3Error as e:
            logging.error(f"Error creating bucket {bucket_name}: {e}")
            return False
            
    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        """Upload a file to the data lake."""
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            logging.info(f"Uploaded {file_path} to {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            logging.error(f"Error uploading {file_path}: {e}")
            return False
            
    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        """Download a file from the data lake."""
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            logging.info(f"Downloaded {bucket_name}/{object_name} to {file_path}")
            return True
        except S3Error as e:
            logging.error(f"Error downloading {object_name}: {e}")
            return False
            
    def list_objects(self, bucket_name: str, prefix: str = "") -> List[str]:
        """List objects in a bucket with optional prefix."""
        try:
            objects = self.client.list_objects(bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logging.error(f"Error listing objects in {bucket_name}: {e}")
            return []
            
    def delete_object(self, bucket_name: str, object_name: str) -> bool:
        """Delete an object from the data lake."""
        try:
            self.client.remove_object(bucket_name, object_name)
            logging.info(f"Deleted {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            logging.error(f"Error deleting {object_name}: {e}")
            return False 