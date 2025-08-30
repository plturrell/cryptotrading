#!/usr/bin/env python3
"""
Secure S3 Storage Service for Cryptocurrency Trading Platform
Uses AWS Secrets Manager for credential management
"""

import os
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, Dict, List, Union, BinaryIO
import logging
from datetime import datetime, timedelta
import mimetypes
from pathlib import Path

from .aws_secrets_manager import SecretsManager

logger = logging.getLogger(__name__)

class S3StorageService:
    """
    Secure S3 storage service with credential management via AWS Secrets Manager
    """
    
    def __init__(self, secret_name: str = "cryptotrading/s3-storage", region_name: str = "us-east-1"):
        """
        Initialize S3 storage service with credentials from Secrets Manager
        
        Args:
            secret_name: Name of secret in AWS Secrets Manager containing S3 credentials
            region_name: AWS region
        """
        self.secret_name = secret_name
        self.region_name = region_name
        self.bucket_name = None
        self.s3_client = None
        
        # Initialize Secrets Manager
        try:
            self.secrets_manager = SecretsManager(region_name)
            self._initialize_s3_client()
        except Exception as e:
            logger.error(f"Failed to initialize S3StorageService: {e}")
            raise
    
    def _initialize_s3_client(self):
        """Initialize S3 client using credentials from Secrets Manager"""
        try:
            # Get credentials from Secrets Manager
            credentials = self.secrets_manager.get_secret(self.secret_name)
            
            if not credentials:
                logger.error(f"Could not retrieve S3 credentials from secret: {self.secret_name}")
                raise ValueError("S3 credentials not found in Secrets Manager")
            
            # Extract S3 configuration
            aws_access_key = credentials.get('aws_access_key_id')
            aws_secret_key = credentials.get('aws_secret_access_key')
            self.bucket_name = credentials.get('bucket_name')
            region = credentials.get('region', self.region_name)
            
            if not all([aws_access_key, aws_secret_key, self.bucket_name]):
                raise ValueError("Missing required S3 credentials in secret")
            
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 client initialized successfully for bucket: {self.bucket_name}")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                logger.error(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                logger.error(f"S3 client initialization failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def upload_file(self, local_file_path: str, s3_key: str, 
                   metadata: Dict[str, str] = None,
                   content_type: str = None) -> bool:
        """
        Upload file to S3 bucket
        
        Args:
            local_file_path: Path to local file
            s3_key: S3 object key (path in bucket)
            metadata: Optional metadata to attach to object
            content_type: Optional content type override
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(local_file_path):
                logger.error(f"Local file not found: {local_file_path}")
                return False
            
            # Auto-detect content type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(local_file_path)
                if not content_type:
                    content_type = 'binary/octet-stream'
            
            # Prepare upload arguments
            upload_args = {
                'ContentType': content_type
            }
            
            if metadata:
                upload_args['Metadata'] = metadata
            
            # Upload file
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=upload_args
            )
            
            logger.info(f"File uploaded successfully: {local_file_path} -> s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {e}")
            return False
    
    def upload_data(self, data: Union[str, bytes], s3_key: str,
                   metadata: Dict[str, str] = None,
                   content_type: str = 'application/json') -> bool:
        """
        Upload data directly to S3 (without creating local file)
        
        Args:
            data: Data to upload (string or bytes)
            s3_key: S3 object key
            metadata: Optional metadata
            content_type: Content type
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert string to bytes if necessary
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Prepare upload arguments
            upload_args = {
                'ContentType': content_type
            }
            
            if metadata:
                upload_args['Metadata'] = metadata
            
            # Upload data
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data,
                **upload_args
            )
            
            logger.info(f"Data uploaded successfully to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload data to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading data: {e}")
            return False
    
    def download_file(self, s3_key: str, local_file_path: str) -> bool:
        """
        Download file from S3 to local path
        
        Args:
            s3_key: S3 object key
            local_file_path: Local file path to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_file_path
            )
            
            logger.info(f"File downloaded successfully: s3://{self.bucket_name}/{s3_key} -> {local_file_path}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 object not found: {s3_key}")
            else:
                logger.error(f"Failed to download file from S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading file: {e}")
            return False
    
    def get_object_data(self, s3_key: str) -> Optional[bytes]:
        """
        Get object data directly from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bytes: Object data or None if not found
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            data = response['Body'].read()
            logger.info(f"Object data retrieved from s3://{self.bucket_name}/{s3_key}")
            return data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 object not found: {s3_key}")
            else:
                logger.error(f"Failed to get object data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting object data: {e}")
            return None
    
    def delete_object(self, s3_key: str) -> bool:
        """
        Delete object from S3
        
        Args:
            s3_key: S3 object key to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            logger.info(f"Object deleted successfully: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete object: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting object: {e}")
            return False
    
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict]:
        """
        List objects in S3 bucket
        
        Args:
            prefix: Object key prefix to filter by
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object metadata dictionaries
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = response.get('Contents', [])
            logger.info(f"Found {len(objects)} objects with prefix: {prefix}")
            
            return [
                {
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag'].strip('"')
                }
                for obj in objects
            ]
            
        except ClientError as e:
            logger.error(f"Failed to list objects: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing objects: {e}")
            return []
    
    def object_exists(self, s3_key: str) -> bool:
        """
        Check if object exists in S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                logger.error(f"Error checking object existence: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error checking object existence: {e}")
            return False
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 3600,
                              http_method: str = 'GET') -> Optional[str]:
        """
        Generate presigned URL for S3 object
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            http_method: HTTP method (GET, PUT, etc.)
            
        Returns:
            str: Presigned URL or None if failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Presigned URL generated for s3://{self.bucket_name}/{s3_key}")
            return url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL: {e}")
            return None

# Utility functions for common crypto trading data operations

def save_market_data_to_s3(storage_service: S3StorageService, 
                          symbol: str, 
                          market_data: Dict,
                          timestamp: datetime = None) -> bool:
    """
    Save market data to S3 with organized structure
    
    Args:
        storage_service: S3StorageService instance
        symbol: Trading symbol (e.g., 'BTC-USD')
        market_data: Market data dictionary
        timestamp: Data timestamp (defaults to now)
        
    Returns:
        bool: True if successful
    """
    if not timestamp:
        timestamp = datetime.utcnow()
    
    # Organize by date for efficient querying
    date_str = timestamp.strftime('%Y/%m/%d')
    hour_str = timestamp.strftime('%H')
    
    s3_key = f"market-data/{symbol}/{date_str}/{hour_str}/{timestamp.isoformat()}.json"
    
    # Add metadata
    metadata = {
        'symbol': symbol,
        'data_type': 'market_data',
        'timestamp': timestamp.isoformat()
    }
    
    return storage_service.upload_data(
        data=json.dumps(market_data, indent=2),
        s3_key=s3_key,
        metadata=metadata,
        content_type='application/json'
    )

def save_user_data_to_s3(storage_service: S3StorageService,
                        user_id: str,
                        data_type: str,
                        user_data: Dict) -> bool:
    """
    Save user-specific data to S3
    
    Args:
        storage_service: S3StorageService instance  
        user_id: User identifier
        data_type: Type of data (portfolio, trades, etc.)
        user_data: User data dictionary
        
    Returns:
        bool: True if successful
    """
    timestamp = datetime.utcnow()
    s3_key = f"user-data/{user_id}/{data_type}/{timestamp.isoformat()}.json"
    
    metadata = {
        'user_id': user_id,
        'data_type': data_type,
        'timestamp': timestamp.isoformat()
    }
    
    return storage_service.upload_data(
        data=json.dumps(user_data, indent=2),
        s3_key=s3_key,
        metadata=metadata,
        content_type='application/json'
    )