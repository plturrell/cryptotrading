#!/usr/bin/env python3
"""
AWS Secrets Manager integration for secure credential management
"""

import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Secure AWS credentials management using AWS Secrets Manager
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize Secrets Manager client
        
        Args:
            region_name: AWS region for Secrets Manager
        """
        self.region_name = region_name
        try:
            self.client = boto3.client(
                'secretsmanager',
                region_name=region_name
            )
            logger.info(f"SecretsManager client initialized for region: {region_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI or IAM role.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SecretsManager client: {e}")
            raise
    
    def create_secret(self, secret_name: str, secret_value: Dict[str, str], 
                     description: str = None) -> bool:
        """
        Create a new secret in AWS Secrets Manager
        
        Args:
            secret_name: Name of the secret
            secret_value: Dictionary containing secret key-value pairs
            description: Optional description
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self.client.create_secret(
                Name=secret_name,
                Description=description or f"Credentials for {secret_name}",
                SecretString=json.dumps(secret_value)
            )
            logger.info(f"Secret '{secret_name}' created successfully. ARN: {response['ARN']}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceExistsException':
                logger.warning(f"Secret '{secret_name}' already exists")
                return self.update_secret(secret_name, secret_value)
            else:
                logger.error(f"Failed to create secret '{secret_name}': {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error creating secret '{secret_name}': {e}")
            return False
    
    def get_secret(self, secret_name: str) -> Optional[Dict[str, str]]:
        """
        Retrieve secret from AWS Secrets Manager
        
        Args:
            secret_name: Name of the secret to retrieve
            
        Returns:
            Dict containing secret values or None if not found
        """
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_string = response['SecretString']
            secret_dict = json.loads(secret_string)
            logger.info(f"Secret '{secret_name}' retrieved successfully")
            return secret_dict
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.error(f"Secret '{secret_name}' not found")
            elif error_code == 'InvalidRequestException':
                logger.error(f"Invalid request for secret '{secret_name}': {e}")
            elif error_code == 'InvalidParameterException':
                logger.error(f"Invalid parameter for secret '{secret_name}': {e}")
            else:
                logger.error(f"Failed to retrieve secret '{secret_name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving secret '{secret_name}': {e}")
            return None
    
    def update_secret(self, secret_name: str, secret_value: Dict[str, str]) -> bool:
        """
        Update existing secret in AWS Secrets Manager
        
        Args:
            secret_name: Name of the secret to update
            secret_value: New secret values
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.update_secret(
                SecretId=secret_name,
                SecretString=json.dumps(secret_value)
            )
            logger.info(f"Secret '{secret_name}' updated successfully")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to update secret '{secret_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating secret '{secret_name}': {e}")
            return False
    
    def delete_secret(self, secret_name: str, force_delete: bool = False) -> bool:
        """
        Delete secret from AWS Secrets Manager
        
        Args:
            secret_name: Name of the secret to delete
            force_delete: If True, delete immediately without recovery window
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            kwargs = {'SecretId': secret_name}
            if force_delete:
                kwargs['ForceDeleteWithoutRecovery'] = True
            else:
                kwargs['RecoveryWindowInDays'] = 7  # 7-day recovery window
            
            response = self.client.delete_secret(**kwargs)
            action = "deleted immediately" if force_delete else "scheduled for deletion"
            logger.info(f"Secret '{secret_name}' {action}. ARN: {response['ARN']}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete secret '{secret_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting secret '{secret_name}': {e}")
            return False
    
    def list_secrets(self, name_filter: str = None) -> list:
        """
        List secrets in AWS Secrets Manager
        
        Args:
            name_filter: Optional filter for secret names
            
        Returns:
            List of secret metadata
        """
        try:
            kwargs = {}
            if name_filter:
                kwargs['Filters'] = [
                    {
                        'Key': 'name',
                        'Values': [name_filter]
                    }
                ]
            
            response = self.client.list_secrets(**kwargs)
            secrets = response.get('SecretList', [])
            logger.info(f"Found {len(secrets)} secrets")
            return secrets
            
        except ClientError as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing secrets: {e}")
            return []

def setup_crypto_trading_secrets():
    """
    Setup function to create required secrets for crypto trading platform
    """
    secrets_manager = SecretsManager()
    
    # Example setup - DO NOT use these values in production
    print("🔐 Setting up AWS Secrets for Crypto Trading Platform")
    print("=" * 60)
    
    # S3 Storage credentials
    s3_secret_name = "cryptotrading/s3-storage"
    s3_credentials = {
        "aws_access_key_id": "PLACEHOLDER_ACCESS_KEY",
        "aws_secret_access_key": "PLACEHOLDER_SECRET_KEY",
        "bucket_name": "tentimecrypto",
        "region": "us-east-1"
    }
    
    print(f"\n📝 To setup S3 storage credentials:")
    print(f"1. Replace PLACEHOLDER values in the code with your actual AWS credentials")
    print(f"2. Run: python3 -c \"from storage.aws_secrets_manager import setup_crypto_trading_secrets; setup_crypto_trading_secrets()\"")
    print(f"3. Secret will be created as: {s3_secret_name}")
    
    # Database credentials  
    db_secret_name = "cryptotrading/database"
    db_credentials = {
        "host": "localhost",
        "port": "5432",
        "database": "cryptotrading",
        "username": "crypto_user",
        "password": "PLACEHOLDER_DB_PASSWORD"
    }
    
    print(f"\n📊 Database credentials will be stored as: {db_secret_name}")
    
    # API keys for crypto exchanges
    exchange_secret_name = "cryptotrading/exchange-keys"
    exchange_credentials = {
        "binance_api_key": "PLACEHOLDER_BINANCE_KEY",
        "binance_secret_key": "PLACEHOLDER_BINANCE_SECRET",
        "coinbase_api_key": "PLACEHOLDER_COINBASE_KEY",
        "coinbase_secret_key": "PLACEHOLDER_COINBASE_SECRET"
    }
    
    print(f"🔑 Exchange API keys will be stored as: {exchange_secret_name}")
    
    print(f"\n⚠️  IMPORTANT: Update placeholder values before creating secrets!")
    print(f"🔒 Never commit actual credentials to version control")
    
    return {
        "s3_secret": s3_secret_name,
        "db_secret": db_secret_name,
        "exchange_secret": exchange_secret_name
    }

if __name__ == "__main__":
    # Setup secrets configuration
    setup_crypto_trading_secrets()