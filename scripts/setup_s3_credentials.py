#!/usr/bin/env python3
"""
Setup S3 credentials in AWS Secrets Manager
IMPORTANT: Only run this after securing your AWS credentials properly
"""

import sys
import os
import getpass
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.cryptotrading.infrastructure.storage.aws_secrets_manager import SecretsManager

def setup_s3_credentials():
    """
    Interactive setup of S3 credentials in AWS Secrets Manager
    """
    
    print("🔐 S3 Credentials Setup for Crypto Trading Platform")
    print("=" * 60)
    
    print("\n⚠️  SECURITY REMINDER:")
    print("   • Never share or commit AWS credentials to version control")
    print("   • Use IAM roles when possible instead of access keys")
    print("   • Regularly rotate your access keys")
    print("   • Use least privilege principle for IAM permissions")
    
    confirm = input("\nHave you followed AWS security best practices? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Please review AWS security best practices before continuing.")
        return False
    
    try:
        # Initialize Secrets Manager
        region = input("\nAWS Region (default: us-east-1): ").strip() or "us-east-1"
        secrets_manager = SecretsManager(region_name=region)
        
        print(f"\n✅ Connected to AWS Secrets Manager in region: {region}")
        
        # Get S3 credentials
        print("\n📝 Enter S3 Credentials:")
        aws_access_key = input("AWS Access Key ID: ").strip()
        if not aws_access_key:
            print("❌ Access Key ID is required")
            return False
        
        aws_secret_key = getpass.getpass("AWS Secret Access Key: ").strip()
        if not aws_secret_key:
            print("❌ Secret Access Key is required")
            return False
        
        bucket_name = input("S3 Bucket Name (default: tentimecrypto): ").strip() or "tentimecrypto"
        
        # Prepare credentials
        s3_credentials = {
            "aws_access_key_id": aws_access_key,
            "aws_secret_access_key": aws_secret_key,
            "bucket_name": bucket_name,
            "region": region
        }
        
        # Create secret
        secret_name = "cryptotrading/s3-storage"
        print(f"\n🔒 Creating secret: {secret_name}")
        
        success = secrets_manager.create_secret(
            secret_name=secret_name,
            secret_value=s3_credentials,
            description="S3 storage credentials for cryptocurrency trading platform"
        )
        
        if success:
            print("✅ S3 credentials stored successfully in AWS Secrets Manager!")
            print(f"   Secret Name: {secret_name}")
            print(f"   Region: {region}")
            print(f"   Bucket: {bucket_name}")
            
            # Verify the secret was created
            print("\n🔍 Verifying secret creation...")
            verify_secret = secrets_manager.get_secret(secret_name)
            if verify_secret:
                print("✅ Secret verification successful")
                print(f"   Stored bucket: {verify_secret.get('bucket_name')}")
                print(f"   Stored region: {verify_secret.get('region')}")
            else:
                print("⚠️ Secret verification failed")
            
            # Test the credentials
            print("\n🧪 Testing S3 connection...")
            test_s3_connection(secret_name, region)
            
        else:
            print("❌ Failed to store S3 credentials")
            return False
        
    except Exception as e:
        print(f"❌ Error setting up S3 credentials: {e}")
        return False
    
    return True

def test_s3_connection(secret_name: str, region: str):
    """Test S3 connection using stored credentials"""
    
    try:
        from src.cryptotrading.infrastructure.storage.s3_storage_service import S3StorageService
        
        # Initialize S3 service
        s3_service = S3StorageService(secret_name=secret_name, region_name=region)
        
        # Test basic operations
        print("   📋 Testing bucket access...")
        objects = s3_service.list_objects(max_keys=1)
        print(f"   ✅ Bucket accessible - found {len(objects)} objects")
        
        # Test upload with sample data
        print("   📤 Testing upload...")
        test_data = {
            "test": True,
            "timestamp": "2024-01-01T00:00:00Z",
            "message": "S3 connection test successful"
        }
        
        test_key = "test-connection/test.json"
        upload_success = s3_service.upload_data(
            data=json.dumps(test_data),
            s3_key=test_key,
            metadata={"test": "connection"},
            content_type="application/json"
        )
        
        if upload_success:
            print("   ✅ Upload test successful")
            
            # Test download
            print("   📥 Testing download...")
            downloaded_data = s3_service.get_object_data(test_key)
            if downloaded_data:
                print("   ✅ Download test successful")
                
                # Cleanup test file
                s3_service.delete_object(test_key)
                print("   🗑️  Test file cleaned up")
            else:
                print("   ⚠️  Download test failed")
        else:
            print("   ⚠️  Upload test failed")
        
        print("\n🎉 S3 integration is ready for use!")
        
    except Exception as e:
        print(f"   ❌ S3 connection test failed: {e}")
        print("   💡 Please verify your credentials and bucket permissions")

def list_existing_secrets():
    """List existing secrets in Secrets Manager"""
    
    try:
        region = input("AWS Region (default: us-east-1): ").strip() or "us-east-1"
        secrets_manager = SecretsManager(region_name=region)
        
        print(f"\n📋 Existing secrets in {region}:")
        secrets = secrets_manager.list_secrets("cryptotrading")
        
        if not secrets:
            print("   No secrets found with 'cryptotrading' prefix")
        else:
            for secret in secrets:
                print(f"   • {secret['Name']} - {secret.get('Description', 'No description')}")
                
    except Exception as e:
        print(f"❌ Error listing secrets: {e}")

def main():
    """Main setup function"""
    
    print("\n🔧 AWS Secrets Manager Setup for S3 Storage:")
    print("=" * 50)
    print("This tool securely stores your AWS credentials in Secrets Manager")
    print("✅ No credentials stored in code or environment variables")
    print("✅ Centralized credential management")
    print("✅ Automatic credential rotation support")
    print("✅ Fine-grained access control")
    print()
    print("Options:")
    print("1. Setup new S3 credentials in Secrets Manager")
    print("2. List existing secrets")
    print("3. Test existing secret")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        setup_s3_credentials()
    elif choice == "2":
        list_existing_secrets()
    elif choice == "3":
        test_existing_secret()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid option")

def test_existing_secret():
    """Test an existing secret in Secrets Manager"""
    try:
        region = input("AWS Region (default: us-east-1): ").strip() or "us-east-1"
        secret_name = input("Secret name (default: cryptotrading/s3-storage): ").strip() or "cryptotrading/s3-storage"
        
        print(f"\n🧪 Testing secret: {secret_name}")
        test_s3_connection(secret_name, region)
        
    except Exception as e:
        print(f"❌ Error testing secret: {e}")

if __name__ == "__main__":
    main()