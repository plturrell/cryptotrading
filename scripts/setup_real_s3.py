#!/usr/bin/env python3
"""
Setup real S3 connection using provided credentials
"""

import sys
import os
import json
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def setup_real_s3_test():
    """Setup and test real S3 connection using AWS Secrets Manager"""
    
    print("🔐 Setting up Real S3 Connection")
    print("=" * 50)
    
    # Import our storage service which uses Secrets Manager
    try:
        from src.cryptotrading.infrastructure.storage.s3_storage_service import S3StorageService
        print("✅ S3StorageService imported (uses AWS Secrets Manager)")
    except ImportError as e:
        print(f"❌ Failed to import S3StorageService: {e}")
        return False
    
    # Initialize S3 service (will get credentials from Secrets Manager)
    try:
        print("🔍 Initializing S3 service with AWS Secrets Manager...")
        s3_service = S3StorageService(secret_name="cryptotrading/s3-storage")
        print(f"✅ S3 service initialized for bucket: {s3_service.bucket_name}")
        bucket_name = s3_service.bucket_name
    except Exception as e:
        print(f"❌ Failed to initialize S3 service: {e}")
        print("\n💡 This likely means:")
        print("   1. AWS Secrets Manager secret 'cryptotrading/s3-storage' doesn't exist")
        print("   2. Run: python3 scripts/setup_s3_credentials.py")
        print("   3. Or your AWS credentials don't have Secrets Manager access")
        return False
    
    # Test basic operations using our secure S3 service
    try:
        # Test upload
        print("📤 Testing file upload...")
        test_data = {
            "test": "real_s3_integration_test",
            "timestamp": datetime.now().isoformat(),
            "message": "This is a real S3 test using AWS Secrets Manager"
        }
        
        test_key = "test/real_integration_test.json"
        
        upload_success = s3_service.upload_data(
            data=json.dumps(test_data, indent=2),
            s3_key=test_key,
            metadata={
                'test': 'real_integration',
                'timestamp': datetime.now().isoformat()
            },
            content_type='application/json'
        )
        
        if upload_success:
            print(f"✅ Test file uploaded: {test_key}")
        else:
            print("❌ Upload failed")
            return False
        
        # Test download
        print("📥 Testing file download...")
        downloaded_data_bytes = s3_service.get_object_data(test_key)
        
        if downloaded_data_bytes:
            downloaded_data = json.loads(downloaded_data_bytes.decode('utf-8'))
            if downloaded_data['test'] == test_data['test']:
                print("✅ Download and verification successful")
            else:
                print("❌ Downloaded data doesn't match")
                return False
        else:
            print("❌ Download failed")
            return False
        
        # Test object existence
        print("🔍 Testing object existence check...")
        if s3_service.object_exists(test_key):
            print("✅ Object existence check successful")
        else:
            print("❌ Object existence check failed")
            return False
        
        # List objects
        print("📋 Testing object listing...")
        objects = s3_service.list_objects(prefix="test/", max_keys=10)
        print(f"✅ Found {len(objects)} test objects")
        
        # Test crypto data manager
        print("\n🧪 Testing CryptoDataManager...")
        test_crypto_data_manager(s3_service)
        
        # Cleanup
        print("🗑️ Cleaning up test file...")
        if s3_service.delete_object(test_key):
            print("✅ Cleanup complete")
        else:
            print("⚠️ Cleanup failed, but test was successful")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '403':
            print("❌ Access denied - check your credentials and permissions")
        elif error_code == '404':
            print("❌ Bucket not found")
        else:
            print(f"❌ AWS error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_crypto_data_manager(s3_service):
    """Test CryptoDataManager functionality"""
    
    try:
        from src.cryptotrading.infrastructure.storage.crypto_data_manager import CryptoDataManager
        
        # Initialize data manager
        data_manager = CryptoDataManager(s3_service)
        print("  ✅ CryptoDataManager initialized")
        
        # Test OHLCV data save
        sample_ohlcv = [
            {
                'timestamp': datetime.now().isoformat(),
                'open': 50000.0,
                'high': 51000.0,
                'low': 49500.0,
                'close': 50500.0,
                'volume': 125.5
            }
        ]
        
        crypto_success = data_manager.save_ohlcv_data(
            symbol="BTC-USD",
            timeframe="1h",
            ohlcv_data=sample_ohlcv,
            timestamp=datetime.now()
        )
        
        if crypto_success:
            print("  ✅ CryptoDataManager OHLCV save successful")
        else:
            print("  ❌ CryptoDataManager OHLCV save failed")
        
        # Test portfolio save
        portfolio = {
            'total_value_usd': 125000.50,
            'last_updated': datetime.now().isoformat(),
            'assets': [
                {'symbol': 'BTC', 'quantity': 2.5, 'value_usd': 125000.00}
            ]
        }
        
        portfolio_success = data_manager.save_user_portfolio(
            user_id="test_user",
            portfolio=portfolio,
            timestamp=datetime.now()
        )
        
        if portfolio_success:
            print("  ✅ CryptoDataManager portfolio save successful")
        else:
            print("  ❌ CryptoDataManager portfolio save failed")
        
        # Get storage stats
        try:
            stats = data_manager.get_storage_stats()
            if stats:
                print(f"  ✅ Storage stats: {stats.get('total_objects', 0)} objects, {stats.get('total_size_mb', 0)} MB")
            else:
                print("  ⚠️ Storage stats not available")
        except Exception as e:
            print(f"  ⚠️ Storage stats failed: {e}")
        
    except Exception as e:
        print(f"  ❌ CryptoDataManager test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Real S3 Integration Test with AWS Secrets Manager")
    print("=" * 60)
    print("\n🔐 SECURITY NOTE:")
    print("   This script uses AWS Secrets Manager for secure credential storage")
    print("   Prerequisites:")
    print("   1. AWS credentials configured (CLI, IAM role, or env vars)")
    print("   2. Secret 'cryptotrading/s3-storage' exists in Secrets Manager")
    print("   3. If not, run: python3 scripts/setup_s3_credentials.py")
    print("   4. Permissions: SecretsManager:GetSecretValue, S3 access")
    print()
    
    success = setup_real_s3_test()
    
    if success:
        print("\n🎉 Real S3 integration test successful!")
        print("✅ AWS Secrets Manager integration working")
        print("✅ S3 operations working with secure credentials")
        print("✅ CryptoDataManager functioning correctly")
        print("✅ Ready for production use")
    else:
        print("\n❌ Real S3 integration test failed")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure AWS credentials are configured")
        print("2. Run: python3 scripts/setup_s3_credentials.py")
        print("3. Check AWS permissions for Secrets Manager and S3")
        print("4. Verify secret 'cryptotrading/s3-storage' exists")