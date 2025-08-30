#!/usr/bin/env python3
"""
Test and demonstration script for S3 storage integration
"""

import sys
import os
import json
from datetime import datetime, timedelta
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_s3_storage():
    """Test S3 storage functionality"""
    
    print("🧪 S3 Storage Integration Test")
    print("=" * 50)
    
    try:
        # Check if boto3 is available
        import boto3
        print("✅ boto3 library available")
    except ImportError:
        print("❌ boto3 not installed. Run: pip install boto3")
        return False
    
    try:
        from src.cryptotrading.infrastructure.storage.s3_storage_service import S3StorageService
        from src.cryptotrading.infrastructure.storage.crypto_data_manager import CryptoDataManager
        
        print("✅ S3 storage modules imported successfully")
        
        # Initialize S3 service (will fail gracefully if credentials not set up)
        print("\n🔐 Testing S3 service initialization...")
        
        try:
            s3_service = S3StorageService()
            print(f"✅ S3 service initialized for bucket: {s3_service.bucket_name}")
            
            # Test basic operations
            test_basic_operations(s3_service)
            
            # Test crypto data manager
            test_crypto_data_operations(s3_service)
            
        except Exception as e:
            print(f"❌ S3 service initialization failed: {e}")
            print("\n💡 To fix this:")
            print("   1. Run: python3 scripts/setup_s3_credentials.py")
            print("   2. Follow the setup instructions")
            print("   3. Ensure your AWS credentials have S3 permissions")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import storage modules: {e}")
        return False
    
    return True

def test_basic_operations(s3_service):
    """Test basic S3 operations"""
    
    print("\n📤 Testing basic S3 operations...")
    
    # Test data upload
    test_data = {
        "test_id": "s3_integration_test",
        "timestamp": datetime.utcnow().isoformat(),
        "data": "This is a test upload to S3",
        "numbers": [1, 2, 3, 4, 5]
    }
    
    test_key = f"test/{datetime.utcnow().strftime('%Y/%m/%d')}/integration_test.json"
    
    print(f"   Uploading test data to: {test_key}")
    upload_success = s3_service.upload_data(
        data=json.dumps(test_data, indent=2),
        s3_key=test_key,
        metadata={"test": "integration", "type": "basic_operation"},
        content_type="application/json"
    )
    
    if upload_success:
        print("   ✅ Upload successful")
        
        # Test data download
        print("   📥 Testing download...")
        downloaded_data = s3_service.get_object_data(test_key)
        
        if downloaded_data:
            downloaded_json = json.loads(downloaded_data.decode('utf-8'))
            if downloaded_json['test_id'] == test_data['test_id']:
                print("   ✅ Download and verification successful")
            else:
                print("   ⚠️  Downloaded data doesn't match uploaded data")
        else:
            print("   ❌ Download failed")
        
        # Test object existence
        if s3_service.object_exists(test_key):
            print("   ✅ Object existence check successful")
        else:
            print("   ⚠️  Object existence check failed")
        
        # Test presigned URL generation
        presigned_url = s3_service.generate_presigned_url(test_key, expiration=3600)
        if presigned_url:
            print("   ✅ Presigned URL generation successful")
            print(f"   🔗 URL: {presigned_url[:50]}...")
        else:
            print("   ⚠️  Presigned URL generation failed")
        
        # Test object listing
        print("   📋 Testing object listing...")
        objects = s3_service.list_objects(prefix="test/", max_keys=10)
        print(f"   ✅ Found {len(objects)} test objects")
        
        # Cleanup test object
        if s3_service.delete_object(test_key):
            print("   🗑️  Test object cleaned up successfully")
        else:
            print("   ⚠️  Failed to cleanup test object")
            
    else:
        print("   ❌ Upload failed")

def test_crypto_data_operations(s3_service):
    """Test crypto-specific data operations"""
    
    print("\n💰 Testing crypto data operations...")
    
    try:
        data_manager = CryptoDataManager(s3_service)
        print("   ✅ CryptoDataManager initialized")
        
        # Test OHLCV data upload
        print("   📊 Testing OHLCV data upload...")
        sample_ohlcv = generate_sample_ohlcv_data()
        
        ohlcv_success = data_manager.save_ohlcv_data(
            symbol="BTC-USD",
            timeframe="1h",
            ohlcv_data=sample_ohlcv,
            timestamp=datetime.utcnow()
        )
        
        if ohlcv_success:
            print("   ✅ OHLCV data upload successful")
        else:
            print("   ❌ OHLCV data upload failed")
        
        # Test user portfolio data
        print("   👤 Testing user portfolio upload...")
        sample_portfolio = generate_sample_portfolio()
        
        portfolio_success = data_manager.save_user_portfolio(
            user_id="test_user_123",
            portfolio=sample_portfolio,
            timestamp=datetime.utcnow()
        )
        
        if portfolio_success:
            print("   ✅ User portfolio upload successful")
        else:
            print("   ❌ User portfolio upload failed")
        
        # Test analytics report
        print("   📈 Testing analytics report upload...")
        sample_analytics = generate_sample_analytics()
        
        analytics_success = data_manager.save_analytics_report(
            report_type="technical_analysis",
            analysis_data=sample_analytics,
            timestamp=datetime.utcnow()
        )
        
        if analytics_success:
            print("   ✅ Analytics report upload successful")
        else:
            print("   ❌ Analytics report upload failed")
        
        # Test storage statistics
        print("   📊 Testing storage statistics...")
        stats = data_manager.get_storage_stats()
        
        if stats:
            print(f"   ✅ Storage stats retrieved:")
            print(f"      Total objects: {stats.get('total_objects', 0)}")
            print(f"      Total size: {stats.get('total_size_mb', 0)} MB")
            print(f"      Data types: {len(stats.get('data_types', {}))}")
        else:
            print("   ⚠️  Failed to retrieve storage statistics")
            
    except Exception as e:
        print(f"   ❌ Crypto data operations test failed: {e}")

def generate_sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    
    data = []
    base_price = 50000  # Base Bitcoin price
    current_time = datetime.utcnow()
    
    for i in range(24):  # 24 hours of data
        timestamp = current_time - timedelta(hours=23-i)
        
        # Generate realistic price movement
        price_change = random.uniform(-0.05, 0.05)  # ±5% change
        open_price = base_price * (1 + price_change)
        
        high = open_price * random.uniform(1.001, 1.02)  # Up to 2% higher
        low = open_price * random.uniform(0.98, 0.999)   # Up to 2% lower
        close = random.uniform(low, high)
        volume = random.uniform(100, 1000)
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(volume, 2)
        })
        
        base_price = close  # Next candle starts where this one ended
    
    return data

def generate_sample_portfolio():
    """Generate sample portfolio data for testing"""
    
    return {
        'total_value_usd': 125000.50,
        'last_updated': datetime.utcnow().isoformat(),
        'assets': [
            {
                'symbol': 'BTC',
                'quantity': 2.5,
                'avg_price': 48000.00,
                'current_price': 50000.00,
                'value_usd': 125000.00,
                'pnl_usd': 5000.00,
                'pnl_percent': 4.17
            },
            {
                'symbol': 'ETH',
                'quantity': 0.5,
                'avg_price': 3000.00,
                'current_price': 3100.00,
                'value_usd': 1550.00,
                'pnl_usd': 50.00,
                'pnl_percent': 3.33
            }
        ],
        'allocation': {
            'BTC': 98.8,
            'ETH': 1.2
        }
    }

def generate_sample_analytics():
    """Generate sample analytics data for testing"""
    
    return {
        'analysis_type': 'technical_indicators',
        'symbol': 'BTC-USD',
        'timeframe': '1h',
        'timestamp': datetime.utcnow().isoformat(),
        'indicators': {
            'rsi': 65.5,
            'macd': {
                'macd': 1250.5,
                'signal': 1100.2,
                'histogram': 150.3
            },
            'bollinger_bands': {
                'upper': 52000,
                'middle': 50000,
                'lower': 48000
            },
            'sma_20': 49500,
            'ema_12': 50200
        },
        'signals': {
            'trend': 'bullish',
            'strength': 'moderate',
            'recommendation': 'hold'
        },
        'confidence': 0.75
    }

def show_usage_examples():
    """Show usage examples for S3 storage"""
    
    print("\n📚 S3 Storage Usage Examples")
    print("=" * 50)
    
    print("\n1. Basic S3 Operations:")
    print("```python")
    print("from storage.s3_storage_service import S3StorageService")
    print("")
    print("# Initialize service (uses credentials from Secrets Manager)")
    print("s3_service = S3StorageService()")
    print("")
    print("# Upload data")
    print("data = {'key': 'value'}")
    print("s3_service.upload_data(json.dumps(data), 'path/to/file.json')")
    print("")
    print("# Download data")
    print("downloaded = s3_service.get_object_data('path/to/file.json')")
    print("```")
    
    print("\n2. Crypto Data Management:")
    print("```python")
    print("from storage.crypto_data_manager import CryptoDataManager")
    print("")
    print("# Initialize manager")
    print("data_manager = CryptoDataManager(s3_service)")
    print("")
    print("# Save market data")
    print("data_manager.save_ohlcv_data('BTC-USD', '1h', ohlcv_data)")
    print("")
    print("# Save user portfolio")
    print("data_manager.save_user_portfolio('user123', portfolio_data)")
    print("```")
    
    print("\n3. Setup S3 Credentials:")
    print("```bash")
    print("python3 scripts/setup_s3_credentials.py")
    print("```")

def main():
    """Main test function"""
    
    print("🚀 Cryptocurrency Trading Platform - S3 Storage Test")
    print("=" * 60)
    
    # Run tests
    success = test_s3_storage()
    
    if success:
        print("\n🎉 S3 storage integration is ready!")
        show_usage_examples()
    else:
        print("\n❌ S3 storage integration needs setup")
    
    print("\n" + "=" * 60)
    print("📝 Next Steps:")
    print("   1. Setup AWS credentials: python3 scripts/setup_s3_credentials.py")
    print("   2. Test integration: python3 scripts/test_s3_storage.py")
    print("   3. Use in your application with secure credential management")

if __name__ == "__main__":
    main()