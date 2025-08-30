#!/usr/bin/env python3
"""
Test script for the updated market data service
Tests database integration and real data functionality
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cryptotrading.infrastructure.data.market_data_service import MarketDataService
from src.cryptotrading.data.database.models import MarketData
from src.cryptotrading.infrastructure.database.unified_database import UnifiedDatabase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_market_data_service():
    """Test the market data service functionality"""
    print("🧪 Testing Market Data Service with Database Integration")
    print("=" * 60)
    
    # Test 1: Initialize service
    print("\n1. Testing service initialization...")
    async with MarketDataService() as service:
        print("✅ Market data service initialized successfully")
        
        # Test 2: Store sample market data
        print("\n2. Testing data storage...")
        sample_data = {
            "price": 45000.0,
            "volume_24h": 2500000.0,
            "high_24h": 46000.0,
            "low_24h": 44000.0,
            "change_24h": 1000.0,
            "change_percent_24h": 2.27,
            "market_cap": 850000000000.0,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        success = await service.store_market_data("BTC", sample_data)
        if success:
            print("✅ Successfully stored BTC market data")
        else:
            print("❌ Failed to store BTC market data")
        
        # Store data for multiple symbols
        symbols_data = {
            "ETH": {
                "price": 2800.0,
                "volume_24h": 1500000.0,
                "high_24h": 2850.0,
                "low_24h": 2750.0,
                "change_24h": 50.0,
                "change_percent_24h": 1.82,
                "market_cap": 340000000000.0,
                "last_updated": datetime.utcnow().isoformat()
            },
            "ADA": {
                "price": 0.35,
                "volume_24h": 500000.0,
                "high_24h": 0.37,
                "low_24h": 0.33,
                "change_24h": 0.02,
                "change_percent_24h": 6.06,
                "market_cap": 12000000000.0,
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
        for symbol, data in symbols_data.items():
            success = await service.store_market_data(symbol, data)
            if success:
                print(f"✅ Successfully stored {symbol} market data")
            else:
                print(f"❌ Failed to store {symbol} market data")
        
        # Test 3: Retrieve single symbol data
        print("\n3. Testing single symbol retrieval...")
        btc_data = await service.get_market_data_dict("BTC")
        if btc_data and btc_data.get("symbol") == "BTC":
            print(f"✅ Retrieved BTC data: ${btc_data['price']:,.2f}")
            print(f"   Volume 24h: ${btc_data['volume_24h']:,.0f}")
            print(f"   Change 24h: {btc_data['change_percent_24h']:.2f}%")
        else:
            print("❌ Failed to retrieve BTC data or got placeholder")
        
        # Test 4: Retrieve multiple symbols
        print("\n4. Testing multiple symbol retrieval...")
        symbols = ["BTC", "ETH", "ADA", "DOGE"]  # Include one that doesn't exist
        multi_data = await service.get_multiple_symbols(symbols)
        
        for symbol in symbols:
            if symbol in multi_data:
                data = multi_data[symbol]
                print(f"✅ {symbol}: ${data['price']:,.2f} ({data['change_percent_24h']:+.2f}%)")
            else:
                print(f"❌ {symbol}: No data found")
        
        # Test 5: Test trending coins
        print("\n5. Testing trending coins...")
        trending = await service.get_trending_coins(limit=5)
        if trending:
            print(f"✅ Found {len(trending)} trending coins:")
            for coin in trending:
                print(f"   {coin['symbol']}: ${coin['price']:,.2f} (Vol: ${coin['volume_24h']:,.0f})")
        else:
            print("❌ No trending coins found")
        
        # Test 6: Test database direct access
        print("\n6. Testing direct database access...")
        try:
            cursor = service.db.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM market_data")
            count = cursor.fetchone()[0]
            print(f"✅ Database contains {count} market data records")
            
            # Get latest records
            cursor.execute("""
                SELECT symbol, close, volume, timestamp 
                FROM market_data 
                ORDER BY timestamp DESC 
                LIMIT 3
            """)
            latest = cursor.fetchall()
            print("   Latest records:")
            for record in latest:
                symbol = record[0] if isinstance(record, tuple) else record['symbol']
                price = record[1] if isinstance(record, tuple) else record['close']
                timestamp = record[3] if isinstance(record, tuple) else record['timestamp']
                print(f"   - {symbol}: ${price:.2f} at {timestamp}")
            cursor.close()
                    
        except Exception as e:
            print(f"❌ Database access failed: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Market Data Service Test Complete!")

async def test_database_connection():
    """Test database connection independently"""
    print("\n🔌 Testing Database Connection...")
    
    try:
        db = UnifiedDatabase()
        await db.initialize()
        print("✅ Database initialized successfully")
        
        # Test basic operations
        with db.get_session() as session:
            # Check if MarketData table exists and is accessible
            try:
                count = session.query(MarketData).count()
                print(f"✅ MarketData table accessible with {count} records")
            except Exception as e:
                print(f"❌ MarketData table access failed: {e}")
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    return True

async def main():
    """Main test function"""
    print("🚀 Starting Market Data Service Tests")
    
    # Test database connection first
    db_ok = await test_database_connection()
    if not db_ok:
        print("❌ Database connection failed, skipping service tests")
        return
    
    # Test the market data service
    try:
        await test_market_data_service()
    except Exception as e:
        print(f"❌ Market data service test failed: {e}")
        logger.error("Test failed", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
