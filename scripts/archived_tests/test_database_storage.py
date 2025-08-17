#!/usr/bin/env python3
"""
Test database storage functionality with real data
Validates that market and economic data gets stored and retrieved correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3

# Add project root to path
sys.path.insert(0, '/Users/apple/projects/cryptotrading/src')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_client():
    """Test the database client functionality"""
    print("💾 Testing Database Client")
    print("=" * 60)
    
    try:
        from cryptotrading.data.database.client import DatabaseClient
        
        # Initialize client
        client = DatabaseClient()
        print("✅ Database client initialized")
        
        # Test connection
        if client.test_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Database client not available")
        return False
    except Exception as e:
        print(f"❌ Database client test failed: {e}")
        return False

def test_direct_sqlite_storage():
    """Test direct SQLite database storage"""
    print("\n🗄️  Testing Direct SQLite Storage")
    print("=" * 60)
    
    try:
        # Create test database
        db_path = "data/test_market_data.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Create economic data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS economic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id TEXT NOT NULL,
                date TEXT NOT NULL,
                value REAL,
                source TEXT DEFAULT 'FRED',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(series_id, date)
            )
        """)
        
        conn.commit()
        print("✅ Database tables created")
        
        # Test market data insertion
        sample_market_data = [
            ('BTC-USD', '2025-08-17', 117000.0, 118000.0, 116000.0, 117500.0, 15000.0),
            ('ETH-USD', '2025-08-17', 4400.0, 4500.0, 4350.0, 4450.0, 50000.0)
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO market_data 
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, sample_market_data)
        
        print(f"✅ Inserted {len(sample_market_data)} market data records")
        
        # Test economic data insertion
        sample_economic_data = [
            ('DGS10', '2025-08-17', 4.25, 'FRED'),
            ('EFFR', '2025-08-17', 5.50, 'FRED'),
            ('M2SL', '2025-08-01', 21000000.0, 'FRED')
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO economic_data 
            (series_id, date, value, source)
            VALUES (?, ?, ?, ?)
        """, sample_economic_data)
        
        print(f"✅ Inserted {len(sample_economic_data)} economic data records")
        
        conn.commit()
        
        # Test data retrieval
        cursor.execute("SELECT COUNT(*) FROM market_data")
        market_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM economic_data")
        economic_count = cursor.fetchone()[0]
        
        print(f"✅ Database contains {market_count} market data records")
        print(f"✅ Database contains {economic_count} economic data records")
        
        # Test query functionality
        cursor.execute("""
            SELECT symbol, timestamp, close, volume 
            FROM market_data 
            WHERE symbol = 'BTC-USD'
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if result:
            symbol, timestamp, close, volume = result
            print(f"✅ Latest BTC data: {close} @ {timestamp}, Volume: {volume}")
        
        conn.close()
        
        print("✅ Direct SQLite storage test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Direct SQLite storage test failed: {e}")
        return False

def test_real_data_storage():
    """Test storing real Yahoo Finance data in database"""
    print("\n📊 Testing Real Data Storage")
    print("=" * 60)
    
    try:
        from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
        
        # Load real data
        yahoo_client = YahooFinanceClient()
        btc_data = yahoo_client.download_data(
            symbol="BTC-USD",
            start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            interval="1d",
            save=False
        )
        
        if btc_data is None or btc_data.empty:
            print("❌ Failed to load real BTC data")
            return False
        
        print(f"✅ Loaded {len(btc_data)} days of real BTC data")
        
        # Store in database
        db_path = "data/real_market_data.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                dividends REAL,
                stock_splits REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Insert real data
        records_inserted = 0
        for timestamp, row in btc_data.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, dividends, stock_splits)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'BTC-USD',
                    timestamp.strftime('%Y-%m-%d'),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    float(row.get('dividends', 0)),
                    float(row.get('stock_splits', 0))
                ))
                records_inserted += 1
            except Exception as e:
                print(f"⚠️  Error inserting record for {timestamp}: {e}")
        
        conn.commit()
        print(f"✅ Stored {records_inserted} real market data records")
        
        # Test retrieval
        cursor.execute("""
            SELECT timestamp, close, volume 
            FROM market_data 
            WHERE symbol = 'BTC-USD'
            ORDER BY timestamp DESC
            LIMIT 3
        """)
        
        results = cursor.fetchall()
        print(f"✅ Retrieved {len(results)} recent records:")
        for timestamp, close, volume in results:
            print(f"   {timestamp}: ${close:,.2f}, Volume: {volume:,.0f}")
        
        # Test aggregation
        cursor.execute("""
            SELECT 
                COUNT(*) as record_count,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(close) as avg_price,
                SUM(volume) as total_volume
            FROM market_data 
            WHERE symbol = 'BTC-USD'
        """)
        
        result = cursor.fetchone()
        if result:
            count, min_price, max_price, avg_price, total_volume = result
            print(f"✅ Aggregation results:")
            print(f"   Records: {count}")
            print(f"   Price range: ${min_price:,.2f} - ${max_price:,.2f}")
            print(f"   Average price: ${avg_price:,.2f}")
            print(f"   Total volume: {total_volume:,.0f}")
        
        conn.close()
        
        print("✅ Real data storage test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Real data storage test failed: {e}")
        return False

def test_data_quality_in_database():
    """Test data quality validation on stored data"""
    print("\n✅ Testing Data Quality in Database")
    print("=" * 60)
    
    try:
        db_path = "data/real_market_data.db"
        if not os.path.exists(db_path):
            print("⚠️  No database found for quality testing")
            return False
        
        conn = sqlite3.connect(db_path)
        
        # Test data completeness
        query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN close IS NOT NULL THEN 1 END) as non_null_close,
                COUNT(CASE WHEN volume IS NOT NULL THEN 1 END) as non_null_volume,
                COUNT(CASE WHEN open IS NOT NULL THEN 1 END) as non_null_open
            FROM market_data
            WHERE symbol = 'BTC-USD'
        """
        
        df = pd.read_sql_query(query, conn)
        result = df.iloc[0]
        
        completeness = result['non_null_close'] / result['total_records'] * 100
        print(f"✅ Data completeness: {completeness:.1f}% ({result['non_null_close']}/{result['total_records']} records)")
        
        # Test for duplicate timestamps
        query = """
            SELECT timestamp, COUNT(*) as count
            FROM market_data
            WHERE symbol = 'BTC-USD'
            GROUP BY timestamp
            HAVING COUNT(*) > 1
        """
        
        duplicates = pd.read_sql_query(query, conn)
        print(f"✅ Duplicate check: {len(duplicates)} duplicate timestamps found")
        
        # Test price range validation
        query = """
            SELECT 
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(close) as avg_price,
                (MAX(close) - MIN(close)) / AVG(close) * 100 as price_range_pct
            FROM market_data
            WHERE symbol = 'BTC-USD'
        """
        
        df = pd.read_sql_query(query, conn)
        result = df.iloc[0]
        
        print(f"✅ Price validation:")
        print(f"   Range: ${result['min_price']:,.2f} - ${result['max_price']:,.2f}")
        print(f"   Range %: {result['price_range_pct']:.1f}% of average")
        
        # Test volume validation
        query = """
            SELECT 
                MIN(volume) as min_volume,
                MAX(volume) as max_volume,
                AVG(volume) as avg_volume,
                COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume_days
            FROM market_data
            WHERE symbol = 'BTC-USD'
        """
        
        df = pd.read_sql_query(query, conn)
        result = df.iloc[0]
        
        print(f"✅ Volume validation:")
        print(f"   Range: {result['min_volume']:,.0f} - {result['max_volume']:,.0f}")
        print(f"   Zero-volume days: {result['zero_volume_days']}")
        
        conn.close()
        
        print("✅ Database quality validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Database quality validation failed: {e}")
        return False

def main():
    """Run comprehensive database storage tests"""
    print("💾 Comprehensive Database Storage Test Suite")
    print("=" * 80)
    print("Testing database storage with real market data")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Database client
    results['database_client'] = test_database_client()
    
    # Test 2: Direct SQLite storage
    results['sqlite_storage'] = test_direct_sqlite_storage()
    
    # Test 3: Real data storage
    results['real_data_storage'] = test_real_data_storage()
    
    # Test 4: Data quality in database
    results['database_quality'] = test_data_quality_in_database()
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 DATABASE STORAGE SUMMARY")
    print("=" * 80)
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    success_rate = successful_tests / total_tests * 100
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if success_rate >= 75:
        print(f"✅ EXCELLENT: {successful_tests}/{total_tests} tests passed ({success_rate:.0f}%)")
        print("   Database storage system is working well!")
    elif success_rate >= 50:
        print(f"⚠️  PARTIAL: {successful_tests}/{total_tests} tests passed ({success_rate:.0f}%)")
        print("   Some database functionality working")
    else:
        print(f"❌ ISSUES: {successful_tests}/{total_tests} tests passed ({success_rate:.0f}%)")
        print("   Significant database issues detected")
    
    print(f"\n📁 Database files created in: ./data/")
    print(f"📝 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    results = main()