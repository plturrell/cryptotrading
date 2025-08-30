#!/usr/bin/env python3
"""Test real data loading system"""

import sys
sys.path.append(".")

from api.data_loader import RealDataLoader, Session
from src.cryptotrading.data.database.models import MarketData
from sqlalchemy import func

def test_yahoo_data_loading():
    print("Testing Real Data Loader with fixed schema...")

    # Test Yahoo Finance data loading
    with RealDataLoader() as loader:
        results = loader.load_yahoo_data(["BTC", "ETH"], "2024-01-01", "2024-01-05", "1d")
        print("Yahoo Finance Results:")
        print(f"  Symbols processed: {results['symbols_processed']}")
        print(f"  Symbols failed: {results['symbols_failed']}")
        print(f"  Total records: {results['total_records']}")
        if results["errors"]:
            print(f"  Errors: {results['errors']}")

    print()

    # Check database
    session = Session()
    count = session.query(MarketData).count()
    print(f"Total MarketData records in database: {count}")

    if count > 0:
        # Get sample records
        samples = session.query(MarketData).limit(3).all()
        print("Sample records:")
        for sample in samples:
            print(f"  {sample.symbol}: ${sample.price} on {sample.timestamp} (Vol: {sample.volume_24h})")
            
        # Get summary by symbol
        print()
        print("Records by symbol:")
        symbol_counts = session.query(
            MarketData.symbol, 
            func.count(MarketData.id).label("count"),
            func.min(MarketData.timestamp).label("start_date"),
            func.max(MarketData.timestamp).label("end_date")
        ).group_by(MarketData.symbol).all()
        
        for symbol, count_val, start_date, end_date in symbol_counts:
            print(f"  {symbol}: {count_val} records from {start_date} to {end_date}")

    session.close()

if __name__ == "__main__":
    test_yahoo_data_loading()