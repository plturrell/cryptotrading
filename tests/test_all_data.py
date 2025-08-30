#!/usr/bin/env python3
"""Test all data sources (Yahoo, FRED, DEX)"""

import sys
sys.path.append(".")

from api.data_loader import RealDataLoader, Session, EconomicData, DEXData
from src.cryptotrading.data.database.models import MarketData
from sqlalchemy import func

def test_all_data_sources():
    print("Testing All Data Sources...")
    print("=" * 50)

    with RealDataLoader() as loader:
        # Test Yahoo Finance data loading
        print("1. Testing Yahoo Finance...")
        yahoo_results = loader.load_yahoo_data(["BTC", "ETH"], "2024-01-01", "2024-01-03", "1d")
        print(f"   Symbols processed: {yahoo_results['symbols_processed']}")
        print(f"   Total records: {yahoo_results['total_records']}")
        if yahoo_results["errors"]:
            print(f"   Errors: {yahoo_results['errors']}")

        print()
        
        # Test FRED data loading
        print("2. Testing FRED Economic Data...")
        fred_results = loader.load_fred_data(["GDPC1", "UNRATE"], "2023-01-01", "2023-12-31")
        print(f"   Series processed: {fred_results['series_processed']}")
        print(f"   Series failed: {fred_results['series_failed']}")
        print(f"   Total records: {fred_results['total_records']}")
        if fred_results["errors"]:
            print(f"   Errors: {fred_results['errors']}")

        print()
        
        # Test DEX data loading
        print("3. Testing DEX Data...")
        dex_results = loader.load_dex_data(["ethereum", "bsc"], 5)
        print(f"   Networks processed: {dex_results['networks_processed']}")
        print(f"   Networks failed: {dex_results['networks_failed']}")
        print(f"   Total records: {dex_results['total_records']}")
        if dex_results["errors"]:
            print(f"   Errors: {dex_results['errors']}")

    print()
    print("Database Summary:")
    print("=" * 50)

    # Check database totals
    session = Session()
    
    # Market Data
    market_count = session.query(MarketData).count()
    print(f"Market Data records: {market_count}")
    if market_count > 0:
        market_symbols = session.query(MarketData.symbol).distinct().all()
        print(f"  Unique symbols: {[s[0] for s in market_symbols]}")
    
    # Economic Data  
    econ_count = session.query(EconomicData).count()
    print(f"Economic Data records: {econ_count}")
    if econ_count > 0:
        econ_series = session.query(EconomicData.series_id).distinct().all()
        print(f"  Unique series: {[s[0] for s in econ_series]}")
    
    # DEX Data
    dex_count = session.query(DEXData).count()
    print(f"DEX Data records: {dex_count}")
    if dex_count > 0:
        dex_networks = session.query(DEXData.network).distinct().all()
        print(f"  Unique networks: {[n[0] for n in dex_networks]}")

    print(f"\nTotal records across all sources: {market_count + econ_count + dex_count}")
    
    # Show sample data from each source
    if market_count > 0:
        print(f"\nSample Market Data:")
        sample_market = session.query(MarketData).first()
        print(f"  {sample_market.symbol}: ${sample_market.price} on {sample_market.timestamp}")
        
    if econ_count > 0:
        print(f"\nSample Economic Data:")
        sample_econ = session.query(EconomicData).first()
        print(f"  {sample_econ.series_id}: {sample_econ.value} on {sample_econ.date}")
        
    if dex_count > 0:
        print(f"\nSample DEX Data:")
        sample_dex = session.query(DEXData).first()
        print(f"  {sample_dex.network}: {sample_dex.base_token_symbol}/{sample_dex.quote_token_symbol} at ${sample_dex.price}")

    session.close()

if __name__ == "__main__":
    test_all_data_sources()