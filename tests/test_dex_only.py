#!/usr/bin/env python3
"""Test DEX data loading only"""

import sys
sys.path.append(".")

from api.data_loader import RealDataLoader, Session, DEXData

def test_dex_data():
    print("Testing DEX Data Loading...")
    
    with RealDataLoader() as loader:
        # Test DEX data loading
        dex_results = loader.load_dex_data(["eth"], 3)  # Start with just ethereum, limit to 3 pools
        print(f"Networks processed: {dex_results['networks_processed']}")
        print(f"Networks failed: {dex_results['networks_failed']}")
        print(f"Total records: {dex_results['total_records']}")
        if dex_results["errors"]:
            print(f"Errors: {dex_results['errors']}")

    # Check database
    session = Session()
    dex_count = session.query(DEXData).count()
    print(f"\nDEX Data records in database: {dex_count}")
    
    if dex_count > 0:
        samples = session.query(DEXData).limit(3).all()
        print("Sample DEX records:")
        for sample in samples:
            print(f"  {sample.network}: {sample.base_token_symbol}/{sample.quote_token_symbol}")
            print(f"    Pool: {sample.pool_address}")
            print(f"    Volume 24h: ${sample.volume_24h:,.2f}" if sample.volume_24h else "    Volume 24h: N/A")
            print(f"    Liquidity: ${sample.liquidity:,.2f}" if sample.liquidity else "    Liquidity: N/A")
            print()

    session.close()

if __name__ == "__main__":
    test_dex_data()