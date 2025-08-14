#!/usr/bin/env python3
"""
Test REAL historical data from Yahoo Finance only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.rex.ml.yfinance_client import get_yfinance_client
from src.rex.historical_data.yahoo_finance import YahooFinanceClient

def test_yfinance_client():
    """Test REAL Yahoo Finance ETH client"""
    print("\n=== Testing REAL Yahoo Finance ETH Client ===")
    client = get_yfinance_client()
    
    # Test getting real ETH data
    print("\n1. Getting real ETH historical data...")
    data = client.get_historical_data(days_back=5)
    if data is not None and not data.empty:
        print(f"✅ Real ETH data: {len(data)} rows")
        print(f"  Latest ETH price: ${data['Close'].iloc[-1]:.2f}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        return True
    else:
        print("❌ No real data returned")
        return False

def test_yahoo_finance_client():
    """Test REAL Yahoo Finance multi-crypto client"""
    print("\n=== Testing REAL Yahoo Finance Multi-Crypto Client ===")
    client = YahooFinanceClient()
    
    # Test BTC data
    print("\n1. Getting real BTC data...")
    btc_data = client.download_data("BTC", save=False)
    if btc_data is not None and not btc_data.empty:
        print(f"✅ Real BTC data: {len(btc_data)} rows")
        print(f"  Latest BTC price: ${btc_data['close'].iloc[-1]:.2f}")
        btc_success = True
    else:
        print("❌ No BTC data")
        btc_success = False
    
    # Test real-time price
    print("\n2. Getting real-time ETH price...")
    eth_price = client.get_realtime_price("ETH")
    if eth_price and eth_price.get('price'):
        print(f"✅ Real ETH price: ${eth_price['price']:.2f}")
        print(f"  Volume: {eth_price.get('volume', 'N/A')}")
        price_success = True
    else:
        print("❌ No real-time price")
        price_success = False
    
    return btc_success and price_success

def main():
    """Run REAL historical data tests only"""
    print("Testing REAL Historical Data Sources Only")
    print("=" * 50)
    
    tests = [
        ("Yahoo Finance ETH Client", test_yfinance_client),
        ("Yahoo Finance Multi-Crypto", test_yahoo_finance_client)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "✅ REAL DATA" if result else "❌ FAILED"))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, f"❌ ERROR: {str(e)[:50]}..."))
    
    # Summary
    print("\n" + "=" * 50)
    print("REAL DATA TEST SUMMARY")
    print("=" * 50)
    for name, status in results:
        print(f"{name:<30} {status}")
    
    passed = sum(1 for _, status in results if "REAL DATA" in status)
    print(f"\nTotal: {passed}/{len(tests)} real data sources working")

if __name__ == "__main__":
    main()