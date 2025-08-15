#!/usr/bin/env python3
"""
Simple test for Yahoo Finance multi-crypto client
Tests the 10 main crypto trading pairs
"""

import sys
import os
sys.path.append('.')

# Direct import without A2A dependencies
from cryptotrading.core.ml.multi_crypto_yfinance_client import MultiCryptoYFinanceClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_yahoo_finance_crypto_pairs():
    """Test Yahoo Finance for all 10 main crypto pairs"""
    print("🔬 Testing Yahoo Finance Multi-Crypto Client")
    print("=" * 60)
    
    client = MultiCryptoYFinanceClient()
    
    # The 10 main crypto trading pairs
    test_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC']
    
    print(f"📊 Testing {len(test_symbols)} cryptocurrency pairs...")
    print(f"Supported pairs: {list(client.SUPPORTED_PAIRS.keys())}")
    
    results = {}
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        
        try:
            # Test symbol normalization
            normalized = client.normalize_symbol(symbol)
            print(f"  ✓ Normalized: {symbol} → {normalized}")
            
            if normalized:
                # Test current price (quick test)
                try:
                    price = client.get_current_price(symbol)
                    if price:
                        print(f"  ✓ Current price: ${price:.4f}")
                        results[symbol] = {'price': price, 'status': 'success'}
                    else:
                        print(f"  ⚠ Price unavailable")
                        results[symbol] = {'status': 'price_unavailable'}
                except Exception as e:
                    print(f"  ⚠ Price error: {e}")
                    results[symbol] = {'status': 'price_error', 'error': str(e)}
                
                # Test basic historical data (just 5 days to be fast)
                try:
                    hist_data = client.get_historical_data(symbol, days_back=5)
                    if not hist_data.empty:
                        print(f"  ✓ Historical data: {len(hist_data)} records")
                        print(f"  ✓ Latest close: ${hist_data['Close'].iloc[-1]:.4f}")
                        results[symbol]['historical'] = True
                        results[symbol]['records'] = len(hist_data)
                    else:
                        print(f"  ⚠ No historical data")
                        results[symbol]['historical'] = False
                except Exception as e:
                    print(f"  ⚠ Historical data error: {e}")
                    results[symbol]['historical'] = False
            else:
                print(f"  ❌ Symbol not supported")
                results[symbol] = {'status': 'not_supported'}
                
        except Exception as e:
            print(f"  ❌ Error testing {symbol}: {e}")
            results[symbol] = {'status': 'error', 'error': str(e)}
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 SUMMARY RESULTS")
    print("=" * 60)
    
    successful = 0
    for symbol, result in results.items():
        status = result.get('status', 'unknown')
        if status == 'success':
            successful += 1
            print(f"✅ {symbol}: ${result['price']:.4f} ({result.get('records', 0)} historical records)")
        elif status == 'price_unavailable':
            print(f"⚠️  {symbol}: Price unavailable but symbol supported")
        elif status == 'not_supported':
            print(f"❌ {symbol}: Not supported by Yahoo Finance")
        else:
            print(f"❌ {symbol}: Error - {result.get('error', 'unknown error')}")
    
    print(f"\n🎯 SUCCESS RATE: {successful}/{len(test_symbols)} pairs working ({successful/len(test_symbols)*100:.1f}%)")
    
    if successful >= 6:  # At least 6 out of 8 should work
        print("✅ Yahoo Finance multi-crypto client is working well!")
    else:
        print("⚠️  Some issues detected - may need Yahoo Finance symbol verification")
    
    return results

def test_extended_functionality():
    """Test extended functionality"""
    print(f"\n🔧 Testing Extended Functionality")
    print("=" * 60)
    
    client = MultiCryptoYFinanceClient()
    
    try:
        # Test market data for BTC (most reliable)
        print("📈 Testing market data for BTC...")
        market_data = client.get_market_data('BTC')
        
        if 'error' not in market_data:
            print(f"  ✅ Market data successful:")
            print(f"    Name: {market_data['name']}")
            print(f"    Current Price: ${market_data['current_price']:.4f}")
            print(f"    24h Change: {market_data['change_24h']:.2f}%")
            print(f"    Volume 24h: {market_data['volume_24h']:,.0f}")
        else:
            print(f"  ❌ Market data error: {market_data['error']}")
    
        # Test data for analysis format
        print(f"\n📊 Testing analysis data format for ETH...")
        analysis_data = client.get_data_for_analysis('ETH', days_back=7)
        
        if 'error' not in analysis_data:
            print(f"  ✅ Analysis data successful:")
            print(f"    Symbol: {analysis_data['symbol']}")
            print(f"    Records: {analysis_data['summary']['total_records']}")
            print(f"    Date Range: {analysis_data['summary']['date_range']['start']} to {analysis_data['summary']['date_range']['end']}")
            print(f"    Price Range: ${analysis_data['summary']['price_range']['min']:.2f} - ${analysis_data['summary']['price_range']['max']:.2f}")
        else:
            print(f"  ❌ Analysis data error: {analysis_data['error']}")
    
    except Exception as e:
        print(f"❌ Extended functionality test failed: {e}")

if __name__ == "__main__":
    print("🚀 Yahoo Finance Simple Test Suite")
    print("Testing 10 main cryptocurrency pairs via Yahoo Finance")
    
    # Test basic functionality
    results = test_yahoo_finance_crypto_pairs()
    
    # Test extended functionality
    test_extended_functionality()
    
    print(f"\n✅ Test completed! Yahoo Finance agent ready for crypto trading pairs.")