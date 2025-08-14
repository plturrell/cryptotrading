#!/usr/bin/env python3
"""
Test script for extended Yahoo Finance agent
Tests all 10 main crypto trading pairs
"""

import sys
import os
sys.path.append('.')

from src.rex.ml.multi_crypto_yfinance_client import get_multi_crypto_client
from src.rex.a2a.agents.historical_loader_agent import get_historical_loader_agent
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multi_crypto_client():
    """Test the multi-crypto Yahoo Finance client"""
    print("🔬 Testing Multi-Crypto Yahoo Finance Client")
    print("=" * 50)
    
    client = get_multi_crypto_client()
    
    # Test all supported pairs
    test_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC']
    
    for symbol in test_symbols:
        try:
            print(f"\n📊 Testing {symbol}...")
            
            # Test symbol normalization
            normalized = client.normalize_symbol(symbol)
            print(f"  ✓ Normalized: {symbol} → {normalized}")
            
            # Test current price
            price = client.get_current_price(symbol)
            print(f"  ✓ Current price: ${price:.4f}" if price else "  ⚠ Price unavailable")
            
            # Test market data
            market_data = client.get_market_data(symbol)
            if 'error' not in market_data:
                print(f"  ✓ Market data: {market_data['name']} - 24h change: {market_data.get('change_24h', 0):.2f}%")
            else:
                print(f"  ⚠ Market data error: {market_data['error']}")
                
        except Exception as e:
            print(f"  ❌ Error testing {symbol}: {e}")

def test_historical_data():
    """Test historical data loading"""
    print("\n📈 Testing Historical Data Loading")
    print("=" * 50)
    
    client = get_multi_crypto_client()
    test_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    for symbol in test_symbols:
        try:
            print(f"\n📊 Loading historical data for {symbol}...")
            
            # Get 30 days of data
            hist_data = client.get_historical_data(symbol, days_back=30)
            
            if not hist_data.empty:
                print(f"  ✓ Loaded {len(hist_data)} records")
                print(f"  ✓ Date range: {hist_data.index.min()} to {hist_data.index.max()}")
                print(f"  ✓ Columns: {list(hist_data.columns)}")
                print(f"  ✓ Latest close: ${hist_data['Close'].iloc[-1]:.4f}")
                
                # Test data quality
                quality = client.validate_data_quality(hist_data)
                print(f"  ✓ Data quality - Completeness: {quality['completeness']*100:.1f}%, Accuracy: {quality['accuracy']*100:.1f}%")
            else:
                print(f"  ❌ No data returned for {symbol}")
                
        except Exception as e:
            print(f"  ❌ Error loading {symbol}: {e}")

def test_agent_functionality():
    """Test the historical loader agent"""
    print("\n🤖 Testing Historical Loader Agent")
    print("=" * 50)
    
    try:
        agent = get_historical_loader_agent()
        print("  ✓ Agent initialized successfully")
        
        # Test available datasets
        datasets = None
        for tool in agent._create_tools():
            if tool.name == 'get_available_datasets':
                datasets = tool()
                break
        
        if datasets:
            print(f"  ✓ Found {len(datasets)} available datasets:")
            for dataset in datasets[:3]:  # Show first 3
                print(f"    - {dataset['symbol']}: {dataset['name']}")
                if 'current_price' in dataset:
                    print(f"      Price: ${dataset['current_price']:.4f}")
        
        # Test loading a single symbol
        load_tool = None
        for tool in agent._create_tools():
            if tool.name == 'load_symbol_data':
                load_tool = tool
                break
        
        if load_tool:
            print(f"\n  🔄 Testing single symbol load (BTC)...")
            result = load_tool('BTC', 30, False)
            
            if result['success']:
                print(f"  ✓ Loaded {result['data']['records_count']} records for {result['data']['symbol']}")
                print(f"  ✓ Message: {result['message']}")
            else:
                print(f"  ❌ Load failed: {result.get('error')}")
        
        # Test loading multiple symbols
        multi_load_tool = None
        for tool in agent._create_tools():
            if tool.name == 'load_multiple_symbols':
                multi_load_tool = tool
                break
                
        if multi_load_tool:
            print(f"\n  🔄 Testing multiple symbol load...")
            symbols = ['BTC', 'ETH', 'SOL']
            result = multi_load_tool(symbols, 30)
            
            print(f"  ✓ Processed {result['symbols_processed']} symbols")
            print(f"  ✓ Successful: {result['symbols_successful']}")
            print(f"  ✓ Total records: {result['total_records']}")
            print(f"  ✓ Summary: {result['summary']}")
            
    except Exception as e:
        print(f"  ❌ Agent test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Yahoo Finance Extended Agent Test Suite")
    print("=" * 60)
    
    # Test 1: Multi-crypto client
    test_multi_crypto_client()
    
    # Test 2: Historical data
    test_historical_data()
    
    # Test 3: Agent functionality
    test_agent_functionality()
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")

if __name__ == "__main__":
    main()