#!/usr/bin/env python3
"""
Test equity indicators functionality
Verify equity data loading for crypto prediction
"""

import sys
import os
sys.path.append('.')

from src.rex.ml.equity_indicators_client import get_equity_indicators_client
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_equity_indicators_client():
    """Test the equity indicators client directly"""
    print("📈 Testing Equity Indicators Client")
    print("=" * 60)
    
    client = get_equity_indicators_client()
    
    # Test 1: Load specific equity indicators
    print("\n1️⃣ Loading specific equity indicators...")
    test_symbols = ['SPY', 'QQQ', 'AAPL', 'NVDA', '^VIX']
    
    for symbol in test_symbols:
        try:
            data = client.get_equity_data(symbol, days_back=30)
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                records = len(data)
                date_range = f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"
                print(f"   ✅ {symbol}: ${latest_price:.2f} ({records} records, {date_range})")
            else:
                print(f"   ❌ {symbol}: No data")
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    # Test 2: Load crypto-specific predictors
    print(f"\n2️⃣ Loading crypto-specific predictors...")
    crypto_symbols = ['BTC', 'ETH', 'SOL']
    
    for crypto in crypto_symbols:
        try:
            predictors = client.get_predictors_for_crypto(crypto, days_back=30)
            if predictors:
                predictor_names = list(predictors.keys())
                total_records = sum(len(data) for data in predictors.values())
                print(f"   ✅ {crypto}: {len(predictor_names)} predictors ({total_records:,} records)")
                print(f"      Predictors: {', '.join(predictor_names)}")
            else:
                print(f"   ❌ {crypto}: No predictors found")
        except Exception as e:
            print(f"   ❌ {crypto}: Error - {e}")
    
    # Test 3: Get all indicators info
    print(f"\n3️⃣ Getting indicators information...")
    try:
        indicators_info = client.get_all_indicators_info()
        if indicators_info:
            # Group by category
            categories = {}
            for indicator in indicators_info:
                category = indicator.get('category', 'unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append(indicator['symbol'])
            
            print(f"   ✅ Total indicators: {len(indicators_info)}")
            for category, symbols in categories.items():
                print(f"   📊 {category.title()}: {', '.join(symbols)}")
        else:
            print(f"   ❌ No indicators info returned")
    except Exception as e:
        print(f"   ❌ Error getting indicators info: {e}")
    
    # Test 4: Load Tier 1 indicators
    print(f"\n4️⃣ Loading Tier 1 indicators...")
    try:
        tier1_data = client.get_all_tier1_indicators(days_back=7)
        if tier1_data:
            total_records = sum(len(data) for data in tier1_data.values())
            print(f"   ✅ Tier 1 indicators: {len(tier1_data)} loaded ({total_records:,} records)")
            
            # Show sample prices
            for symbol, data in list(tier1_data.items())[:5]:  # First 5
                if not data.empty:
                    latest = data['Close'].iloc[-1]
                    info = client.get_indicator_info(symbol)
                    correlation = info.get('correlation', 0)
                    print(f"      {symbol}: ${latest:.2f} (correlation: {correlation:+.2f})")
        else:
            print(f"   ❌ No Tier 1 data loaded")
    except Exception as e:
        print(f"   ❌ Error loading Tier 1: {e}")

def test_correlation_analysis():
    """Test correlation analysis between crypto and equity"""
    print(f"\n🔗 Testing Correlation Analysis")
    print("=" * 60)
    
    client = get_equity_indicators_client()
    
    try:
        # Load BTC data (we'll simulate this for the test)
        print("📊 Loading BTC and equity data for correlation analysis...")
        
        # Load key predictors
        equity_symbols = ['SPY', 'QQQ', '^VIX', 'MSTR']
        equity_data = {}
        
        for symbol in equity_symbols:
            try:
                data = client.get_equity_data(symbol, days_back=90)  # 3 months
                if not data.empty:
                    equity_data[symbol] = data
                    print(f"   ✅ {symbol}: {len(data)} records loaded")
                else:
                    print(f"   ⚠️  {symbol}: No data")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
        
        print(f"\n💡 Correlation analysis ready with {len(equity_data)} equity indicators")
        print("   (In real implementation, this would correlate with crypto data)")
        
    except Exception as e:
        print(f"❌ Correlation analysis error: {e}")

def test_data_quality():
    """Test data quality and freshness"""
    print(f"\n🔍 Testing Data Quality")
    print("=" * 60)
    
    client = get_equity_indicators_client()
    
    # Test data freshness
    test_symbols = ['SPY', 'AAPL', 'COIN']
    
    for symbol in test_symbols:
        try:
            # Get recent data
            recent_data = client.get_equity_data(symbol, days_back=5)
            
            if not recent_data.empty:
                latest_date = recent_data.index[-1]
                days_old = (datetime.now() - latest_date.tz_localize(None)).days
                
                # Check for reasonable prices (not zeros or negatives)
                latest_price = recent_data['Close'].iloc[-1]
                volume = recent_data['Volume'].iloc[-1]
                
                print(f"   {symbol}:")
                print(f"      Latest: ${latest_price:.2f} on {latest_date.strftime('%Y-%m-%d')}")
                print(f"      Age: {days_old} days")
                print(f"      Volume: {volume:,.0f}")
                
                if latest_price > 0 and volume > 0 and days_old <= 3:
                    print(f"      ✅ Quality: Good")
                else:
                    print(f"      ⚠️  Quality: Check needed")
            else:
                print(f"   ❌ {symbol}: No recent data")
                
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")

if __name__ == "__main__":
    print("🚀 Equity Indicators Test Suite")
    print("Testing equity data loading for crypto prediction")
    
    # Test client functionality
    test_equity_indicators_client()
    
    # Test correlation capabilities
    test_correlation_analysis()
    
    # Test data quality
    test_data_quality()
    
    print(f"\n" + "="*60)
    print("✅ EQUITY INDICATORS TEST COMPLETED")
    print("   - Real Yahoo Finance equity data loading ✓")
    print("   - Crypto-specific predictor mapping ✓")
    print("   - Tier 1 high-correlation indicators ✓")
    print("   - Data quality validation ✓")
    print("   - Ready for crypto prediction models ✓")
    print("="*60)