#!/usr/bin/env python3
"""
Test dynamic Yahoo Finance configuration
Shows how to use different time periods and granularities
"""

import sys
import os
sys.path.append('.')

from src.rex.ml.multi_crypto_yfinance_client import get_multi_crypto_client
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_configurations():
    """Test various dynamic configurations"""
    print("⚙️  Testing Dynamic Yahoo Finance Configurations")
    print("=" * 60)
    
    client = get_multi_crypto_client()
    
    # Test different configurations
    test_configs = [
        # Format: (symbol, days_back, interval, description)
        ('BTC', 7, '1d', '1 week daily data'),
        ('ETH', 730, '1d', '2 years daily data'),
        ('BTC', 1, '1h', '24 hours hourly data'), 
        ('SOL', 30, '1d', '1 month daily data'),
        ('ADA', 90, '1d', '3 months daily data'),
    ]
    
    results = {}
    
    for symbol, days_back, interval, description in test_configs:
        print(f"\n🔧 Testing: {description}")
        print(f"   Symbol: {symbol}, Days: {days_back}, Interval: {interval}")
        
        try:
            hist_data = client.get_historical_data(
                symbol=symbol,
                days_back=days_back, 
                interval=interval,
                prepost=False,
                auto_adjust=True
            )
            
            if not hist_data.empty:
                start_date = hist_data.index.min().strftime('%Y-%m-%d %H:%M:%S')
                end_date = hist_data.index.max().strftime('%Y-%m-%d %H:%M:%S')
                records = len(hist_data)
                latest_price = hist_data['Close'].iloc[-1]
                
                print(f"   ✅ Success: {records} records")
                print(f"   📅 Period: {start_date} to {end_date}")
                print(f"   💰 Latest: ${latest_price:.4f}")
                
                results[f"{symbol}_{days_back}d_{interval}"] = {
                    'success': True,
                    'records': records,
                    'start_date': start_date,
                    'end_date': end_date,
                    'latest_price': latest_price
                }
            else:
                print(f"   ❌ No data returned")
                results[f"{symbol}_{days_back}d_{interval}"] = {'success': False, 'error': 'No data'}
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[f"{symbol}_{days_back}d_{interval}"] = {'success': False, 'error': str(e)}
    
    return results

def test_all_intervals():
    """Test all supported intervals on BTC"""
    print(f"\n📊 Testing All Supported Intervals (BTC, 7 days)")
    print("=" * 60)
    
    client = get_multi_crypto_client()
    
    # All supported intervals from Yahoo Finance
    intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    working_intervals = []
    
    for interval in intervals:
        try:
            print(f"🔍 Testing {interval}...", end=' ')
            
            # Use shorter time for intraday intervals to avoid limits
            days = 5 if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'] else 30
            
            hist_data = client.get_historical_data('BTC', days_back=days, interval=interval)
            
            if not hist_data.empty:
                print(f"✅ {len(hist_data)} records")
                working_intervals.append(interval)
            else:
                print(f"❌ No data")
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}...")
    
    print(f"\n📈 Working intervals: {working_intervals}")
    return working_intervals

def demonstrate_usage():
    """Demonstrate how to use the dynamic configuration"""
    print(f"\n💡 Usage Examples")
    print("=" * 60)
    
    examples = [
        "# Load 2 years of BTC daily data:",
        "load_symbol_data('BTC', days_back=730, interval='1d')",
        "",
        "# Load 24 hours of ETH hourly data:",
        "load_symbol_data('ETH', days_back=1, interval='1h')", 
        "",
        "# Load 5 days of SOL 15-minute data:",
        "load_symbol_data('SOL', days_back=5, interval='15m')",
        "",
        "# Load multiple symbols with custom config:",
        "load_multiple_symbols(['BTC', 'ETH'], days_back=90, interval='1d')",
        "",
        "# Create training dataset with custom period:",
        "create_training_dataset('ADA', days_back=365, interval='1d')"
    ]
    
    for example in examples:
        print(example)

if __name__ == "__main__":
    print("🚀 Dynamic Yahoo Finance Configuration Test")
    
    # Test various configurations
    config_results = test_dynamic_configurations()
    
    # Test all intervals
    working_intervals = test_all_intervals()
    
    # Show usage examples
    demonstrate_usage()
    
    print(f"\n✅ Dynamic configuration testing completed!")
    
    successful_configs = sum(1 for r in config_results.values() if r.get('success'))
    total_configs = len(config_results)
    
    print(f"📊 Results: {successful_configs}/{total_configs} configurations worked")
    print(f"📊 Intervals: {len(working_intervals)} out of 13 intervals supported")
    print(f"💡 Yahoo Finance agent now supports flexible time periods and granularities!")