#!/usr/bin/env python3
"""
Test FRED data loading with API key
Set FRED_API_KEY environment variable to test
Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, '/Users/apple/projects/cryptotrading/src')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fred_with_api_key():
    """Test FRED data loading if API key is available"""
    print("🏦 Testing FRED Data Loading with API Key")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("⚠️  FRED_API_KEY not found in environment variables")
        print("")
        print("To test FRED data loading:")
        print("1. Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Set environment variable: export FRED_API_KEY=your_api_key_here")
        print("3. Run this test again")
        print("")
        print("Example:")
        print("  export FRED_API_KEY=abcd1234567890")
        print("  python3 test_fred_with_api_key.py")
        return {}
    
    try:
        from cryptotrading.data.historical.fred_client import FREDClient
        
        client = FREDClient(api_key=api_key)
        results = {}
        
        # Test individual series
        test_series = {
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "EFFR": "Effective Federal Funds Rate",
            "M2SL": "M2 Money Stock",
            "WALCL": "Fed Balance Sheet Total Assets",
            "CPIAUCSL": "Consumer Price Index"
        }
        
        print(f"🔍 Testing {len(test_series)} FRED economic series...")
        
        for series_id, description in test_series.items():
            print(f"\n📊 Testing {series_id} - {description}")
            
            try:
                data = client.get_series_data(
                    series_id=series_id,
                    start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    save=True
                )
                
                if data is not None and not data.empty:
                    print(f"   ✅ Success: {len(data)} observations")
                    print(f"   📅 Date range: {data.index[0]} to {data.index[-1]}")
                    print(f"   📈 Value range: {data[series_id].min():.4f} to {data[series_id].max():.4f}")
                    print(f"   🔢 Latest value: {data[series_id].iloc[-1]:.4f}")
                    
                    results[series_id] = {
                        'status': 'success',
                        'records': len(data),
                        'latest_value': data[series_id].iloc[-1],
                        'data': data
                    }
                else:
                    print(f"   ❌ No data returned for {series_id}")
                    results[series_id] = {'status': 'no_data'}
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[series_id] = {'status': 'error', 'error': str(e)}
        
        # Test crypto-relevant data batch loading
        print(f"\n💰 Testing crypto-relevant economic indicators...")
        try:
            crypto_data = client.get_crypto_relevant_data(
                start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if crypto_data:
                print(f"   ✅ Success: {len(crypto_data)} series loaded")
                for series_id, data in crypto_data.items():
                    if data is not None and not data.empty:
                        print(f"     {series_id}: {len(data)} observations")
                results['crypto_relevant'] = {
                    'status': 'success',
                    'series_count': len(crypto_data)
                }
            else:
                print(f"   ❌ No crypto-relevant data loaded")
                results['crypto_relevant'] = {'status': 'no_data'}
                
        except Exception as e:
            print(f"   ❌ Error loading crypto-relevant data: {e}")
            results['crypto_relevant'] = {'status': 'error', 'error': str(e)}
        
        # Test liquidity metrics calculation
        print(f"\n🌊 Testing liquidity metrics calculation...")
        try:
            liquidity_data = client.get_liquidity_metrics(
                start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if liquidity_data is not None and not liquidity_data.empty:
                print(f"   ✅ Success: {len(liquidity_data)} observations")
                print(f"   📊 Metrics: {list(liquidity_data.columns)}")
                if 'NET_LIQUIDITY' in liquidity_data.columns:
                    net_liq = liquidity_data['NET_LIQUIDITY'].iloc[-1]
                    print(f"   💰 Latest net liquidity: ${net_liq:,.0f} million")
                
                results['liquidity_metrics'] = {
                    'status': 'success',
                    'records': len(liquidity_data),
                    'metrics': list(liquidity_data.columns)
                }
            else:
                print(f"   ❌ No liquidity metrics calculated")
                results['liquidity_metrics'] = {'status': 'no_data'}
                
        except Exception as e:
            print(f"   ❌ Error calculating liquidity metrics: {e}")
            results['liquidity_metrics'] = {'status': 'error', 'error': str(e)}
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    except Exception as e:
        print(f"❌ FRED test failed: {e}")
        return {}

def test_economic_data_quality():
    """Test quality of loaded economic data"""
    print(f"\n✅ Testing Economic Data Quality")
    print("=" * 60)
    
    try:
        # Check if we have cached FRED data
        fred_data_dir = Path("data/historical/fred")
        if not fred_data_dir.exists():
            print("⚠️  No FRED data directory found")
            return {}
        
        csv_files = list(fred_data_dir.glob("*.csv"))
        if not csv_files:
            print("⚠️  No FRED data files found")
            return {}
        
        print(f"📊 Found {len(csv_files)} FRED data files")
        
        results = {}
        for file_path in csv_files[:3]:  # Test first 3 files
            try:
                # Extract series ID from filename
                series_id = file_path.stem.split('_')[0]
                
                # Load data
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                if data.empty:
                    continue
                
                print(f"\n🔍 Analyzing {series_id}...")
                
                # Data completeness
                total_rows = len(data)
                non_null_values = data.iloc[:, 0].notna().sum()
                completeness = non_null_values / total_rows * 100
                
                print(f"   📊 Completeness: {completeness:.1f}% ({non_null_values}/{total_rows})")
                
                # Data freshness
                latest_date = data.index[-1]
                days_old = (datetime.now().date() - latest_date.date()).days
                print(f"   📅 Freshness: {days_old} days old (latest: {latest_date.date()})")
                
                # Value range
                values = data.iloc[:, 0].dropna()
                if len(values) > 0:
                    print(f"   📈 Range: {values.min():.4f} to {values.max():.4f}")
                    print(f"   🔢 Latest: {values.iloc[-1]:.4f}")
                
                # Missing data gaps
                if len(values) > 1:
                    missing_count = total_rows - len(values)
                    print(f"   ⚠️  Missing values: {missing_count} ({missing_count/total_rows*100:.1f}%)")
                
                results[series_id] = {
                    'completeness': completeness,
                    'days_old': days_old,
                    'records': total_rows,
                    'missing_values': total_rows - len(values)
                }
                
            except Exception as e:
                print(f"   ❌ Error analyzing {file_path}: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Economic data quality test failed: {e}")
        return {}

def main():
    """Run FRED API tests"""
    print("🏦 FRED Economic Data Testing Suite")
    print("=" * 80)
    print("Testing FRED data loading with real API calls")
    print("=" * 80)
    
    # Test FRED data loading
    fred_results = test_fred_with_api_key()
    
    # Test data quality
    quality_results = test_economic_data_quality()
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 FRED DATA TESTING SUMMARY")
    print("=" * 80)
    
    api_key = os.getenv("FRED_API_KEY")
    
    if not api_key:
        print("⚠️  FRED_API_KEY not set - Cannot test FRED data loading")
        print("   Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   Set with: export FRED_API_KEY=your_api_key_here")
    else:
        print("✅ FRED_API_KEY found - API testing enabled")
        
        if fred_results:
            successful_series = sum(1 for r in fred_results.values() 
                                  if isinstance(r, dict) and r.get('status') == 'success')
            total_series = len([r for r in fred_results.values() 
                              if isinstance(r, dict) and 'status' in r])
            
            print(f"📊 FRED Series: {successful_series}/{total_series} loaded successfully")
            
            # Show sample data
            for series_id, result in fred_results.items():
                if isinstance(result, dict) and result.get('status') == 'success':
                    latest = result.get('latest_value', 'N/A')
                    records = result.get('records', 0)
                    print(f"   {series_id}: {records} records, latest = {latest}")
        
        if quality_results:
            avg_completeness = np.mean([r['completeness'] for r in quality_results.values()])
            avg_freshness = np.mean([r['days_old'] for r in quality_results.values()])
            
            print(f"✅ Data Quality:")
            print(f"   Average completeness: {avg_completeness:.1f}%")
            print(f"   Average freshness: {avg_freshness:.0f} days")
    
    print(f"\n📁 FRED data files saved to: ./data/historical/fred/")
    print(f"📝 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {'fred_results': fred_results, 'quality_results': quality_results}

if __name__ == "__main__":
    results = main()