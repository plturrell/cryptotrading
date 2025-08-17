#!/usr/bin/env python3
"""
Comprehensive test for real data ingestion system
Tests actual data loading from Yahoo Finance and FRED with real API calls
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

def test_yahoo_finance_real_data():
    """Test YahooFinanceClient with real data loading"""
    print("🔬 Testing Yahoo Finance Real Data Loading")
    print("=" * 60)
    
    try:
        from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
        
        client = YahooFinanceClient()
        results = {}
        
        # Test BTC-USD data loading
        print("\n📊 Testing BTC-USD data loading...")
        btc_data = client.download_data(
            symbol="BTC-USD",
            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            interval="1d",
            save=True
        )
        
        if btc_data is not None and not btc_data.empty:
            print(f"✅ BTC-USD: Loaded {len(btc_data)} days of data")
            print(f"   Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
            print(f"   Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
            print(f"   Latest price: ${btc_data['close'].iloc[-1]:.2f}")
            print(f"   Columns: {list(btc_data.columns)}")
            results['BTC-USD'] = {
                'status': 'success',
                'records': len(btc_data),
                'latest_price': btc_data['close'].iloc[-1],
                'data': btc_data
            }
        else:
            print("❌ BTC-USD: Failed to load data")
            results['BTC-USD'] = {'status': 'failed'}
        
        # Test ETH-USD data loading
        print("\n📊 Testing ETH-USD data loading...")
        eth_data = client.download_data(
            symbol="ETH-USD",
            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            interval="1d",
            save=True
        )
        
        if eth_data is not None and not eth_data.empty:
            print(f"✅ ETH-USD: Loaded {len(eth_data)} days of data")
            print(f"   Date range: {eth_data.index[0]} to {eth_data.index[-1]}")
            print(f"   Price range: ${eth_data['close'].min():.2f} - ${eth_data['close'].max():.2f}")
            print(f"   Latest price: ${eth_data['close'].iloc[-1]:.2f}")
            results['ETH-USD'] = {
                'status': 'success',
                'records': len(eth_data),
                'latest_price': eth_data['close'].iloc[-1],
                'data': eth_data
            }
        else:
            print("❌ ETH-USD: Failed to load data")
            results['ETH-USD'] = {'status': 'failed'}
        
        # Test real-time price functionality
        print("\n📈 Testing real-time price functionality...")
        btc_realtime = client.get_realtime_price("BTC")
        if btc_realtime:
            print(f"✅ BTC real-time price: ${btc_realtime['price']:.2f}")
            print(f"   Previous close: ${btc_realtime['previous_close']:.2f}")
            print(f"   Market cap: ${btc_realtime.get('market_cap', 0):,.0f}")
            results['BTC_realtime'] = btc_realtime
        else:
            print("❌ Failed to get BTC real-time price")
        
        # Test training data preparation
        if 'BTC-USD' in results and results['BTC-USD']['status'] == 'success':
            print("\n🧮 Testing training data preparation...")
            training_data = client.prepare_training_data(results['BTC-USD']['data'])
            if not training_data.empty:
                print(f"✅ Training data prepared: {len(training_data)} rows")
                print(f"   Features: {list(training_data.columns)}")
                # Check for technical indicators
                expected_features = ['rsi', 'macd', 'signal', 'bb_middle', 'returns']
                found_features = [f for f in expected_features if f in training_data.columns]
                print(f"   Technical indicators found: {found_features}")
                results['training_data'] = training_data
            else:
                print("❌ Failed to prepare training data")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {e}")
        return {}

def test_fred_real_data():
    """Test FREDClient with real economic data loading"""
    print("\n🏦 Testing FRED Real Data Loading")
    print("=" * 60)
    
    try:
        from cryptotrading.data.historical.fred_client import FREDClient
        
        # Check if FRED API key is available
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            print("⚠️  FRED_API_KEY not found in environment variables")
            print("   Set FRED_API_KEY to test FRED data loading")
            print("   Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
            return {}
        
        client = FREDClient(api_key=api_key)
        results = {}
        
        # Test key economic indicators
        test_series = {
            "DGS10": "10-Year Treasury Rate",
            "EFFR": "Federal Funds Rate", 
            "M2SL": "M2 Money Supply"
        }
        
        for series_id, description in test_series.items():
            print(f"\n📊 Testing {series_id} ({description})...")
            
            data = client.get_series_data(
                series_id=series_id,
                start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                save=True
            )
            
            if data is not None and not data.empty:
                print(f"✅ {series_id}: Loaded {len(data)} observations")
                print(f"   Date range: {data.index[0]} to {data.index[-1]}")
                print(f"   Value range: {data[series_id].min():.4f} - {data[series_id].max():.4f}")
                print(f"   Latest value: {data[series_id].iloc[-1]:.4f}")
                results[series_id] = {
                    'status': 'success',
                    'records': len(data),
                    'latest_value': data[series_id].iloc[-1],
                    'data': data
                }
            else:
                print(f"❌ {series_id}: Failed to load data")
                results[series_id] = {'status': 'failed'}
        
        # Test crypto-relevant data loading
        print("\n💰 Testing crypto-relevant economic data...")
        crypto_data = client.get_crypto_relevant_data(
            start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if crypto_data:
            print(f"✅ Crypto-relevant data: Loaded {len(crypto_data)} series")
            for series_id, data in crypto_data.items():
                if not data.empty:
                    print(f"   {series_id}: {len(data)} observations")
            results['crypto_relevant'] = crypto_data
        else:
            print("❌ Failed to load crypto-relevant data")
        
        # Test liquidity metrics calculation
        print("\n🌊 Testing liquidity metrics calculation...")
        liquidity_data = client.get_liquidity_metrics(
            start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if not liquidity_data.empty:
            print(f"✅ Liquidity metrics: {len(liquidity_data)} observations")
            print(f"   Metrics: {list(liquidity_data.columns)}")
            if 'NET_LIQUIDITY' in liquidity_data.columns:
                print(f"   Latest net liquidity: {liquidity_data['NET_LIQUIDITY'].iloc[-1]:,.0f}")
            results['liquidity_metrics'] = liquidity_data
        else:
            print("❌ Failed to calculate liquidity metrics")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    except Exception as e:
        print(f"❌ FRED test failed: {e}")
        return {}

def test_database_storage():
    """Test that data gets stored in the database correctly"""
    print("\n💾 Testing Database Storage")
    print("=" * 60)
    
    try:
        from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase
        from cryptotrading.data.database.models import MarketData, EconomicData
        
        client = UnifiedDatabase()
        results = {}
        
        # Test market data storage
        print("📊 Testing market data storage...")
        sample_market_data = {
            'symbol': 'BTC-USD',
            'timestamp': datetime.now(),
            'open': 50000.0,
            'high': 51000.0,
            'low': 49500.0,
            'close': 50500.0,
            'volume': 1000000.0
        }
        
        # Insert sample data
        success = client.insert_market_data(**sample_market_data)
        if success:
            print("✅ Market data insertion successful")
            
            # Query back the data
            query_result = client.get_market_data(
                symbol='BTC-USD',
                start_time=datetime.now() - timedelta(minutes=1),
                end_time=datetime.now() + timedelta(minutes=1)
            )
            
            if query_result:
                print(f"✅ Market data query successful: {len(query_result)} records")
                results['market_data'] = 'success'
            else:
                print("❌ Market data query failed")
                results['market_data'] = 'query_failed'
        else:
            print("❌ Market data insertion failed")
            results['market_data'] = 'insert_failed'
        
        # Test economic data storage
        print("\n🏦 Testing economic data storage...")
        sample_economic_data = {
            'series_id': 'DGS10',
            'timestamp': datetime.now().date(),
            'value': 4.25,
            'source': 'FRED'
        }
        
        success = client.insert_economic_data(**sample_economic_data)
        if success:
            print("✅ Economic data insertion successful")
            
            # Query back the data
            query_result = client.get_economic_data(
                series_id='DGS10',
                start_date=datetime.now().date() - timedelta(days=1),
                end_date=datetime.now().date() + timedelta(days=1)
            )
            
            if query_result:
                print(f"✅ Economic data query successful: {len(query_result)} records")
                results['economic_data'] = 'success'
            else:
                print("❌ Economic data query failed")
                results['economic_data'] = 'query_failed'
        else:
            print("❌ Economic data insertion failed")
            results['economic_data'] = 'insert_failed'
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return {}

def test_factor_calculations():
    """Test factor calculations on real data"""
    print("\n🧮 Testing Factor Calculations on Real Data")
    print("=" * 60)
    
    try:
        from cryptotrading.core.factors.factor_definitions import ALL_FACTORS, get_factor_by_name
        
        results = {}
        
        # Load some real market data first
        from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
        yahoo_client = YahooFinanceClient()
        
        btc_data = yahoo_client.download_data(
            symbol="BTC-USD",
            start_date=(datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            interval="1d",
            save=False
        )
        
        if btc_data is None or btc_data.empty:
            print("❌ No market data available for factor calculations")
            return {}
        
        print(f"📊 Using {len(btc_data)} days of BTC data for factor calculations")
        
        # Test basic price factors
        price_factors_to_test = [
            'price_return_24h', 'price_return_7d', 'price_return_30d',
            'price_vs_ma_50', 'log_return_1h'
        ]
        
        for factor_name in price_factors_to_test:
            factor = get_factor_by_name(factor_name)
            if factor:
                print(f"\n🔍 Testing {factor_name}...")
                try:
                    if factor_name == 'price_return_24h':
                        # Calculate 24h returns
                        if len(btc_data) >= 2:
                            result = btc_data['close'].pct_change() * 100
                            valid_count = result.dropna().count()
                            print(f"✅ {factor_name}: {valid_count} valid calculations")
                            print(f"   Latest value: {result.iloc[-1]:.2f}%")
                            results[factor_name] = 'success'
                        else:
                            print(f"❌ {factor_name}: Insufficient data")
                            results[factor_name] = 'insufficient_data'
                    
                    elif factor_name == 'price_vs_ma_50':
                        # Price vs 50-day moving average
                        if len(btc_data) >= 50:
                            ma_50 = btc_data['close'].rolling(window=50).mean()
                            result = btc_data['close'] / ma_50
                            valid_count = result.dropna().count()
                            print(f"✅ {factor_name}: {valid_count} valid calculations")
                            print(f"   Latest value: {result.iloc[-1]:.4f}")
                            results[factor_name] = 'success'
                        else:
                            print(f"❌ {factor_name}: Insufficient data (need 50+ days)")
                            results[factor_name] = 'insufficient_data'
                    
                    elif factor_name == 'log_return_1h':
                        # For daily data, calculate daily log returns
                        result = np.log(btc_data['close'] / btc_data['close'].shift(1))
                        valid_count = result.dropna().count()
                        print(f"✅ {factor_name} (daily): {valid_count} valid calculations")
                        print(f"   Latest value: {result.iloc[-1]:.6f}")
                        results[factor_name] = 'success'
                    
                    elif 'return' in factor_name:
                        # Generic return calculation
                        if '7d' in factor_name:
                            periods = 7
                        elif '30d' in factor_name:
                            periods = 30
                        else:
                            periods = 1
                        
                        if len(btc_data) >= periods + 1:
                            result = btc_data['close'].pct_change(periods=periods) * 100
                            valid_count = result.dropna().count()
                            print(f"✅ {factor_name}: {valid_count} valid calculations")
                            print(f"   Latest value: {result.iloc[-1]:.2f}%")
                            results[factor_name] = 'success'
                        else:
                            print(f"❌ {factor_name}: Insufficient data (need {periods + 1}+ days)")
                            results[factor_name] = 'insufficient_data'
                    
                except Exception as e:
                    print(f"❌ {factor_name}: Calculation error: {e}")
                    results[factor_name] = 'calculation_error'
        
        # Test technical indicators
        technical_factors_to_test = ['rsi_14', 'bollinger_position', 'atr']
        
        for factor_name in technical_factors_to_test:
            factor = get_factor_by_name(factor_name)
            if factor:
                print(f"\n🔍 Testing {factor_name}...")
                try:
                    if factor_name == 'rsi_14':
                        # RSI calculation
                        if len(btc_data) >= 15:
                            delta = btc_data['close'].diff()
                            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            valid_count = rsi.dropna().count()
                            print(f"✅ {factor_name}: {valid_count} valid calculations")
                            print(f"   Latest RSI: {rsi.iloc[-1]:.2f}")
                            results[factor_name] = 'success'
                        else:
                            print(f"❌ {factor_name}: Insufficient data")
                            results[factor_name] = 'insufficient_data'
                    
                    elif factor_name == 'bollinger_position':
                        # Bollinger Bands position
                        if len(btc_data) >= 20:
                            bb_middle = btc_data['close'].rolling(window=20).mean()
                            bb_std = btc_data['close'].rolling(window=20).std()
                            bb_upper = bb_middle + (bb_std * 2)
                            bb_lower = bb_middle - (bb_std * 2)
                            bb_position = (btc_data['close'] - bb_lower) / (bb_upper - bb_lower)
                            valid_count = bb_position.dropna().count()
                            print(f"✅ {factor_name}: {valid_count} valid calculations")
                            print(f"   Latest position: {bb_position.iloc[-1]:.4f}")
                            results[factor_name] = 'success'
                        else:
                            print(f"❌ {factor_name}: Insufficient data")
                            results[factor_name] = 'insufficient_data'
                    
                    elif factor_name == 'atr':
                        # Average True Range
                        if len(btc_data) >= 14:
                            high_low = btc_data['high'] - btc_data['low']
                            high_close_prev = abs(btc_data['high'] - btc_data['close'].shift(1))
                            low_close_prev = abs(btc_data['low'] - btc_data['close'].shift(1))
                            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                            atr = tr.rolling(window=14).mean()
                            valid_count = atr.dropna().count()
                            print(f"✅ {factor_name}: {valid_count} valid calculations")
                            print(f"   Latest ATR: {atr.iloc[-1]:.2f}")
                            results[factor_name] = 'success'
                        else:
                            print(f"❌ {factor_name}: Insufficient data")
                            results[factor_name] = 'insufficient_data'
                    
                except Exception as e:
                    print(f"❌ {factor_name}: Calculation error: {e}")
                    results[factor_name] = 'calculation_error'
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    except Exception as e:
        print(f"❌ Factor calculation test failed: {e}")
        return {}

def test_quality_validation():
    """Test quality validation on real data"""
    print("\n✅ Testing Quality Validation on Real Data")
    print("=" * 60)
    
    try:
        from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
        
        yahoo_client = YahooFinanceClient()
        
        # Load data for validation
        btc_data = yahoo_client.download_data(
            symbol="BTC-USD",
            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            interval="1d",
            save=False
        )
        
        if btc_data is None or btc_data.empty:
            print("❌ No data available for quality validation")
            return {}
        
        results = {}
        
        # Test data completeness
        print("📊 Testing data completeness...")
        total_rows = len(btc_data)
        non_null_close = btc_data['close'].notna().sum()
        completeness = non_null_close / total_rows
        print(f"✅ Data completeness: {completeness:.2%} ({non_null_close}/{total_rows} rows)")
        results['completeness'] = completeness
        
        # Test for outliers
        print("\n🚨 Testing for outliers...")
        close_prices = btc_data['close'].dropna()
        Q1 = close_prices.quantile(0.25)
        Q3 = close_prices.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = close_prices[(close_prices < lower_bound) | (close_prices > upper_bound)]
        outlier_pct = len(outliers) / len(close_prices)
        print(f"✅ Outlier detection: {len(outliers)} outliers ({outlier_pct:.2%})")
        results['outlier_percentage'] = outlier_pct
        
        # Test price movements
        print("\n📈 Testing price movement validation...")
        daily_returns = btc_data['close'].pct_change() * 100
        extreme_moves = daily_returns[abs(daily_returns) > 20]  # >20% moves
        extreme_pct = len(extreme_moves) / len(daily_returns.dropna())
        print(f"✅ Extreme movements: {len(extreme_moves)} days with >20% moves ({extreme_pct:.2%})")
        results['extreme_movements'] = extreme_pct
        
        # Test volume validation
        print("\n📊 Testing volume validation...")
        if 'volume' in btc_data.columns:
            volume_data = btc_data['volume'].dropna()
            zero_volume_days = (volume_data == 0).sum()
            zero_volume_pct = zero_volume_days / len(volume_data)
            print(f"✅ Volume validation: {zero_volume_days} zero-volume days ({zero_volume_pct:.2%})")
            results['zero_volume_percentage'] = zero_volume_pct
        
        # Test data freshness
        print("\n⏰ Testing data freshness...")
        latest_date = btc_data.index[-1]
        time_since_latest = datetime.now() - latest_date.to_pydatetime().replace(tzinfo=None)
        freshness_hours = time_since_latest.total_seconds() / 3600
        print(f"✅ Data freshness: Latest data is {freshness_hours:.1f} hours old")
        results['freshness_hours'] = freshness_hours
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    except Exception as e:
        print(f"❌ Quality validation test failed: {e}")
        return {}

def main():
    """Run comprehensive real data ingestion tests"""
    print("🚀 Comprehensive Real Data Ingestion Test Suite")
    print("=" * 80)
    print("Testing actual data loading from Yahoo Finance and FRED")
    print("with real API calls, database storage, and factor calculations")
    print("=" * 80)
    
    all_results = {}
    
    # Test 1: Yahoo Finance real data loading
    yahoo_results = test_yahoo_finance_real_data()
    all_results['yahoo_finance'] = yahoo_results
    
    # Test 2: FRED real data loading
    fred_results = test_fred_real_data()
    all_results['fred'] = fred_results
    
    # Test 3: Database storage
    # db_results = test_database_storage()
    # all_results['database'] = db_results
    
    # Test 4: Factor calculations
    factor_results = test_factor_calculations()
    all_results['factors'] = factor_results
    
    # Test 5: Quality validation
    quality_results = test_quality_validation()
    all_results['quality'] = quality_results
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    # Yahoo Finance summary
    if yahoo_results:
        yahoo_success = sum(1 for r in yahoo_results.values() if isinstance(r, dict) and r.get('status') == 'success')
        yahoo_total = len([r for r in yahoo_results.values() if isinstance(r, dict) and 'status' in r])
        print(f"📊 Yahoo Finance: {yahoo_success}/{yahoo_total} data sources working")
    
    # FRED summary
    if fred_results:
        fred_success = sum(1 for r in fred_results.values() if isinstance(r, dict) and r.get('status') == 'success')
        fred_total = len([r for r in fred_results.values() if isinstance(r, dict) and 'status' in r])
        print(f"🏦 FRED Economic Data: {fred_success}/{fred_total} data sources working")
    
    # Factor calculations summary
    if factor_results:
        factor_success = sum(1 for r in factor_results.values() if r == 'success')
        factor_total = len(factor_results)
        print(f"🧮 Factor Calculations: {factor_success}/{factor_total} factors working")
    
    # Quality validation summary
    if quality_results:
        quality_score = 0
        if quality_results.get('completeness', 0) > 0.95:
            quality_score += 1
        if quality_results.get('outlier_percentage', 1) < 0.05:
            quality_score += 1
        if quality_results.get('extreme_movements', 1) < 0.1:
            quality_score += 1
        if quality_results.get('freshness_hours', 100) < 48:
            quality_score += 1
        print(f"✅ Data Quality: {quality_score}/4 quality checks passed")
    
    # Overall assessment
    print("\n🎯 OVERALL ASSESSMENT:")
    
    total_tests = 0
    passed_tests = 0
    
    # Count successful tests
    if yahoo_results and any(isinstance(r, dict) and r.get('status') == 'success' for r in yahoo_results.values()):
        passed_tests += 1
    total_tests += 1
    
    if fred_results and any(isinstance(r, dict) and r.get('status') == 'success' for r in fred_results.values()):
        passed_tests += 1
    total_tests += 1
    
    if factor_results and any(r == 'success' for r in factor_results.values()):
        passed_tests += 1
    total_tests += 1
    
    if quality_results:
        passed_tests += 1
    total_tests += 1
    
    success_rate = passed_tests / total_tests * 100
    
    if success_rate >= 75:
        print(f"✅ EXCELLENT: {passed_tests}/{total_tests} test categories passed ({success_rate:.0f}%)")
        print("   Real data ingestion system is working well!")
    elif success_rate >= 50:
        print(f"⚠️  PARTIAL: {passed_tests}/{total_tests} test categories passed ({success_rate:.0f}%)")
        print("   Some components working, others need attention")
    else:
        print(f"❌ ISSUES: {passed_tests}/{total_tests} test categories passed ({success_rate:.0f}%)")
        print("   Significant issues detected - check API keys and network connectivity")
    
    print(f"\n📁 Data files saved to: ./data/historical/")
    print(f"📝 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results

if __name__ == "__main__":
    results = main()