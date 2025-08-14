#!/usr/bin/env python3
"""
Load 2 years of historical data for main crypto trading pairs
Uses the extended Yahoo Finance agent
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

def load_2year_crypto_data():
    """Load 2 years of data for all supported crypto pairs"""
    print("📈 Loading 2 Years of Cryptocurrency Data")
    print("=" * 60)
    
    client = get_multi_crypto_client()
    
    # Main crypto pairs to load
    crypto_pairs = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC']
    
    results = {}
    total_records = 0
    
    print(f"🔄 Loading 730 days of data for {len(crypto_pairs)} cryptocurrency pairs...")
    print(f"Start date: ~{datetime.now().replace(year=datetime.now().year-2).strftime('%Y-%m-%d')}")
    print(f"End date: {datetime.now().strftime('%Y-%m-%d')}")
    
    for i, symbol in enumerate(crypto_pairs, 1):
        print(f"\n[{i}/{len(crypto_pairs)}] 📊 Loading {symbol}...")
        
        try:
            # Load 2 years (730 days) of historical data
            hist_data = client.get_historical_data(symbol, days_back=730, interval='1d')
            
            if not hist_data.empty:
                records_count = len(hist_data)
                total_records += records_count
                
                # Get data summary
                start_date = hist_data.index.min().strftime('%Y-%m-%d')
                end_date = hist_data.index.max().strftime('%Y-%m-%d')
                first_price = hist_data['Close'].iloc[0]
                last_price = hist_data['Close'].iloc[-1]
                price_change = ((last_price - first_price) / first_price * 100)
                
                # Volume statistics
                avg_volume = hist_data['Volume'].mean()
                max_volume = hist_data['Volume'].max()
                
                # Price statistics
                min_price = hist_data['Low'].min()
                max_price = hist_data['High'].max()
                
                results[symbol] = {
                    'success': True,
                    'records': records_count,
                    'start_date': start_date,
                    'end_date': end_date,
                    'first_price': first_price,
                    'last_price': last_price,
                    'price_change_pct': price_change,
                    'min_price': min_price,
                    'max_price': max_price,
                    'avg_volume': avg_volume,
                    'max_volume': max_volume,
                    'data': hist_data
                }
                
                print(f"  ✅ Success: {records_count} records")
                print(f"     📅 Period: {start_date} to {end_date}")
                print(f"     💰 Price: ${first_price:.4f} → ${last_price:.4f} ({price_change:+.1f}%)")
                print(f"     📊 Range: ${min_price:.4f} - ${max_price:.4f}")
                print(f"     📈 Avg Volume: {avg_volume:,.0f}")
                
            else:
                results[symbol] = {
                    'success': False,
                    'error': 'No data returned',
                    'records': 0
                }
                print(f"  ❌ No data available for {symbol}")
                
        except Exception as e:
            results[symbol] = {
                'success': False,
                'error': str(e),
                'records': 0
            }
            print(f"  ❌ Error loading {symbol}: {e}")
    
    # Summary report
    print(f"\n" + "=" * 60)
    print("📊 2-YEAR DATA LOADING SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(crypto_pairs) - successful
    
    print(f"✅ Successfully loaded: {successful}/{len(crypto_pairs)} pairs")
    print(f"❌ Failed: {failed} pairs")
    print(f"📈 Total records: {total_records:,}")
    print(f"📊 Average records per pair: {total_records//successful if successful > 0 else 0:,}")
    
    # Top performers (by price change)
    successful_pairs = {k: v for k, v in results.items() if v['success']}
    if successful_pairs:
        print(f"\n🏆 TOP PERFORMERS (2-year price change):")
        sorted_pairs = sorted(successful_pairs.items(), 
                            key=lambda x: x[1]['price_change_pct'], 
                            reverse=True)
        
        for i, (symbol, data) in enumerate(sorted_pairs[:3], 1):
            print(f"  {i}. {symbol}: {data['price_change_pct']:+.1f}% "
                  f"(${data['first_price']:.4f} → ${data['last_price']:.4f})")
    
    # Data quality check
    print(f"\n🔍 DATA QUALITY CHECK:")
    for symbol, data in results.items():
        if data['success']:
            quality = client.validate_data_quality(data['data'])
            print(f"  {symbol}: Completeness {quality['completeness']*100:.1f}%, "
                  f"Accuracy {quality['accuracy']*100:.1f}%")
    
    return results

def save_data_summary(results):
    """Save a summary of loaded data to CSV"""
    print(f"\n💾 Saving data summary...")
    
    summary_data = []
    for symbol, data in results.items():
        if data['success']:
            summary_data.append({
                'Symbol': symbol,
                'Records': data['records'],
                'Start_Date': data['start_date'],
                'End_Date': data['end_date'],
                'First_Price': data['first_price'],
                'Last_Price': data['last_price'],
                'Price_Change_Pct': data['price_change_pct'],
                'Min_Price': data['min_price'],
                'Max_Price': data['max_price'],
                'Avg_Volume': data['avg_volume'],
                'Max_Volume': data['max_volume']
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        filename = f"crypto_2year_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_summary.to_csv(filename, index=False)
        print(f"  ✅ Summary saved to: {filename}")
        
        # Show preview
        print(f"\n📋 Data Summary Preview:")
        print(df_summary.to_string(index=False))
    
    return summary_data

if __name__ == "__main__":
    print("🚀 2-Year Cryptocurrency Data Loader")
    print("Using Yahoo Finance API via extended agent")
    
    # Load the data
    results = load_2year_crypto_data()
    
    # Save summary
    summary = save_data_summary(results)
    
    print(f"\n✅ 2-year data loading completed!")
    print(f"💡 Data is now available for analysis and trading strategies")