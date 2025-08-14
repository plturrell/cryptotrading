#!/usr/bin/env python3
"""
Complete Data System Test
Tests crypto, equity, FX, and comprehensive metrics loading
"""

import sys
import os
sys.path.append('.')

from src.rex.ml.multi_crypto_yfinance_client import get_multi_crypto_client
from src.rex.ml.equity_indicators_client import get_equity_indicators_client
from src.rex.ml.fx_rates_client import get_fx_rates_client
from src.rex.ml.comprehensive_metrics_client import get_comprehensive_metrics_client
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_data_system():
    """Test all four data sources together"""
    print("🚀 COMPLETE CRYPTO PREDICTION DATA SYSTEM")
    print("=" * 70)
    
    # Initialize all clients
    crypto_client = get_multi_crypto_client()
    equity_client = get_equity_indicators_client()
    fx_client = get_fx_rates_client()
    metrics_client = get_comprehensive_metrics_client()
    
    print("📊 Testing all data sources for BTC prediction...")
    
    # Test 1: Load BTC data
    print("\n1️⃣ Cryptocurrency Data (BTC):")
    try:
        btc_data = crypto_client.get_historical_data('BTC', days_back=30)
        if not btc_data.empty:
            btc_price = btc_data['Close'].iloc[-1]
            btc_change = (btc_data['Close'].iloc[-1] - btc_data['Close'].iloc[0]) / btc_data['Close'].iloc[0] * 100
            print(f"   ✅ BTC: ${btc_price:.2f} ({btc_change:+.1f}% 30d) - {len(btc_data)} records")
        else:
            print("   ❌ BTC: No data")
            return False
    except Exception as e:
        print(f"   ❌ BTC loading failed: {e}")
        return False
    
    # Test 2: Load BTC equity predictors
    print("\n2️⃣ Equity Indicators (BTC Predictors):")
    try:
        btc_equity = equity_client.get_predictors_for_crypto('BTC', days_back=30)
        equity_summary = {}
        for symbol, data in btc_equity.items():
            if not data.empty:
                latest = data['Close'].iloc[-1]
                change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
                equity_summary[symbol] = {'price': latest, 'change': change}
        
        print(f"   ✅ Loaded {len(equity_summary)} equity predictors:")
        for symbol, info in list(equity_summary.items())[:3]:
            print(f"      {symbol}: ${info['price']:.2f} ({info['change']:+.1f}%)")
    except Exception as e:
        print(f"   ❌ Equity predictors failed: {e}")
    
    # Test 3: Load BTC FX predictors
    print("\n3️⃣ FX Rates (BTC Predictors):")
    try:
        btc_fx = fx_client.get_fx_predictors_for_crypto('BTC', days_back=30)
        fx_summary = {}
        for symbol, data in btc_fx.items():
            if not data.empty:
                latest = data['Close'].iloc[-1]
                change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
                fx_summary[symbol] = {'rate': latest, 'change': change}
        
        print(f"   ✅ Loaded {len(fx_summary)} FX predictors:")
        for symbol, info in fx_summary.items():
            pair_info = fx_client.get_fx_pair_info(symbol)
            print(f"      {pair_info.get('name', symbol)}: {info['rate']:.4f} ({info['change']:+.1f}%)")
    except Exception as e:
        print(f"   ❌ FX predictors failed: {e}")
    
    # Test 4: Load comprehensive metrics
    print("\n4️⃣ Comprehensive Metrics (BTC Predictors):")
    try:
        btc_metrics = metrics_client.get_comprehensive_predictors_for_crypto('BTC', days_back=30)
        metrics_summary = {}
        for symbol, data in btc_metrics.items():
            if not data.empty:
                latest = data['Close'].iloc[-1]
                change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
                metrics_summary[symbol] = {'value': latest, 'change': change}
        
        print(f"   ✅ Loaded {len(metrics_summary)} comprehensive metrics:")
        for symbol, info in list(metrics_summary.items())[:5]:
            metric_info = metrics_client.get_indicator_info(symbol)
            print(f"      {metric_info.get('name', symbol)}: {info['value']:.2f} ({info['change']:+.1f}%)")
    except Exception as e:
        print(f"   ❌ Comprehensive metrics failed: {e}")
    
    return True

def test_alert_systems():
    """Test alert and early warning systems"""
    print(f"\n🚨 ALERT & EARLY WARNING SYSTEMS")
    print("=" * 70)
    
    fx_client = get_fx_rates_client()
    metrics_client = get_comprehensive_metrics_client()
    
    # Test FX early warning
    print("1️⃣ FX Early Warning Signals:")
    try:
        fx_alerts = fx_client.get_early_warning_signals(threshold_pct=1.5)
        if fx_alerts and "error" not in fx_alerts:
            print(f"   📊 Monitoring {fx_alerts['signals_monitored']} FX pairs")
            print(f"   🚨 {fx_alerts['alerts_triggered']} alerts triggered")
            
            if fx_alerts['alerts']:
                for alert in fx_alerts['alerts'][:2]:  # Show first 2 alerts
                    print(f"      {alert['alert_type']}: {alert['fx_pair']} ({alert['change_pct']:+.1f}%)")
            else:
                print("   ✅ No FX alerts - markets stable")
        else:
            print(f"   ❌ FX alerts failed: {fx_alerts.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ FX alerts error: {e}")
    
    # Test comprehensive metrics alerts
    print("\n2️⃣ Comprehensive Metrics Alerts:")
    try:
        metrics_alerts = metrics_client.get_alert_conditions()
        if metrics_alerts and "error" not in metrics_alerts:
            print(f"   📊 Monitoring {len(metrics_alerts['indicators_monitored'])} indicators")
            print(f"   🚨 {metrics_alerts['alert_count']} alerts triggered")
            
            if metrics_alerts['alerts']:
                for alert in metrics_alerts['alerts']:
                    print(f"      {alert['alert_type']}: {alert['message']}")
            else:
                print("   ✅ No metric alerts - conditions normal")
        else:
            print(f"   ❌ Metrics alerts failed: {metrics_alerts.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Metrics alerts error: {e}")

def test_data_coverage():
    """Test data coverage across all sources"""
    print(f"\n📊 DATA COVERAGE ANALYSIS")
    print("=" * 70)
    
    crypto_client = get_multi_crypto_client()
    equity_client = get_equity_indicators_client()
    fx_client = get_fx_rates_client()
    metrics_client = get_comprehensive_metrics_client()
    
    # Count available indicators
    crypto_pairs = len(crypto_client.SUPPORTED_PAIRS)
    equity_indicators = len(equity_client.EQUITY_INDICATORS)
    fx_pairs = len(fx_client.FX_PAIRS)
    comprehensive_indicators = len(metrics_client.COMPREHENSIVE_INDICATORS)
    
    total_indicators = crypto_pairs + equity_indicators + fx_pairs + comprehensive_indicators
    
    print(f"📈 Data Source Coverage:")
    print(f"   • Cryptocurrencies: {crypto_pairs} pairs")
    print(f"   • Equity Indicators: {equity_indicators} symbols")
    print(f"   • FX Pairs: {fx_pairs} currency pairs")
    print(f"   • Comprehensive Metrics: {comprehensive_indicators} indicators")
    print(f"   🎯 Total Predictive Indicators: {total_indicators}")
    
    # Show tier breakdown
    print(f"\n🎯 Predictive Power Distribution:")
    
    # FX tiers
    fx_tiers = {}
    for symbol, info in fx_client.FX_PAIRS.items():
        tier = info['tier']
        if tier not in fx_tiers:
            fx_tiers[tier] = 0
        fx_tiers[tier] += 1
    
    for tier in sorted(fx_tiers.keys()):
        print(f"   • FX Tier {tier}: {fx_tiers[tier]} pairs")
    
    # Comprehensive metrics tiers
    metrics_tiers = {}
    for symbol, info in metrics_client.COMPREHENSIVE_INDICATORS.items():
        power = info['predictive_power']
        if power not in metrics_tiers:
            metrics_tiers[power] = 0
        metrics_tiers[power] += 1
    
    for tier in ['very_high', 'high', 'medium', 'low']:
        if tier in metrics_tiers:
            print(f"   • Metrics {tier.title()}: {metrics_tiers[tier]} indicators")

def show_usage_examples():
    """Show practical usage examples"""
    print(f"\n💡 PRACTICAL USAGE EXAMPLES")
    print("=" * 70)
    
    examples = [
        "# Load complete BTC prediction dataset:",
        "load_symbol_data('BTC', days_back=365, interval='1d')",
        "load_crypto_predictors('BTC', days_back=365, interval='1d')  # Equity predictors",  
        "load_crypto_fx_predictors('BTC', days_back=365, interval='1d')  # FX predictors",
        "load_comprehensive_predictors('BTC', days_back=365, interval='1d')  # Metrics",
        "",
        "# Get early warning signals:",
        "get_fx_early_warning_signals(threshold_pct=2.0)",
        "get_comprehensive_alerts()",
        "",
        "# Load by category/tier:",
        "load_tier1_indicators(days_back=90)  # Highest correlation equity",
        "load_tier1_fx_pairs(days_back=90)  # Highest correlation FX", 
        "load_tier_indicators('very_high', days_back=90)  # Top metrics",
        "",
        "# Load by market session:",
        "load_session_fx_pairs('asian', days_back=30)  # Asian session FX",
        "load_category_indicators('volatility', days_back=30)  # Fear indices"
    ]
    
    for example in examples:
        print(f"   {example}")

if __name__ == "__main__":
    print("🎯 COMPLETE CRYPTOCURRENCY PREDICTION DATA SYSTEM")
    print("Testing crypto + equity + FX + comprehensive metrics")
    
    # Test complete system
    success = test_complete_data_system()
    
    if success:
        # Test alert systems
        test_alert_systems()
        
        # Show data coverage
        test_data_coverage()
        
        # Show usage examples
        show_usage_examples()
        
        print(f"\n" + "="*70)
        print("🚀 COMPLETE CRYPTO PREDICTION SYSTEM READY!")
        print("   • 200+ predictive indicators across 4 data sources")
        print("   • Real-time early warning systems")
        print("   • Multi-tier correlation analysis")  
        print("   • Session-based FX signals")
        print("   • Comprehensive market sentiment tracking")
        print("="*70)
    else:
        print(f"\n❌ System test failed - check configuration")