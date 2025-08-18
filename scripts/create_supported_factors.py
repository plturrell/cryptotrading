#!/usr/bin/env python3
"""
Create realistic factor definitions based on available Yahoo Finance and FRED data
"""

import sys
sys.path.append('src')

def analyze_yahoo_finance_capabilities():
    """What can we calculate from Yahoo Finance OHLCV data"""
    print("📊 YAHOO FINANCE DATA CAPABILITIES")
    print("=" * 60)
    print("Available data: Open, High, Low, Close, Volume, AdjClose")
    print("Symbols: BTC-USD, ETH-USD, and 15+ other major cryptos")
    print("Historical: 2+ years of daily data loaded")
    print("Frequency: Daily (can get hourly/minute from API)")
    
    yahoo_supported_factors = [
        # Price Factors - Can calculate from Close price
        "spot_price",           # Direct from Close
        "price_return_1h",      # Close.pct_change()
        "price_return_24h",     # Close.pct_change(24)
        "price_return_7d",      # Close.pct_change(7*24)
        "price_return_30d",     # Close.pct_change(30*24)
        "log_return_1h",        # np.log(Close/Close.shift())
        "vwap_1h",             # (Close * Volume).rolling().sum() / Volume.rolling().sum()
        "twap_1h",             # Close.rolling().mean()
        "price_vs_ma_50",      # Close / Close.rolling(50).mean()
        "price_vs_ma_200",     # Close / Close.rolling(200).mean()
        
        # Volume Factors - From Volume column
        "spot_volume",         # Direct from Volume
        "volume_24h",          # Volume.rolling(24).sum()
        "volume_ratio_1h_24h", # Volume.rolling(1).sum() / Volume.rolling(24).mean()
        "volume_momentum",     # Volume.pct_change()
        "obv",                 # On-Balance Volume calculation
        "volume_profile",      # Volume distribution by price levels
        
        # Technical Factors - All calculable from OHLCV
        "rsi_14",              # Relative Strength Index
        "macd_signal",         # MACD signal line
        "bollinger_position",  # Position within Bollinger Bands
        "stochastic_k",        # Stochastic %K
        "williams_r",          # Williams %R
        "adx",                 # Average Directional Index
        "cci",                 # Commodity Channel Index
        "mfi",                 # Money Flow Index (uses High, Low, Close, Volume)
        "ichimoku_cloud",      # Ichimoku cloud calculation
        "parabolic_sar",       # Parabolic SAR
        
        # Volatility Factors - From price returns
        "volatility_1h",       # Returns.rolling().std()
        "volatility_24h",      # 24-hour realized volatility
        "garch_volatility",    # GARCH model from returns
        "volatility_ratio",    # Short vs long-term volatility
        "parkinson_volatility", # High-Low volatility estimator
        "atr",                 # Average True Range
        "volatility_skew",     # Skewness of returns
        
        # Macro Factors - Cross-correlations with other Yahoo symbols
        "dxy_correlation",     # Correlation with DX-Y.NYB (Dollar Index)
        "gold_correlation",    # Correlation with GC=F (Gold futures)
        "spy_correlation",     # Correlation with SPY (S&P 500 ETF)
    ]
    
    print(f"\n✅ Yahoo Finance can support {len(yahoo_supported_factors)} factors:")
    for i, factor in enumerate(yahoo_supported_factors, 1):
        print(f"{i:2d}. {factor}")
    
    return yahoo_supported_factors

def analyze_fred_capabilities():
    """What can we calculate from FRED economic data"""
    print("\n\n📈 FRED DATA CAPABILITIES")  
    print("=" * 60)
    print("Available series: DGS10, WALCL, RRPONTSYD, WTREGEN")
    print("Economic indicators: Treasury rates, Fed balance sheet, etc.")
    
    fred_supported_factors = [
        # Macro correlations with crypto
        "us_treasury_10y",        # Direct from DGS10
        "fed_balance_sheet",      # Direct from WALCL
        "reverse_repo",           # Direct from RRPONTSYD
        "treasury_general_account", # Direct from WTREGEN
        
        # Derived economic factors
        "liquidity_conditions",   # Combination of Fed metrics
        "monetary_policy_stance", # Fed balance sheet / treasury rates
        "dollar_strength_proxy",  # Treasury rate differentials
    ]
    
    print(f"\n✅ FRED can support {len(fred_supported_factors)} additional factors:")
    for i, factor in enumerate(fred_supported_factors, 1):
        print(f"{i:2d}. {factor}")
    
    return fred_supported_factors

def create_realistic_factor_definitions():
    """Create factor definitions for what we can actually implement"""
    
    yahoo_factors = analyze_yahoo_finance_capabilities()
    fred_factors = analyze_fred_capabilities()
    
    total_supported = len(yahoo_factors) + len(fred_factors)
    
    print(f"\n\n🎯 TOTAL REALISTIC FACTOR COUNT")
    print("=" * 60)
    print(f"Yahoo Finance factors: {len(yahoo_factors)}")
    print(f"FRED factors: {len(fred_factors)}")
    print(f"Total supported factors: {total_supported}")
    print(f"Original factor count: 58")
    print(f"Coverage: {total_supported/58*100:.1f}%")
    
    print("\n📝 FACTOR CATEGORIES:")
    print("-" * 30)
    print("✅ Price Analysis (10 factors)")
    print("✅ Volume Analysis (6 factors)")  
    print("✅ Technical Indicators (10 factors)")
    print("✅ Volatility Measures (7 factors)")
    print("✅ Macro Correlations (3 factors)")
    print("✅ Economic Indicators (7 factors)")
    print("❌ Market Structure (0 factors) - needs order book data")
    print("❌ On-Chain Metrics (0 factors) - needs blockchain data")
    print("❌ Sentiment Analysis (0 factors) - needs social data")
    print("❌ DeFi Metrics (0 factors) - needs protocol data")
    
    print(f"\n💡 IMPLEMENTATION STRATEGY:")
    print("-" * 40)
    print("1. Update factor definitions to use YAHOO/FRED sources only")
    print("2. Focus on the 43 factors we can actually calculate")
    print("3. Implement factor calculation engine using existing data")
    print("4. Add other data sources incrementally to reach 58 factors")
    
    print(f"\n🚀 IMMEDIATE NEXT STEPS:")
    print("-" * 40)
    print("1. Create factor calculation functions for OHLCV-based indicators")
    print("2. Build correlation engine for macro factors")
    print("3. Implement real-time factor updates")
    print("4. Test factor quality with loaded 2-year dataset")
    
    return {
        'yahoo_factors': yahoo_factors,
        'fred_factors': fred_factors,
        'total_count': total_supported
    }

if __name__ == "__main__":
    results = create_realistic_factor_definitions()
    
    print(f"\n✅ CONCLUSION: We can realistically support {results['total_count']} factors")
    print("   This provides excellent coverage for crypto trading strategies!")