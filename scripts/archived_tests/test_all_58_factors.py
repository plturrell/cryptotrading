#!/usr/bin/env python3
"""
Test all 58 factor calculations with real data
Validates that each factor can be computed from available data sources
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

def load_test_data():
    """Load real market data for testing"""
    try:
        from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        # Load 200 days of BTC data for comprehensive testing
        btc_data = client.download_data(
            symbol="BTC-USD",
            start_date=(datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            interval="1d",
            save=False
        )
        
        if btc_data is None or btc_data.empty:
            logger.error("Failed to load BTC test data")
            return None
        
        # Ensure we have OHLCV data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in btc_data.columns for col in required_cols):
            logger.error(f"Missing required columns. Available: {list(btc_data.columns)}")
            return None
        
        logger.info(f"Loaded {len(btc_data)} days of BTC data for factor testing")
        return btc_data
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None

def calculate_price_factors(data):
    """Calculate all price factors (1-10)"""
    results = {}
    
    try:
        # 1. spot_price - current close price
        results['spot_price'] = data['close'].iloc[-1] if not data.empty else np.nan
        
        # 2. price_return_1h - For daily data, use 1-day return
        results['price_return_1h'] = data['close'].pct_change().iloc[-1] * 100 if len(data) > 1 else np.nan
        
        # 3. price_return_24h - same as 1-day return for daily data
        results['price_return_24h'] = data['close'].pct_change().iloc[-1] * 100 if len(data) > 1 else np.nan
        
        # 4. price_return_7d - 7-day return
        if len(data) >= 8:
            results['price_return_7d'] = data['close'].pct_change(periods=7).iloc[-1] * 100
        else:
            results['price_return_7d'] = np.nan
        
        # 5. price_return_30d - 30-day return
        if len(data) >= 31:
            results['price_return_30d'] = data['close'].pct_change(periods=30).iloc[-1] * 100
        else:
            results['price_return_30d'] = np.nan
        
        # 6. log_return_1h - logarithmic return
        if len(data) > 1:
            results['log_return_1h'] = np.log(data['close'].iloc[-1] / data['close'].iloc[-2])
        else:
            results['log_return_1h'] = np.nan
        
        # 7. vwap_1h - volume weighted average price (using daily OHLC)
        if 'volume' in data.columns and len(data) > 0:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            volume_price = typical_price * data['volume']
            results['vwap_1h'] = volume_price.iloc[-1] / data['volume'].iloc[-1] if data['volume'].iloc[-1] > 0 else np.nan
        else:
            results['vwap_1h'] = np.nan
        
        # 8. twap_1h - time weighted average price (simple average for daily data)
        results['twap_1h'] = (data['open'].iloc[-1] + data['high'].iloc[-1] + data['low'].iloc[-1] + data['close'].iloc[-1]) / 4 if not data.empty else np.nan
        
        # 9. price_vs_ma_50 - price relative to 50-day MA
        if len(data) >= 50:
            ma_50 = data['close'].rolling(window=50).mean()
            results['price_vs_ma_50'] = data['close'].iloc[-1] / ma_50.iloc[-1] if not pd.isna(ma_50.iloc[-1]) else np.nan
        else:
            results['price_vs_ma_50'] = np.nan
        
        # 10. price_vs_ma_200 - price relative to 200-day MA
        if len(data) >= 200:
            ma_200 = data['close'].rolling(window=200).mean()
            results['price_vs_ma_200'] = data['close'].iloc[-1] / ma_200.iloc[-1] if not pd.isna(ma_200.iloc[-1]) else np.nan
        else:
            results['price_vs_ma_200'] = np.nan
        
    except Exception as e:
        logger.error(f"Error calculating price factors: {e}")
    
    return results

def calculate_volume_factors(data):
    """Calculate all volume factors (11-18)"""
    results = {}
    
    try:
        if 'volume' not in data.columns:
            logger.warning("Volume data not available")
            return {f'volume_factor_{i}': np.nan for i in range(11, 19)}
        
        # 11. spot_volume - current volume
        results['spot_volume'] = data['volume'].iloc[-1] if not data.empty else np.nan
        
        # 12. volume_24h - for daily data, this is the daily volume
        results['volume_24h'] = data['volume'].iloc[-1] if not data.empty else np.nan
        
        # 13. volume_ratio_1h_24h - current vs average volume
        if len(data) >= 24:
            avg_volume_24h = data['volume'].rolling(window=24).mean().iloc[-1]
            results['volume_ratio_1h_24h'] = data['volume'].iloc[-1] / avg_volume_24h if avg_volume_24h > 0 else np.nan
        else:
            results['volume_ratio_1h_24h'] = np.nan
        
        # 14. buy_sell_ratio - simplified as volume/avg_volume
        if len(data) >= 10:
            avg_volume = data['volume'].rolling(window=10).mean().iloc[-1]
            results['buy_sell_ratio'] = data['volume'].iloc[-1] / avg_volume if avg_volume > 0 else np.nan
        else:
            results['buy_sell_ratio'] = np.nan
        
        # 15. large_trade_volume - proxy using high volume days
        volume_threshold = data['volume'].quantile(0.9)  # Top 10% of volume
        results['large_trade_volume'] = data['volume'].iloc[-1] if data['volume'].iloc[-1] > volume_threshold else 0
        
        # 16. volume_momentum - rate of change in volume
        if len(data) >= 5:
            results['volume_momentum'] = data['volume'].pct_change(periods=5).iloc[-1] if len(data) > 5 else np.nan
        else:
            results['volume_momentum'] = np.nan
        
        # 17. obv - On Balance Volume
        if len(data) >= 2:
            price_change = data['close'].diff()
            obv = np.where(price_change > 0, data['volume'], 
                          np.where(price_change < 0, -data['volume'], 0)).cumsum()
            results['obv'] = obv[-1]
        else:
            results['obv'] = np.nan
        
        # 18. volume_profile - simplified volume profile metric
        if len(data) >= 20:
            price_levels = pd.cut(data['close'], bins=10, labels=False)
            volume_by_level = data.groupby(price_levels)['volume'].sum()
            max_volume_level = volume_by_level.idxmax()
            results['volume_profile'] = max_volume_level / 10.0  # Normalized
        else:
            results['volume_profile'] = np.nan
        
    except Exception as e:
        logger.error(f"Error calculating volume factors: {e}")
    
    return results

def calculate_technical_factors(data):
    """Calculate all technical factors (19-28)"""
    results = {}
    
    try:
        # 19. rsi_14 - Relative Strength Index
        if len(data) >= 15:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            results['rsi_14'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else np.nan
        else:
            results['rsi_14'] = np.nan
        
        # 20. macd_signal - MACD Signal
        if len(data) >= 35:
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            results['macd_signal'] = 1 if macd.iloc[-1] > signal.iloc[-1] else 0
        else:
            results['macd_signal'] = np.nan
        
        # 21. bollinger_position - Position within Bollinger Bands
        if len(data) >= 20:
            bb_middle = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            results['bollinger_position'] = bb_position.iloc[-1] if not pd.isna(bb_position.iloc[-1]) else np.nan
        else:
            results['bollinger_position'] = np.nan
        
        # 22. stochastic_k - Stochastic %K
        if len(data) >= 14:
            low_14 = data['low'].rolling(window=14).min()
            high_14 = data['high'].rolling(window=14).max()
            k_percent = 100 * (data['close'] - low_14) / (high_14 - low_14)
            results['stochastic_k'] = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else np.nan
        else:
            results['stochastic_k'] = np.nan
        
        # 23. williams_r - Williams %R
        if len(data) >= 14:
            low_14 = data['low'].rolling(window=14).min()
            high_14 = data['high'].rolling(window=14).max()
            williams_r = -100 * (high_14 - data['close']) / (high_14 - low_14)
            results['williams_r'] = williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else np.nan
        else:
            results['williams_r'] = np.nan
        
        # 24. adx - Average Directional Index (simplified)
        if len(data) >= 14:
            high_diff = data['high'].diff()
            low_diff = data['low'].diff()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            tr = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)),
                                    abs(data['low'] - data['close'].shift(1))))
            plus_di = 100 * pd.Series(plus_dm).rolling(window=14).sum() / pd.Series(tr).rolling(window=14).sum()
            minus_di = 100 * pd.Series(minus_dm).rolling(window=14).sum() / pd.Series(tr).rolling(window=14).sum()
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            results['adx'] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else np.nan
        else:
            results['adx'] = np.nan
        
        # 25. cci - Commodity Channel Index
        if len(data) >= 20:
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - sma) / (0.015 * mad)
            results['cci'] = cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else np.nan
        else:
            results['cci'] = np.nan
        
        # 26. mfi - Money Flow Index
        if len(data) >= 14 and 'volume' in data.columns:
            tp = (data['high'] + data['low'] + data['close']) / 3
            mf = tp * data['volume']
            pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=14).sum()
            neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=14).sum()
            mfi = 100 - (100 / (1 + pos_mf / neg_mf))
            results['mfi'] = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else np.nan
        else:
            results['mfi'] = np.nan
        
        # 27. ichimoku_cloud - Ichimoku Cloud position
        if len(data) >= 52:
            tenkan = (data['high'].rolling(9).max() + data['low'].rolling(9).min()) / 2
            kijun = (data['high'].rolling(26).max() + data['low'].rolling(26).min()) / 2
            senkou_a = ((tenkan + kijun) / 2).shift(26)
            senkou_b = ((data['high'].rolling(52).max() + data['low'].rolling(52).min()) / 2).shift(26)
            # Simplified: 1 if above cloud, 0 if below, 0.5 if in cloud
            current_price = data['close'].iloc[-1]
            if not pd.isna(senkou_a.iloc[-1]) and not pd.isna(senkou_b.iloc[-1]):
                cloud_top = max(senkou_a.iloc[-1], senkou_b.iloc[-1])
                cloud_bottom = min(senkou_a.iloc[-1], senkou_b.iloc[-1])
                if current_price > cloud_top:
                    results['ichimoku_cloud'] = 1.0
                elif current_price < cloud_bottom:
                    results['ichimoku_cloud'] = 0.0
                else:
                    results['ichimoku_cloud'] = 0.5
            else:
                results['ichimoku_cloud'] = np.nan
        else:
            results['ichimoku_cloud'] = np.nan
        
        # 28. parabolic_sar - Parabolic SAR (simplified)
        if len(data) >= 5:
            # Simplified version - use recent high/low
            recent_high = data['high'].rolling(window=5).max().iloc[-1]
            recent_low = data['low'].rolling(window=5).min().iloc[-1]
            current_price = data['close'].iloc[-1]
            # SAR position relative to price
            sar_value = recent_low if current_price > (recent_high + recent_low) / 2 else recent_high
            results['parabolic_sar'] = sar_value
        else:
            results['parabolic_sar'] = np.nan
        
    except Exception as e:
        logger.error(f"Error calculating technical factors: {e}")
    
    return results

def calculate_volatility_factors(data):
    """Calculate all volatility factors (29-35)"""
    results = {}
    
    try:
        # 29. volatility_1h - 1-hour realized volatility (using daily returns)
        if len(data) >= 2:
            returns = data['close'].pct_change().dropna()
            results['volatility_1h'] = returns.std() if len(returns) > 0 else np.nan
        else:
            results['volatility_1h'] = np.nan
        
        # 30. volatility_24h - 24-hour realized volatility
        if len(data) >= 24:
            returns = data['close'].pct_change().dropna()
            results['volatility_24h'] = returns.rolling(window=24).std().iloc[-1] if len(returns) >= 24 else np.nan
        else:
            results['volatility_24h'] = np.nan
        
        # 31. garch_volatility - GARCH(1,1) simplified
        if len(data) >= 30:
            returns = data['close'].pct_change().dropna()
            if len(returns) >= 30:
                # Simplified GARCH: exponentially weighted volatility
                results['garch_volatility'] = returns.ewm(alpha=0.1).std().iloc[-1]
            else:
                results['garch_volatility'] = np.nan
        else:
            results['garch_volatility'] = np.nan
        
        # 32. volatility_ratio - Short vs long term volatility
        if len(data) >= 30:
            returns = data['close'].pct_change().dropna()
            if len(returns) >= 30:
                short_vol = returns.rolling(window=5).std().iloc[-1]
                long_vol = returns.rolling(window=30).std().iloc[-1]
                results['volatility_ratio'] = short_vol / long_vol if long_vol > 0 else np.nan
            else:
                results['volatility_ratio'] = np.nan
        else:
            results['volatility_ratio'] = np.nan
        
        # 33. parkinson_volatility - High-Low volatility estimator
        if len(data) >= 7:
            hl_ratio = np.log(data['high'] / data['low'])
            parkinson_vol = np.sqrt(hl_ratio.rolling(window=7).apply(lambda x: (x**2).sum() / (4 * len(x) * np.log(2))))
            results['parkinson_volatility'] = parkinson_vol.iloc[-1] if not pd.isna(parkinson_vol.iloc[-1]) else np.nan
        else:
            results['parkinson_volatility'] = np.nan
        
        # 34. atr - Average True Range
        if len(data) >= 14:
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            results['atr'] = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else np.nan
        else:
            results['atr'] = np.nan
        
        # 35. volatility_skew - Volatility skewness
        if len(data) >= 30:
            returns = data['close'].pct_change().dropna()
            if len(returns) >= 30:
                results['volatility_skew'] = returns.rolling(window=30).skew().iloc[-1]
            else:
                results['volatility_skew'] = np.nan
        else:
            results['volatility_skew'] = np.nan
        
    except Exception as e:
        logger.error(f"Error calculating volatility factors: {e}")
    
    return results

def calculate_market_structure_factors(data):
    """Calculate market structure factors (36-42) - simplified versions"""
    results = {}
    
    try:
        # 36. bid_ask_spread - estimated from high-low
        if not data.empty:
            latest_high = data['high'].iloc[-1]
            latest_low = data['low'].iloc[-1]
            latest_close = data['close'].iloc[-1]
            estimated_spread = (latest_high - latest_low) / latest_close
            results['bid_ask_spread'] = estimated_spread
        else:
            results['bid_ask_spread'] = np.nan
        
        # 37. order_book_imbalance - proxy using volume and price movement
        if len(data) >= 2:
            price_change = data['close'].pct_change().iloc[-1]
            volume_change = data['volume'].pct_change().iloc[-1] if 'volume' in data.columns else 0
            # Simple proxy: positive when price and volume move together
            results['order_book_imbalance'] = np.tanh(price_change * volume_change) if not pd.isna(price_change) else np.nan
        else:
            results['order_book_imbalance'] = np.nan
        
        # 38. market_depth_ratio - proxy using volume to volatility ratio
        if len(data) >= 7 and 'volume' in data.columns:
            avg_volume = data['volume'].rolling(window=7).mean().iloc[-1]
            price_volatility = data['close'].pct_change().rolling(window=7).std().iloc[-1]
            results['market_depth_ratio'] = avg_volume / (price_volatility * 1000000) if price_volatility > 0 else np.nan
        else:
            results['market_depth_ratio'] = np.nan
        
        # 39. trade_size_distribution - proxy using volume quartiles
        if len(data) >= 20 and 'volume' in data.columns:
            volume_q75 = data['volume'].quantile(0.75)
            volume_q25 = data['volume'].quantile(0.25)
            current_volume = data['volume'].iloc[-1]
            if current_volume > volume_q75:
                results['trade_size_distribution'] = 0.75  # Large trades
            elif current_volume < volume_q25:
                results['trade_size_distribution'] = 0.25  # Small trades
            else:
                results['trade_size_distribution'] = 0.5   # Medium trades
        else:
            results['trade_size_distribution'] = np.nan
        
        # 40. price_impact - estimated from volatility
        if len(data) >= 5:
            volatility = data['close'].pct_change().rolling(window=5).std().iloc[-1]
            # Assume $100k trade impact proportional to volatility
            results['price_impact'] = volatility * 0.1 if not pd.isna(volatility) else np.nan
        else:
            results['price_impact'] = np.nan
        
        # 41. exchange_flow - proxy using volume momentum
        if len(data) >= 5 and 'volume' in data.columns:
            volume_trend = data['volume'].pct_change(periods=5).iloc[-1]
            results['exchange_flow'] = np.tanh(volume_trend) if not pd.isna(volume_trend) else np.nan
        else:
            results['exchange_flow'] = np.nan
        
        # 42. liquidation_levels - proxy using support/resistance
        if len(data) >= 20:
            recent_lows = data['low'].rolling(window=20).min()
            recent_highs = data['high'].rolling(window=20).max()
            current_price = data['close'].iloc[-1]
            support_distance = (current_price - recent_lows.iloc[-1]) / current_price
            resistance_distance = (recent_highs.iloc[-1] - current_price) / current_price
            # Closer to support = higher liquidation risk
            results['liquidation_levels'] = 1 - support_distance if not pd.isna(support_distance) else np.nan
        else:
            results['liquidation_levels'] = np.nan
        
    except Exception as e:
        logger.error(f"Error calculating market structure factors: {e}")
    
    return results

def calculate_remaining_factors():
    """Calculate remaining factors (43-58) - placeholders for data not available"""
    results = {}
    
    # On-chain factors (43-48) - would need blockchain data
    onchain_factors = [
        'network_hashrate', 'active_addresses', 'transaction_volume',
        'exchange_balance', 'nvt_ratio', 'whale_movements'
    ]
    
    # Sentiment factors (49-52) - would need social media data
    sentiment_factors = [
        'social_volume', 'social_sentiment', 'fear_greed_index', 'reddit_sentiment'
    ]
    
    # Macro factors (53-55) - would need FRED data
    macro_factors = [
        'dxy_correlation', 'gold_correlation', 'spy_correlation'
    ]
    
    # DeFi factors (56-58) - would need DeFi protocol data
    defi_factors = [
        'tvl_ratio', 'staking_ratio', 'defi_dominance'
    ]
    
    all_missing = onchain_factors + sentiment_factors + macro_factors + defi_factors
    
    for factor in all_missing:
        results[factor] = 'DATA_SOURCE_UNAVAILABLE'
    
    return results

def test_all_58_factors():
    """Test calculation of all 58 factors"""
    print("🧮 Testing All 58 Factor Calculations")
    print("=" * 80)
    
    # Load test data
    data = load_test_data()
    if data is None:
        print("❌ Failed to load test data")
        return {}
    
    all_results = {}
    
    # Test price factors (1-10)
    print("\n📊 Testing Price Factors (1-10)...")
    price_results = calculate_price_factors(data)
    all_results.update(price_results)
    successful_price = sum(1 for v in price_results.values() if not pd.isna(v))
    print(f"✅ Price factors: {successful_price}/{len(price_results)} calculated successfully")
    
    # Test volume factors (11-18)
    print("\n📈 Testing Volume Factors (11-18)...")
    volume_results = calculate_volume_factors(data)
    all_results.update(volume_results)
    successful_volume = sum(1 for v in volume_results.values() if not pd.isna(v))
    print(f"✅ Volume factors: {successful_volume}/{len(volume_results)} calculated successfully")
    
    # Test technical factors (19-28)
    print("\n🔧 Testing Technical Factors (19-28)...")
    technical_results = calculate_technical_factors(data)
    all_results.update(technical_results)
    successful_technical = sum(1 for v in technical_results.values() if not pd.isna(v))
    print(f"✅ Technical factors: {successful_technical}/{len(technical_results)} calculated successfully")
    
    # Test volatility factors (29-35)
    print("\n📊 Testing Volatility Factors (29-35)...")
    volatility_results = calculate_volatility_factors(data)
    all_results.update(volatility_results)
    successful_volatility = sum(1 for v in volatility_results.values() if not pd.isna(v))
    print(f"✅ Volatility factors: {successful_volatility}/{len(volatility_results)} calculated successfully")
    
    # Test market structure factors (36-42)
    print("\n🏗️  Testing Market Structure Factors (36-42)...")
    market_results = calculate_market_structure_factors(data)
    all_results.update(market_results)
    successful_market = sum(1 for v in market_results.values() if not pd.isna(v))
    print(f"✅ Market structure factors: {successful_market}/{len(market_results)} calculated successfully")
    
    # Test remaining factors (43-58)
    print("\n⚠️  Testing Remaining Factors (43-58)...")
    remaining_results = calculate_remaining_factors()
    all_results.update(remaining_results)
    unavailable_count = sum(1 for v in remaining_results.values() if v == 'DATA_SOURCE_UNAVAILABLE')
    print(f"⚠️  Remaining factors: {unavailable_count}/{len(remaining_results)} require external data sources")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 ALL 58 FACTORS SUMMARY")
    print("=" * 80)
    
    computable_factors = successful_price + successful_volume + successful_technical + successful_volatility + successful_market
    total_testable_factors = len(price_results) + len(volume_results) + len(technical_results) + len(volatility_results) + len(market_results)
    
    print(f"✅ Computable from available data: {computable_factors}/{total_testable_factors} ({computable_factors/total_testable_factors*100:.1f}%)")
    print(f"⚠️  Require external data sources: {unavailable_count}/16 ({unavailable_count/16*100:.1f}%)")
    print(f"📊 Total factor coverage: {len(all_results)}/58 factors defined")
    
    # Show sample calculations
    print("\n📋 Sample Factor Values:")
    for i, (factor_name, value) in enumerate(list(all_results.items())[:10]):
        if not pd.isna(value) and value != 'DATA_SOURCE_UNAVAILABLE':
            if isinstance(value, float):
                print(f"   {factor_name}: {value:.4f}")
            else:
                print(f"   {factor_name}: {value}")
        else:
            print(f"   {factor_name}: {value}")
    
    # Assessment
    if computable_factors >= 35:  # At least 35 out of 42 computable factors
        print(f"\n✅ EXCELLENT: Factor calculation system working well!")
        print(f"   {computable_factors} factors can be computed from available data")
        print(f"   External data integration needed for remaining {unavailable_count} factors")
    else:
        print(f"\n⚠️  PARTIAL: Factor calculation needs improvement")
        print(f"   Only {computable_factors} factors working from available data")
    
    return all_results

def main():
    """Run the comprehensive 58-factor test"""
    print("🚀 Comprehensive 58-Factor Calculation Test")
    print("=" * 80)
    print("Testing all 58 factor calculations with real market data")
    print("=" * 80)
    
    results = test_all_58_factors()
    
    print(f"\n📝 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 {len(results)} factor calculations attempted")
    
    return results

if __name__ == "__main__":
    results = main()