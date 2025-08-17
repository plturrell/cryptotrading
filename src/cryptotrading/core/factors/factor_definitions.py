"""
Comprehensive Factor Definitions for Cryptocurrency Trading

Defines all 58 factors with metadata for data sources, validation rules,
and quality checks. Organized into 9 categories covering price, volume,
technical, volatility, market structure, on-chain, sentiment, macro, and DeFi.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import timedelta


class FactorCategory(Enum):
    """Factor categories for organization and analysis"""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    VOLATILITY = "volatility"
    MARKET_STRUCTURE = "market_structure"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    DEFI = "defi"


class FactorFrequency(Enum):
    """Data frequency requirements for factors"""
    TICK = "tick"  # Every trade
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"


class DataSource(Enum):
    """Available data sources for factor calculation"""
    # Exchange data
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    
    # Market data aggregators
    YAHOO = "yahoo"
    COINGECKO = "coingecko"
    MESSARI = "messari"
    CRYPTOCOMPARE = "cryptocompare"
    
    # On-chain data
    GLASSNODE = "glassnode"
    SANTIMENT = "santiment"
    INTOTHEBLOCK = "intotheblock"
    
    # DeFi data (removed - not approved)
    DUNE = "dune"
    
    # Sentiment data
    TWITTER = "twitter"
    REDDIT = "reddit"
    LUNARCRUSH = "lunarcrush"
    
    # Macro data
    FRED = "fred"
    TRADINGECONOMICS = "tradingeconomics"


@dataclass
class Factor:
    """Complete factor definition with metadata"""
    name: str
    category: FactorCategory
    description: str
    
    # Data requirements
    min_frequency: FactorFrequency
    optimal_frequency: FactorFrequency
    required_sources: List[DataSource]
    optional_sources: List[DataSource]
    
    # Calculation metadata
    lookback_period: timedelta
    dependencies: List[str]  # Other factors this depends on
    
    # Validation rules
    validation_rules: Dict[str, Any]
    quality_thresholds: Dict[str, float]
    
    # Storage optimization
    is_derived: bool  # Can be calculated from other factors
    cache_ttl: timedelta  # How long to cache calculated values


# Price Factors (1-10)
PRICE_FACTORS = [
    Factor(
        name="spot_price",
        category=FactorCategory.PRICE,
        description="Current spot price from major exchanges",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.SECOND,
        required_sources=[DataSource.BINANCE, DataSource.COINBASE],
        optional_sources=[DataSource.KRAKEN],
        lookback_period=timedelta(minutes=1),
        dependencies=[],
        validation_rules={
            "min_value": 0.0,
            "max_change_percent": 50.0,  # Flag if price changes >50% in 1 minute
            "required_exchanges": 2  # Need at least 2 exchange prices
        },
        quality_thresholds={
            "staleness_seconds": 60,
            "price_deviation_percent": 5.0  # Max deviation between exchanges
        },
        is_derived=False,
        cache_ttl=timedelta(seconds=1)
    ),
    
    Factor(
        name="price_return_1h",
        category=FactorCategory.PRICE,
        description="1-hour price return percentage",
        min_frequency=FactorFrequency.FIVE_MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[DataSource.YAHOO, DataSource.COINGECKO],
        lookback_period=timedelta(hours=1),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -100.0,
            "max_value": 200.0
        },
        quality_thresholds={
            "min_data_points": 12,  # Need at least 12 5-min candles
            "max_gaps": 2
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="price_return_24h",
        category=FactorCategory.PRICE,
        description="24-hour price return percentage",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[DataSource.YAHOO, DataSource.COINGECKO],
        lookback_period=timedelta(hours=24),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -100.0,
            "max_value": 500.0
        },
        quality_thresholds={
            "min_data_points": 24,
            "max_gaps": 4
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=5)
    ),
    
    Factor(
        name="price_return_7d",
        category=FactorCategory.PRICE,
        description="7-day price return percentage",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.YAHOO],
        optional_sources=[DataSource.COINGECKO],
        lookback_period=timedelta(days=7),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -100.0,
            "max_value": 1000.0
        },
        quality_thresholds={
            "min_data_points": 7,
            "max_gaps": 1
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="price_return_30d",
        category=FactorCategory.PRICE,
        description="30-day price return percentage",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.YAHOO],
        optional_sources=[DataSource.COINGECKO],
        lookback_period=timedelta(days=30),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -100.0,
            "max_value": 2000.0
        },
        quality_thresholds={
            "min_data_points": 30,
            "max_gaps": 3
        },
        is_derived=True,
        cache_ttl=timedelta(hours=6)
    ),
    
    Factor(
        name="log_return_1h",
        category=FactorCategory.PRICE,
        description="1-hour logarithmic return",
        min_frequency=FactorFrequency.FIVE_MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=1),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -2.0,
            "max_value": 2.0
        },
        quality_thresholds={
            "min_data_points": 12
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="vwap_1h",
        category=FactorCategory.PRICE,
        description="1-hour volume-weighted average price",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[DataSource.COINBASE],
        lookback_period=timedelta(hours=1),
        dependencies=["spot_price", "spot_volume"],
        validation_rules={
            "min_value": 0.0,
            "max_deviation_from_spot": 0.1  # 10% max deviation
        },
        quality_thresholds={
            "min_volume": 1000.0,
            "min_trades": 100
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="twap_1h",
        category=FactorCategory.PRICE,
        description="1-hour time-weighted average price",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=1),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0
        },
        quality_thresholds={
            "min_data_points": 60
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="price_vs_ma_50",
        category=FactorCategory.PRICE,
        description="Price relative to 50-period moving average",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[DataSource.YAHOO],
        lookback_period=timedelta(hours=50),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.5,
            "max_value": 2.0
        },
        quality_thresholds={
            "min_data_points": 45
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="price_vs_ma_200",
        category=FactorCategory.PRICE,
        description="Price relative to 200-period moving average",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.YAHOO],
        optional_sources=[DataSource.COINGECKO],
        lookback_period=timedelta(days=200),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.2,
            "max_value": 5.0
        },
        quality_thresholds={
            "min_data_points": 180
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
]

# Volume Factors (11-18)
VOLUME_FACTORS = [
    Factor(
        name="spot_volume",
        category=FactorCategory.VOLUME,
        description="Current trading volume across exchanges",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE, DataSource.COINBASE],
        optional_sources=[DataSource.KRAKEN],
        lookback_period=timedelta(minutes=1),
        dependencies=[],
        validation_rules={
            "min_value": 0.0,
            "max_spike_ratio": 100.0  # Flag if volume spikes 100x
        },
        quality_thresholds={
            "min_exchanges": 2
        },
        is_derived=False,
        cache_ttl=timedelta(seconds=30)
    ),
    
    Factor(
        name="volume_24h",
        category=FactorCategory.VOLUME,
        description="24-hour rolling volume",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIVE_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[DataSource.COINGECKO],
        lookback_period=timedelta(hours=24),
        dependencies=["spot_volume"],
        validation_rules={
            "min_value": 0.0
        },
        quality_thresholds={
            "min_data_completeness": 0.9
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=5)
    ),
    
    Factor(
        name="volume_ratio_1h_24h",
        category=FactorCategory.VOLUME,
        description="Ratio of 1-hour volume to 24-hour average",
        min_frequency=FactorFrequency.FIVE_MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=24),
        dependencies=["spot_volume", "volume_24h"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 50.0
        },
        quality_thresholds={
            "min_24h_volume": 10000.0
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="buy_sell_ratio",
        category=FactorCategory.VOLUME,
        description="Ratio of buy volume to sell volume",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(minutes=5),
        dependencies=[],
        validation_rules={
            "min_value": 0.1,
            "max_value": 10.0
        },
        quality_thresholds={
            "min_trades": 50
        },
        is_derived=False,
        cache_ttl=timedelta(seconds=30)
    ),
    
    Factor(
        name="large_trade_volume",
        category=FactorCategory.VOLUME,
        description="Volume from trades >$100k",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[DataSource.COINBASE],
        lookback_period=timedelta(hours=1),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "large_trade_threshold": 100000.0
        },
        quality_thresholds={
            "min_sample_trades": 1000
        },
        is_derived=False,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="volume_momentum",
        category=FactorCategory.VOLUME,
        description="Rate of change in volume",
        min_frequency=FactorFrequency.FIVE_MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=2),
        dependencies=["spot_volume"],
        validation_rules={
            "min_value": -10.0,
            "max_value": 10.0
        },
        quality_thresholds={
            "min_periods": 20
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="obv",
        category=FactorCategory.VOLUME,
        description="On-balance volume indicator",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(days=30),
        dependencies=["spot_price", "spot_volume"],
        validation_rules={
            "allow_negative": True
        },
        quality_thresholds={
            "min_periods": 100
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="volume_profile",
        category=FactorCategory.VOLUME,
        description="Volume distribution by price level",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=24),
        dependencies=["spot_price", "spot_volume"],
        validation_rules={
            "price_bins": 100,
            "min_volume_per_bin": 0.0
        },
        quality_thresholds={
            "min_total_volume": 100000.0
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=5)
    ),
]

# Technical Factors (19-28)
TECHNICAL_FACTORS = [
    Factor(
        name="rsi_14",
        category=FactorCategory.TECHNICAL,
        description="14-period Relative Strength Index",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 100.0
        },
        quality_thresholds={
            "min_periods": 14
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="macd_signal",
        category=FactorCategory.TECHNICAL,
        description="MACD signal line crossover",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=26),
        dependencies=["spot_price"],
        validation_rules={
            "ema_fast": 12,
            "ema_slow": 26,
            "signal_period": 9
        },
        quality_thresholds={
            "min_periods": 35
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="bollinger_position",
        category=FactorCategory.TECHNICAL,
        description="Position within Bollinger Bands (0-1)",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=20),
        dependencies=["spot_price"],
        validation_rules={
            "period": 20,
            "std_dev": 2.0,
            "min_value": -0.5,
            "max_value": 1.5
        },
        quality_thresholds={
            "min_periods": 20
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="stochastic_k",
        category=FactorCategory.TECHNICAL,
        description="Stochastic oscillator %K",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 100.0,
            "period": 14
        },
        quality_thresholds={
            "min_periods": 14
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="williams_r",
        category=FactorCategory.TECHNICAL,
        description="Williams %R indicator",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -100.0,
            "max_value": 0.0,
            "period": 14
        },
        quality_thresholds={
            "min_periods": 14
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="adx",
        category=FactorCategory.TECHNICAL,
        description="Average Directional Index",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 100.0,
            "period": 14
        },
        quality_thresholds={
            "min_periods": 28  # Need 2x period for smoothing
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="cci",
        category=FactorCategory.TECHNICAL,
        description="Commodity Channel Index",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=20),
        dependencies=["spot_price"],
        validation_rules={
            "period": 20,
            "constant": 0.015
        },
        quality_thresholds={
            "min_periods": 20
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="mfi",
        category=FactorCategory.TECHNICAL,
        description="Money Flow Index",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price", "spot_volume"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 100.0,
            "period": 14
        },
        quality_thresholds={
            "min_periods": 14,
            "min_volume": 1000.0
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="ichimoku_cloud",
        category=FactorCategory.TECHNICAL,
        description="Ichimoku cloud position",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(days=52),
        dependencies=["spot_price"],
        validation_rules={
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_span_b_period": 52
        },
        quality_thresholds={
            "min_periods": 52
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="parabolic_sar",
        category=FactorCategory.TECHNICAL,
        description="Parabolic SAR indicator",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(days=5),
        dependencies=["spot_price"],
        validation_rules={
            "initial_af": 0.02,
            "max_af": 0.2
        },
        quality_thresholds={
            "min_periods": 50
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
]

# Volatility Factors (29-35)
VOLATILITY_FACTORS = [
    Factor(
        name="volatility_1h",
        category=FactorCategory.VOLATILITY,
        description="1-hour realized volatility",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=1),
        dependencies=["log_return_1h"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 10.0  # 1000% annualized
        },
        quality_thresholds={
            "min_returns": 30
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="volatility_24h",
        category=FactorCategory.VOLATILITY,
        description="24-hour realized volatility",
        min_frequency=FactorFrequency.FIVE_MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=24),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 5.0
        },
        quality_thresholds={
            "min_returns": 200
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=5)
    ),
    
    Factor(
        name="garch_volatility",
        category=FactorCategory.VOLATILITY,
        description="GARCH(1,1) volatility forecast",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(days=30),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 5.0,
            "alpha": 0.1,
            "beta": 0.85,
            "omega": 0.05
        },
        quality_thresholds={
            "min_periods": 500
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="volatility_ratio",
        category=FactorCategory.VOLATILITY,
        description="Short-term vs long-term volatility ratio",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(days=7),
        dependencies=["volatility_1h", "volatility_24h"],
        validation_rules={
            "min_value": 0.1,
            "max_value": 10.0
        },
        quality_thresholds={
            "min_long_term_vol": 0.01
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="parkinson_volatility",
        category=FactorCategory.VOLATILITY,
        description="Parkinson high-low volatility estimator",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(days=7),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 5.0
        },
        quality_thresholds={
            "min_periods": 50
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="atr",
        category=FactorCategory.VOLATILITY,
        description="Average True Range",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "period": 14
        },
        quality_thresholds={
            "min_periods": 14
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="volatility_skew",
        category=FactorCategory.VOLATILITY,
        description="Volatility skewness",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(days=7),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -3.0,
            "max_value": 3.0
        },
        quality_thresholds={
            "min_returns": 100
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
]

# Market Structure Factors (36-42)
MARKET_STRUCTURE_FACTORS = [
    Factor(
        name="bid_ask_spread",
        category=FactorCategory.MARKET_STRUCTURE,
        description="Best bid-ask spread percentage",
        min_frequency=FactorFrequency.SECOND,
        optimal_frequency=FactorFrequency.SECOND,
        required_sources=[DataSource.BINANCE],
        optional_sources=[DataSource.COINBASE],
        lookback_period=timedelta(seconds=1),
        dependencies=[],
        validation_rules={
            "min_value": 0.0,
            "max_value": 0.1  # 10% max spread
        },
        quality_thresholds={
            "max_staleness_ms": 1000
        },
        is_derived=False,
        cache_ttl=timedelta(seconds=1)
    ),
    
    Factor(
        name="order_book_imbalance",
        category=FactorCategory.MARKET_STRUCTURE,
        description="Buy vs sell order book imbalance",
        min_frequency=FactorFrequency.SECOND,
        optimal_frequency=FactorFrequency.SECOND,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(seconds=1),
        dependencies=[],
        validation_rules={
            "min_value": -1.0,
            "max_value": 1.0,
            "depth_levels": 20
        },
        quality_thresholds={
            "min_depth_usd": 10000.0
        },
        is_derived=False,
        cache_ttl=timedelta(seconds=1)
    ),
    
    Factor(
        name="market_depth_ratio",
        category=FactorCategory.MARKET_STRUCTURE,
        description="Ratio of market depth to daily volume",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(minutes=1),
        dependencies=["volume_24h"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 1.0,
            "depth_percentage": 0.02  # 2% depth
        },
        quality_thresholds={
            "min_depth_usd": 50000.0
        },
        is_derived=True,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="trade_size_distribution",
        category=FactorCategory.MARKET_STRUCTURE,
        description="Distribution of trade sizes (whale indicator)",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(hours=1),
        dependencies=[],
        validation_rules={
            "size_buckets": [100, 1000, 10000, 100000],
            "min_trades": 100
        },
        quality_thresholds={
            "min_sample_size": 500
        },
        is_derived=False,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="price_impact",
        category=FactorCategory.MARKET_STRUCTURE,
        description="Estimated price impact for $100k trade",
        min_frequency=FactorFrequency.MINUTE,
        optimal_frequency=FactorFrequency.SECOND,
        required_sources=[DataSource.BINANCE],
        optional_sources=[],
        lookback_period=timedelta(seconds=1),
        dependencies=["spot_price"],
        validation_rules={
            "test_size_usd": 100000.0,
            "min_value": 0.0,
            "max_value": 0.05  # 5% max impact
        },
        quality_thresholds={
            "min_depth_levels": 20
        },
        is_derived=False,
        cache_ttl=timedelta(seconds=30)
    ),
    
    Factor(
        name="exchange_flow",
        category=FactorCategory.MARKET_STRUCTURE,
        description="Net flow between exchanges",
        min_frequency=FactorFrequency.FIVE_MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE, DataSource.COINBASE],
        optional_sources=[DataSource.KRAKEN],
        lookback_period=timedelta(hours=1),
        dependencies=[],
        validation_rules={
            "min_exchanges": 2
        },
        quality_thresholds={
            "min_volume_usd": 10000.0
        },
        is_derived=False,
        cache_ttl=timedelta(minutes=1)
    ),
    
    Factor(
        name="liquidation_levels",
        category=FactorCategory.MARKET_STRUCTURE,
        description="Nearby liquidation price levels",
        min_frequency=FactorFrequency.FIVE_MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.BINANCE],
        optional_sources=[DataSource.COINGECKO],
        lookback_period=timedelta(minutes=5),
        dependencies=["spot_price"],
        validation_rules={
            "price_range_percent": 0.1,  # Look within 10%
            "min_position_size": 100000.0
        },
        quality_thresholds={
            "data_freshness_minutes": 5
        },
        is_derived=False,
        cache_ttl=timedelta(minutes=1)
    ),
]

# On-Chain Factors (43-48)
ONCHAIN_FACTORS = [
    Factor(
        name="network_hashrate",
        category=FactorCategory.ONCHAIN,
        description="Network hash rate (PoW only)",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.GLASSNODE],
        optional_sources=[DataSource.INTOTHEBLOCK],
        lookback_period=timedelta(hours=1),
        dependencies=[],
        validation_rules={
            "min_value": 0.0,
            "applicable_to": ["BTC", "ETH", "LTC"]
        },
        quality_thresholds={
            "max_change_percent": 50.0
        },
        is_derived=False,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="active_addresses",
        category=FactorCategory.ONCHAIN,
        description="Daily active addresses",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.GLASSNODE],
        optional_sources=[DataSource.SANTIMENT],
        lookback_period=timedelta(days=1),
        dependencies=[],
        validation_rules={
            "min_value": 0
        },
        quality_thresholds={
            "min_addresses": 1000
        },
        is_derived=False,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="transaction_volume",
        category=FactorCategory.ONCHAIN,
        description="On-chain transaction volume in USD",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.GLASSNODE],
        optional_sources=[DataSource.INTOTHEBLOCK],
        lookback_period=timedelta(hours=1),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0
        },
        quality_thresholds={
            "min_transactions": 100
        },
        is_derived=False,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="exchange_balance",
        category=FactorCategory.ONCHAIN,
        description="Total balance on exchanges",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.GLASSNODE],
        optional_sources=[DataSource.SANTIMENT],
        lookback_period=timedelta(hours=1),
        dependencies=[],
        validation_rules={
            "min_value": 0.0
        },
        quality_thresholds={
            "tracked_exchanges": 10
        },
        is_derived=False,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="nvt_ratio",
        category=FactorCategory.ONCHAIN,
        description="Network Value to Transactions ratio",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.GLASSNODE],
        optional_sources=[],
        lookback_period=timedelta(days=1),
        dependencies=["spot_price", "transaction_volume"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 1000.0
        },
        quality_thresholds={
            "min_tx_volume": 1000000.0
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="whale_movements",
        category=FactorCategory.ONCHAIN,
        description="Large wallet transfers (>$1M)",
        min_frequency=FactorFrequency.FIVE_MINUTE,
        optimal_frequency=FactorFrequency.MINUTE,
        required_sources=[DataSource.SANTIMENT],
        optional_sources=[DataSource.INTOTHEBLOCK],
        lookback_period=timedelta(hours=1),
        dependencies=["spot_price"],
        validation_rules={
            "whale_threshold_usd": 1000000.0
        },
        quality_thresholds={
            "confirmation_blocks": 1
        },
        is_derived=False,
        cache_ttl=timedelta(minutes=1)
    ),
]

# Sentiment Factors (49-52)
SENTIMENT_FACTORS = [
    Factor(
        name="social_volume",
        category=FactorCategory.SENTIMENT,
        description="Social media mention volume",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.LUNARCRUSH],
        optional_sources=[DataSource.SANTIMENT],
        lookback_period=timedelta(hours=1),
        dependencies=[],
        validation_rules={
            "min_value": 0,
            "platforms": ["twitter", "reddit", "telegram"]
        },
        quality_thresholds={
            "min_mentions": 100
        },
        is_derived=False,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="social_sentiment",
        category=FactorCategory.SENTIMENT,
        description="Weighted social media sentiment score",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.FIFTEEN_MINUTE,
        required_sources=[DataSource.LUNARCRUSH],
        optional_sources=[DataSource.SANTIMENT],
        lookback_period=timedelta(hours=1),
        dependencies=["social_volume"],
        validation_rules={
            "min_value": -1.0,
            "max_value": 1.0
        },
        quality_thresholds={
            "min_scored_posts": 50
        },
        is_derived=False,
        cache_ttl=timedelta(minutes=15)
    ),
    
    Factor(
        name="fear_greed_index",
        category=FactorCategory.SENTIMENT,
        description="Crypto Fear & Greed Index",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.CRYPTOCOMPARE],
        optional_sources=[],
        lookback_period=timedelta(days=1),
        dependencies=[],
        validation_rules={
            "min_value": 0,
            "max_value": 100
        },
        quality_thresholds={
            "components": ["volatility", "momentum", "social", "surveys"]
        },
        is_derived=False,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="reddit_sentiment",
        category=FactorCategory.SENTIMENT,
        description="Reddit-specific sentiment score",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.REDDIT],
        optional_sources=[],
        lookback_period=timedelta(hours=1),
        dependencies=[],
        validation_rules={
            "min_value": -1.0,
            "max_value": 1.0,
            "subreddits": ["cryptocurrency", "bitcoin", "ethtrader"]
        },
        quality_thresholds={
            "min_comments": 100
        },
        is_derived=False,
        cache_ttl=timedelta(hours=1)
    ),
]

# Macro Factors (53-55)
MACRO_FACTORS = [
    Factor(
        name="dxy_correlation",
        category=FactorCategory.MACRO,
        description="Correlation with US Dollar Index",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.FRED],
        optional_sources=[DataSource.TRADINGECONOMICS],
        lookback_period=timedelta(days=30),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -1.0,
            "max_value": 1.0,
            "correlation_window": 30
        },
        quality_thresholds={
            "min_observations": 20
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="gold_correlation",
        category=FactorCategory.MACRO,
        description="Correlation with gold prices",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.YAHOO],
        optional_sources=[],
        lookback_period=timedelta(days=30),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -1.0,
            "max_value": 1.0,
            "symbol": "GC=F"
        },
        quality_thresholds={
            "min_observations": 20
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="spy_correlation",
        category=FactorCategory.MACRO,
        description="Correlation with S&P 500",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.YAHOO],
        optional_sources=[],
        lookback_period=timedelta(days=30),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": -1.0,
            "max_value": 1.0,
            "symbol": "SPY"
        },
        quality_thresholds={
            "min_observations": 20
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
]

# DeFi Factors (56-58)
DEFI_FACTORS = [
    Factor(
        name="tvl_ratio",
        category=FactorCategory.DEFI,
        description="Total Value Locked as ratio of market cap",
        min_frequency=FactorFrequency.HOURLY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.DEFILLAMA],
        optional_sources=[],
        lookback_period=timedelta(hours=1),
        dependencies=["spot_price"],
        validation_rules={
            "min_value": 0.0,
            "max_value": 10.0,
            "applicable_to": ["ETH", "SOL", "AVAX", "MATIC"]
        },
        quality_thresholds={
            "min_tvl": 1000000.0
        },
        is_derived=True,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="staking_ratio",
        category=FactorCategory.DEFI,
        description="Percentage of supply staked",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.SANTIMENT],
        optional_sources=[DataSource.GLASSNODE],
        lookback_period=timedelta(hours=1),
        dependencies=[],
        validation_rules={
            "min_value": 0.0,
            "max_value": 1.0,
            "applicable_to": ["ETH", "SOL", "ADA", "DOT"]
        },
        quality_thresholds={
            "data_freshness_hours": 24
        },
        is_derived=False,
        cache_ttl=timedelta(hours=1)
    ),
    
    Factor(
        name="defi_dominance",
        category=FactorCategory.DEFI,
        description="DeFi market cap as % of total crypto",
        min_frequency=FactorFrequency.DAILY,
        optimal_frequency=FactorFrequency.HOURLY,
        required_sources=[DataSource.DEFILLAMA],
        optional_sources=[DataSource.COINGECKO],
        lookback_period=timedelta(hours=1),
        dependencies=[],
        validation_rules={
            "min_value": 0.0,
            "max_value": 1.0
        },
        quality_thresholds={
            "tracked_protocols": 100
        },
        is_derived=False,
        cache_ttl=timedelta(hours=1)
    ),
]

# Combine all factors
ALL_FACTORS = (
    PRICE_FACTORS +
    VOLUME_FACTORS +
    TECHNICAL_FACTORS +
    VOLATILITY_FACTORS +
    MARKET_STRUCTURE_FACTORS +
    ONCHAIN_FACTORS +
    SENTIMENT_FACTORS +
    MACRO_FACTORS +
    DEFI_FACTORS
)

# Helper functions
def get_factor_by_name(name: str) -> Optional[Factor]:
    """Get factor definition by name"""
    for factor in ALL_FACTORS:
        if factor.name == name:
            return factor
    return None


def get_factors_by_category(category: FactorCategory) -> List[Factor]:
    """Get all factors in a category"""
    return [f for f in ALL_FACTORS if f.category == category]


def get_required_data_sources() -> Set[DataSource]:
    """Get all required data sources across all factors"""
    sources = set()
    for factor in ALL_FACTORS:
        sources.update(factor.required_sources)
    return sources