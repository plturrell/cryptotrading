"""
Factor Definitions for Crypto Trading Data Analysis
Defines standardized factors for quantitative analysis
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class FactorCategory(Enum):
    """Categories for financial factors"""
    
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    SENTIMENT = "sentiment"
    MACRO_ECONOMIC = "macro_economic"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    LIQUIDITY = "liquidity"


class FactorFrequency(Enum):
    """Factor update frequencies"""
    
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


@dataclass
class FactorDefinition:
    """Definition of a quantitative factor"""
    
    name: str
    description: str
    category: FactorCategory
    frequency: FactorFrequency
    calculation_fn: Optional[Callable] = None
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    data_requirements: Optional[List[str]] = None
    is_active: bool = True


# Technical Factors
TECHNICAL_FACTORS = {
    "rsi_14": FactorDefinition(
        name="rsi_14",
        description="14-period Relative Strength Index",
        category=FactorCategory.TECHNICAL,
        frequency=FactorFrequency.MINUTE,
        parameters={"period": 14},
        data_requirements=["close"]
    ),
    
    "macd_signal": FactorDefinition(
        name="macd_signal",
        description="MACD Signal Line Crossover",
        category=FactorCategory.TECHNICAL,
        frequency=FactorFrequency.MINUTE,
        parameters={"fast": 12, "slow": 26, "signal": 9},
        data_requirements=["close"]
    ),
    
    "bollinger_position": FactorDefinition(
        name="bollinger_position",
        description="Position within Bollinger Bands",
        category=FactorCategory.TECHNICAL,
        frequency=FactorFrequency.MINUTE,
        parameters={"period": 20, "std_dev": 2},
        data_requirements=["close"]
    ),
    
    "sma_20": FactorDefinition(
        name="sma_20",
        description="20-period Simple Moving Average",
        category=FactorCategory.TECHNICAL,
        frequency=FactorFrequency.MINUTE,
        parameters={"period": 20},
        data_requirements=["close"]
    ),
    
    "ema_50": FactorDefinition(
        name="ema_50",
        description="50-period Exponential Moving Average",
        category=FactorCategory.TECHNICAL,
        frequency=FactorFrequency.MINUTE,
        parameters={"period": 50},
        data_requirements=["close"]
    )
}

# Volatility Factors
VOLATILITY_FACTORS = {
    "realized_volatility": FactorDefinition(
        name="realized_volatility",
        description="Realized volatility over rolling window",
        category=FactorCategory.VOLATILITY,
        frequency=FactorFrequency.HOUR,
        parameters={"window": 24},
        data_requirements=["close"]
    ),
    
    "garch_volatility": FactorDefinition(
        name="garch_volatility",
        description="GARCH(1,1) conditional volatility",
        category=FactorCategory.VOLATILITY,
        frequency=FactorFrequency.DAILY,
        data_requirements=["close"]
    ),
    
    "vix_factor": FactorDefinition(
        name="vix_factor",
        description="Volatility Index Factor",
        category=FactorCategory.VOLATILITY,
        frequency=FactorFrequency.HOUR,
        data_requirements=["options_data"]
    )
}

# Momentum Factors
MOMENTUM_FACTORS = {
    "price_momentum_1h": FactorDefinition(
        name="price_momentum_1h",
        description="1-hour price momentum",
        category=FactorCategory.MOMENTUM,
        frequency=FactorFrequency.MINUTE,
        parameters={"lookback": 60},
        data_requirements=["close"]
    ),
    
    "volume_momentum": FactorDefinition(
        name="volume_momentum",
        description="Volume-weighted momentum",
        category=FactorCategory.MOMENTUM,
        frequency=FactorFrequency.MINUTE,
        parameters={"lookback": 20},
        data_requirements=["close", "volume"]
    ),
    
    "momentum_reversal": FactorDefinition(
        name="momentum_reversal",
        description="Short-term momentum reversal signal",
        category=FactorCategory.MEAN_REVERSION,
        frequency=FactorFrequency.MINUTE,
        parameters={"short_window": 5, "long_window": 20},
        data_requirements=["close"]
    )
}

# Market Structure Factors  
MARKET_STRUCTURE_FACTORS = {
    "market_depth": FactorDefinition(
        name="market_depth",
        description="Market depth at various price levels",
        category=FactorCategory.MARKET_MICROSTRUCTURE,
        frequency=FactorFrequency.SECOND,
        data_requirements=["order_book"]
    ),
    
    "price_impact_linear": FactorDefinition(
        name="price_impact_linear",
        description="Linear price impact coefficient",
        category=FactorCategory.MARKET_MICROSTRUCTURE,
        frequency=FactorFrequency.MINUTE,
        data_requirements=["trades", "order_book"]
    ),
    
    "effective_spread": FactorDefinition(
        name="effective_spread",
        description="Effective bid-ask spread",
        category=FactorCategory.MARKET_MICROSTRUCTURE,
        frequency=FactorFrequency.SECOND,
        data_requirements=["bid", "ask", "trades"]
    )
}

# Market Microstructure Factors
MICROSTRUCTURE_FACTORS = {
    "bid_ask_spread": FactorDefinition(
        name="bid_ask_spread",
        description="Bid-ask spread as percentage of mid",
        category=FactorCategory.MARKET_MICROSTRUCTURE,
        frequency=FactorFrequency.SECOND,
        data_requirements=["bid", "ask"]
    ),
    
    "order_flow_imbalance": FactorDefinition(
        name="order_flow_imbalance",
        description="Order flow imbalance indicator",
        category=FactorCategory.MARKET_MICROSTRUCTURE,
        frequency=FactorFrequency.SECOND,
        data_requirements=["buy_volume", "sell_volume"]
    ),
    
    "market_impact": FactorDefinition(
        name="market_impact",
        description="Estimated market impact cost",
        category=FactorCategory.MARKET_MICROSTRUCTURE,
        frequency=FactorFrequency.MINUTE,
        data_requirements=["volume", "volatility", "spread"]
    )
}

# Sentiment Factors
SENTIMENT_FACTORS = {
    "fear_greed_index": FactorDefinition(
        name="fear_greed_index",
        description="Market fear and greed sentiment index",
        category=FactorCategory.SENTIMENT,
        frequency=FactorFrequency.DAILY,
        data_requirements=["sentiment_data"]
    ),
    
    "social_sentiment": FactorDefinition(
        name="social_sentiment",
        description="Social media sentiment aggregation",
        category=FactorCategory.SENTIMENT,
        frequency=FactorFrequency.HOUR,
        data_requirements=["social_data"]
    ),
    
    "news_sentiment": FactorDefinition(
        name="news_sentiment",
        description="News sentiment analysis",
        category=FactorCategory.SENTIMENT,
        frequency=FactorFrequency.HOUR,
        data_requirements=["news_data"]
    )
}

# DeFi Factors
DEFI_FACTORS = {
    "defi_tvl_change": FactorDefinition(
        name="defi_tvl_change",
        description="DeFi Total Value Locked daily change rate",
        category=FactorCategory.MACRO_ECONOMIC,
        frequency=FactorFrequency.DAILY,
        data_requirements=["defi_tvl_data"]
    ),
    
    "defi_protocol_dominance": FactorDefinition(
        name="defi_protocol_dominance",
        description="Top protocol dominance in DeFi space",
        category=FactorCategory.MACRO_ECONOMIC,
        frequency=FactorFrequency.DAILY,
        data_requirements=["defi_protocol_data"]
    ),
    
    "yield_farming_apy": FactorDefinition(
        name="yield_farming_apy",
        description="Average yield farming APY across protocols",
        category=FactorCategory.MACRO_ECONOMIC,
        frequency=FactorFrequency.DAILY,
        data_requirements=["yield_data"]
    )
}

# On-Chain Factors
ONCHAIN_FACTORS = {
    "blockchain_activity": FactorDefinition(
        name="blockchain_activity",
        description="On-chain transaction activity metrics",
        category=FactorCategory.MACRO_ECONOMIC,
        frequency=FactorFrequency.HOUR,
        data_requirements=["blockchain_data"]
    ),
    
    "network_hash_rate": FactorDefinition(
        name="network_hash_rate",
        description="Network computational hash rate",
        category=FactorCategory.MACRO_ECONOMIC,
        frequency=FactorFrequency.HOUR,
        data_requirements=["mining_data"]
    ),
    
    "whale_activity": FactorDefinition(
        name="whale_activity", 
        description="Large holder transaction patterns",
        category=FactorCategory.MARKET_MICROSTRUCTURE,
        frequency=FactorFrequency.HOUR,
        data_requirements=["whale_tracking_data"]
    )
}

# Macro Economic Factors
MACRO_FACTORS = {
    "btc_dominance": FactorDefinition(
        name="btc_dominance",
        description="Bitcoin market dominance percentage",
        category=FactorCategory.MACRO_ECONOMIC,
        frequency=FactorFrequency.HOUR,
        data_requirements=["market_cap_data"]
    ),
    
    "defi_tvl": FactorDefinition(
        name="defi_tvl",
        description="DeFi Total Value Locked",
        category=FactorCategory.MACRO_ECONOMIC,
        frequency=FactorFrequency.DAILY,
        data_requirements=["defi_data"]
    ),
    
    "stablecoin_supply": FactorDefinition(
        name="stablecoin_supply",
        description="Total stablecoin supply growth",
        category=FactorCategory.MACRO_ECONOMIC,
        frequency=FactorFrequency.DAILY,
        data_requirements=["stablecoin_data"]
    )
}

# All factors registry
ALL_FACTORS = {
    **TECHNICAL_FACTORS,
    **VOLATILITY_FACTORS,
    **MOMENTUM_FACTORS,
    **MARKET_STRUCTURE_FACTORS,
    **MICROSTRUCTURE_FACTORS,
    **SENTIMENT_FACTORS,
    **DEFI_FACTORS,
    **ONCHAIN_FACTORS,
    **MACRO_FACTORS
}

# Factor groups for easy access
FACTOR_GROUPS = {
    FactorCategory.TECHNICAL: TECHNICAL_FACTORS,
    FactorCategory.VOLATILITY: VOLATILITY_FACTORS,
    FactorCategory.MOMENTUM: MOMENTUM_FACTORS,
    FactorCategory.MARKET_MICROSTRUCTURE: {**MARKET_STRUCTURE_FACTORS, **MICROSTRUCTURE_FACTORS},
    FactorCategory.SENTIMENT: SENTIMENT_FACTORS,
    FactorCategory.MACRO_ECONOMIC: {**DEFI_FACTORS, **MACRO_FACTORS}
}


def get_factor_definition(factor_name: str) -> Optional[FactorDefinition]:
    """Get factor definition by name"""
    return ALL_FACTORS.get(factor_name)


def get_factors_by_category(category: FactorCategory) -> Dict[str, FactorDefinition]:
    """Get all factors for a specific category"""
    return FACTOR_GROUPS.get(category, {})


def get_factors_by_frequency(frequency: FactorFrequency) -> Dict[str, FactorDefinition]:
    """Get all factors for a specific frequency"""
    return {
        name: factor for name, factor in ALL_FACTORS.items()
        if factor.frequency == frequency
    }


def get_active_factors() -> Dict[str, FactorDefinition]:
    """Get all active factors"""
    return {
        name: factor for name, factor in ALL_FACTORS.items()
        if factor.is_active
    }