"""
Technical Analysis Factors for Week 2
Uses only Yahoo Finance data - no fake blockchain or exchange data
"""
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List


class FactorCategory(Enum):
    """Factor categories"""

    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    VOLATILITY = "volatility"


@dataclass
class Factor:
    """Factor definition"""

    name: str
    category: FactorCategory
    description: str
    lookback_period: timedelta
    dependencies: List[str]


# Technical Analysis Factors (from Yahoo Finance data only)
FACTORS = [
    # Price-based factors
    Factor(
        name="spot_price",
        category=FactorCategory.PRICE,
        description="Current price",
        lookback_period=timedelta(minutes=1),
        dependencies=[],
    ),
    Factor(
        name="sma_20",
        category=FactorCategory.PRICE,
        description="20-period Simple Moving Average",
        lookback_period=timedelta(hours=20),
        dependencies=["spot_price"],
    ),
    Factor(
        name="ema_50",
        category=FactorCategory.PRICE,
        description="50-period Exponential Moving Average",
        lookback_period=timedelta(hours=50),
        dependencies=["spot_price"],
    ),
    # Volume factors
    Factor(
        name="volume",
        category=FactorCategory.VOLUME,
        description="Trading volume",
        lookback_period=timedelta(minutes=1),
        dependencies=[],
    ),
    Factor(
        name="obv",
        category=FactorCategory.VOLUME,
        description="On-Balance Volume",
        lookback_period=timedelta(days=30),
        dependencies=["spot_price", "volume"],
    ),
    # Technical indicators
    Factor(
        name="rsi_14",
        category=FactorCategory.TECHNICAL,
        description="14-period RSI",
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price"],
    ),
    Factor(
        name="macd",
        category=FactorCategory.TECHNICAL,
        description="MACD indicator",
        lookback_period=timedelta(hours=26),
        dependencies=["spot_price"],
    ),
    Factor(
        name="bollinger_bands",
        category=FactorCategory.TECHNICAL,
        description="Bollinger Bands position",
        lookback_period=timedelta(hours=20),
        dependencies=["spot_price"],
    ),
    Factor(
        name="stochastic",
        category=FactorCategory.TECHNICAL,
        description="Stochastic oscillator",
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price"],
    ),
    # Volatility measures
    Factor(
        name="atr",
        category=FactorCategory.VOLATILITY,
        description="Average True Range",
        lookback_period=timedelta(hours=14),
        dependencies=["spot_price"],
    ),
    Factor(
        name="volatility",
        category=FactorCategory.VOLATILITY,
        description="Price volatility",
        lookback_period=timedelta(hours=24),
        dependencies=["spot_price"],
    ),
]
