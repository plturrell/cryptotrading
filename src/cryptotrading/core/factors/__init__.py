"""
Comprehensive Factor Model for Cryptocurrency Trading

Defines all 58 factors organized by categories with metadata for 
data ingestion, validation, and quality checks.
"""

from .factor_definitions import (
    Factor,
    FactorCategory,
    FactorFrequency,
    PRICE_FACTORS,
    VOLUME_FACTORS,
    TECHNICAL_FACTORS,
    VOLATILITY_FACTORS,
    MARKET_STRUCTURE_FACTORS,
    ONCHAIN_FACTORS,
    SENTIMENT_FACTORS,
    MACRO_FACTORS,
    DEFI_FACTORS,
    ALL_FACTORS,
    get_factor_by_name,
    get_factors_by_category,
    get_required_data_sources
)

__all__ = [
    'Factor',
    'FactorCategory',
    'FactorFrequency',
    'PRICE_FACTORS',
    'VOLUME_FACTORS',
    'TECHNICAL_FACTORS',
    'VOLATILITY_FACTORS',
    'MARKET_STRUCTURE_FACTORS',
    'ONCHAIN_FACTORS',
    'SENTIMENT_FACTORS',
    'MACRO_FACTORS',
    'DEFI_FACTORS',
    'ALL_FACTORS',
    'get_factor_by_name',
    'get_factors_by_category',
    'get_required_data_sources'
]