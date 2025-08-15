"""
Professional Trading Configuration Module
Institutional-grade indicator sets and trading strategies based on Two Sigma, Deribit, Jump Trading, and Galaxy Digital
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    CRISIS = "crisis"
    EUPHORIA = "euphoria"


class TradingStrategy(Enum):
    """Professional trading strategy types"""
    MACRO_DRIVEN = "macro_driven"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"
    CROSS_ASSET_MOMENTUM = "cross_asset_momentum"
    MEAN_REVERSION = "mean_reversion"
    SECTOR_ROTATION = "sector_rotation"
    CARRY_TRADE = "carry_trade"


@dataclass
class IndicatorSet:
    """Professional indicator set configuration"""
    name: str
    description: str
    symbols: List[str]
    weights: Dict[str, float]
    min_correlation: float
    strategy_type: TradingStrategy
    institutional_reference: str


class ProfessionalTradingConfig:
    """Configuration for institutional-grade crypto trading strategies"""
    
    # Two Sigma Factor Lens Model
    TWO_SIGMA_FACTOR_MODEL = IndicatorSet(
        name="Two Sigma Factor Lens",
        description="Multi-factor model with 0.74 beta to global equity",
        symbols=[
            '^GSPC',  # Global equity factor
            '^TNX',   # 10-year Treasury (15% correlation)
            'TIP',    # Inflation breakevens (0.76 beta)
            'DX-Y.NYB',  # USD strength
            'GC=F',   # Gold (9% correlation vs 15% for TIPS)
            'CL=F',   # Oil prices
            'EEM',    # Emerging markets (negative exposure)
            '^VIX',   # Volatility
            'XLK',    # Technology sector
            'HYG'     # Risk appetite
        ],
        weights={
            '^GSPC': 0.20,
            '^TNX': 0.15,
            'TIP': 0.15,
            'DX-Y.NYB': 0.10,
            'GC=F': 0.08,
            'CL=F': 0.07,
            'EEM': 0.10,
            '^VIX': 0.05,
            'XLK': 0.05,
            'HYG': 0.05
        },
        min_correlation=0.09,  # Only 9% of BTC risk explained by traditional factors
        strategy_type=TradingStrategy.MACRO_DRIVEN,
        institutional_reference="Two Sigma Advisers LLC"
    )
    
    # Deribit Volatility Trading Model
    DERIBIT_VOLATILITY_MODEL = IndicatorSet(
        name="Deribit DVOL Strategy",
        description="Volatility-focused trading using VIX-equivalent indices",
        symbols=[
            '^VIX',     # Primary volatility
            '^VVIX',    # Vol of vol
            '^VIX9D',   # Short-term vol
            '^TNX',     # 10Y Treasury yields
            'VIXY',     # VIX futures ETF
            'DX-Y.NYB', # Dollar strength
            'GC=F',     # Gold
            '^SKEW',    # Tail risk
            '^OVX',     # Oil volatility
            'TLT'       # Treasury exposure
        ],
        weights={
            '^VIX': 0.25,
            '^VVIX': 0.15,
            '^VIX9D': 0.15,
            '^TNX': 0.10,
            'VIXY': 0.10,
            'DX-Y.NYB': 0.08,
            'GC=F': 0.07,
            '^SKEW': 0.05,
            '^OVX': 0.03,
            'TLT': 0.02
        },
        min_correlation=-0.75,
        strategy_type=TradingStrategy.VOLATILITY_ARBITRAGE,
        institutional_reference="Deribit Exchange"
    )
    
    # Jump Trading Cross-Asset Arbitrage
    JUMP_TRADING_ARBITRAGE = IndicatorSet(
        name="Jump Trading Cross-Asset",
        description="Statistical arbitrage across futures, options, and equities",
        symbols=[
            '^GSPC',    # Equity futures
            '^VIX',     # Options market
            'GC=F',     # Commodity futures
            'CL=F',     # Energy futures
            'DX-Y.NYB', # Currency futures
            '^TNX',     # Bond futures
            'XLK',      # Tech sector
            'XLF',      # Financial sector
            'QQQ',      # NASDAQ exposure
            'IWM'       # Small cap (Russell 2000 ETF)
        ],
        weights={
            '^GSPC': 0.15,
            '^VIX': 0.15,
            'GC=F': 0.10,
            'CL=F': 0.10,
            'DX-Y.NYB': 0.10,
            '^TNX': 0.10,
            'XLK': 0.10,
            'XLF': 0.08,
            'QQQ': 0.07,
            'IWM': 0.05
        },
        min_correlation=0.50,
        strategy_type=TradingStrategy.CROSS_ASSET_MOMENTUM,
        institutional_reference="Jump Trading LLC"
    )
    
    # Galaxy Digital Comparative Analysis
    GALAXY_DIGITAL_COMPARATIVE = IndicatorSet(
        name="Galaxy Digital Strategy",
        description="BTC vs traditional assets comparative framework",
        symbols=[
            'GC=F',     # Gold comparison ($17.8T market)
            '^GSPC',    # Equity correlation monitoring
            'TLT',      # Treasury flows
            'DX-Y.NYB', # Dollar weakness plays
            'XLF',      # Banking sector (crypto adoption)
            'EEM',      # Emerging markets
            'HYG',      # High yield (risk correlation)
            'VGK',      # European exposure
            'FXE',      # Euro strength
            'XLK'       # Tech correlation
        ],
        weights={
            'GC=F': 0.20,
            '^GSPC': 0.15,
            'TLT': 0.12,
            'DX-Y.NYB': 0.10,
            'XLF': 0.10,
            'EEM': 0.08,
            'HYG': 0.08,
            'VGK': 0.07,
            'FXE': 0.05,
            'XLK': 0.05
        },
        min_correlation=0.60,  # Warning level for structural fragility
        strategy_type=TradingStrategy.MACRO_DRIVEN,
        institutional_reference="Galaxy Digital Holdings"
    )
    
    # Professional Sector Rotation Model
    SECTOR_ROTATION_MODEL = IndicatorSet(
        name="Institutional Sector Rotation",
        description="Track sector flows to predict crypto movements",
        symbols=[
            'XLK',  # Technology
            'XLF',  # Financials
            'XLE',  # Energy
            'XLU',  # Utilities (defensive)
            'XLI',  # Industrials
            'XLY',  # Consumer Discretionary
            'XLP',  # Consumer Staples
            'XLV',  # Healthcare
            'XLB',  # Materials
            'XLRE', # Real Estate
            'IYR'   # Real Estate ETF
        ],
        weights={
            'XLK': 0.20,
            'XLF': 0.15,
            'XLE': 0.10,
            'XLU': 0.08,
            'XLI': 0.08,
            'XLY': 0.08,
            'XLP': 0.08,
            'XLV': 0.08,
            'XLB': 0.08,
            'XLRE': 0.04,
            'IYR': 0.03
        },
        min_correlation=0.70,
        strategy_type=TradingStrategy.SECTOR_ROTATION,
        institutional_reference="Multiple Hedge Funds"
    )
    
    # Enhanced Risk Indicators Set
    RISK_MANAGEMENT_SET = IndicatorSet(
        name="Professional Risk Management",
        description="Comprehensive risk indicators for position sizing",
        symbols=[
            '^VIX',     # Market fear
            '^VVIX',    # Vol of vol
            '^SKEW',    # Tail risk
            'TLT',      # Flight to quality
            'UUP',      # Dollar strength
            'HYG',      # Credit spreads
            'EMB',      # EM credit risk
            'BKLN',     # Floating rate risk
            'MBB',      # MBS liquidity
            '^VIX9D'    # Short-term fear
        ],
        weights={
            '^VIX': 0.20,
            '^VVIX': 0.15,
            '^SKEW': 0.10,
            'TLT': 0.10,
            'UUP': 0.10,
            'HYG': 0.10,
            'EMB': 0.10,
            'BKLN': 0.05,
            'MBB': 0.05,
            '^VIX9D': 0.05
        },
        min_correlation=-0.60,
        strategy_type=TradingStrategy.MEAN_REVERSION,
        institutional_reference="Risk Management Best Practices"
    )
    
    @classmethod
    def get_all_indicator_sets(cls) -> Dict[str, IndicatorSet]:
        """Get all configured indicator sets"""
        return {
            'two_sigma': cls.TWO_SIGMA_FACTOR_MODEL,
            'deribit': cls.DERIBIT_VOLATILITY_MODEL,
            'jump_trading': cls.JUMP_TRADING_ARBITRAGE,
            'galaxy_digital': cls.GALAXY_DIGITAL_COMPARATIVE,
            'sector_rotation': cls.SECTOR_ROTATION_MODEL,
            'risk_management': cls.RISK_MANAGEMENT_SET
        }
    
    @classmethod
    def get_indicators_by_strategy(cls, strategy: TradingStrategy) -> List[IndicatorSet]:
        """Get indicator sets by trading strategy type"""
        all_sets = cls.get_all_indicator_sets()
        return [
            indicator_set for indicator_set in all_sets.values()
            if indicator_set.strategy_type == strategy
        ]
    
    @classmethod
    def get_regime_indicators(cls, regime: MarketRegime) -> List[str]:
        """Get indicators for specific market regime"""
        regime_map = {
            MarketRegime.RISK_ON: ['XLK', 'QQQ', 'EEM', 'HYG', '^RUT', 'XLY'],
            MarketRegime.RISK_OFF: ['^VIX', 'TLT', 'UUP', 'GC=F', 'XLU', '^SKEW'],
            MarketRegime.NEUTRAL: ['^GSPC', '^TNX', 'DX-Y.NYB', 'XLF', 'IWM'],
            MarketRegime.CRISIS: ['^VIX', '^VVIX', 'TLT', 'UUP', 'GC=F', '^SKEW', 'SHY'],
            MarketRegime.EUPHORIA: ['XLK', 'QQQ', '^RUT', 'HYG', 'EEM', 'VIXY']
        }
        return regime_map.get(regime, [])
    
    @staticmethod
    def get_correlation_windows() -> Dict[str, str]:
        """Get recommended correlation analysis windows"""
        return {
            'intraday': '1h',
            'short_term': '1d',
            'medium_term': '1w',
            'long_term': '1mo',
            'regime_detection': '3mo'
        }
    
    @staticmethod
    def get_weighting_model() -> Dict[str, float]:
        """Get institutional weighting model"""
        return {
            'macro_factors': 0.40,
            'liquidity_factors': 0.35,
            'crypto_native': 0.25
        }
    
    @staticmethod
    def get_critical_thresholds() -> Dict[str, Dict[str, float]]:
        """Get critical thresholds for risk management"""
        return {
            '^VIX': {'warning': 25, 'critical': 35, 'extreme': 50},
            'DX-Y.NYB': {'weak': 90, 'neutral': 95, 'strong': 100},
            '^TNX': {'low': 1.5, 'normal': 3.0, 'high': 4.5},
            'HYG': {'tight': 85, 'normal': 80, 'wide': 75},
            '^SKEW': {'normal': 120, 'elevated': 135, 'extreme': 150}
        }
    
    @classmethod
    def get_institutional_validation(cls) -> Dict[str, str]:
        """Get institutional validation references"""
        return {
            'TIP': "Two Sigma: 0.76 beta to Bitcoin vs 0.09 for gold",
            '^TNX': "Deribit: Explicitly analyzes in daily market reports",
            'DX-Y.NYB': "CoinGlass: Confirmed inverse relationship with crypto",
            '^VIX': "Academic research: Bitcoin options best evaluated with VIX",
            'XLK': "60% S&P 500 correlation during COVID crisis",
            'GC=F': "Galaxy Digital: Compares BTC to gold's $17.8T market"
        }