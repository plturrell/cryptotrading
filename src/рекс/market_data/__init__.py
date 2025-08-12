"""
Market data providers for рекс.com
"""

from .geckoterminal import GeckoTerminalClient
from .coingecko import CoinGeckoClient
from .coinmarketcap import CoinMarketCapClient
from .bitquery import BitqueryClient
from .aggregator import MarketDataAggregator

__all__ = [
    'GeckoTerminalClient',
    'CoinGeckoClient', 
    'CoinMarketCapClient',
    'BitqueryClient',
    'MarketDataAggregator'
]