"""
Historical data providers for model training
"""

from .cryptodatadownload import CryptoDataDownloadClient
from .yahoo_finance import YahooFinanceClient
from .bitget import BitgetHistoricalClient
from .aggregator import HistoricalDataAggregator

__all__ = [
    'CryptoDataDownloadClient',
    'YahooFinanceClient',
    'BitgetHistoricalClient',
    'HistoricalDataAggregator'
]