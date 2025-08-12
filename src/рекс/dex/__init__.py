"""
DEX (Decentralized Exchange) integration for рекс.com
Trade directly through MetaMask wallet
"""

from .uniswap_client import UniswapClient
from .dex_aggregator import DEXAggregator

__all__ = ['UniswapClient', 'DEXAggregator']