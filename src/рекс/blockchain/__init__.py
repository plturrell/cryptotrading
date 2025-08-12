"""
Blockchain integration for рекс.com
MetaMask and Web3 connectivity
"""

from .metamask_client import MetaMaskClient
from .eth_client import EthereumClient

__all__ = ['MetaMaskClient', 'EthereumClient']