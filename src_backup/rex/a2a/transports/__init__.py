"""
Transport layer implementations for A2A messages
"""

from .base_transport import MessageTransport

# Lazy import to avoid circular dependency issues
def get_blockchain_transport():
    from .blockchain_transport import BlockchainTransport
    return BlockchainTransport

__all__ = ['MessageTransport', 'get_blockchain_transport']