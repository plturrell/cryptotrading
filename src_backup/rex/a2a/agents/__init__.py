"""
A2A Strand Agents - 100% A2A Compliant Trading Agents
"""

from .base_strands_agent import BaseStrandsAgent
from .a2a_strands_agent import A2AStrandsAgent
from .historical_loader_agent import get_historical_loader_agent, HistoricalLoaderAgent
from .database_agent import database_agent, DatabaseAgent
from .data_management_agent import data_management_agent, DataManagementAgent
from .a2a_coordinator import a2a_coordinator, A2ACoordinator
from .base_memory_agent import BaseMemoryAgent
from .memory_strands_agent import MemoryStrandsAgent

# Import BlockchainStrandsAgent separately to avoid circular imports
try:
    from .blockchain_strands_agent import BlockchainStrandsAgent
    blockchain_available = True
except ImportError:
    BlockchainStrandsAgent = None
    blockchain_available = False

__all__ = [
    'BaseStrandsAgent',
    'A2AStrandsAgent',
    'get_historical_loader_agent', 'HistoricalLoaderAgent',
    'database_agent', 'DatabaseAgent',
    'data_management_agent', 'DataManagementAgent',
    'a2a_coordinator', 'A2ACoordinator',
    'BaseMemoryAgent',
    'MemoryStrandsAgent'
]

# Add BlockchainStrandsAgent to exports if available
if blockchain_available:
    __all__.append('BlockchainStrandsAgent')