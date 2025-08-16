"""
Core agents module
"""
from .base import BaseAgent
from .memory import MemoryAgent
from .strands import StrandsAgent

# Try to import specialized agents
try:
    from .specialized.mcts_calculation_agent_v2 import ProductionMCTSCalculationAgent
    specialized_available = True
except ImportError:
    specialized_available = False

__all__ = [
    'BaseAgent',
    'MemoryAgent',
    'StrandsAgent'
]

if specialized_available:
    __all__.append('ProductionMCTSCalculationAgent')