"""
рекс.com - Professional Cryptocurrency AI Trading Platform
"""

__version__ = "0.1.0"
__author__ = "Paul Turrell"
__email__ = "plturrell@gmail.com"

from .core import TradingEngine
from .agents import AIAgent
from .strategies import Strategy
from .portfolio import Portfolio
from .risk import RiskManager

__all__ = [
    "TradingEngine",
    "AIAgent", 
    "Strategy",
    "Portfolio",
    "RiskManager",
]