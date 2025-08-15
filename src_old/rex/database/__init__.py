"""
SQLite database module for rex.com
"""

from .client import DatabaseClient, get_db
from .models import (
    Base, User, Trade, Portfolio, AIAnalysis, ConversationSession, 
    ConversationMessage, AgentContext, MemoryFragment, SemanticMemory
)

__all__ = [
    'DatabaseClient', 'get_db', 'Base', 'User', 'Trade', 'Portfolio', 'AIAnalysis',
    'ConversationSession', 'ConversationMessage', 'AgentContext', 'MemoryFragment', 'SemanticMemory'
]