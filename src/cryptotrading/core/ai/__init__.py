"""
AI and Machine Learning Integration Package
Provides AI-powered analysis and decision making capabilities
"""

from .ai_gateway_client import AIGatewayClient
from .analysis import SentimentAnalyzer, TechnicalAnalyzer
from .decision_engine import TradingDecisionEngine
from .grok4_client import Grok4Client, Grok4ClientFactory, close_grok4_client, get_grok4_client
from .models import AIModel, PredictionModel

__all__ = [
    "AIModel",
    "PredictionModel",
    "TechnicalAnalyzer",
    "SentimentAnalyzer",
    "TradingDecisionEngine",
    "Grok4Client",
    "get_grok4_client",
    "close_grok4_client",
    "Grok4ClientFactory",
    "AIGatewayClient",  # Backwards compatibility
]
