"""
API v2 - Enhanced endpoints with improved performance and features
NO TRADING - Focus on analytics, technical analysis, and news correlation
"""
from flask import Blueprint
from flask_restx import Api

from .market import market_v2_ns
from .ai import ai_v2_ns
from .ml import ml_v2_ns
from .analytics import analytics_v2_ns
from .news import news_v2_ns

# Create v2 blueprint
api_v2_bp = Blueprint('api_v2', __name__, url_prefix='/api/v2')

# Create API instance for v2
api_v2 = Api(
    api_v2_bp,
    version='2.0',
    title='CryptoTrading Analytics API v2',
    description='Enhanced API for market analytics, technical analysis, and news correlation using STRANDS agents and MCP tools',
    doc='/docs'
)

# Register namespaces
api_v2.add_namespace(market_v2_ns, path='/market')
api_v2.add_namespace(ai_v2_ns, path='/ai')
api_v2.add_namespace(ml_v2_ns, path='/ml')
api_v2.add_namespace(analytics_v2_ns, path='/analytics')
api_v2.add_namespace(news_v2_ns, path='/news')

__all__ = ['api_v2_bp']
