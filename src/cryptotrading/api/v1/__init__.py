"""
API v1 - Legacy endpoints for backward compatibility
"""
from flask import Blueprint
from flask_restx import Api

from .market import market_ns
from .ai import ai_ns
from .ml import ml_ns
from .trading import trading_ns

# Create v1 blueprint
api_v1_bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')

# Create API instance for v1
api_v1 = Api(
    api_v1_bp,
    version='1.0',
    title='CryptoTrading API v1',
    description='Legacy API endpoints - maintained for backward compatibility',
    doc='/docs'
)

# Register namespaces
api_v1.add_namespace(market_ns, path='/market')
api_v1.add_namespace(ai_ns, path='/ai')
api_v1.add_namespace(ml_ns, path='/ml')
api_v1.add_namespace(trading_ns, path='/trading')

__all__ = ['api_v1_bp']
