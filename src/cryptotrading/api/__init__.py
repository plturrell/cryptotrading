"""
API Module - Modularized Flask Blueprints
Replaces the monolithic app.py structure
"""

from .v1 import api_v1_bp
from .v2 import api_v2_bp
from .websocket import websocket_bp

__all__ = ["api_v1_bp", "api_v2_bp", "websocket_bp"]
