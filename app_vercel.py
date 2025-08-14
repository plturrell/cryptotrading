"""
Minimal WSGI entry point for Vercel deployment
Serves SAP Fiori frontend with basic API endpoints
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__, 
           static_folder='webapp',
           template_folder='webapp')

# CORS configuration
CORS(app, origins=['*'])

# SAP UI5 routes
@app.route('/')
def index():
    """Serve SAP UI5 application"""
    return send_from_directory('webapp', 'index.html')

@app.route('/manifest.json')
def manifest():
    """Serve SAP UI5 manifest"""
    return send_from_directory('.', 'manifest.json')

@app.route('/webapp/<path:filename>')
def webapp_files(filename):
    """Serve SAP UI5 webapp files"""
    return send_from_directory('webapp', filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'platform': 'rex.com', 'version': '0.1.0'}

# Minimal API endpoints for the frontend
@app.route('/api/market/overview')
def market_overview():
    """Mock market overview for frontend"""
    return jsonify({
        "symbols": {
            "bitcoin": {
                "prices": {
                    "average": 65432,
                    "median": 65400,
                    "min": 65000,
                    "max": 66000
                },
                "volume_24h_total": 28500000000,
                "sources": 3
            },
            "ethereum": {
                "prices": {
                    "average": 3456,
                    "median": 3450,
                    "min": 3400,
                    "max": 3500
                },
                "volume_24h_total": 15600000000,
                "sources": 3
            }
        }
    })

@app.route('/api/wallet/balance')
def wallet_balance():
    """Mock wallet balance"""
    return jsonify({
        "balance": {
            "ETH": 1.2345,
            "USD": 4265.43
        },
        "address": "0x88bE2a6408934e32a0Ad63c368Be5b257ca63cC1"
    })

@app.route('/api/market/dex/trending')
def dex_trending():
    """Mock trending DEX pools"""
    return jsonify({
        "data": [
            {"name": "WETH/USDC", "liquidity": 125000000},
            {"name": "WBTC/WETH", "liquidity": 89000000},
            {"name": "PEPE/WETH", "liquidity": 45000000}
        ]
    })

@app.route('/api/limits')
def api_limits():
    """Mock API limits"""
    return jsonify({
        "limits": {
            "geckoterminal": {"remaining": 25, "limit": 30},
            "coingecko": {"remaining": 8, "limit": 10}
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return {'error': 'Not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500

# WSGI application for Vercel
application = app

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)