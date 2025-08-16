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
    """Real-time market overview from live data sources"""
    try:
        # TODO: Integrate with real market data API (CoinGecko, CryptoCompare, etc.)
        return jsonify({"error": "Real market data integration required", "status": "not_implemented"}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/wallet/balance')
def wallet_balance():
    """Real wallet balance from blockchain"""
    try:
        # TODO: Integrate with Web3 provider for real wallet balance
        return jsonify({"error": "Blockchain wallet integration required", "status": "not_implemented"}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/market/dex/trending')
def dex_trending():
    """Real DEX trending pools from Uniswap/DEX APIs"""
    try:
        # TODO: Integrate with DEX APIs (Uniswap, SushiSwap, etc.)
        return jsonify({"error": "DEX API integration required", "status": "not_implemented"}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/limits')
def api_limits():
    """Real API rate limits from configured providers"""
    try:
        # TODO: Query actual API limits from configured providers
        return jsonify({"error": "API limit monitoring not implemented", "status": "not_implemented"}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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