"""
WSGI entry point for рекс.com on GoDaddy hosting
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from flask_restx import Api
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, 
           static_folder='webapp',
           template_folder='webapp')

# CORS configuration for рекс.com
CORS(app, origins=[
    'https://рекс.com',
    'https://www.рекс.com',
    'https://xn--e1afmkfd.com',  # Punycode for рекс.com
    'https://www.xn--e1afmkfd.com'
])

# API configuration
api = Api(
    app,
    version='1.0',
    title='рекс.com API',
    description='Professional cryptocurrency trading platform API',
    doc='/api/'
)

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
    return {'status': 'healthy', 'platform': 'рекс.com', 'version': '0.1.0'}

# API endpoints
@api.route('/api/trading/status')
class TradingStatus(api.Resource):
    def get(self):
        """Get trading system status"""
        return {'status': 'active', 'platform': 'рекс.com'}

@api.route('/api/market/data')
class MarketData(api.Resource):
    def get(self):
        """Get market data"""
        return {'data': 'market_feed', 'timestamp': 'current'}

@api.route('/api/ai/analyze')
class AIAnalysis(api.Resource):
    def post(self):
        """AI market analysis using DeepSeek R1"""
        from src.рекс.ml.deepseek import DeepSeekR1
        
        data = api.payload
        ai = DeepSeekR1()
        analysis = ai.analyze_market(data)
        
        return {
            'analysis': analysis,
            'model': 'deepseek-r1',
            'symbol': data.get('symbol', 'BTC')
        }

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return {'error': 'Not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500

if __name__ == '__main__':
    # Development server
    app.run(debug=False, host='0.0.0.0', port=5000)

# WSGI application for GoDaddy
application = app