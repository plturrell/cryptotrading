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

# Import unified bootstrap
from cryptotrading.core.bootstrap_unified import setup_flask_app

app = Flask(__name__, 
           static_folder='webapp',
           template_folder='webapp')

# Setup unified monitoring and storage
bootstrap = setup_flask_app(app, "cryptotrading-vercel")
monitor = bootstrap.get_monitor()
storage = bootstrap.get_storage()
feature_flags = bootstrap.get_feature_flags()

# Log Vercel startup
monitor.log_info("Vercel app starting", {
    "platform": "vercel",
    "features": feature_flags.get_feature_flags()
})

# CORS configuration
CORS(app, origins=['*'])

# Register shared routes (SAP UI5 and health check)
# Note: For Vercel, we'll override the health route to add monitoring
from cryptotrading.core.web.shared_routes import register_shared_routes
register_shared_routes(app, include_health=False)  # We'll define a custom health route

@app.route('/health')
def health():
    """Health check endpoint with monitoring"""
    monitor.increment_counter("health_check")
    return {'status': 'healthy', 'platform': 'cryptotrading.com', 'version': '0.1.0'}

# Minimal API endpoints for the frontend
@app.route('/api/market/overview')
def market_overview():
    """Real-time market overview from live data sources"""
    with monitor.span("market_overview") as span:
        span.set_attribute("endpoint", "/api/market/overview")
        try:
            monitor.increment_counter("api.market.overview.requests")
            
            from cryptotrading.data.providers.real_only_provider import RealOnlyDataProvider
            import asyncio
            from datetime import datetime
            
            symbols = ['BTC', 'ETH', 'BNB']
            provider = RealOnlyDataProvider()
            
            # Run async in sync context
            async def get_overview():
                results = {}
                for symbol in symbols:
                    try:
                        price_data = await provider.get_real_time_price(symbol)
                        results[symbol] = price_data
                    except Exception as e:
                        monitor.record_error(e, {"symbol": symbol})
                        continue
                return results
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            overview = loop.run_until_complete(get_overview())
            
            if not overview:
                monitor.increment_counter("api.market.overview.no_data")
                return jsonify({"error": "No market data available"}), 503
            
            monitor.increment_counter("api.market.overview.success")
            return jsonify({
                "symbols": list(overview.keys()),
                "data": overview,
                "timestamp": datetime.now().isoformat(),
                "source": "real_only_provider"
            })
            
        except Exception as e:
            monitor.record_error(e, {"endpoint": "/api/market/overview"})
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

# Error handlers are registered by register_shared_routes()

# WSGI application for Vercel
application = app

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)