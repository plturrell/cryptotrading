"""
Minimal WSGI entry point for Vercel deployment
Serves SAP Fiori frontend with basic API endpoints
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from flask import Flask, jsonify, request
from flask_cors import CORS
from cryptotrading.core.bootstrap_unified import setup_flask_app
from cryptotrading.core.web.shared_routes import register_shared_routes

app = Flask(__name__, static_folder="webapp", template_folder="webapp")

# Setup unified monitoring and storage
bootstrap = setup_flask_app(app, "cryptotrading-vercel")
monitor = bootstrap.get_monitor()
storage = bootstrap.get_storage()
feature_flags = bootstrap.get_feature_flags()

# Log Vercel startup
monitor.log_info(
    "Vercel app starting", {"platform": "vercel", "features": feature_flags.get_feature_flags()}
)

# CORS configuration
CORS(app, origins=["*"])

# Register shared routes (SAP UI5 and health check)
# Note: For Vercel, we'll override the health route to add monitoring
register_shared_routes(app, include_health=False)  # We'll define a custom health route


@app.route("/health")
def health():
    """Health check endpoint with monitoring"""
    monitor.increment_counter("health_check")
    return {"status": "healthy", "platform": "cryptotrading.com", "version": "0.1.0"}


# Minimal API endpoints for the frontend
@app.route("/api/market/overview")
def market_overview():
    """Real-time market overview from live data sources"""
    with monitor.span("market_overview") as span:
        span.set_attribute("endpoint", "/api/market/overview")
        try:
            monitor.increment_counter("api.market.overview.requests")

            from cryptotrading.data.providers.real_only_provider import RealOnlyDataProvider
            import asyncio
            from datetime import datetime

            symbols = ["BTC", "ETH", "BNB"]
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
            return jsonify(
                {
                    "symbols": list(overview.keys()),
                    "data": overview,
                    "timestamp": datetime.now().isoformat(),
                    "source": "real_only_provider",
                }
            )

        except Exception as e:
            monitor.record_error(e, {"endpoint": "/api/market/overview"})
            return jsonify({"error": str(e)}), 500


@app.route("/api/wallet/balance")
async def wallet_balance():
    """Real wallet balance from blockchain"""
    try:
        from src.cryptotrading.infrastructure.blockchain.web3_service import get_web3_service

        # Get wallet address from query params or use demo address
        address = request.args.get("address", "0x742d35Cc6634C0532925a3b8D4C9db96c6b8b8b8")
        include_tokens = request.args.get("tokens", "USDT,USDC,WETH").split(",")

        web3_service = get_web3_service()
        wallet_summary = await web3_service.get_wallet_summary(address, include_tokens)

        return jsonify(wallet_summary)

    except Exception as e:
        return (
            jsonify(
                {"error": str(e), "status": "error", "message": "Failed to fetch wallet balance"}
            ),
            500,
        )


@app.route("/api/market/dex/trending")
async def dex_trending():
    """Real DEX trending pools from Uniswap/DEX APIs"""
    try:
        from src.cryptotrading.infrastructure.defi.dex_service import DEXService

        # Get limit from query params
        limit_per_dex = int(request.args.get("limit", 5))

        async with DEXService() as dex_service:
            trending_data = await dex_service.get_trending_dict(limit_per_dex)

        return jsonify(trending_data)

    except Exception as e:
        return (
            jsonify(
                {"error": str(e), "status": "error", "message": "Failed to fetch DEX trending data"}
            ),
            500,
        )


@app.route("/api/limits")
def api_limits():
    """Real API rate limits from configured providers"""
    try:
        from src.cryptotrading.infrastructure.monitoring.api_rate_monitor import get_rate_monitor

        rate_monitor = get_rate_monitor()
        usage_data = rate_monitor.get_usage_dict()

        return jsonify(usage_data)

    except Exception as e:
        return (
            jsonify(
                {"error": str(e), "status": "error", "message": "Failed to fetch API rate limits"}
            ),
            500,
        )


# Error handlers are registered by register_shared_routes()

# WSGI application for Vercel
application = app


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
