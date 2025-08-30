"""
News API endpoints for Perplexity integration
Provides REST endpoints for cryptocurrency news data
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from flask import Blueprint, request, jsonify

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cryptotrading.infrastructure.data.news_service import PerplexityNewsService
from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)

# Create blueprint for news API
news_bp = Blueprint("news_api", __name__, url_prefix="/api/news")


def run_async(coro):
    """Helper to run async functions in sync Flask routes"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


@news_bp.route("/latest", methods=["GET"])
def get_latest_news():
    """Get latest cryptocurrency news"""
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(max(limit, 1), 50)  # Clamp between 1 and 50

        async def fetch_news():
            async with PerplexityNewsService() as service:
                articles = await service.get_latest_news(limit)
                return [article.to_dict() for article in articles]

        articles = run_async(fetch_news())

        return (
            jsonify(
                {
                    "success": True,
                    "data": articles,
                    "count": len(articles),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error fetching latest news: {e}")
        return jsonify({"success": False, "error": str(e), "data": [], "count": 0}), 500


@news_bp.route("/category/<string:category>", methods=["GET"])
def get_news_by_category(category):
    """Get news by category"""
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(max(limit, 1), 50)

        async def fetch_news():
            async with PerplexityNewsService() as service:
                articles = await service.get_news_by_category(category, limit)
                return [article.to_dict() for article in articles]

        articles = run_async(fetch_news())

        return (
            jsonify(
                {
                    "success": True,
                    "category": category,
                    "data": articles,
                    "count": len(articles),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error fetching news for category {category}: {e}")
        return (
            jsonify(
                {"success": False, "error": str(e), "category": category, "data": [], "count": 0}
            ),
            500,
        )


@news_bp.route("/symbol/<string:symbol>", methods=["GET"])
def get_news_by_symbol(symbol):
    """Get news for specific cryptocurrency symbol"""
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(max(limit, 1), 50)

        async def fetch_news():
            async with PerplexityNewsService() as service:
                articles = await service.get_news_by_symbol(symbol, limit)
                return [article.to_dict() for article in articles]

        articles = run_async(fetch_news())

        return (
            jsonify(
                {
                    "success": True,
                    "symbol": symbol.upper(),
                    "data": articles,
                    "count": len(articles),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error fetching news for symbol {symbol}: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "symbol": symbol.upper(),
                    "data": [],
                    "count": 0,
                }
            ),
            500,
        )


@news_bp.route("/sentiment", methods=["GET"])
def get_market_sentiment_news():
    """Get market sentiment and analysis news"""
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(max(limit, 1), 50)

        async def fetch_news():
            async with PerplexityNewsService() as service:
                articles = await service.get_market_sentiment_news(limit)
                return [article.to_dict() for article in articles]

        articles = run_async(fetch_news())

        # Calculate sentiment distribution
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for article in articles:
            sentiment_counts[article.get("sentiment", "neutral")] += 1

        return (
            jsonify(
                {
                    "success": True,
                    "data": articles,
                    "count": len(articles),
                    "sentiment_distribution": sentiment_counts,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error fetching sentiment news: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "data": [],
                    "count": 0,
                    "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                }
            ),
            500,
        )


@news_bp.route("/search", methods=["GET"])
def search_news():
    """Search news with custom query"""
    try:
        query = request.args.get("q", "").strip()
        if not query:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": 'Query parameter "q" is required',
                        "data": [],
                        "count": 0,
                    }
                ),
                400,
            )

        limit = request.args.get("limit", 10, type=int)
        limit = min(max(limit, 1), 50)

        async def fetch_news():
            async with PerplexityNewsService() as service:
                articles = await service.search_news(query, limit)
                return [article.to_dict() for article in articles]

        articles = run_async(fetch_news())

        return (
            jsonify(
                {
                    "success": True,
                    "query": query,
                    "data": articles,
                    "count": len(articles),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error searching news with query '{query}': {e}")
        return (
            jsonify({"success": False, "error": str(e), "query": query, "data": [], "count": 0}),
            500,
        )


@news_bp.route("/categories", methods=["GET"])
def get_categories():
    """Get available news categories"""
    try:

        async def fetch_categories():
            async with PerplexityNewsService() as service:
                return service.get_available_categories()

        categories = run_async(fetch_categories())

        return jsonify({"success": True, "categories": categories, "count": len(categories)}), 200

    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        return jsonify({"success": False, "error": str(e), "categories": {}, "count": 0}), 500


@news_bp.route("/symbols", methods=["GET"])
def get_tracked_symbols():
    """Get list of tracked cryptocurrency symbols"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def get_symbols():
            async with PerplexityNewsService() as service:
                return service.get_tracked_symbols()

        symbols = loop.run_until_complete(get_symbols())
        loop.close()

        return {"success": True, "symbols": symbols, "count": len(symbols)}

    except Exception as e:
        logger.error("Error getting tracked symbols: %s", str(e))
        return {"success": False, "error": str(e), "symbols": []}


@news_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for news service"""
    try:

        async def check_service():
            async with PerplexityNewsService() as service:
                # Try to fetch one article to test the service
                articles = await service.get_latest_news(1)
                return len(articles) > 0

        is_healthy = run_async(check_service())

        return (
            jsonify(
                {
                    "success": True,
                    "healthy": is_healthy,
                    "service": "Perplexity News API",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "healthy": False,
                    "service": "Perplexity News API",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


def register_news_api(app):
    """Register news API blueprint with Flask app"""
    app.register_blueprint(news_bp)
    logger.info("News API registered successfully")
    return news_bp
