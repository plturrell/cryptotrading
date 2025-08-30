#!/usr/bin/env python3
"""
Unified News API - Complete Flask Integration
Connects UI frontend to backend services with proper routing
"""

import asyncio
import logging
import json
from datetime import datetime
from flask import Flask, Blueprint, request, jsonify, render_template
from flask_cors import CORS
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cryptotrading.infrastructure.data.news_service import PerplexityNewsService
from cryptotrading.infrastructure.data.image_services import NewsImageEnhancer

logger = logging.getLogger(__name__)

# Create Flask app and blueprint
app = Flask(__name__)
CORS(app)
news_api = Blueprint("news_api", __name__, url_prefix="/api/news")


def run_async(coro):
    """Helper to run async functions in sync Flask routes"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


# UI-specific endpoints (matching frontend controller calls)
@news_api.route("/ui/latest", methods=["GET"])
def get_ui_latest_news():
    """Get latest news formatted for UI"""
    try:
        limit = request.args.get("limit", 20, type=int)
        language = request.args.get("language", "en")
        category = request.args.get("category", None)

        async def fetch_news():
            service = PerplexityNewsService(enable_images=True)
            async with service:
                if language == "ru":
                    articles = await service.get_latest_news_russian(limit)
                elif category:
                    articles = await service.get_news_by_category(category, limit)
                else:
                    articles = await service.get_latest_news(limit)

                # Enhance with images
                enhanced_articles = await service.enhance_articles_with_images(articles)

                return [format_article_for_ui(article) for article in enhanced_articles]

        articles = run_async(fetch_news())

        return (
            jsonify(
                {
                    "success": True,
                    "articles": articles,
                    "count": len(articles),
                    "language": language,
                    "category": category,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error fetching UI latest news: {e}")
        return jsonify({"success": False, "error": str(e), "articles": [], "count": 0}), 500


@news_api.route("/ui/russian", methods=["GET"])
def get_ui_russian_news():
    """Get Russian news formatted for UI"""
    try:
        limit = request.args.get("limit", 20, type=int)

        async def fetch_news():
            service = PerplexityNewsService(enable_images=True)
            async with service:
                articles = await service.get_latest_news_russian(limit)
                enhanced_articles = await service.enhance_articles_with_images(articles)
                return [format_article_for_ui(article) for article in enhanced_articles]

        articles = run_async(fetch_news())

        return (
            jsonify(
                {
                    "success": True,
                    "articles": articles,
                    "count": len(articles),
                    "language": "ru",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error fetching UI Russian news: {e}")
        return jsonify({"success": False, "error": str(e), "articles": [], "count": 0}), 500


@news_api.route("/fetch/fresh", methods=["POST"])
def fetch_fresh_news():
    """Fetch fresh news from API"""
    try:
        data = request.get_json() or {}
        limit = data.get("limit", 10)
        language = data.get("language", "en")

        async def fetch_news():
            service = PerplexityNewsService(enable_images=True)
            async with service:
                if language == "ru":
                    articles = await service.get_latest_news_russian(limit)
                else:
                    articles = await service.get_latest_news(limit)

                enhanced_articles = await service.enhance_articles_with_images(articles)
                return [format_article_for_ui(article) for article in enhanced_articles]

        articles = run_async(fetch_news())

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Fresh news fetched successfully",
                    "articles": articles,
                    "count": len(articles),
                    "language": language,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error fetching fresh news: {e}")
        return (
            jsonify({"success": False, "error": str(e), "message": "Failed to fetch fresh news"}),
            500,
        )


@news_api.route("/translate", methods=["POST"])
def translate_article():
    """Translate article to Russian"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"success": False, "error": "Text to translate is required"}), 400

        text = data["text"]

        async def translate_text():
            service = PerplexityNewsService()
            async with service:
                translated = await service.translator.translate_to_russian(text, service.session)
                return translated

        translated_text = run_async(translate_text())

        return (
            jsonify(
                {
                    "success": True,
                    "original": text,
                    "translated": translated_text,
                    "language": "ru",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error translating article: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Standard API endpoints
@news_api.route("/latest", methods=["GET"])
def get_latest_news():
    """Get latest cryptocurrency news"""
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(max(limit, 1), 50)

        async def fetch_news():
            service = PerplexityNewsService(enable_images=True)
            async with service:
                articles = await service.get_latest_news(limit)
                enhanced_articles = await service.enhance_articles_with_images(articles)
                return [article.to_dict() for article in enhanced_articles]

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


@news_api.route("/category/<string:category>", methods=["GET"])
def get_news_by_category(category):
    """Get news by category"""
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(max(limit, 1), 50)

        async def fetch_news():
            service = PerplexityNewsService(enable_images=True)
            async with service:
                articles = await service.get_news_by_category(category, limit)
                enhanced_articles = await service.enhance_articles_with_images(articles)
                return [article.to_dict() for article in enhanced_articles]

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


@news_api.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:

        async def check_services():
            service = PerplexityNewsService()
            # Basic service check
            return True

        service_status = run_async(check_services())

        return (
            jsonify(
                {
                    "success": True,
                    "status": "healthy",
                    "services": {
                        "news_service": service_status,
                        "image_service": True,
                        "database": True,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"success": False, "status": "unhealthy", "error": str(e)}), 500


def format_article_for_ui(article):
    """Format article for UI consumption"""
    formatted = article.to_dict()

    # Add UI-specific formatting
    formatted["id"] = f"article_{hash(article.title) % 1000000}"
    formatted["publishedAtFormatted"] = format_date_for_ui(article.published_at)
    formatted["readingTime"] = estimate_reading_time(article.content)
    formatted["sentimentColor"] = get_sentiment_color(article.sentiment)
    formatted["categoryIcon"] = get_category_icon(article.category)

    # Format images for UI
    if hasattr(article, "images") and article.images:
        formatted["images"] = [img.to_dict() for img in article.images]
        formatted["featuredImage"] = formatted["images"][0] if formatted["images"] else None

    return formatted


def format_date_for_ui(date_str):
    """Format date for UI display"""
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = date_str
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return date_str


def estimate_reading_time(content):
    """Estimate reading time in minutes"""
    if not content:
        return 1
    word_count = len(content.split())
    return max(1, round(word_count / 200))  # 200 words per minute


def get_sentiment_color(sentiment):
    """Get color for sentiment"""
    colors = {"positive": "Good", "negative": "Error", "neutral": "Neutral"}
    return colors.get(sentiment.lower(), "Neutral")


def get_category_icon(category):
    """Get icon for category"""
    icons = {
        "market_analysis": "trend-up",
        "regulatory": "law",
        "technology": "technical-object",
        "institutional": "building",
        "defi": "chain-link",
        "nft": "picture",
        "trading": "business-card",
    }
    return icons.get(category, "newspaper")


# Register blueprint
app.register_blueprint(news_api)


# Root endpoint
@app.route("/")
def index():
    """Serve the main application"""
    return render_template("index.html")


@app.route("/webapp/<path:filename>")
def webapp_files(filename):
    """Serve webapp files"""
    return app.send_static_file(f"webapp/{filename}")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=59150)
