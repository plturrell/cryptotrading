"""
News Service Adapter - Integrates Perplexity News Service with CDS and Database
Provides on-demand news retrieval, storage, and beautiful UI data formatting
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify
from dataclasses import asdict

# Import our services
import sys

sys.path.insert(0, "src")
from src.cryptotrading.infrastructure.data.news_service import PerplexityNewsService, NewsArticle
from src.cryptotrading.infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)

# Create Flask blueprint for news service adapter
adapter_bp = Blueprint('news_adapter', __name__, url_prefix='/api/adapter')

def run_async(coro):
    """Helper to run async functions in sync Flask routes"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


class NewsServiceAdapter:
    """Adapter that connects Perplexity News Service to CDS and Database"""

    def __init__(self):
        self.news_service = PerplexityNewsService()
        self.db_adapter = UnifiedDatabase()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.news_service.__aenter__()
        await self.db_adapter.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.news_service.__aexit__(exc_type, exc_val, exc_tb)
        await self.db_adapter.__aexit__(exc_type, exc_val, exc_tb)

    async def fetch_and_store_latest_news(
        self, limit: int = 10, language: str = "en"
    ) -> Dict[str, Any]:
        """Fetch latest news and store in database"""
        try:
            # Fetch from Perplexity API
            if language == "ru":
                articles = await self.news_service.get_latest_news_russian(limit)
            else:
                articles = await self.news_service.get_latest_news(limit)

            # Store in database
            stored_count = 0
            for article in articles:
                article_data = {
                    "title": article.title,
                    "content": article.content,
                    "summary": article.summary,
                    "url": article.url,
                    "source": article.source,
                    "author": article.author,
                    "published_at": article.published_at,
                    "language": article.language,
                    "category": article.category,
                    "symbols": json.dumps(article.symbols) if article.symbols else "[]",
                    "sentiment": article.sentiment,
                    "relevance_score": article.relevance_score,
                    "translated_title": article.translated_title,
                    "translated_content": article.translated_content,
                    "translation_status": "COMPLETED"
                    if article.translated_title
                    else "NOT_REQUIRED",
                    "tags": json.dumps(article.tags) if hasattr(article, "tags") else "[]",
                    "is_active": True,
                    "view_count": 0,
                }

                # Store in database
                await self.db_adapter.store_data("news_articles", article_data)
                stored_count += 1

            return {
                "success": True,
                "fetched_count": len(articles),
                "stored_count": stored_count,
                "language": language,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error fetching and storing news: {e}")
            return {"success": False, "error": str(e), "fetched_count": 0, "stored_count": 0}

    async def get_news_for_ui(
        self, limit: int = 10, category: str = None, language: str = "en"
    ) -> Dict[str, Any]:
        """Get news formatted for beautiful UI display"""
        try:
            # Try to get from database first
            query_conditions = {"is_active": True, "language": language}
            if category:
                query_conditions["category"] = category

            stored_articles = await self.db_adapter.query_data(
                "news_articles",
                conditions=query_conditions,
                limit=limit,
                order_by="published_at DESC",
            )

            # If not enough in database, fetch fresh from API
            if len(stored_articles) < limit:
                await self.fetch_and_store_latest_news(limit, language)
                stored_articles = await self.db_adapter.query_data(
                    "news_articles",
                    conditions=query_conditions,
                    limit=limit,
                    order_by="published_at DESC",
                )

            # Format for beautiful UI
            ui_articles = []
            for article in stored_articles:
                ui_article = {
                    "id": article.get("id"),
                    "title": article.get("translated_title") or article.get("title"),
                    "summary": article.get("translated_content", "")[:200] + "..."
                    if article.get("translated_content")
                    else article.get("content", "")[:200] + "...",
                    "source": article.get("source", "Unknown"),
                    "publishedAt": article.get("published_at"),
                    "category": article.get("category", "general"),
                    "sentiment": article.get("sentiment", "neutral"),
                    "symbols": json.loads(article.get("symbols", "[]")),
                    "language": article.get("language", "en"),
                    "url": article.get("url"),
                    "viewCount": article.get("view_count", 0),
                    "relevanceScore": article.get("relevance_score", 0.5),
                    # UI-specific formatting
                    "sentimentColor": self._get_sentiment_color(
                        article.get("sentiment", "neutral")
                    ),
                    "categoryIcon": self._get_category_icon(article.get("category", "general")),
                    "timeAgo": self._format_time_ago(article.get("published_at")),
                    "isTranslated": bool(article.get("translated_title")),
                    "readingTime": self._estimate_reading_time(article.get("content", "")),
                }
                ui_articles.append(ui_article)

            return {
                "success": True,
                "articles": ui_articles,
                "count": len(ui_articles),
                "language": language,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "totalInDatabase": len(stored_articles),
                    "freshlyFetched": len(stored_articles) == limit,
                    "supportedLanguages": ["en", "ru"],
                    "availableCategories": await self._get_available_categories(),
                },
            }

        except Exception as e:
            logger.error(f"Error getting news for UI: {e}")
            return {"success": False, "error": str(e), "articles": [], "count": 0}

    async def get_russian_crypto_news_for_ui(self, limit: int = 10) -> Dict[str, Any]:
        """Get Russian crypto news formatted for UI"""
        try:
            # Fetch fresh Russian crypto news
            articles = await self.news_service.get_russian_crypto_news(limit)

            # Store in database
            for article in articles:
                article_data = {
                    "title": article.title,
                    "content": article.content,
                    "summary": article.summary,
                    "url": article.url,
                    "source": article.source,
                    "published_at": article.published_at,
                    "language": "ru",
                    "category": article.category or "russian_crypto",
                    "symbols": json.dumps(article.symbols) if article.symbols else "[]",
                    "sentiment": article.sentiment,
                    "relevance_score": article.relevance_score,
                    "is_active": True,
                    "view_count": 0,
                }
                await self.db_adapter.store_data("news_articles", article_data)

            # Format for UI
            ui_articles = []
            for article in articles:
                ui_article = {
                    "id": f"ru_{hash(article.title)}",
                    "title": article.title,
                    "summary": article.content[:200] + "..." if article.content else "",
                    "source": article.source,
                    "publishedAt": article.published_at,
                    "category": "russian_crypto",
                    "sentiment": article.sentiment,
                    "symbols": article.symbols or [],
                    "language": "ru",
                    "url": article.url,
                    "sentimentColor": self._get_sentiment_color(article.sentiment),
                    "categoryIcon": "🇷🇺",
                    "timeAgo": self._format_time_ago(article.published_at),
                    "isRussianSpecific": True,
                    "readingTime": self._estimate_reading_time(article.content),
                }
                ui_articles.append(ui_article)

            return {
                "success": True,
                "articles": ui_articles,
                "count": len(ui_articles),
                "language": "ru",
                "category": "russian_crypto",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting Russian crypto news: {e}")
            return {"success": False, "error": str(e), "articles": [], "count": 0}

    async def translate_and_store_article(self, article_id: str) -> Dict[str, Any]:
        """Translate specific article to Russian and store"""
        try:
            # Get article from database
            article = await self.db_adapter.get_by_id("news_articles", article_id)
            if not article:
                return {"success": False, "error": "Article not found"}

            # Translate using news service
            async with self.news_service:
                translated_title = await self.news_service.ai_client.translate_to_russian(
                    article["title"], self.news_service.session
                )
                translated_content = await self.news_service.ai_client.translate_to_russian(
                    article["content"], self.news_service.session
                )

            # Update in database
            update_data = {
                "translated_title": translated_title,
                "translated_content": translated_content,
                "translation_status": "COMPLETED",
            }
            await self.db_adapter.update_data("news_articles", article_id, update_data)

            return {
                "success": True,
                "article_id": article_id,
                "translated_title": translated_title,
                "translation_status": "COMPLETED",
            }

        except Exception as e:
            logger.error(f"Error translating article {article_id}: {e}")
            return {"success": False, "error": str(e), "article_id": article_id}

    def _get_sentiment_color(self, sentiment: str) -> str:
        """Get color for sentiment display"""
        colors = {
            "positive": "#4CAF50",  # Green
            "negative": "#F44336",  # Red
            "neutral": "#9E9E9E",  # Gray
        }
        return colors.get(sentiment.lower(), colors["neutral"])

    def _get_category_icon(self, category: str) -> str:
        """Get icon for category display"""
        icons = {
            "market_analysis": "📊",
            "regulatory": "⚖️",
            "technology": "🔧",
            "institutional": "🏛️",
            "defi": "🔗",
            "nft": "🎨",
            "trading": "💹",
            "russian_crypto": "🇷🇺",
            "general": "📰",
        }
        return icons.get(category, icons["general"])

    def _format_time_ago(self, published_at) -> str:
        """Format time ago for UI display"""
        if not published_at:
            return "Unknown"

        try:
            if isinstance(published_at, str):
                pub_time = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            else:
                pub_time = published_at

            now = datetime.now(pub_time.tzinfo) if pub_time.tzinfo else datetime.now()
            diff = now - pub_time

            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60}m ago"
            else:
                return "Just now"
        except:
            return "Unknown"

    def _estimate_reading_time(self, content: str) -> str:
        """Estimate reading time for article"""
        if not content:
            return "1 min"

        words = len(content.split())
        minutes = max(1, words // 200)  # Average 200 words per minute
        return f"{minutes} min"

    async def _get_available_categories(self) -> List[str]:
        """Get available news categories"""
        return [
            "market_analysis",
            "regulatory",
            "technology",
            "institutional",
            "defi",
            "nft",
            "trading",
            "russian_crypto",
        ]


# Flask Blueprint for News Service API
news_service_bp = Blueprint("news_service_api", __name__, url_prefix="/api/news")


def run_async(coro):
    """Helper to run async functions in sync Flask routes"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


@news_service_bp.route("/ui/latest", methods=["GET"])
def get_latest_news_ui():
    """Get latest news formatted for beautiful UI display"""
    try:
        limit = min(int(request.args.get("limit", 10)), 50)
        language = request.args.get("language", "en")
        category = request.args.get("category")

        async def fetch_data():
            async with NewsServiceAdapter() as adapter:
                return await adapter.get_news_for_ui(limit, category, language)

        result = run_async(fetch_data())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error in UI latest news endpoint: {e}")
        return jsonify({"success": False, "error": str(e), "articles": [], "count": 0}), 500


@news_service_bp.route("/ui/russian", methods=["GET"])
def get_russian_news_ui():
    """Get Russian crypto news formatted for UI"""
    try:
        limit = min(int(request.args.get("limit", 10)), 50)

        async def fetch_data():
            async with NewsServiceAdapter() as adapter:
                return await adapter.get_russian_crypto_news_for_ui(limit)

        result = run_async(fetch_data())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error in Russian news UI endpoint: {e}")
        return jsonify({"success": False, "error": str(e), "articles": [], "count": 0}), 500


@news_service_bp.route("/translate/<article_id>", methods=["POST"])
def translate_article(article_id):
    """Translate specific article to Russian"""
    try:

        async def translate_data():
            async with NewsServiceAdapter() as adapter:
                return await adapter.translate_and_store_article(article_id)

        result = run_async(translate_data())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error translating article {article_id}: {e}")
        return jsonify({"success": False, "error": str(e), "article_id": article_id}), 500


@news_service_bp.route("/fetch/fresh", methods=["POST"])
def fetch_fresh_news():
    """Fetch fresh news from API and store in database"""
    try:
        limit = min(int(request.json.get("limit", 10)), 50)
        language = request.json.get("language", "en")

        async def fetch_data():
            async with NewsServiceAdapter() as adapter:
                return await adapter.fetch_and_store_latest_news(limit, language)

        result = run_async(fetch_data())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error fetching fresh news: {e}")
        return (
            jsonify({"success": False, "error": str(e), "fetched_count": 0, "stored_count": 0}),
            500,
        )


@news_service_bp.route("/health", methods=["GET"])
def news_service_health():
    """Health check for news service adapter"""
    try:

        async def check_health():
            async with NewsServiceAdapter() as adapter:
                # Test database connection
                db_healthy = await adapter.db_adapter.health_check()

                # Test news service
                news_healthy = True
                try:
                    async with adapter.news_service:
                        test_articles = await adapter.news_service.get_latest_news(1)
                        news_healthy = len(test_articles) >= 0  # Even 0 is ok for health check
                except:
                    news_healthy = False

                return {
                    "success": True,
                    "database_healthy": db_healthy,
                    "news_service_healthy": news_healthy,
                    "overall_healthy": db_healthy and news_healthy,
                    "timestamp": datetime.now().isoformat(),
                }

        result = run_async(check_health())
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "database_healthy": False,
                    "news_service_healthy": False,
                    "overall_healthy": False,
                }
            ),
            500,
        )


# Flask routes for adapter API
@adapter_bp.route('/fetch-store', methods=['POST'])
def fetch_and_store():
    """Fetch and store news via adapter"""
    try:
        data = request.get_json() or {}
        limit = data.get('limit', 10)
        language = data.get('language', 'en')
        
        async def fetch_store():
            async with NewsServiceAdapter() as adapter:
                return await adapter.fetch_and_store_latest_news(limit, language)
        
        result = run_async(fetch_store())
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        logger.error(f"Error in fetch and store: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@adapter_bp.route('/health', methods=['GET'])
def adapter_health():
    """Health check for adapter"""
    try:
        async def check_health():
            async with NewsServiceAdapter() as adapter:
                return await adapter.health_check()
        
        result = run_async(check_health())
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Adapter health check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'database_healthy': False,
            'news_service_healthy': False,
            'overall_healthy': False
        }), 500

def register_news_adapter(app):
    """Register news service adapter with Flask app"""
    app.register_blueprint(adapter_bp)
    logger.info("News Service Adapter registered successfully")
    return adapter_bp
