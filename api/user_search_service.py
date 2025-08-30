"""
User Search Service - Handles user-initiated news searches with storage and history
Provides search functionality, saves searches, and manages search history
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from flask import Flask, Blueprint, request, jsonify
from dataclasses import asdict
import hashlib

# Import our services
import sys

sys.path.insert(0, "src")
from src.cryptotrading.infrastructure.data.news_service import PerplexityNewsService, NewsArticle
from src.cryptotrading.infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)

# Create Flask blueprint for search API
search_api = Blueprint("search_api", __name__, url_prefix="/api/search")


def run_async(coro):
    """Helper to run async functions in sync Flask routes"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


class UserSearchService:
    """Service for handling user news searches with storage and history"""

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

    async def perform_user_search(
        self,
        user_id: int,
        query: str,
        language: str = "en",
        limit: int = 10,
        save_search: bool = True,
    ) -> Dict[str, Any]:
        """Perform user-initiated news search and optionally save it"""
        try:
            # Generate search ID
            search_id = hashlib.md5(
                f"{user_id}_{query}_{datetime.now().isoformat()}".encode()
            ).hexdigest()

            # Perform the search using Perplexity
            if language == "ru":
                articles = await self.news_service.search_news(
                    f"{query} криптовалюта новости", limit
                )
            else:
                articles = await self.news_service.search_news(query, limit)

            # Store search in database if requested
            if save_search:
                search_data = {
                    "search_id": search_id,
                    "user_id": user_id,
                    "query": query,
                    "language": language,
                    "results_count": len(articles),
                    "search_timestamp": datetime.now(),
                    "search_type": "user_initiated",
                    "parameters": json.dumps({"limit": limit, "language": language}),
                }
                await self.db_adapter.store_data("user_searches", search_data)

            # Store articles with search reference
            stored_articles = []
            for article in articles:
                article_data = {
                    "title": article.title,
                    "content": article.content,
                    "summary": getattr(article, 'summary', ''),
                    "url": article.url,
                    "source": article.source,
                    "author": getattr(article, 'author', ''),
                    "published_at": article.published_at,
                    "language": article.language,
                    "category": article.category,
                    "symbols": json.dumps(article.symbols) if article.symbols else "[]",
                    "sentiment": article.sentiment,
                    "relevance_score": article.relevance_score,
                    "translated_title": getattr(article, 'translated_title', ''),
                    "translated_content": getattr(article, 'translated_content', ''),
                    "search_id": search_id,
                    "found_by_user_search": True,
                    "is_active": True,
                    "view_count": 0,
                }

                article_id = await self.db_adapter.store_data("news_articles", article_data)
                article_data["id"] = article_id
                stored_articles.append(article_data)

            # Format for UI
            ui_articles = []
            for article_data in stored_articles:
                ui_article = {
                    "id": article_data["id"],
                    "title": article_data.get("translated_title") or article_data.get("title"),
                    "summary": article_data.get("translated_content", "")[:200] + "..."
                    if article_data.get("translated_content")
                    else article_data.get("content", "")[:200] + "...",
                    "source": article_data.get("source", "Unknown"),
                    "publishedAt": article_data.get("published_at"),
                    "category": article_data.get("category", "search_result"),
                    "sentiment": article_data.get("sentiment", "neutral"),
                    "symbols": json.loads(article_data.get("symbols", "[]")),
                    "language": article_data.get("language", "en"),
                    "url": article_data.get("url"),
                    "relevanceScore": article_data.get("relevance_score", 0.5),
                    "searchQuery": query,
                    "searchId": search_id,
                    "isSearchResult": True,
                    "sentimentColor": self._get_sentiment_color(
                        article_data.get("sentiment", "neutral")
                    ),
                    "categoryIcon": "🔍",  # Search icon for search results
                    "timeAgo": self._format_time_ago(article_data.get("published_at")),
                    "readingTime": self._estimate_reading_time(article_data.get("content", "")),
                }
                ui_articles.append(ui_article)

            return {
                "success": True,
                "search_id": search_id,
                "query": query,
                "language": language,
                "articles": ui_articles,
                "count": len(ui_articles),
                "saved": save_search,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error performing user search: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_id": None,
                "articles": [],
                "count": 0,
            }

    async def get_user_search_history(self, user_id: int, limit: int = 20) -> Dict[str, Any]:
        """Get user's search history"""
        try:
            searches = await self.db_adapter.query_data(
                "user_searches",
                conditions={"user_id": user_id},
                limit=limit,
                order_by="search_timestamp DESC",
            )

            # Format search history
            search_history = []
            for search in searches:
                search_item = {
                    "search_id": search.get("search_id"),
                    "query": search.get("query"),
                    "language": search.get("language"),
                    "results_count": search.get("results_count"),
                    "search_timestamp": search.get("search_timestamp"),
                    "timeAgo": self._format_time_ago(search.get("search_timestamp")),
                    "can_rerun": True,
                }
                search_history.append(search_item)

            return {
                "success": True,
                "user_id": user_id,
                "searches": search_history,
                "count": len(search_history),
            }

        except Exception as e:
            logger.error(f"Error getting search history: {e}")
            return {"success": False, "error": str(e), "searches": [], "count": 0}

    async def rerun_saved_search(self, user_id: int, search_id: str) -> Dict[str, Any]:
        """Rerun a previously saved search"""
        try:
            # Get original search
            original_search = await self.db_adapter.query_data(
                "user_searches", conditions={"search_id": search_id, "user_id": user_id}, limit=1
            )

            if not original_search:
                return {"success": False, "error": "Search not found", "articles": [], "count": 0}

            search_data = original_search[0]
            parameters = json.loads(search_data.get("parameters", "{}"))

            # Perform new search with same parameters
            result = await self.perform_user_search(
                user_id=user_id,
                query=search_data["query"],
                language=search_data["language"],
                limit=parameters.get("limit", 10),
                save_search=True,  # Save as new search
            )

            # Add rerun information
            if result["success"]:
                result["rerun_from"] = search_id
                result["original_query"] = search_data["query"]

            return result

        except Exception as e:
            logger.error(f"Error rerunning search: {e}")
            return {"success": False, "error": str(e), "articles": [], "count": 0}

    async def save_search_as_alert(
        self, user_id: int, search_id: str, alert_name: str
    ) -> Dict[str, Any]:
        """Save a search as a recurring alert"""
        try:
            # Get original search
            original_search = await self.db_adapter.query_data(
                "user_searches", conditions={"search_id": search_id, "user_id": user_id}, limit=1
            )

            if not original_search:
                return {"success": False, "error": "Search not found"}

            search_data = original_search[0]

            # Create search alert
            alert_data = {
                "user_id": user_id,
                "alert_name": alert_name,
                "search_query": search_data["query"],
                "language": search_data["language"],
                "parameters": search_data["parameters"],
                "is_active": True,
                "frequency": "daily",  # Default frequency
                "last_run": None,
                "next_run": datetime.now() + timedelta(days=1),
                "created_at": datetime.now(),
            }

            alert_id = await self.db_adapter.store_data("search_alerts", alert_data)

            return {
                "success": True,
                "alert_id": alert_id,
                "alert_name": alert_name,
                "message": "Search saved as recurring alert",
            }

        except Exception as e:
            logger.error(f"Error saving search alert: {e}")
            return {"success": False, "error": str(e)}

    def _get_sentiment_color(self, sentiment: str) -> str:
        """Get color for sentiment display"""
        colors = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#9E9E9E"}
        return colors.get(sentiment.lower(), colors["neutral"])

    def _format_time_ago(self, timestamp) -> str:
        """Format time ago for UI display"""
        if not timestamp:
            return "Unknown"

        try:
            if isinstance(timestamp, str):
                time_obj = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                time_obj = timestamp

            now = datetime.now(time_obj.tzinfo) if time_obj.tzinfo else datetime.now()
            diff = now - time_obj

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
        minutes = max(1, words // 200)
        return f"{minutes} min"


# Flask Blueprint for User Search API
user_search_bp = Blueprint("user_search_api", __name__, url_prefix="/api/search")


def run_async(coro):
    """Helper to run async functions in sync Flask routes"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


@user_search_bp.route("/news", methods=["POST"])
def search_news():
    """User-initiated news search"""
    try:
        data = request.get_json()
        user_id = data.get("user_id", 1)  # Default user for demo
        query = data.get("query", "").strip()
        language = data.get("language", "en")
        limit = min(int(data.get("limit", 10)), 50)
        save_search = data.get("save_search", True)

        if not query:
            return (
                jsonify(
                    {"success": False, "error": "Query is required", "articles": [], "count": 0}
                ),
                400,
            )

        async def perform_search():
            async with UserSearchService() as service:
                return await service.perform_user_search(
                    user_id=user_id,
                    query=query,
                    language=language,
                    limit=limit,
                    save_search=save_search,
                )

        result = run_async(perform_search())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return jsonify({"success": False, "error": str(e), "articles": [], "count": 0}), 500


@user_search_bp.route("/history/<int:user_id>", methods=["GET"])
def get_search_history(user_id):
    """Get user's search history"""
    try:
        limit = min(int(request.args.get("limit", 20)), 100)

        async def get_history():
            async with UserSearchService() as service:
                return await service.get_user_search_history(user_id, limit)

        result = run_async(get_history())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error getting search history: {e}")
        return jsonify({"success": False, "error": str(e), "searches": [], "count": 0}), 500


@user_search_bp.route("/rerun", methods=["POST"])
def rerun_search():
    """Rerun a saved search"""
    try:
        data = request.get_json()
        user_id = data.get("user_id", 1)
        search_id = data.get("search_id", "").strip()

        if not search_id:
            return jsonify({"success": False, "error": "Search ID is required"}), 400

        async def rerun():
            async with UserSearchService() as service:
                return await service.rerun_saved_search(user_id, search_id)

        result = run_async(rerun())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error rerunning search: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@user_search_bp.route("/save-alert", methods=["POST"])
def save_search_alert():
    """Save search as recurring alert"""
    try:
        data = request.get_json()
        user_id = data.get("user_id", 1)
        search_id = data.get("search_id", "").strip()
        alert_name = data.get("alert_name", "").strip()

        if not search_id or not alert_name:
            return (
                jsonify({"success": False, "error": "Search ID and alert name are required"}),
                400,
            )

        async def save_alert():
            async with UserSearchService() as service:
                return await service.save_search_as_alert(user_id, search_id, alert_name)

        result = run_async(save_alert())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error saving search alert: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Flask routes for search API
@search_api.route("/news", methods=["POST"])
def search_news():
    """Search news with user tracking"""
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id", 1)
        query = data.get("query", "").strip()
        language = data.get("language", "en")
        limit = data.get("limit", 10)
        save_search = data.get("save_search", True)

        if not query:
            return jsonify({"success": False, "error": "Search query is required"}), 400

        async def perform_search():
            async with UserSearchService() as service:
                return await service.perform_user_search(
                    user_id=user_id,
                    query=query,
                    language=language,
                    limit=limit,
                    save_search=save_search,
                )

        result = run_async(perform_search())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error searching news: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@search_api.route("/history/<int:user_id>", methods=["GET"])
def get_search_history(user_id):
    """Get user search history"""
    try:
        limit = request.args.get("limit", 20, type=int)

        async def get_history():
            async with UserSearchService() as service:
                return await service.get_user_search_history(user_id, limit)

        result = run_async(get_history())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error getting search history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@search_api.route("/rerun/<int:search_id>", methods=["POST"])
def rerun_search(search_id):
    """Rerun a saved search"""
    try:
        async def rerun():
            async with UserSearchService() as service:
                return await service.rerun_saved_search(search_id)

        result = run_async(rerun())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error rerunning search: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@search_api.route("/alerts", methods=["POST"])
def save_search_alert():
    """Save a search alert for notifications"""
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        query = data.get("query", "").strip()
        frequency = data.get("frequency", "daily")
        active = data.get("active", True)

        if not user_id or not query:
            return jsonify({"success": False, "error": "User ID and query are required"}), 400

        async def save_alert():
            async with UserSearchService() as service:
                return await service.save_search_alert(
                    user_id=user_id,
                    query=query,
                    frequency=frequency,
                    active=active,
                )

        result = run_async(save_alert())
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error saving search alert: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def register_user_search_service(app):
    """Register user search service with Flask app"""
    app.register_blueprint(search_api)
    logger.info("User Search Service registered successfully")
    return search_api
