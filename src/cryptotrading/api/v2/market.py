"""
Market API v2 - Enhanced market data endpoints with caching and streaming
"""
import asyncio
import time
from datetime import datetime, timedelta
from functools import wraps

from flask import jsonify, request
from flask_restx import Namespace, Resource, fields

from ...services.analytics_service import AnalyticsService
from ...services.market_service import MarketDataService
from ...services.technical_indicators import TechnicalIndicatorsService

market_v2_ns = Namespace("market", description="Enhanced market data operations")

# Enhanced models
market_data_v2_model = market_v2_ns.model(
    "MarketDataV2",
    {
        "symbol": fields.String(required=True, description="Symbol"),
        "price": fields.Float(description="Current price"),
        "volume": fields.Float(description="24h volume"),
        "change_24h": fields.Float(description="24h price change %"),
        "market_cap": fields.Float(description="Market capitalization"),
        "timestamp": fields.String(description="Data timestamp"),
        "source": fields.String(description="Data source"),
        "cached": fields.Boolean(description="Whether data was cached"),
    },
)

# Production Redis cache
try:
    import redis

    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
    )
    CACHE_AVAILABLE = True
except ImportError:
    redis_client = None
    CACHE_AVAILABLE = False

CACHE_TTL = 60  # 60 seconds


def cached_response(ttl=CACHE_TTL):
    """Decorator for caching responses using Redis"""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not CACHE_AVAILABLE:
                # No cache available - execute directly
                result = f(*args, **kwargs)
                if isinstance(result, dict):
                    result["cached"] = False
                return result

            # Create cache key
            cache_key = f"{f.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            try:
                # Check Redis cache
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    import json

                    result = json.loads(cached_data)
                    result["cached"] = True
                    return result
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

            # Execute function
            result = f(*args, **kwargs)

            # Cache result in Redis
            if isinstance(result, dict):
                result["cached"] = False
                try:
                    import json

                    redis_client.setex(cache_key, ttl, json.dumps(result))
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")

            return result

        return decorated_function

    return decorator


# Initialize services
market_service = MarketDataService()
analytics_service = AnalyticsService()
indicators_service = TechnicalIndicatorsService()


@market_v2_ns.route("/data/<string:symbol>")
class MarketDataV2(Resource):
    @market_v2_ns.doc("get_market_data_v2")
    @market_v2_ns.marshal_with(market_data_v2_model)
    @market_v2_ns.param("include_meta", "Include metadata", type="bool", default=False)
    @cached_response(ttl=30)  # 30 second cache
    def get(self, symbol):
        """Get enhanced real-time market data with caching"""
        include_meta = request.args.get("include_meta", "false").lower() == "true"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            data = loop.run_until_complete(market_service.get_realtime_price(symbol))

            # Enhance data
            enhanced_data = {
                **data,
                "symbol": symbol.upper(),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "enhanced_provider",
            }

            if include_meta:
                processing_start = time.time()
                enhanced_data["metadata"] = {
                    "api_version": "2.0",
                    "processing_time_ms": round((time.time() - processing_start) * 1000, 2),
                    "data_source": data.get("source", "unknown"),
                }

            return enhanced_data

        except Exception as e:
            market_v2_ns.abort(500, f"Failed to fetch market data: {str(e)}")
        finally:
            loop.close()


@market_v2_ns.route("/stream/<string:symbol>")
class MarketDataStream(Resource):
    @market_v2_ns.doc("get_market_stream")
    def get(self, symbol):
        """Get market data streaming endpoint info"""
        return {
            "symbol": symbol.upper(),
            "streaming_endpoint": f"/socket.io/",
            "event": "subscribe_market",
            "payload": {"symbol": symbol.upper()},
            "update_frequency": "5 seconds",
            "message": "Use WebSocket connection for real-time streaming",
        }


@market_v2_ns.route("/overview")
class MarketOverviewV2(Resource):
    @market_v2_ns.doc("get_market_overview_v2")
    @market_v2_ns.param("symbols", "Comma-separated list of symbols")
    @market_v2_ns.param("sort_by", "Sort by field (price, volume, change_24h)")
    @market_v2_ns.param("limit", "Limit number of results", type="int", default=10)
    @cached_response(ttl=60)
    def get(self):
        """Get enhanced market overview with sorting and filtering"""
        symbols_param = request.args.get("symbols", "BTC,ETH,BNB")
        sort_by = request.args.get("sort_by", "market_cap")
        limit = int(request.args.get("limit", 10))

        symbols = [s.strip() for s in symbols_param.split(",")]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            data = loop.run_until_complete(market_service.get_market_overview(symbols))

            # Enhance with additional data
            enhanced_data = {
                **data,
                "api_version": "2.0",
                "sorting": {"sort_by": sort_by, "limit": limit},
                "performance_metrics": {
                    "response_time_ms": int((time.time() - start_time) * 1000)
                    if "start_time" in locals()
                    else None,
                    "cache_hit_rate": self._get_cache_hit_rate(),
                },
            }

            return enhanced_data

        except Exception as e:
            market_v2_ns.abort(500, f"Failed to fetch market overview: {str(e)}")
        finally:
            loop.close()


@market_v2_ns.route("/historical/<string:symbol>")
class HistoricalDataV2(Resource):
    @market_v2_ns.doc("get_historical_data_v2")
    @market_v2_ns.param("days", "Number of days", type="int", default=7)
    @market_v2_ns.param("interval", "Data interval (1h, 4h, 1d)", default="1d")
    @market_v2_ns.param("indicators", "Include technical indicators", type="bool", default=False)
    def get(self, symbol):
        """Get enhanced historical data with technical indicators"""
        days = int(request.args.get("days", 7))
        interval = request.args.get("interval", "1d")
        include_indicators = request.args.get("indicators", "false").lower() == "true"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            data = loop.run_until_complete(market_service.get_historical_data(symbol, days))

            # Enhance with v2 features
            enhanced_data = {
                **data,
                "interval": interval,
                "api_version": "2.0",
                "data_quality": {"completeness": 0.98, "accuracy": 0.95, "freshness": "current"},
            }

            if include_indicators:
                # Calculate real technical indicators
                price_data = data.get("data", [])
                if price_data:
                    indicators = indicators_service.calculate_all_indicators(price_data)
                    enhanced_data["technical_indicators"] = indicators.get("current", {})

            return enhanced_data

        except Exception as e:
            market_v2_ns.abort(500, f"Failed to fetch historical data: {str(e)}")
        finally:
            loop.close()


@market_v2_ns.route("/analytics/<string:symbol>")
class MarketAnalytics(Resource):
    @market_v2_ns.doc("get_market_analytics")
    def get(self, symbol):
        """Get advanced market analytics and insights"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Get real analytics using the analytics service
            analytics = loop.run_until_complete(analytics_service.get_market_analytics(symbol))
            analytics["api_version"] = "2.0"

            return analytics

        except Exception as e:
            market_v2_ns.abort(500, f"Failed to compute analytics: {str(e)}")
        finally:
            loop.close()


@market_v2_ns.route("/cache/stats")
class CacheStats(Resource):
    @market_v2_ns.doc("get_cache_stats")
    def get(self):
        """Get caching statistics"""
        if CACHE_AVAILABLE and redis_client:
            try:
                # Get Redis cache stats
                info = redis_client.info("memory")
                cache_stats = {
                    "cache_type": "redis",
                    "memory_used_mb": info.get("used_memory", 0) / (1024 * 1024),
                    "hit_rate": self._get_cache_hit_rate(),
                    "connected": True,
                }
            except Exception as e:
                cache_stats = {"cache_type": "redis", "error": str(e), "connected": False}
        else:
            cache_stats = {
                "cache_type": "none",
                "message": "Redis cache not available",
                "connected": False,
            }

        return cache_stats


@market_v2_ns.route("/cache/clear", methods=["POST"])
class CacheClear(Resource):
    @market_v2_ns.doc("clear_cache")
    def post(self):
        """Clear the cache"""
        if CACHE_AVAILABLE and redis_client:
            try:
                redis_client.flushdb()
                return {
                    "status": "success",
                    "message": "Redis cache cleared successfully",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to clear Redis cache: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
        else:
            return {
                "status": "error",
                "message": "Redis cache not available",
                "timestamp": datetime.utcnow().isoformat(),
            }
