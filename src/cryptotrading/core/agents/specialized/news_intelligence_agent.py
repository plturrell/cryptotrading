"""
News Intelligence Agent - A2A Strands Agent with MCP Tools
Handles news collection, analysis, sentiment, and translation through MCP tools only.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...ai.grok4_client import Grok4Client
from ...protocols.a2a.a2a_protocol import A2AMessage, MessageType
from ...protocols.cds.cds_client import A2AAgentCDSMixin
from ..base import AgentStatus
from ..strands import StrandsAgent

# Enhanced CDS integration imports
try:
    from ...infrastructure.monitoring.cds_integration_monitor import get_cds_monitor, CDSOperationType
    from ...infrastructure.transactions.cds_transactional_client import CDSTransactionalMixin
    from ...infrastructure.transactions.agent_transaction_manager import transactional, TransactionIsolation
    CDS_ENHANCED_FEATURES = True
except ImportError:
    # Fallback classes for compatibility
    class CDSTransactionalMixin:
        pass
    def transactional(transaction_type=None, isolation_level=None):
        def decorator(func):
            return func
        return decorator
    class TransactionIsolation:
        READ_COMMITTED = "READ_COMMITTED"
    get_cds_monitor = None
    CDSOperationType = None
    CDS_ENHANCED_FEATURES = False

logger = logging.getLogger(__name__)


class NewsSentiment(Enum):
    """News sentiment categories."""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class NewsArticle:
    """News article with analysis metadata."""

    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    sentiment: NewsSentiment
    relevance_score: float
    symbols: List[str] = field(default_factory=list)
    category: str = "general"
    language: str = "en"
    translated_title: Optional[str] = None
    translated_content: Optional[str] = None
    images: List[Dict[str, Any]] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsAlert:
    """News-based alert."""

    alert_id: str
    alert_type: str  # breaking, sentiment_shift, high_impact
    title: str
    summary: str
    symbols: List[str]
    sentiment_change: Optional[float] = None
    urgency: str = "normal"  # low, normal, high, critical
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class NewsIntelligenceAgent(StrandsAgent, A2AAgentCDSMixin, CDSTransactionalMixin):
    """
    News Intelligence Agent - ALL functionality through MCP tools.

    This agent handles:
    - News collection from multiple sources
    - Sentiment analysis
    - Multi-language translation
    - News correlation with market events
    - Alert generation for significant news
    - News summarization and insights
    """

    def __init__(self, agent_id: str = "news_intelligence_agent"):
        super().__init__(
            agent_id=agent_id,
            agent_type="news_intelligence",
            capabilities=[
                "news_collection",
                "sentiment_analysis", 
                "translation",
                "market_correlation",
                "alert_generation",
                "news_summarization"
            ]
        )

        # Initialize CDS monitoring if available
        if CDS_ENHANCED_FEATURES and get_cds_monitor:
            self._cds_monitor = get_cds_monitor()
        else:
            self._cds_monitor = None

        # Load MCP tools configuration
        self.mcp_tools = self._load_mcp_tools()
        self.mcp_handlers = self._initialize_mcp_handlers()

        # Initialize AI client for analysis
        self.ai_client = Grok4Client()

        # News storage and tracking
        self.news_cache: Dict[str, NewsArticle] = {}
        self.alerts: List[NewsAlert] = []
        self.sentiment_history: Dict[str, List[tuple]] = {}  # symbol -> [(time, sentiment)]

        # Configuration
        self.config = {
            "max_cache_size": 1000,
            "cache_ttl_hours": 24,
            "sentiment_window_hours": 6,
            "alert_threshold": 0.8,
            "languages": ["en", "ru", "zh", "es"],
            "sources": ["perplexity", "newsapi", "social", "blogs"],
        }

    async def initialize(self) -> bool:
        """Initialize the News Intelligence Agent with CDS integration"""
        try:
            logger.info(f"Initializing News Intelligence Agent {self.agent_id} with CDS")

            # Initialize CDS connection
            await self.initialize_cds()

            # Register agent capabilities with CDS if available
            if hasattr(self, '_cds_client') and self._cds_client:
                await self.register_with_cds(capabilities={
                    "news_collection": True,
                    "sentiment_analysis": True, 
                    "translation": True,
                    "market_correlation": True,
                    "alert_generation": True,
                    "news_summarization": True
                })

            logger.info(f"News Intelligence Agent {self.agent_id} initialized successfully with CDS")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize News Intelligence Agent {self.agent_id}: {e}")
            return False

    def _load_mcp_tools(self) -> Dict[str, Any]:
        """Load MCP tools configuration."""
        tools_path = Path(__file__).parent.parent / "mcp_tools" / "news_intelligence_tools.json"

        # Create the tools configuration if it doesn't exist
        if not tools_path.exists():
            tools_config = self._get_default_mcp_tools()
            tools_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tools_path, "w") as f:
                json.dump(tools_config, f, indent=2)
            return tools_config

        with open(tools_path, "r") as f:
            return json.load(f)

    def _get_default_mcp_tools(self) -> Dict[str, Any]:
        """Get default MCP tools configuration."""
        return {
            "name": "news-intelligence",
            "version": "1.0.0",
            "description": "MCP tools for news intelligence and analysis",
            "tools": {
                "fetch_news": {
                    "description": "Fetch news from multiple sources",
                    "parameters": {
                        "query": {"type": "string", "description": "Search query"},
                        "symbols": {
                            "type": "array",
                            "items": "string",
                            "description": "Crypto symbols to search",
                        },
                        "sources": {
                            "type": "array",
                            "items": "string",
                            "description": "News sources",
                            "optional": True,
                        },
                        "language": {
                            "type": "string",
                            "description": "Language code",
                            "default": "en",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of articles",
                            "default": 10,
                        },
                    },
                },
                "analyze_sentiment": {
                    "description": "Analyze news sentiment",
                    "parameters": {
                        "article": {"type": "object", "description": "News article to analyze"},
                        "symbols": {
                            "type": "array",
                            "items": "string",
                            "description": "Symbols to focus on",
                            "optional": True,
                        },
                    },
                },
                "translate_article": {
                    "description": "Translate news article",
                    "parameters": {
                        "article": {"type": "object", "description": "Article to translate"},
                        "target_language": {
                            "type": "string",
                            "description": "Target language code",
                        },
                        "preserve_technical": {
                            "type": "boolean",
                            "description": "Preserve technical terms",
                            "default": True,
                        },
                    },
                },
                "correlate_with_market": {
                    "description": "Correlate news with market events",
                    "parameters": {
                        "articles": {"type": "array", "description": "News articles"},
                        "market_data": {
                            "type": "object",
                            "description": "Market data for correlation",
                        },
                        "time_window": {
                            "type": "integer",
                            "description": "Hours to analyze",
                            "default": 24,
                        },
                    },
                },
                "generate_alert": {
                    "description": "Generate news-based alert",
                    "parameters": {
                        "article": {"type": "object", "description": "Triggering article"},
                        "alert_type": {"type": "string", "description": "Type of alert"},
                        "urgency": {"type": "string", "description": "Alert urgency level"},
                    },
                },
                "summarize_news": {
                    "description": "Summarize multiple news articles",
                    "parameters": {
                        "articles": {"type": "array", "description": "Articles to summarize"},
                        "focus": {"type": "string", "description": "Focus area", "optional": True},
                        "max_length": {
                            "type": "integer",
                            "description": "Max summary length",
                            "default": 500,
                        },
                    },
                },
                "extract_insights": {
                    "description": "Extract trading insights from news",
                    "parameters": {
                        "articles": {"type": "array", "description": "Articles to analyze"},
                        "symbols": {
                            "type": "array",
                            "items": "string",
                            "description": "Symbols of interest",
                        },
                    },
                },
                "monitor_sentiment_shift": {
                    "description": "Monitor for sentiment shifts",
                    "parameters": {
                        "symbol": {"type": "string", "description": "Symbol to monitor"},
                        "threshold": {
                            "type": "number",
                            "description": "Shift threshold",
                            "default": 0.3,
                        },
                    },
                },
                "get_trending_topics": {
                    "description": "Get trending crypto news topics",
                    "parameters": {
                        "time_range": {
                            "type": "string",
                            "description": "Time range (1h, 24h, 7d)",
                            "default": "24h",
                        },
                        "min_mentions": {
                            "type": "integer",
                            "description": "Minimum mentions",
                            "default": 5,
                        },
                    },
                },
                "analyze_source_credibility": {
                    "description": "Analyze news source credibility",
                    "parameters": {
                        "source": {"type": "string", "description": "News source to analyze"},
                        "articles": {
                            "type": "array",
                            "description": "Sample articles from source",
                            "optional": True,
                        },
                    },
                },
            },
        }

    def _initialize_mcp_handlers(self) -> Dict[str, Any]:
        """Initialize MCP tool handlers - ALL functionality through these."""
        return {
            "fetch_news": self._mcp_fetch_news,
            "analyze_sentiment": self._mcp_analyze_sentiment,
            "translate_article": self._mcp_translate_article,
            "correlate_with_market": self._mcp_correlate_with_market,
            "generate_alert": self._mcp_generate_alert,
            "summarize_news": self._mcp_summarize_news,
            "extract_insights": self._mcp_extract_insights,
            "monitor_sentiment_shift": self._mcp_monitor_sentiment_shift,
            "get_trending_topics": self._mcp_get_trending_topics,
            "analyze_source_credibility": self._mcp_analyze_source_credibility,
        }

    async def process_mcp_request(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for ALL agent functionality.
        All requests MUST come through MCP tool invocations.
        """
        logger.info(f"Processing MCP tool request: {tool_name}")

        if tool_name not in self.mcp_handlers:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        handler = self.mcp_handlers[tool_name]
        result = await handler(**parameters)

        return result

    # ============= MCP Tool Handlers =============

    @transactional(transaction_type="NEWS_COLLECTION", isolation_level=TransactionIsolation.READ_COMMITTED)
    async def _mcp_fetch_news(
        self,
        query: str,
        symbols: List[str],
        sources: Optional[List[str]] = None,
        language: str = "en",
        limit: int = 10,
        transaction=None,
    ) -> Dict[str, Any]:
        """MCP handler for fetching news."""
        # Track operation with CDS monitoring
        if self._cds_monitor and CDSOperationType:
            async with self._cds_monitor.track_operation(self.agent_id, CDSOperationType.DATA_ACCESS):
                return await self._fetch_news_internal(query, symbols, sources, language, limit)
        else:
            # Fallback without monitoring
            return await self._fetch_news_internal(query, symbols, sources, language, limit)

    async def _fetch_news_internal(
        self, query: str, symbols: List[str], sources: Optional[List[str]], language: str, limit: int
    ) -> Dict[str, Any]:
        """Internal method for fetching news."""
        # In production, this would call actual news APIs
        # For now, simulate with AI-generated news

        articles = []

        # Generate search query
        search_query = f"{query} {' '.join(symbols)} cryptocurrency trading"

        # Simulate fetching from sources
        for i in range(min(limit, 5)):
            article = NewsArticle(
                title=f"Breaking: {symbols[0] if symbols else 'Crypto'} Market Update",
                content=f"Latest developments in {query}...",
                url=f"https://news.example.com/article_{i}",
                source=sources[i % len(sources)] if sources else "newsapi",
                published_at=datetime.now() - timedelta(hours=i),
                sentiment=NewsSentiment.NEUTRAL,
                relevance_score=0.8 - (i * 0.1),
                symbols=symbols,
                language=language,
            )

            articles.append(
                {
                    "title": article.title,
                    "content": article.content,
                    "url": article.url,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "symbols": article.symbols,
                    "language": article.language,
                }
            )

            # Cache the article
            self.news_cache[article.url] = article

        return {
            "articles": articles,
            "total": len(articles),
            "query": search_query,
            "sources": sources or self.config["sources"],
        }

    async def _mcp_analyze_sentiment(
        self, article: Dict[str, Any], symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """MCP handler for sentiment analysis."""
        # Analyze sentiment using AI
        content = article.get("content", "")
        title = article.get("title", "")

        # Simple sentiment scoring (in production, use NLP model)
        positive_words = ["bullish", "surge", "gain", "rise", "profit", "growth"]
        negative_words = ["bearish", "crash", "loss", "fall", "decline", "risk"]

        text = f"{title} {content}".lower()
        positive_score = sum(1 for word in positive_words if word in text)
        negative_score = sum(1 for word in negative_words if word in text)

        if positive_score > negative_score + 2:
            sentiment = NewsSentiment.VERY_BULLISH
            score = 0.9
        elif positive_score > negative_score:
            sentiment = NewsSentiment.BULLISH
            score = 0.7
        elif negative_score > positive_score + 2:
            sentiment = NewsSentiment.VERY_BEARISH
            score = 0.1
        elif negative_score > positive_score:
            sentiment = NewsSentiment.BEARISH
            score = 0.3
        else:
            sentiment = NewsSentiment.NEUTRAL
            score = 0.5

        # Track sentiment history for symbols
        if symbols:
            for symbol in symbols:
                if symbol not in self.sentiment_history:
                    self.sentiment_history[symbol] = []
                self.sentiment_history[symbol].append((datetime.now(), score))
                # Keep only recent history
                cutoff = datetime.now() - timedelta(hours=self.config["sentiment_window_hours"])
                self.sentiment_history[symbol] = [
                    (t, s) for t, s in self.sentiment_history[symbol] if t > cutoff
                ]

        return {
            "sentiment": sentiment.value,
            "sentiment_score": score,
            "confidence": 0.75,
            "factors": {
                "positive_indicators": positive_score,
                "negative_indicators": negative_score,
            },
            "symbols_mentioned": symbols or [],
        }

    async def _mcp_translate_article(
        self, article: Dict[str, Any], target_language: str, preserve_technical: bool = True
    ) -> Dict[str, Any]:
        """MCP handler for article translation."""
        # In production, use translation API
        # For now, simulate translation

        original_title = article.get("title", "")
        original_content = article.get("content", "")

        # Simulate translation
        language_prefixes = {"ru": "РУ: ", "zh": "中文: ", "es": "ES: ", "en": ""}

        prefix = language_prefixes.get(target_language, "")

        translated = {
            "original_language": article.get("language", "en"),
            "target_language": target_language,
            "translated_title": f"{prefix}{original_title}",
            "translated_content": f"{prefix}{original_content}",
            "technical_terms_preserved": preserve_technical,
            "translation_confidence": 0.95,
        }

        return translated

    async def _mcp_correlate_with_market(
        self, articles: List[Dict[str, Any]], market_data: Dict[str, Any], time_window: int = 24
    ) -> Dict[str, Any]:
        """MCP handler for news-market correlation."""
        correlations = []

        for article in articles:
            # Extract publish time
            pub_time = datetime.fromisoformat(
                article.get("published_at", datetime.now().isoformat())
            )

            # Find market events around publication time
            market_events = []

            # Simulate correlation analysis
            correlation = {
                "article_url": article.get("url"),
                "publish_time": pub_time.isoformat(),
                "market_impact": {
                    "price_change_1h": 0.02,  # Mock 2% change
                    "volume_spike": 1.5,  # Mock 50% volume increase
                    "volatility_change": 0.1,
                },
                "correlation_score": 0.65,
                "lag_minutes": 30,  # Market reacted 30 mins after news
                "confidence": 0.7,
            }

            correlations.append(correlation)

        return {
            "correlations": correlations,
            "time_window_hours": time_window,
            "significant_correlations": len(
                [c for c in correlations if c["correlation_score"] > 0.6]
            ),
            "average_lag_minutes": 30,
        }

    async def _mcp_generate_alert(
        self, article: Dict[str, Any], alert_type: str, urgency: str
    ) -> Dict[str, Any]:
        """MCP handler for alert generation."""
        alert = NewsAlert(
            alert_id=f"alert_{datetime.now().timestamp()}",
            alert_type=alert_type,
            title=f"News Alert: {article.get('title', 'Unknown')}",
            summary=article.get("content", "")[:200],
            symbols=article.get("symbols", []),
            urgency=urgency,
        )

        self.alerts.append(alert)

        # Keep only recent alerts
        self.alerts = self.alerts[-100:]  # Keep last 100 alerts

        return {
            "alert_id": alert.alert_id,
            "alert_type": alert.alert_type,
            "title": alert.title,
            "urgency": alert.urgency,
            "symbols": alert.symbols,
            "created_at": alert.created_at.isoformat(),
            "status": "generated",
        }

    async def _mcp_summarize_news(
        self, articles: List[Dict[str, Any]], focus: Optional[str] = None, max_length: int = 500
    ) -> Dict[str, Any]:
        """MCP handler for news summarization."""
        if not articles:
            return {"summary": "", "article_count": 0}

        # Combine article content
        combined_text = "\n".join(
            [f"{a.get('title', '')}: {a.get('content', '')[:200]}" for a in articles]
        )

        # Generate summary (in production, use AI summarization)
        summary = f"Summary of {len(articles)} articles"
        if focus:
            summary += f" focusing on {focus}"
        summary += f": {combined_text[:max_length]}..."

        # Extract key points
        key_points = [
            f"Article {i+1}: {a.get('title', 'Unknown')}" for i, a in enumerate(articles[:3])
        ]

        return {
            "summary": summary,
            "article_count": len(articles),
            "key_points": key_points,
            "focus_area": focus,
            "generated_at": datetime.now().isoformat(),
        }

    async def _mcp_extract_insights(
        self, articles: List[Dict[str, Any]], symbols: List[str]
    ) -> Dict[str, Any]:
        """MCP handler for extracting trading insights."""
        insights = {
            "bullish_signals": [],
            "bearish_signals": [],
            "neutral_observations": [],
            "opportunities": [],
            "risks": [],
        }

        for article in articles:
            # Analyze each article for insights
            sentiment_result = await self._mcp_analyze_sentiment(article, symbols)

            if sentiment_result["sentiment_score"] > 0.7:
                insights["bullish_signals"].append(
                    {
                        "source": article.get("url"),
                        "signal": "Positive sentiment detected",
                        "confidence": sentiment_result["confidence"],
                    }
                )
            elif sentiment_result["sentiment_score"] < 0.3:
                insights["bearish_signals"].append(
                    {
                        "source": article.get("url"),
                        "signal": "Negative sentiment detected",
                        "confidence": sentiment_result["confidence"],
                    }
                )

        # Generate trading opportunities
        if len(insights["bullish_signals"]) > len(insights["bearish_signals"]):
            insights["opportunities"].append(
                {
                    "type": "long_position",
                    "symbols": symbols,
                    "reason": "Positive news sentiment",
                    "confidence": 0.65,
                }
            )

        return {
            "insights": insights,
            "symbols_analyzed": symbols,
            "article_count": len(articles),
            "recommendation": "monitor" if not insights["opportunities"] else "consider_position",
            "generated_at": datetime.now().isoformat(),
        }

    async def _mcp_monitor_sentiment_shift(
        self, symbol: str, threshold: float = 0.3
    ) -> Dict[str, Any]:
        """MCP handler for monitoring sentiment shifts."""
        if symbol not in self.sentiment_history:
            return {
                "symbol": symbol,
                "shift_detected": False,
                "message": "No sentiment history for symbol",
            }

        history = self.sentiment_history[symbol]
        if len(history) < 2:
            return {"symbol": symbol, "shift_detected": False, "message": "Insufficient history"}

        # Calculate sentiment change
        recent_sentiments = [s for _, s in history[-5:]]
        older_sentiments = (
            [s for _, s in history[-10:-5]] if len(history) > 5 else [s for _, s in history[:-1]]
        )

        if not older_sentiments:
            return {
                "symbol": symbol,
                "shift_detected": False,
                "message": "Insufficient history for comparison",
            }

        recent_avg = sum(recent_sentiments) / len(recent_sentiments)
        older_avg = sum(older_sentiments) / len(older_sentiments)
        shift = recent_avg - older_avg

        shift_detected = abs(shift) > threshold

        return {
            "symbol": symbol,
            "shift_detected": shift_detected,
            "shift_magnitude": shift,
            "current_sentiment": recent_avg,
            "previous_sentiment": older_avg,
            "direction": "bullish" if shift > 0 else "bearish",
            "confidence": min(abs(shift) / threshold, 1.0) if shift_detected else 0,
        }

    async def _mcp_get_trending_topics(
        self, time_range: str = "24h", min_mentions: int = 5
    ) -> Dict[str, Any]:
        """MCP handler for getting trending topics."""
        # Analyze cached news for trending topics
        topics = {}

        # Parse time range
        hours = {"1h": 1, "24h": 24, "7d": 168}.get(time_range, 24)
        cutoff = datetime.now() - timedelta(hours=hours)

        # Count topic mentions
        for url, article in self.news_cache.items():
            if article.published_at > cutoff:
                # Extract topics (simplified)
                words = article.title.lower().split() + article.content.lower().split()[:50]
                for word in words:
                    if len(word) > 4:  # Skip short words
                        topics[word] = topics.get(word, 0) + 1

        # Filter by minimum mentions
        trending = {topic: count for topic, count in topics.items() if count >= min_mentions}

        # Sort by mentions
        sorted_topics = sorted(trending.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "trending_topics": [
                {"topic": topic, "mentions": count} for topic, count in sorted_topics
            ],
            "time_range": time_range,
            "min_mentions": min_mentions,
            "total_articles_analyzed": len(
                [a for a in self.news_cache.values() if a.published_at > cutoff]
            ),
        }

    async def _mcp_analyze_source_credibility(
        self, source: str, articles: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """MCP handler for analyzing source credibility."""
        # Analyze news source credibility
        known_sources = {
            "perplexity": 0.9,
            "newsapi": 0.85,
            "reuters": 0.95,
            "bloomberg": 0.95,
            "coindesk": 0.8,
            "cointelegraph": 0.75,
            "social": 0.5,
            "blogs": 0.6,
        }

        base_score = known_sources.get(source.lower(), 0.5)

        # Analyze articles if provided
        consistency_score = 0.7  # Default
        if articles:
            # Check for consistency, accuracy, etc.
            consistency_score = 0.8  # Simplified

        final_score = (base_score + consistency_score) / 2

        return {
            "source": source,
            "credibility_score": final_score,
            "reputation": "high" if final_score > 0.8 else "medium" if final_score > 0.6 else "low",
            "factors": {
                "known_reputation": base_score,
                "content_consistency": consistency_score,
                "sample_size": len(articles) if articles else 0,
            },
            "recommendation": "trusted" if final_score > 0.7 else "verify",
        }

    async def process_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """
        Process incoming A2A messages.
        ALL functionality routed through MCP tools.
        """
        if message.type == MessageType.ANALYSIS_REQUEST:
            # Extract MCP tool request
            tool_name = message.payload.get("mcp_tool")
            parameters = message.payload.get("parameters", {})

            if not tool_name:
                # Legacy support
                action = message.payload.get("action")
                if action == "fetch_news":
                    tool_name = "fetch_news"
                    parameters = message.payload
                elif action == "analyze":
                    tool_name = "analyze_sentiment"
                    parameters = message.payload

            if tool_name:
                try:
                    result = await self.process_mcp_request(tool_name, parameters)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": "No MCP tool specified"}

            return A2AMessage(
                type=MessageType.ANALYSIS_RESPONSE,
                sender=self.agent_id,
                receiver=message.sender,
                payload=result,
            )

        return await super().process_message(message)

    async def _process_message_impl(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Implementation of abstract method."""
        return await self.process_message(message)
