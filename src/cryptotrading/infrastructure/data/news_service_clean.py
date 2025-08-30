#!/usr/bin/env python3
"""
Perplexity News Service for Crypto Trading Platform
Integrates with Perplexity API to fetch relevant cryptocurrency and trading news
"""

import asyncio
import logging
import re
import ssl
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import certifi

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Data class for news articles"""

    title: str
    content: str
    url: str
    published_at: str
    source: str
    relevance_score: float = 0.0
    sentiment: str = "neutral"
    symbols: List[str] = None
    category: str = "general"

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": self.published_at,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "sentiment": self.sentiment,
            "symbols": self.symbols,
            "category": self.category,
        }


class PerplexityNewsService:
    """Service for fetching cryptocurrency news from Perplexity API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or "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"
        self.base_url = "https://api.perplexity.ai"
        self.session = None

        # News categories for crypto trading
        self.categories = {
            "market_analysis": "cryptocurrency market analysis and price predictions",
            "regulatory": "cryptocurrency regulation and legal developments",
            "technology": "blockchain technology and cryptocurrency innovations",
            "institutional": "institutional cryptocurrency adoption and investments",
            "defi": "decentralized finance DeFi protocols and developments",
            "nft": "NFT non-fungible token market and trends",
            "trading": "cryptocurrency trading strategies and market movements",
        }

        # Major cryptocurrency symbols to track
        self.tracked_symbols = [
            "BTC",
            "ETH",
            "BNB",
            "XRP",
            "ADA",
            "SOL",
            "DOGE",
            "DOT",
            "MATIC",
            "AVAX",
            "SHIB",
            "LTC",
            "UNI",
            "LINK",
            "ATOM",
            "XLM",
            "ALGO",
            "VET",
            "ICP",
            "FIL",
        ]

    async def __aenter__(self):
        """Async context manager entry"""
        # Create SSL context with proper certificate handling
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        connector = aiohttp.TCPConnector(
            ssl=ssl_context, limit=100, limit_per_host=30, ttl_dns_cache=300, use_dns_cache=True
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "CryptoTrading-NewsService/1.0",
            },
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _make_request(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Make request to Perplexity API"""
        try:
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency news analyst. Provide recent, accurate news about cryptocurrency markets, blockchain technology, and digital assets. Format responses as structured data with title, content, source, and publication date.",
                    },
                    {
                        "role": "user",
                        "content": f"Find the latest {max_results} news articles about: {query}. Include title, summary, source, and publication date for each article.",
                    },
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_citations": True,
                "search_domain_filter": [
                    "coindesk.com",
                    "cointelegraph.com",
                    "decrypt.co",
                    "theblock.co",
                    "bloomberg.com",
                    "reuters.com",
                ],
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "day",
            }

            async with self.session.post(
                f"{self.base_url}/chat/completions", json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    logger.error("Perplexity API error %d: %s", response.status, error_text)
                    return {"error": f"API request failed with status {response.status}"}

        except asyncio.TimeoutError:
            logger.error("Perplexity API request timeout")
            return {"error": "Request timeout"}
        except aiohttp.ClientError as e:
            logger.error("Perplexity API client error: %s", str(e))
            return {"error": f"Client error: {str(e)}"}
        except Exception as e:
            logger.error("Perplexity API request failed: %s", str(e))
            return {"error": str(e)}

    def _parse_news_response(
        self, response: Dict[str, Any], category: str = "general"
    ) -> List[NewsArticle]:
        """Parse Perplexity API response into NewsArticle objects"""
        articles = []

        try:
            if "error" in response:
                logger.warning("API response contains error: %s", response["error"])
                return articles

            # Extract content from Perplexity response
            choices = response.get("choices", [])
            if not choices:
                return articles

            content = choices[0].get("message", {}).get("content", "")
            citations = response.get("citations", [])

            # Parse the structured response
            lines = content.split("\n")
            current_article = {}

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for article markers
                if line.startswith("**") and line.endswith("**"):
                    # This might be a title
                    if current_article:
                        # Save previous article
                        article = self._create_article_from_dict(current_article, category)
                        if article:
                            articles.append(article)
                    current_article = {"title": line.strip("*")}
                elif line.startswith("Source:"):
                    current_article["source"] = line.replace("Source:", "").strip()
                elif line.startswith("Date:") or line.startswith("Published:"):
                    current_article["published_at"] = line.split(":", 1)[1].strip()
                elif len(line) > 50:  # Assume this is content
                    current_article["content"] = current_article.get("content", "") + " " + line

            # Don't forget the last article
            if current_article:
                article = self._create_article_from_dict(current_article, category)
                if article:
                    articles.append(article)

            # If parsing failed, create a single article from the entire response
            if not articles and content:
                articles.append(
                    NewsArticle(
                        title="Latest Cryptocurrency News",
                        content=content[:500] + "..." if len(content) > 500 else content,
                        url=citations[0] if citations else "",
                        published_at=datetime.now().isoformat(),
                        source="Perplexity AI",
                        category=category,
                        symbols=self._extract_symbols(content),
                    )
                )

        except Exception as e:
            logger.error("Error parsing news response: %s", str(e))

        return articles

    def _create_article_from_dict(
        self, article_dict: Dict[str, str], category: str
    ) -> Optional[NewsArticle]:
        """Create NewsArticle from parsed dictionary"""
        try:
            title = article_dict.get("title", "Untitled")
            content = article_dict.get("content", "")

            if not title or not content:
                return None

            return NewsArticle(
                title=title,
                content=content,
                url=article_dict.get("url", ""),
                published_at=article_dict.get("published_at", datetime.now().isoformat()),
                source=article_dict.get("source", "Unknown"),
                category=category,
                symbols=self._extract_symbols(f"{title} {content}"),
            )
        except Exception as e:
            logger.error("Error creating article: %s", str(e))
            return None

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract cryptocurrency symbols from text"""
        symbols = []
        text_upper = text.upper()

        for symbol in self.tracked_symbols:
            if symbol in text_upper or f"${symbol}" in text_upper:
                symbols.append(symbol)

        return list(set(symbols))  # Remove duplicates

    def _calculate_relevance_score(self, article: NewsArticle, query: str) -> float:
        """Calculate relevance score for an article"""
        score = 0.0
        query_lower = query.lower()
        title_lower = article.title.lower()
        content_lower = article.content.lower()

        # Title matches
        if query_lower in title_lower:
            score += 0.5

        # Content matches
        query_words = query_lower.split()
        for word in query_words:
            if word in title_lower:
                score += 0.2
            if word in content_lower:
                score += 0.1

        # Symbol matches
        if article.symbols:
            score += 0.3

        # Recent articles get higher scores
        try:
            pub_date = datetime.fromisoformat(article.published_at.replace("Z", "+00:00"))
            hours_ago = (datetime.now() - pub_date.replace(tzinfo=None)).total_seconds() / 3600
            if hours_ago < 24:
                score += 0.2
        except (ValueError, TypeError, AttributeError):
            pass

        return min(score, 1.0)

    async def get_latest_news(self, limit: int = 10) -> List[NewsArticle]:
        """Get latest cryptocurrency news"""
        query = "latest cryptocurrency news Bitcoin Ethereum blockchain digital assets"
        response = await self._make_request(query, limit)
        articles = self._parse_news_response(response, "general")

        # Calculate relevance scores
        for article in articles:
            article.relevance_score = self._calculate_relevance_score(article, query)

        # Sort by relevance score
        articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return articles[:limit]

    async def get_news_by_category(self, category: str, limit: int = 10) -> List[NewsArticle]:
        """Get news by specific category"""
        if category not in self.categories:
            logger.warning("Unknown category: %s", category)
            return await self.get_latest_news(limit)

        query = self.categories[category]
        response = await self._make_request(query, limit)
        articles = self._parse_news_response(response, category)

        # Calculate relevance scores
        for article in articles:
            article.relevance_score = self._calculate_relevance_score(article, query)

        # Sort by relevance score
        articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return articles[:limit]

    async def get_news_by_symbol(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Get news for a specific cryptocurrency symbol"""
        symbol = symbol.upper()

        # Map common symbols to full names for better search
        symbol_names = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "BNB": "Binance Coin",
            "XRP": "Ripple",
            "ADA": "Cardano",
            "SOL": "Solana",
            "DOGE": "Dogecoin",
            "DOT": "Polkadot",
            "MATIC": "Polygon",
            "AVAX": "Avalanche",
        }

        full_name = symbol_names.get(symbol, symbol)
        query = f"{full_name} {symbol} cryptocurrency news price analysis market"

        response = await self._make_request(query, limit)
        articles = self._parse_news_response(response, "symbol_specific")

        # Filter articles that mention the symbol
        filtered_articles = []
        for article in articles:
            if (
                symbol in article.symbols
                or symbol.lower() in article.title.lower()
                or symbol.lower() in article.content.lower()
            ):
                article.relevance_score = self._calculate_relevance_score(article, query)
                filtered_articles.append(article)

        # Sort by relevance score
        filtered_articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return filtered_articles[:limit]

    async def get_market_sentiment_news(self, limit: int = 10) -> List[NewsArticle]:
        """Get news focused on market sentiment and analysis"""
        query = "cryptocurrency market sentiment analysis bullish bearish price prediction technical analysis"
        response = await self._make_request(query, limit)
        articles = self._parse_news_response(response, "market_analysis")

        # Calculate relevance scores
        for article in articles:
            article.relevance_score = self._calculate_relevance_score(article, query)
            # Simple sentiment analysis based on keywords
            content_lower = f"{article.title} {article.content}".lower()
            if any(
                word in content_lower
                for word in ["bullish", "bull", "surge", "rally", "pump", "moon"]
            ):
                article.sentiment = "positive"
            elif any(
                word in content_lower
                for word in ["bearish", "bear", "crash", "dump", "decline", "fall"]
            ):
                article.sentiment = "negative"

        # Sort by relevance score
        articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return articles[:limit]

    async def search_news(self, query: str, limit: int = 10) -> List[NewsArticle]:
        """Search for news with custom query"""
        search_query = f"cryptocurrency {query} blockchain digital assets"
        response = await self._make_request(search_query, limit)
        articles = self._parse_news_response(response, "search")

        # Calculate relevance scores
        for article in articles:
            article.relevance_score = self._calculate_relevance_score(article, query)

        # Sort by relevance score
        articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return articles[:limit]

    def get_available_categories(self) -> Dict[str, str]:
        """Get available news categories"""
        return self.categories.copy()

    def get_tracked_symbols(self) -> List[str]:
        """Get list of tracked cryptocurrency symbols"""
        return self.tracked_symbols.copy()


# Convenience function for quick access
async def get_crypto_news(
    category: str = "general", limit: int = 10, api_key: str = None
) -> List[Dict[str, Any]]:
    """Quick function to get crypto news"""
    async with PerplexityNewsService(api_key) as service:
        if category == "general":
            articles = await service.get_latest_news(limit)
        else:
            articles = await service.get_news_by_category(category, limit)

        return [article.to_dict() for article in articles]
