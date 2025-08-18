"""
News Collection Agent using STRANDS framework and MCP tools
Collects news from Perplexity AI API and builds links to historical data
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from ...infrastructure.monitoring import get_logger, trace_context
from ..strands import StrandsAgent, StrandConfig
from ...ml.perplexity import PerplexityClient

logger = get_logger("agents.news_collection")


@dataclass
class NewsItem:
    """News item data structure"""
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    symbol: str
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    impact_level: Optional[str] = None
    keywords: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'keywords': self.keywords or []
        }


@dataclass
class NewsMarketCorrelation:
    """Correlation between news and market movements"""
    symbol: str
    news_timestamp: datetime
    market_price_before: float
    market_price_after: float
    price_change_percent: float
    volume_change_percent: float
    correlation_strength: float
    correlation_type: str  # 'positive', 'negative', 'neutral'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'news_timestamp': self.news_timestamp.isoformat()
        }


class NewsCollectionAgent(StrandsAgent):
    """
    STRANDS agent for collecting news using Perplexity AI API
    Integrates with MCP tools for analysis and correlation
    """
    
    def __init__(self, config: Optional[StrandConfig] = None):
        self.agent_id = "news_collection_agent"
        self.config = config or StrandConfig(
            name="NewsCollectionAgent",
            description="Collects crypto news and correlates with market data",
            capabilities=["news_collection", "sentiment_analysis", "market_correlation"],
            dependencies=["perplexity_api", "market_data", "mcp_tools"]
        )
        
        super().__init__(self.agent_id, self.config)
        
        # Initialize components
        self.perplexity_client = PerplexityClient()
        self.news_cache = {}
        self.correlation_cache = {}
        
        # Symbols to monitor
        self.monitored_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'MATIC', 'DOT', 'LINK']
        
    async def initialize(self) -> bool:
        """Initialize the news collection agent"""
        try:
            with trace_context("news_agent_init") as span:
                span.set_attribute("agent_id", self.agent_id)
                
                logger.info("Initializing News Collection Agent")
                
                # Test Perplexity API connection
                test_result = await self._test_perplexity_connection()
                if not test_result:
                    logger.error("Failed to connect to Perplexity API")
                    return False
                
                # Initialize MCP tools integration
                await self._initialize_mcp_tools()
                
                # Initialize database schemas for news storage
                await self._initialize_news_schemas()
                
                logger.info("News Collection Agent initialized successfully")
                span.set_attribute("initialization", "success")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize News Collection Agent: {e}")
            return False
    
    async def _test_perplexity_connection(self) -> bool:
        """Test connection to Perplexity API"""
        try:
            # Simple test query
            test_news = self.perplexity_client.search_crypto_news("BTC")
            return test_news is not None
        except Exception as e:
            logger.error(f"Perplexity API test failed: {e}")
            return False
    
    async def _initialize_mcp_tools(self):
        """Initialize MCP tools for news analysis"""
        try:
            # This would initialize the MCP tools for pattern matching and analysis
            logger.info("Initializing MCP tools for news analysis")
            # Implementation would connect to existing MCP infrastructure
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
    
    async def _initialize_news_schemas(self):
        """Initialize database schemas for news storage"""
        try:
            from ....infrastructure.database.unified_database import UnifiedDatabase
            
            db = UnifiedDatabase()
            await db.initialize()
            
            # Create news tables
            cursor = db.db_conn.cursor()
            
            # News items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    url TEXT,
                    published_at TIMESTAMP,
                    source TEXT,
                    symbol TEXT,
                    sentiment_score REAL,
                    relevance_score REAL,
                    impact_level TEXT,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # News market correlations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_market_correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    news_timestamp TIMESTAMP,
                    market_price_before REAL,
                    market_price_after REAL,
                    price_change_percent REAL,
                    volume_change_percent REAL,
                    correlation_strength REAL,
                    correlation_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            db.db_conn.commit()
            cursor.close()
            
            logger.info("News database schemas initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize news schemas: {e}")
    
    async def collect_news_for_symbol(self, symbol: str, hours_back: int = 24) -> List[NewsItem]:
        """Collect news for a specific symbol using Perplexity API"""
        with trace_context(f"collect_news_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("hours_back", hours_back)
                
                logger.info(f"Collecting news for {symbol} (last {hours_back} hours)")
                
                # Use Perplexity to search for crypto news
                raw_news = self.perplexity_client.search_crypto_news(symbol)
                
                if not raw_news:
                    logger.warning(f"No news found for {symbol}")
                    return []
                
                # Process and structure the news data
                news_items = await self._process_raw_news(raw_news, symbol)
                
                # Analyze sentiment for each news item
                for news_item in news_items:
                    news_item.sentiment_score = await self._analyze_sentiment(news_item)
                    news_item.relevance_score = await self._calculate_relevance(news_item, symbol)
                    news_item.impact_level = await self._assess_impact_level(news_item)
                
                # Store news items in database
                await self._store_news_items(news_items)
                
                # Cache the results
                cache_key = f"{symbol}_{hours_back}h"
                self.news_cache[cache_key] = {
                    'items': news_items,
                    'collected_at': datetime.utcnow(),
                    'symbol': symbol
                }
                
                span.set_attribute("news_items_collected", len(news_items))
                logger.info(f"Collected {len(news_items)} news items for {symbol}")
                
                return news_items
                
            except Exception as e:
                span.set_attribute("error", str(e))
                logger.error(f"Failed to collect news for {symbol}: {e}")
                return []
    
    async def _process_raw_news(self, raw_news: Any, symbol: str) -> List[NewsItem]:
        """Process raw news data from Perplexity into structured NewsItem objects"""
        news_items = []
        
        try:
            # Handle different response formats from Perplexity
            if isinstance(raw_news, dict) and 'articles' in raw_news:
                articles = raw_news['articles']
            elif isinstance(raw_news, list):
                articles = raw_news
            else:
                logger.warning(f"Unexpected news format for {symbol}")
                return []
            
            for article in articles:
                try:
                    # Extract keywords using MCP tools pattern matching
                    keywords = await self._extract_keywords(article.get('title', ''), article.get('content', ''))
                    
                    news_item = NewsItem(
                        title=article.get('title', ''),
                        content=article.get('content', article.get('description', '')),
                        url=article.get('url', ''),
                        published_at=self._parse_datetime(article.get('published_at', article.get('publishedAt'))),
                        source=article.get('source', 'perplexity'),
                        symbol=symbol,
                        keywords=keywords
                    )
                    
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"Failed to process article: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Failed to process raw news: {e}")
            return []
    
    async def _extract_keywords(self, title: str, content: str) -> List[str]:
        """Extract keywords using MCP tools pattern matching"""
        try:
            # This would use MCP tools for keyword extraction
            # For now, implement basic keyword extraction
            text = f"{title} {content}".lower()
            
            # Crypto-specific keywords
            crypto_keywords = [
                'bitcoin', 'btc', 'ethereum', 'eth', 'blockchain', 'crypto', 'cryptocurrency',
                'defi', 'nft', 'mining', 'wallet', 'exchange', 'trading', 'bull', 'bear',
                'adoption', 'regulation', 'sec', 'etf', 'institutional', 'whale', 'hodl'
            ]
            
            found_keywords = [keyword for keyword in crypto_keywords if keyword in text]
            
            return found_keywords[:10]  # Limit to top 10 keywords
            
        except Exception as e:
            logger.error(f"Failed to extract keywords: {e}")
            return []
    
    def _parse_datetime(self, date_str: Any) -> Optional[datetime]:
        """Parse datetime from various formats"""
        if not date_str:
            return datetime.utcnow()
        
        if isinstance(date_str, datetime):
            return date_str
        
        try:
            # Try different datetime formats
            formats = [
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(str(date_str), fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return current time
            return datetime.utcnow()
            
        except Exception as e:
            logger.warning(f"Failed to parse datetime {date_str}: {e}")
            return datetime.utcnow()
    
    async def _analyze_sentiment(self, news_item: NewsItem) -> float:
        """Analyze sentiment of news item using MCP tools"""
        try:
            # This would use MCP tools for sentiment analysis
            # For now, implement basic sentiment scoring
            text = f"{news_item.title} {news_item.content}".lower()
            
            positive_words = ['bull', 'bullish', 'rise', 'up', 'gain', 'positive', 'adoption', 'growth']
            negative_words = ['bear', 'bearish', 'fall', 'down', 'loss', 'negative', 'crash', 'decline']
            
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count + negative_count == 0:
                return 0.5  # Neutral
            
            sentiment_score = positive_count / (positive_count + negative_count)
            return round(sentiment_score, 3)
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return 0.5  # Default to neutral
    
    async def _calculate_relevance(self, news_item: NewsItem, symbol: str) -> float:
        """Calculate relevance score of news to specific symbol"""
        try:
            text = f"{news_item.title} {news_item.content}".lower()
            symbol_lower = symbol.lower()
            
            # Direct symbol mentions
            direct_mentions = text.count(symbol_lower)
            
            # Symbol-specific terms
            symbol_terms = {
                'btc': ['bitcoin'],
                'eth': ['ethereum'],
                'bnb': ['binance'],
                'ada': ['cardano'],
                'sol': ['solana'],
                'matic': ['polygon'],
                'dot': ['polkadot'],
                'link': ['chainlink']
            }
            
            related_mentions = 0
            if symbol_lower in symbol_terms:
                for term in symbol_terms[symbol_lower]:
                    related_mentions += text.count(term)
            
            # Calculate relevance score
            total_mentions = direct_mentions * 2 + related_mentions  # Weight direct mentions more
            relevance_score = min(total_mentions / 5.0, 1.0)  # Normalize to 0-1
            
            return round(relevance_score, 3)
            
        except Exception as e:
            logger.error(f"Failed to calculate relevance: {e}")
            return 0.5
    
    async def _assess_impact_level(self, news_item: NewsItem) -> str:
        """Assess the potential market impact level of news"""
        try:
            text = f"{news_item.title} {news_item.content}".lower()
            
            high_impact_terms = ['sec', 'regulation', 'etf', 'institutional', 'whale', 'hack', 'crash']
            medium_impact_terms = ['adoption', 'partnership', 'upgrade', 'launch', 'announcement']
            
            high_count = sum(1 for term in high_impact_terms if term in text)
            medium_count = sum(1 for term in medium_impact_terms if term in text)
            
            if high_count > 0:
                return 'high'
            elif medium_count > 0:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Failed to assess impact level: {e}")
            return 'low'
    
    async def _store_news_items(self, news_items: List[NewsItem]):
        """Store news items in database"""
        try:
            from ....infrastructure.database.unified_database import UnifiedDatabase
            
            db = UnifiedDatabase()
            cursor = db.db_conn.cursor()
            
            for news_item in news_items:
                cursor.execute("""
                    INSERT INTO news_items 
                    (title, content, url, published_at, source, symbol, 
                     sentiment_score, relevance_score, impact_level, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    news_item.title,
                    news_item.content,
                    news_item.url,
                    news_item.published_at,
                    news_item.source,
                    news_item.symbol,
                    news_item.sentiment_score,
                    news_item.relevance_score,
                    news_item.impact_level,
                    json.dumps(news_item.keywords or [])
                ))
            
            db.db_conn.commit()
            cursor.close()
            
            logger.info(f"Stored {len(news_items)} news items in database")
            
        except Exception as e:
            logger.error(f"Failed to store news items: {e}")
    
    async def collect_news_for_all_symbols(self) -> Dict[str, List[NewsItem]]:
        """Collect news for all monitored symbols"""
        all_news = {}
        
        logger.info(f"Collecting news for {len(self.monitored_symbols)} symbols")
        
        # Collect news for each symbol in parallel
        tasks = [
            self.collect_news_for_symbol(symbol) 
            for symbol in self.monitored_symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            symbol = self.monitored_symbols[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to collect news for {symbol}: {result}")
                all_news[symbol] = []
            else:
                all_news[symbol] = result
        
        return all_news
    
    async def get_cached_news(self, symbol: str, max_age_hours: int = 1) -> Optional[List[NewsItem]]:
        """Get cached news if available and fresh"""
        cache_key = f"{symbol}_24h"
        
        if cache_key in self.news_cache:
            cached_data = self.news_cache[cache_key]
            age = datetime.utcnow() - cached_data['collected_at']
            
            if age.total_seconds() < max_age_hours * 3600:
                logger.info(f"Returning cached news for {symbol} (age: {age})")
                return cached_data['items']
        
        return None
    
    async def get_news_summary(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get news summary with analytics"""
        try:
            # Try to get cached news first
            news_items = await self.get_cached_news(symbol)
            
            if not news_items:
                # Collect fresh news
                news_items = await self.collect_news_for_symbol(symbol, hours_back)
            
            if not news_items:
                return {
                    'symbol': symbol,
                    'news_count': 0,
                    'message': 'No news found'
                }
            
            # Calculate summary statistics
            sentiment_scores = [item.sentiment_score for item in news_items if item.sentiment_score is not None]
            relevance_scores = [item.relevance_score for item in news_items if item.relevance_score is not None]
            
            impact_counts = {}
            for item in news_items:
                impact = item.impact_level or 'unknown'
                impact_counts[impact] = impact_counts.get(impact, 0) + 1
            
            summary = {
                'symbol': symbol,
                'news_count': len(news_items),
                'time_range_hours': hours_back,
                'average_sentiment': round(sum(sentiment_scores) / len(sentiment_scores), 3) if sentiment_scores else 0.5,
                'average_relevance': round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else 0.5,
                'impact_distribution': impact_counts,
                'latest_news': [item.to_dict() for item in news_items[:5]],  # Latest 5 items
                'collected_at': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get news summary for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
