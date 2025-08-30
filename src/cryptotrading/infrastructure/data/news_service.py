#!/usr/bin/env python3
"""
Perplexity News Service for Crypto Trading Platform
Integrates with Perplexity API to fetch relevant cryptocurrency and trading news
"""

import asyncio
import logging
import ssl
import certifi
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import aiohttp
import json

logger = logging.getLogger(__name__)

@dataclass
class NewsImage:
    """Data class for news article images"""
    url: str
    alt_text: str = ""
    type: str = "photo"  # photo, chart, infographic, logo
    source: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    caption: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'alt_text': self.alt_text,
            'type': self.type,
            'source': self.source,
            'width': self.width,
            'height': self.height,
            'caption': self.caption
        }

@dataclass
class NewsArticle:
    """Data class for news articles with image support"""
    title: str
    content: str
    url: str
    published_at: str
    source: str
    relevance_score: float = 0.0
    sentiment: str = "neutral"
    symbols: List[str] = None
    category: str = "general"
    language: str = "en"
    translated_title: str = ""
    translated_content: str = ""
    images: List[NewsImage] = None
    has_images: bool = False
    image_count: int = 0
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []
        if self.images is None:
            self.images = []
        self.image_count = len(self.images)
        self.has_images = self.image_count > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'published_at': self.published_at,
            'source': self.source,
            'relevance_score': self.relevance_score,
            'sentiment': self.sentiment,
            'symbols': self.symbols,
            'category': self.category,
            'language': self.language,
            'translated_title': self.translated_title,
            'translated_content': self.translated_content,
            'images': [img.to_dict() for img in self.images],
            'has_images': self.has_images,
            'image_count': self.image_count
        }

class AITranslationClient:
    """AI client for translating news content to Russian"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "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"
        self.base_url = "https://api.perplexity.ai"
    
    async def translate_to_russian(self, text: str, session: aiohttp.ClientSession) -> str:
        """Translate text to Russian using AI"""
        try:
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator specializing in cryptocurrency and financial content. Translate the given text to Russian while preserving technical terms and maintaining accuracy."
                    },
                    {
                        "role": "user",
                        "content": f"Translate this cryptocurrency news text to Russian: {text}"
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", text)
                else:
                    logger.error("Translation API error %d", response.status)
                    return text
                    
        except Exception as e:
            logger.error("Translation failed: %s", str(e))
            return text

class PerplexityNewsService:
    """Service for fetching cryptocurrency news from Perplexity API with Russian support"""
    
    def __init__(self, api_key: str = None, enable_images: bool = True):
        self.api_key = api_key or "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"
        self.base_url = "https://api.perplexity.ai"
        self.session = None
        self.translator = AITranslationClient(api_key)
        self.enable_images = enable_images
        
        # Initialize image enhancer if enabled
        if self.enable_images:
            try:
                from .image_services import NewsImageEnhancer
                self.image_enhancer = NewsImageEnhancer()
            except ImportError as e:
                logger.warning(f"Image services not available: {e}")
                self.enable_images = False
                self.image_enhancer = None
        else:
            self.image_enhancer = None
        
        # News categories for crypto trading (English and Russian)
        self.categories = {
            'market_analysis': 'cryptocurrency market analysis and price predictions',
            'regulatory': 'cryptocurrency regulation and legal developments',
            'technology': 'blockchain technology and cryptocurrency innovations',
            'institutional': 'institutional cryptocurrency adoption and investments',
            'defi': 'decentralized finance DeFi protocols and developments',
            'nft': 'NFT non-fungible token market and trends',
            'trading': 'cryptocurrency trading strategies and market movements'
        }
        
        # Russian categories
        self.russian_categories = {
            'market_analysis': 'анализ криптовалютного рынка и прогнозы цен',
            'regulatory': 'регулирование криптовалют и правовые изменения',
            'technology': 'технология блокчейн и криптовалютные инновации',
            'institutional': 'институциональное принятие криптовалют и инвестиции',
            'defi': 'децентрализованные финансы DeFi протоколы и разработки',
            'nft': 'рынок NFT невзаимозаменяемых токенов и тренды',
            'trading': 'стратегии торговли криптовалютами и движения рынка'
        }
        
        # Major cryptocurrency symbols to track
        self.tracked_symbols = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'MATIC', 'AVAX',
            'SHIB', 'LTC', 'UNI', 'LINK', 'ATOM', 'XLM', 'ALGO', 'VET', 'ICP', 'FIL'
        ]
        
        # Russian crypto news sources
        self.russian_sources = [
            'forklog.com', 'coinspot.io', 'bits.media', 'cryptonews.net',
            'incrypted.com', 'mining-cryptocurrency.ru'
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Create SSL context with proper certificate handling
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'CryptoTrading-NewsService/1.0'
            },
            timeout=aiohttp.ClientTimeout(total=30)
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
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency news analyst. Provide recent, accurate news about cryptocurrency markets, blockchain technology, and digital assets. Format responses as structured data with title, content, source, and publication date."
                    },
                    {
                        "role": "user", 
                        "content": f"Find the latest {max_results} news articles about: {query}. Include title, summary, source, and publication date for each article."
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_citations": True,
                "search_domain_filter": ["coindesk.com", "cointelegraph.com", "decrypt.co", "theblock.co", "bloomberg.com", "reuters.com"],
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "day"
            }
            
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
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
    
    def _parse_news_response(self, response: Dict[str, Any], category: str = "general") -> List[NewsArticle]:
        """Parse Perplexity API response into NewsArticle objects"""
        articles = []
        
        try:
            if "error" in response:
                logger.warning("API response contains error: %s", response['error'])
                return articles
            
            # Extract content from Perplexity response
            choices = response.get("choices", [])
            if not choices:
                return articles
            
            content = choices[0].get("message", {}).get("content", "")
            citations = response.get("citations", [])
            
            # Parse the structured response
            lines = content.split('\n')
            current_article = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for article markers
                if line.startswith('**') and line.endswith('**'):
                    # This might be a title
                    if current_article:
                        # Save previous article
                        article = self._create_article_from_dict(current_article, category)
                        if article:
                            articles.append(article)
                    current_article = {'title': line.strip('*')}
                elif line.startswith('Source:'):
                    current_article['source'] = line.replace('Source:', '').strip()
                elif line.startswith('Date:') or line.startswith('Published:'):
                    current_article['published_at'] = line.split(':', 1)[1].strip()
                elif len(line) > 50:  # Assume this is content
                    current_article['content'] = current_article.get('content', '') + ' ' + line
            
            # Don't forget the last article
            if current_article:
                article = self._create_article_from_dict(current_article, category)
                if article:
                    articles.append(article)
            
            # If parsing failed, create a single article from the entire response
            if not articles and content:
                articles.append(NewsArticle(
                    title="Latest Cryptocurrency News",
                    content=content[:500] + "..." if len(content) > 500 else content,
                    url=citations[0] if citations else "",
                    published_at=datetime.now().isoformat(),
                    source="Perplexity AI",
                    category=category,
                    symbols=self._extract_symbols(content)
                ))
                
        except Exception as e:
            logger.error("Error parsing news response: %s", str(e))
        
        return articles
    
    def _create_article_from_dict(self, article_dict: Dict[str, str], category: str) -> Optional[NewsArticle]:
        """Create NewsArticle from parsed dictionary"""
        try:
            title = article_dict.get('title', 'Untitled')
            content = article_dict.get('content', '')
            
            if not title or not content:
                return None
            
            return NewsArticle(
                title=title,
                content=content,
                url=article_dict.get('url', ''),
                published_at=article_dict.get('published_at', datetime.now().isoformat()),
                source=article_dict.get('source', 'Unknown'),
                category=category,
                symbols=self._extract_symbols(f"{title} {content}")
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
            pub_date = datetime.fromisoformat(article.published_at.replace('Z', '+00:00'))
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
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'BNB': 'Binance Coin',
            'XRP': 'Ripple',
            'ADA': 'Cardano',
            'SOL': 'Solana',
            'DOGE': 'Dogecoin',
            'DOT': 'Polkadot',
            'MATIC': 'Polygon',
            'AVAX': 'Avalanche'
        }
        
        full_name = symbol_names.get(symbol, symbol)
        query = f"{full_name} {symbol} cryptocurrency news price analysis market"
        
        response = await self._make_request(query, limit)
        articles = self._parse_news_response(response, "symbol_specific")
        
        # Filter articles that mention the symbol
        filtered_articles = []
        for article in articles:
            if symbol in article.symbols or symbol.lower() in article.title.lower() or symbol.lower() in article.content.lower():
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
            if any(word in content_lower for word in ['bullish', 'bull', 'surge', 'rally', 'pump', 'moon']):
                article.sentiment = "positive"
            elif any(word in content_lower for word in ['bearish', 'bear', 'crash', 'dump', 'decline', 'fall']):
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
    
    async def translate_articles_to_russian(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Translate articles to Russian using AI"""
        translated_articles = []
        
        for article in articles:
            try:
                # Translate title and content
                translated_title = await self.translator.translate_to_russian(article.title, self.session)
                translated_content = await self.translator.translate_to_russian(article.content, self.session)
                
                # Create new article with translations
                translated_article = NewsArticle(
                    title=article.title,
                    content=article.content,
                    url=article.url,
                    published_at=article.published_at,
                    source=article.source,
                    relevance_score=article.relevance_score,
                    sentiment=article.sentiment,
                    symbols=article.symbols,
                    category=article.category,
                    language="ru",
                    translated_title=translated_title,
                    translated_content=translated_content
                )
                
                translated_articles.append(translated_article)
                
            except Exception as e:
                logger.error("Failed to translate article: %s", str(e))
                # Keep original article if translation fails
                translated_articles.append(article)
        
        return translated_articles
    
    async def get_russian_crypto_news(self, limit: int = 10) -> List[NewsArticle]:
        """Get Russian cryptocurrency news specifically"""
        query = "российские криптовалютные новости Bitcoin Ethereum блокчейн цифровые активы Россия"
        
        try:
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency news analyst focusing on Russian crypto market. Provide recent, accurate news about cryptocurrency in Russia, Russian regulations, and Russian crypto market developments."
                    },
                    {
                        "role": "user", 
                        "content": f"Find the latest {limit} Russian cryptocurrency news articles. Include news about Russian crypto regulations, Russian crypto exchanges, Russian blockchain projects, and crypto adoption in Russia. Provide title, summary, source, and publication date for each article."
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_citations": True,
                "search_domain_filter": self.russian_sources,
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "day"
            }
            
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = self._parse_news_response(data, "russian_crypto")
                    
                    # Mark articles as Russian language
                    for article in articles:
                        article.language = "ru"
                        article.relevance_score = self._calculate_relevance_score(article, query)
                    
                    # Sort by relevance score
                    articles.sort(key=lambda x: x.relevance_score, reverse=True)
                    return articles[:limit]
                else:
                    error_text = await response.text()
                    logger.error("Russian news API error %d: %s", response.status, error_text)
                    return []
                    
        except Exception as e:
            logger.error("Russian news request failed: %s", str(e))
            return []
    
    async def get_latest_news_russian(self, limit: int = 10) -> List[NewsArticle]:
        """Get latest news and translate to Russian"""
        articles = await self.get_latest_news(limit)
        return await self.translate_articles_to_russian(articles)
    
    async def get_news_by_category_russian(self, category: str, limit: int = 10) -> List[NewsArticle]:
        """Get news by category and translate to Russian"""
        if category not in self.categories:
            logger.warning("Unknown category: %s", category)
            return await self.get_latest_news_russian(limit)
        
        # Use Russian category query
        russian_query = self.russian_categories.get(category, self.categories[category])
        
        try:
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency news analyst. Provide recent, accurate news about cryptocurrency markets, blockchain technology, and digital assets in Russian language."
                    },
                    {
                        "role": "user", 
                        "content": f"Найдите последние {limit} новостных статей о: {russian_query}. Включите заголовок, краткое содержание, источник и дату публикации для каждой статьи."
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_citations": True,
                "search_domain_filter": ["coindesk.com", "cointelegraph.com", "decrypt.co", "theblock.co"] + self.russian_sources,
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "day"
            }
            
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = self._parse_news_response(data, category)
                    
                    # Mark articles as Russian and calculate relevance
                    for article in articles:
                        article.language = "ru"
                        article.relevance_score = self._calculate_relevance_score(article, russian_query)
                    
                    # Sort by relevance score
                    articles.sort(key=lambda x: x.relevance_score, reverse=True)
                    return articles[:limit]
                else:
                    error_text = await response.text()
                    logger.error("Russian category news API error %d: %s", response.status, error_text)
                    return []
                    
        except Exception as e:
            logger.error("Russian category news request failed: %s", str(e))
            return []
    
    async def enhance_articles_with_images(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Enhance articles with images from multiple sources"""
        if not self.enable_images or not self.image_enhancer:
            return articles
        
        try:
            enhanced_articles = []
            for article in articles:
                # Get images for this article
                images = await self.image_enhancer.enhance_article_with_images(article, max_images=5)
                
                # Update article with images
                article.images = images
                article.image_count = len(images)
                article.has_images = len(images) > 0
                
                enhanced_articles.append(article)
                
            logger.info(f"Enhanced {len(enhanced_articles)} articles with images")
            return enhanced_articles
            
        except Exception as e:
            logger.error(f"Error enhancing articles with images: {str(e)}")
            return articles
    
    async def get_news_by_symbol_russian(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Get news for a specific cryptocurrency symbol in Russian"""
        symbol = symbol.upper()
        
        # Map symbols to Russian names
        russian_symbol_names = {
            'BTC': 'Биткоин Bitcoin',
            'ETH': 'Эфириум Ethereum',
            'BNB': 'Бинанс Коин Binance Coin',
            'XRP': 'Риппл Ripple',
            'ADA': 'Кардано Cardano',
            'SOL': 'Солана Solana',
            'DOGE': 'Догикоин Dogecoin',
            'DOT': 'Полкадот Polkadot',
            'MATIC': 'Полигон Polygon',
            'AVAX': 'Авалanche Avalanche'
        }
        
        russian_name = russian_symbol_names.get(symbol, symbol)
        query = f"Russian cryptocurrency news {symbol} {russian_name} криптовалюты новости России регулирование блокчейн"
        
        try:
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency news analyst specializing in Russian crypto market. Provide recent, accurate news about specific cryptocurrencies in Russian language."
                    },
                    {
                        "role": "user", 
                        "content": f"Найдите последние {limit} новостных статей о: {query}. Включите заголовок, краткое содержание, источник и дату публикации для каждой статьи."
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_citations": True,
                "search_domain_filter": ["coindesk.com", "cointelegraph.com", "decrypt.co", "theblock.co"] + self.russian_sources,
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "day"
            }
            
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = self._parse_news_response(data, "symbol_specific_russian")
                    
                    # Filter and mark articles
                    filtered_articles = []
                    for article in articles:
                        if symbol in article.symbols or symbol.lower() in article.title.lower() or symbol.lower() in article.content.lower():
                            article.language = "ru"
                            article.relevance_score = self._calculate_relevance_score(article, query)
                            filtered_articles.append(article)
                    
                    # Sort by relevance score
                    filtered_articles.sort(key=lambda x: x.relevance_score, reverse=True)
                    return filtered_articles[:limit]
                else:
                    error_text = await response.text()
                    logger.error("Russian symbol news API error %d: %s", response.status, error_text)
                    return []
                    
        except Exception as e:
            logger.error("Russian symbol news request failed: %s", str(e))
            return []

# Convenience functions for quick access
async def get_crypto_news(category: str = "general", limit: int = 10, api_key: str = None) -> List[Dict[str, Any]]:
    """Quick function to get crypto news"""
    async with PerplexityNewsService(api_key) as service:
        if category == "general":
            articles = await service.get_latest_news(limit)
        else:
            articles = await service.get_news_by_category(category, limit)
        
        return [article.to_dict() for article in articles]

async def get_crypto_news_russian(category: str = "general", limit: int = 10, api_key: str = None) -> List[Dict[str, Any]]:
    """Quick function to get crypto news in Russian"""
    async with PerplexityNewsService(api_key) as service:
        if category == "general":
            articles = await service.get_latest_news_russian(limit)
        elif category == "russian_specific":
            articles = await service.get_russian_crypto_news(limit)
        else:
            articles = await service.get_news_by_category_russian(category, limit)
        
        return [article.to_dict() for article in articles]
