"""
AI Analysis Service - Business Logic for AI Operations
Extracted from app.py for better modularity
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from ..infrastructure.monitoring import get_logger, get_business_metrics, trace_context, ErrorSeverity, ErrorCategory
from ..core.ai import AIGatewayClient
from ..core.ml.perplexity import PerplexityClient

logger = get_logger("services.ai")


class AIAnalysisService:
    """Service for AI analysis operations"""
    
    def __init__(self):
        self.ai_gateway = AIGatewayClient()
        self.perplexity = PerplexityClient()
        self.business_metrics = get_business_metrics()
    
    async def analyze_market(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI market analysis using Claude-4-Sonnet"""
        start_time = time.time()
        symbol = data.get('symbol', 'BTC')
        
        with trace_context(f"ai_analysis_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("ai.model", "claude-4-sonnet")
                span.set_attribute("service", "ai_analysis")
                
                logger.info(f"Starting AI analysis for {symbol}")
                
                analysis = self.ai_gateway.analyze_market(data)
                
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="market_analysis",
                    model="claude-4-sonnet",
                    symbol=symbol,
                    success=True,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("success", "true")
                span.set_attribute("analysis_length", len(str(analysis)) if analysis else 0)
                
                logger.info(f"AI analysis completed for {symbol}")
                
                return {
                    'analysis': analysis,
                    'model': 'claude-4-sonnet',
                    'symbol': symbol,
                    'duration_ms': duration_ms,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="market_analysis",
                    model="claude-4-sonnet",
                    symbol=symbol,
                    success=False,
                    duration_ms=duration_ms
                )
                
                # Track error
                from ..infrastructure.monitoring import get_error_tracker
                error_tracker = get_error_tracker()
                error_tracker.track_error(e, severity=ErrorSeverity.HIGH, category=ErrorCategory.AI_ERROR)
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"AI analysis failed for {symbol}: {e}")
                raise
    
    async def get_crypto_news(self, symbol: str) -> Dict[str, Any]:
        """Get real-time crypto news via Perplexity"""
        start_time = time.time()
        
        with trace_context(f"crypto_news_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("service", "crypto_news")
                
                logger.info(f"Fetching crypto news for {symbol}")
                
                news = self.perplexity.search_crypto_news(symbol.upper())
                
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="news_search",
                    model="perplexity",
                    symbol=symbol,
                    success=True,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("success", "true")
                span.set_attribute("news_count", len(news.get('articles', [])) if isinstance(news, dict) else 0)
                
                return news
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="news_search",
                    model="perplexity",
                    symbol=symbol,
                    success=False,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"Crypto news fetch failed for {symbol}: {e}")
                raise
    
    async def generate_trading_strategy(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized trading strategy using Claude-4"""
        start_time = time.time()
        
        with trace_context("ai_strategy_generation") as span:
            try:
                span.set_attribute("service", "strategy_generation")
                span.set_attribute("ai.model", "claude-4")
                
                logger.info("Generating AI trading strategy")
                
                strategy = self.ai_gateway.generate_trading_strategy(user_profile)
                
                # Store in blob storage
                try:
                    from ..data.storage import put_json_blob
                    blob_result = put_json_blob(
                        f"strategies/user_{user_profile.get('user_id', 'anonymous')}.json",
                        strategy
                    )
                    strategy['storage_url'] = blob_result.get('url')
                except Exception as storage_error:
                    logger.warning(f"Failed to store strategy in blob storage: {storage_error}")
                
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="strategy_generation",
                    model="claude-4",
                    symbol="strategy",
                    success=True,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("success", "true")
                
                return strategy
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="strategy_generation",
                    model="claude-4",
                    symbol="strategy",
                    success=False,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"Strategy generation failed: {e}")
                raise
    
    async def analyze_news_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze news sentiment using Claude-4"""
        start_time = time.time()
        
        with trace_context("ai_sentiment_analysis") as span:
            try:
                span.set_attribute("service", "sentiment_analysis")
                span.set_attribute("news_count", len(news_items))
                
                logger.info(f"Analyzing sentiment for {len(news_items)} news items")
                
                sentiment = self.ai_gateway.analyze_news_sentiment(news_items)
                
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="sentiment_analysis",
                    model="claude-4",
                    symbol="sentiment",
                    success=True,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("success", "true")
                
                return sentiment
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_ai_operation(
                    operation="sentiment_analysis",
                    model="claude-4",
                    symbol="sentiment",
                    success=False,
                    duration_ms=duration_ms
                )
                
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"Sentiment analysis failed: {e}")
                raise
