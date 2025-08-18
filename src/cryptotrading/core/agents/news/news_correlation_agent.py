"""
News Correlation Agent using STRANDS framework
Links news events to historical market data and identifies correlations
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ...infrastructure.monitoring import get_logger, trace_context
from ..strands import StrandsAgent, StrandConfig
from .news_collection_agent import NewsItem, NewsMarketCorrelation
from ....services.market_service import MarketDataService

logger = get_logger("agents.news_correlation")


@dataclass
class NewsEventImpact:
    """Impact of a news event on market price"""
    news_id: str
    symbol: str
    news_timestamp: datetime
    price_before_1h: float
    price_before_4h: float
    price_before_24h: float
    price_after_1h: float
    price_after_4h: float
    price_after_24h: float
    volume_before_24h: float
    volume_after_24h: float
    immediate_impact: float  # 1h price change
    short_term_impact: float  # 4h price change
    long_term_impact: float  # 24h price change
    volume_impact: float
    correlation_strength: float
    significance_level: float


@dataclass
class CorrelationPattern:
    """Pattern of news-market correlation"""
    pattern_type: str
    keywords: List[str]
    typical_impact: float
    impact_timeframe: str
    occurrence_count: int
    confidence_score: float
    examples: List[str]


class NewsCorrelationAgent(StrandsAgent):
    """
    STRANDS agent for correlating news events with market movements
    Uses historical data analysis to identify patterns and impacts
    """
    
    def __init__(self, config: Optional[StrandConfig] = None):
        self.agent_id = "news_correlation_agent"
        self.config = config or StrandConfig(
            name="NewsCorrelationAgent",
            description="Correlates news events with market movements using historical data",
            capabilities=["correlation_analysis", "pattern_recognition", "impact_assessment"],
            dependencies=["news_collection", "market_data", "mcp_tools", "technical_analysis"]
        )
        
        super().__init__(self.agent_id, self.config)
        
        # Initialize components
        self.market_service = MarketDataService()
        self.correlation_cache = {}
        self.pattern_cache = {}
        
        # Analysis parameters
        self.analysis_timeframes = [
            timedelta(hours=1),
            timedelta(hours=4), 
            timedelta(hours=24)
        ]
        
    async def initialize(self) -> bool:
        """Initialize the news correlation agent"""
        try:
            with trace_context("correlation_agent_init") as span:
                span.set_attribute("agent_id", self.agent_id)
                
                logger.info("Initializing News Correlation Agent")
                
                # Initialize database schemas for correlation data
                await self._initialize_correlation_schemas()
                
                # Load existing correlation patterns
                await self._load_correlation_patterns()
                
                logger.info("News Correlation Agent initialized successfully")
                span.set_attribute("initialization", "success")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize News Correlation Agent: {e}")
            return False
    
    async def _initialize_correlation_schemas(self):
        """Initialize database schemas for correlation analysis"""
        try:
            from ....infrastructure.database.unified_database import UnifiedDatabase
            
            db = UnifiedDatabase()
            await db.initialize()
            
            cursor = db.db_conn.cursor()
            
            # News event impacts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_event_impacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    news_timestamp TIMESTAMP,
                    price_before_1h REAL,
                    price_before_4h REAL,
                    price_before_24h REAL,
                    price_after_1h REAL,
                    price_after_4h REAL,
                    price_after_24h REAL,
                    volume_before_24h REAL,
                    volume_after_24h REAL,
                    immediate_impact REAL,
                    short_term_impact REAL,
                    long_term_impact REAL,
                    volume_impact REAL,
                    correlation_strength REAL,
                    significance_level REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Correlation patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS correlation_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    keywords TEXT,
                    typical_impact REAL,
                    impact_timeframe TEXT,
                    occurrence_count INTEGER,
                    confidence_score REAL,
                    examples TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            db.db_conn.commit()
            cursor.close()
            
            logger.info("Correlation database schemas initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize correlation schemas: {e}")
    
    async def _load_correlation_patterns(self):
        """Load existing correlation patterns from database"""
        try:
            from ....infrastructure.database.unified_database import UnifiedDatabase
            
            db = UnifiedDatabase()
            cursor = db.db_conn.cursor()
            
            cursor.execute("SELECT * FROM correlation_patterns")
            patterns = cursor.fetchall()
            
            for pattern in patterns:
                pattern_key = f"{pattern[1]}_{pattern[4]}"  # pattern_type_timeframe
                self.pattern_cache[pattern_key] = CorrelationPattern(
                    pattern_type=pattern[1],
                    keywords=pattern[2].split(',') if pattern[2] else [],
                    typical_impact=pattern[3],
                    impact_timeframe=pattern[4],
                    occurrence_count=pattern[5],
                    confidence_score=pattern[6],
                    examples=pattern[7].split('|') if pattern[7] else []
                )
            
            cursor.close()
            logger.info(f"Loaded {len(patterns)} correlation patterns")
            
        except Exception as e:
            logger.error(f"Failed to load correlation patterns: {e}")
    
    async def analyze_news_market_correlation(self, symbol: str, 
                                            days_back: int = 30) -> Dict[str, Any]:
        """Analyze correlation between news and market movements for a symbol"""
        with trace_context(f"correlation_analysis_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("days_back", days_back)
                
                logger.info(f"Analyzing news-market correlation for {symbol} ({days_back} days)")
                
                # Get news items for the period
                news_items = await self._get_news_for_period(symbol, days_back)
                
                if not news_items:
                    logger.warning(f"No news found for {symbol} in last {days_back} days")
                    return {'symbol': symbol, 'correlations': [], 'message': 'No news data'}
                
                # Get historical market data for the same period
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days_back)
                
                market_data = await self.market_service.get_historical_data(symbol, days_back)
                
                if not market_data or not market_data.get('data'):
                    logger.warning(f"No market data found for {symbol}")
                    return {'symbol': symbol, 'correlations': [], 'message': 'No market data'}
                
                # Analyze correlation for each news item
                correlations = []
                for news_item in news_items:
                    correlation = await self._analyze_single_news_impact(
                        news_item, market_data['data'], symbol
                    )
                    if correlation:
                        correlations.append(correlation)
                
                # Calculate overall correlation statistics
                correlation_stats = self._calculate_correlation_statistics(correlations)
                
                # Identify correlation patterns
                patterns = await self._identify_correlation_patterns(correlations, symbol)
                
                # Store results
                await self._store_correlation_results(correlations, patterns)
                
                span.set_attribute("correlations_found", len(correlations))
                span.set_attribute("patterns_identified", len(patterns))
                
                result = {
                    'symbol': symbol,
                    'analysis_period_days': days_back,
                    'news_items_analyzed': len(news_items),
                    'correlations_found': len(correlations),
                    'correlation_statistics': correlation_stats,
                    'identified_patterns': [self._pattern_to_dict(p) for p in patterns],
                    'significant_correlations': [
                        self._correlation_to_dict(c) for c in correlations 
                        if c.significance_level > 0.7
                    ][:10],  # Top 10 significant correlations
                    'analyzed_at': datetime.utcnow().isoformat()
                }
                
                return result
                
            except Exception as e:
                span.set_attribute("error", str(e))
                logger.error(f"Failed to analyze correlation for {symbol}: {e}")
                return {'symbol': symbol, 'error': str(e)}
    
    async def _get_news_for_period(self, symbol: str, days_back: int) -> List[NewsItem]:
        """Get news items for a specific period from database"""
        try:
            from ....infrastructure.database.unified_database import UnifiedDatabase
            
            db = UnifiedDatabase()
            cursor = db.db_conn.cursor()
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            cursor.execute("""
                SELECT title, content, url, published_at, source, symbol,
                       sentiment_score, relevance_score, impact_level, keywords
                FROM news_items 
                WHERE symbol = ? AND published_at > ?
                ORDER BY published_at DESC
            """, (symbol, cutoff_date))
            
            rows = cursor.fetchall()
            cursor.close()
            
            news_items = []
            for row in rows:
                news_item = NewsItem(
                    title=row[0],
                    content=row[1],
                    url=row[2],
                    published_at=datetime.fromisoformat(row[3]) if row[3] else datetime.utcnow(),
                    source=row[4],
                    symbol=row[5],
                    sentiment_score=row[6],
                    relevance_score=row[7],
                    impact_level=row[8],
                    keywords=row[9].split(',') if row[9] else []
                )
                news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Failed to get news for period: {e}")
            return []
    
    async def _analyze_single_news_impact(self, news_item: NewsItem, 
                                        market_data: List[Dict], 
                                        symbol: str) -> Optional[NewsEventImpact]:
        """Analyze the market impact of a single news event"""
        try:
            news_timestamp = news_item.published_at
            if not news_timestamp:
                return None
            
            # Convert market data to DataFrame for easier analysis
            df = pd.DataFrame(market_data)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                # If no date column, assume chronological order
                logger.warning("No date column found in market data")
                return None
            
            # Get price column
            price_col = 'Close' if 'Close' in df.columns else 'price'
            volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
            
            if price_col not in df.columns:
                logger.warning(f"No price column found in market data for {symbol}")
                return None
            
            # Find prices before and after news event
            prices_before = {}
            prices_after = {}
            
            for timeframe in self.analysis_timeframes:
                before_time = news_timestamp - timeframe
                after_time = news_timestamp + timeframe
                
                # Find closest price data points
                price_before = self._find_closest_price(df, before_time, price_col)
                price_after = self._find_closest_price(df, after_time, price_col)
                
                timeframe_key = f"{int(timeframe.total_seconds() / 3600)}h"
                prices_before[timeframe_key] = price_before
                prices_after[timeframe_key] = price_after
            
            # Calculate volume impact (24h before/after)
            volume_before = self._find_closest_price(df, news_timestamp - timedelta(hours=24), volume_col)
            volume_after = self._find_closest_price(df, news_timestamp + timedelta(hours=24), volume_col)
            
            # Calculate impacts
            immediate_impact = self._calculate_price_change(prices_before['1h'], prices_after['1h'])
            short_term_impact = self._calculate_price_change(prices_before['4h'], prices_after['4h'])
            long_term_impact = self._calculate_price_change(prices_before['24h'], prices_after['24h'])
            volume_impact = self._calculate_price_change(volume_before, volume_after)
            
            # Calculate correlation strength based on sentiment and impact alignment
            correlation_strength = self._calculate_correlation_strength(
                news_item.sentiment_score, immediate_impact, news_item.relevance_score
            )
            
            # Calculate significance level
            significance_level = self._calculate_significance_level(
                immediate_impact, short_term_impact, long_term_impact, 
                news_item.relevance_score, news_item.impact_level
            )
            
            impact = NewsEventImpact(
                news_id=f"{symbol}_{news_timestamp.isoformat()}",
                symbol=symbol,
                news_timestamp=news_timestamp,
                price_before_1h=prices_before['1h'],
                price_before_4h=prices_before['4h'],
                price_before_24h=prices_before['24h'],
                price_after_1h=prices_after['1h'],
                price_after_4h=prices_after['4h'],
                price_after_24h=prices_after['24h'],
                volume_before_24h=volume_before,
                volume_after_24h=volume_after,
                immediate_impact=immediate_impact,
                short_term_impact=short_term_impact,
                long_term_impact=long_term_impact,
                volume_impact=volume_impact,
                correlation_strength=correlation_strength,
                significance_level=significance_level
            )
            
            return impact
            
        except Exception as e:
            logger.error(f"Failed to analyze news impact: {e}")
            return None
    
    def _find_closest_price(self, df: pd.DataFrame, target_time: datetime, 
                           price_col: str) -> Optional[float]:
        """Find the closest price to a target time"""
        try:
            if df.empty:
                return None
            
            # Find the row with timestamp closest to target_time
            time_diff = abs(df.index - target_time)
            closest_idx = time_diff.idxmin()
            
            price = df.loc[closest_idx, price_col]
            return float(price) if price is not None else None
            
        except Exception as e:
            logger.warning(f"Failed to find closest price: {e}")
            return None
    
    def _calculate_price_change(self, price_before: Optional[float], 
                               price_after: Optional[float]) -> float:
        """Calculate percentage price change"""
        if price_before is None or price_after is None or price_before == 0:
            return 0.0
        
        return ((price_after - price_before) / price_before) * 100
    
    def _calculate_correlation_strength(self, sentiment_score: Optional[float], 
                                      price_impact: float, 
                                      relevance_score: Optional[float]) -> float:
        """Calculate correlation strength between sentiment and price movement"""
        if sentiment_score is None or relevance_score is None:
            return 0.0
        
        # Normalize sentiment score (-1 to 1, where 0.5 = neutral)
        normalized_sentiment = (sentiment_score - 0.5) * 2
        
        # Check if sentiment and price movement align
        if (normalized_sentiment > 0 and price_impact > 0) or \
           (normalized_sentiment < 0 and price_impact < 0):
            # Positive correlation
            correlation = abs(normalized_sentiment) * relevance_score
        else:
            # Negative correlation or misalignment
            correlation = -abs(normalized_sentiment) * relevance_score
        
        return round(correlation, 3)
    
    def _calculate_significance_level(self, immediate_impact: float, 
                                    short_term_impact: float,
                                    long_term_impact: float,
                                    relevance_score: Optional[float],
                                    impact_level: Optional[str]) -> float:
        """Calculate significance level of the correlation"""
        significance = 0.0
        
        # Impact magnitude
        max_impact = max(abs(immediate_impact), abs(short_term_impact), abs(long_term_impact))
        if max_impact > 5:  # >5% price change
            significance += 0.4
        elif max_impact > 2:  # >2% price change
            significance += 0.2
        
        # Relevance score
        if relevance_score:
            significance += relevance_score * 0.3
        
        # Impact level assessment
        impact_weights = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
        if impact_level in impact_weights:
            significance += impact_weights[impact_level]
        
        return min(significance, 1.0)  # Cap at 1.0
    
    def _calculate_correlation_statistics(self, correlations: List[NewsEventImpact]) -> Dict[str, Any]:
        """Calculate overall correlation statistics"""
        if not correlations:
            return {}
        
        immediate_impacts = [c.immediate_impact for c in correlations]
        short_term_impacts = [c.short_term_impact for c in correlations]
        long_term_impacts = [c.long_term_impact for c in correlations]
        correlation_strengths = [c.correlation_strength for c in correlations]
        significance_levels = [c.significance_level for c in correlations]
        
        stats = {
            'total_correlations': len(correlations),
            'average_immediate_impact': round(np.mean(immediate_impacts), 3),
            'average_short_term_impact': round(np.mean(short_term_impacts), 3),
            'average_long_term_impact': round(np.mean(long_term_impacts), 3),
            'average_correlation_strength': round(np.mean(correlation_strengths), 3),
            'average_significance_level': round(np.mean(significance_levels), 3),
            'significant_correlations_count': len([c for c in correlations if c.significance_level > 0.7]),
            'strong_correlations_count': len([c for c in correlations if abs(c.correlation_strength) > 0.5]),
            'positive_correlations_count': len([c for c in correlations if c.correlation_strength > 0]),
            'negative_correlations_count': len([c for c in correlations if c.correlation_strength < 0])
        }
        
        return stats
    
    async def _identify_correlation_patterns(self, correlations: List[NewsEventImpact], 
                                           symbol: str) -> List[CorrelationPattern]:
        """Identify patterns in news-market correlations"""
        patterns = []
        
        try:
            # Group correlations by impact level and timeframe
            pattern_groups = {}
            
            for correlation in correlations:
                # Create pattern key based on impact characteristics
                impact_magnitude = 'high' if abs(correlation.immediate_impact) > 3 else \
                                 'medium' if abs(correlation.immediate_impact) > 1 else 'low'
                impact_direction = 'positive' if correlation.immediate_impact > 0 else 'negative'
                
                pattern_key = f"{impact_magnitude}_{impact_direction}"
                
                if pattern_key not in pattern_groups:
                    pattern_groups[pattern_key] = []
                pattern_groups[pattern_key].append(correlation)
            
            # Create patterns from groups with sufficient data
            for pattern_key, group_correlations in pattern_groups.items():
                if len(group_correlations) >= 3:  # Minimum 3 occurrences for a pattern
                    
                    typical_impact = np.mean([c.immediate_impact for c in group_correlations])
                    confidence_score = len(group_correlations) / len(correlations)
                    
                    pattern = CorrelationPattern(
                        pattern_type=pattern_key,
                        keywords=[],  # Would extract from news content
                        typical_impact=round(typical_impact, 3),
                        impact_timeframe='1h',
                        occurrence_count=len(group_correlations),
                        confidence_score=round(confidence_score, 3),
                        examples=[c.news_id for c in group_correlations[:3]]
                    )
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to identify correlation patterns: {e}")
            return []
    
    async def _store_correlation_results(self, correlations: List[NewsEventImpact], 
                                       patterns: List[CorrelationPattern]):
        """Store correlation analysis results in database"""
        try:
            from ....infrastructure.database.unified_database import UnifiedDatabase
            
            db = UnifiedDatabase()
            cursor = db.db_conn.cursor()
            
            # Store event impacts
            for correlation in correlations:
                cursor.execute("""
                    INSERT INTO news_event_impacts 
                    (news_id, symbol, news_timestamp, price_before_1h, price_before_4h, 
                     price_before_24h, price_after_1h, price_after_4h, price_after_24h,
                     volume_before_24h, volume_after_24h, immediate_impact, short_term_impact,
                     long_term_impact, volume_impact, correlation_strength, significance_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    correlation.news_id, correlation.symbol, correlation.news_timestamp,
                    correlation.price_before_1h, correlation.price_before_4h, correlation.price_before_24h,
                    correlation.price_after_1h, correlation.price_after_4h, correlation.price_after_24h,
                    correlation.volume_before_24h, correlation.volume_after_24h,
                    correlation.immediate_impact, correlation.short_term_impact, correlation.long_term_impact,
                    correlation.volume_impact, correlation.correlation_strength, correlation.significance_level
                ))
            
            # Store or update patterns
            for pattern in patterns:
                cursor.execute("""
                    INSERT OR REPLACE INTO correlation_patterns 
                    (pattern_type, keywords, typical_impact, impact_timeframe, 
                     occurrence_count, confidence_score, examples)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_type, ','.join(pattern.keywords), pattern.typical_impact,
                    pattern.impact_timeframe, pattern.occurrence_count, 
                    pattern.confidence_score, '|'.join(pattern.examples)
                ))
            
            db.db_conn.commit()
            cursor.close()
            
            logger.info(f"Stored {len(correlations)} correlations and {len(patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to store correlation results: {e}")
    
    def _correlation_to_dict(self, correlation: NewsEventImpact) -> Dict[str, Any]:
        """Convert NewsEventImpact to dictionary"""
        return {
            'news_id': correlation.news_id,
            'symbol': correlation.symbol,
            'news_timestamp': correlation.news_timestamp.isoformat(),
            'immediate_impact': correlation.immediate_impact,
            'short_term_impact': correlation.short_term_impact,
            'long_term_impact': correlation.long_term_impact,
            'volume_impact': correlation.volume_impact,
            'correlation_strength': correlation.correlation_strength,
            'significance_level': correlation.significance_level
        }
    
    def _pattern_to_dict(self, pattern: CorrelationPattern) -> Dict[str, Any]:
        """Convert CorrelationPattern to dictionary"""
        return {
            'pattern_type': pattern.pattern_type,
            'keywords': pattern.keywords,
            'typical_impact': pattern.typical_impact,
            'impact_timeframe': pattern.impact_timeframe,
            'occurrence_count': pattern.occurrence_count,
            'confidence_score': pattern.confidence_score,
            'examples': pattern.examples
        }
    
    async def get_correlation_patterns_for_symbol(self, symbol: str) -> List[CorrelationPattern]:
        """Get identified correlation patterns for a specific symbol"""
        try:
            # This would query patterns specific to the symbol
            # For now, return cached patterns
            return list(self.pattern_cache.values())
            
        except Exception as e:
            logger.error(f"Failed to get correlation patterns for {symbol}: {e}")
            return []
    
    async def predict_news_impact(self, news_item: NewsItem) -> Dict[str, Any]:
        """Predict market impact of a news item based on historical patterns"""
        try:
            # Find matching patterns based on news characteristics
            matching_patterns = []
            
            for pattern in self.pattern_cache.values():
                # Simple pattern matching based on keywords and sentiment
                if any(keyword in news_item.title.lower() or 
                      keyword in news_item.content.lower() 
                      for keyword in pattern.keywords):
                    matching_patterns.append(pattern)
            
            if not matching_patterns:
                return {
                    'predicted_impact': 0.0,
                    'confidence': 0.0,
                    'message': 'No matching patterns found'
                }
            
            # Calculate weighted prediction based on pattern confidence
            total_weight = sum(p.confidence_score for p in matching_patterns)
            if total_weight == 0:
                return {
                    'predicted_impact': 0.0,
                    'confidence': 0.0,
                    'message': 'No reliable patterns'
                }
            
            weighted_impact = sum(p.typical_impact * p.confidence_score 
                                for p in matching_patterns) / total_weight
            
            prediction_confidence = min(total_weight, 1.0)
            
            return {
                'predicted_impact': round(weighted_impact, 3),
                'confidence': round(prediction_confidence, 3),
                'matching_patterns': len(matching_patterns),
                'impact_timeframe': '1h',
                'pattern_details': [self._pattern_to_dict(p) for p in matching_patterns]
            }
            
        except Exception as e:
            logger.error(f"Failed to predict news impact: {e}")
            return {
                'predicted_impact': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
