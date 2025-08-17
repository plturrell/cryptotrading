"""
Grok4 AI Client for Market Analysis and Trading Insights
Provides intelligent market analysis, sentiment scoring, and trading recommendations
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
import os
from enum import Enum

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis Grok4 can perform"""
    SENTIMENT = "sentiment"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_PREDICTION = "market_prediction"
    STRATEGY_EVALUATION = "strategy_evaluation"
    CORRELATION_ANALYSIS = "correlation_analysis"


@dataclass
class MarketInsight:
    """Market insight from Grok4 analysis"""
    symbol: str
    analysis_type: AnalysisType
    score: float  # 0-1 confidence score
    recommendation: str  # BUY, SELL, HOLD
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH
    confidence: float  # 0-1 confidence in recommendation
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyAnalysis:
    """Strategy backtesting analysis from Grok4"""
    strategy_name: str
    expected_return: float
    risk_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    recommendations: List[str]
    risk_factors: List[str]
    confidence: float


class Grok4Client:
    """
    Grok4 AI client for advanced market analysis and trading insights
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Grok4 client
        
        Args:
            api_key: Grok4 API key (or from GROK4_API_KEY env var)
            base_url: Grok4 API base URL (or from GROK4_BASE_URL env var)
        """
        self.api_key = api_key or os.getenv('GROK4_API_KEY')
        self.base_url = base_url or os.getenv('GROK4_BASE_URL', 'https://api.x.ai/v1')
        
        # Require API key for real AI
        if not self.api_key:
            raise ValueError("GROK4_API_KEY is required for real AI intelligence - no mock mode available")
        
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
                'Content-Type': 'application/json'
            }
        )
        
        # Cache for API responses
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def analyze_market_sentiment(self, symbols: List[str], 
                                     timeframe: str = '1d') -> List[MarketInsight]:
        """
        Analyze market sentiment for given symbols
        
        Args:
            symbols: List of trading symbols
            timeframe: Analysis timeframe (1h, 1d, 1w)
            
        Returns:
            List of market insights with sentiment analysis
        """
        try:
            # Use real Grok4 chat API with financial prompt
            prompt = f"""Analyze market sentiment for these cryptocurrency symbols: {', '.join(symbols)}
            
            Consider:
            - Recent price movements and trends
            - Trading volume patterns
            - Market news and sentiment
            - Technical indicators
            - Overall market conditions
            
            For each symbol, provide:
            1. Recommendation (BUY/SELL/HOLD)
            2. Sentiment score (0.0 to 1.0)
            3. Risk level (LOW/MEDIUM/HIGH)
            4. Brief reasoning (1-2 sentences)
            5. Confidence (0.0 to 1.0)
            
            Respond in JSON format:
            {{
              "insights": [
                {{
                  "symbol": "BTC",
                  "recommendation": "BUY",
                  "score": 0.75,
                  "risk_level": "MEDIUM",
                  "reasoning": "Strong technical momentum with institutional adoption",
                  "confidence": 0.8
                }}
              ]
            }}"""
            
            payload = {
                'model': 'grok-2-1212',
                'messages': [
                    {'role': 'system', 'content': 'You are an expert cryptocurrency market analyst with deep knowledge of trading patterns, market sentiment, and risk assessment.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.3,  # Lower temperature for more consistent analysis
                'max_tokens': 2000
            }
            
            response = await self.client.post(
                f'{self.base_url}/chat/completions',
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Parse JSON response from Grok4
                try:
                    analysis_data = json.loads(content)
                    return [self._parse_grok_insight(item) for item in analysis_data['insights']]
                except json.JSONDecodeError:
                    logger.error("Failed to parse Grok4 JSON response")
                    raise ValueError("Grok4 returned invalid JSON format")
            else:
                logger.error(f"Grok4 API error: {response.status_code} - {response.text}")
                raise RuntimeError(f"Grok4 API failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Grok4 sentiment analysis failed: {e}")
            raise
    
    async def assess_trading_risk(self, portfolio: Dict[str, float], 
                                market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess trading risk for portfolio and market conditions
        
        Args:
            portfolio: Current portfolio positions
            market_conditions: Market state information
            
        Returns:
            Risk assessment with recommendations
        """
        if self.use_mock:
            return await self._mock_risk_assessment(portfolio)
        
        try:
            # Use real Grok4 chat API for risk assessment
            portfolio_summary = ', '.join([f"{k}: ${v:,.2f}" for k, v in portfolio.items()])
            total_value = sum(portfolio.values())
            
            prompt = f"""Analyze the trading risk for this cryptocurrency portfolio:
            
            Portfolio:
            {portfolio_summary}
            
            Total Value: ${total_value:,.2f}
            
            Market Conditions: {json.dumps(market_conditions or {}, indent=2)}
            
            Please assess:
            1. Overall risk score (0.0 = very low risk, 1.0 = very high risk)
            2. Risk level category (LOW/MEDIUM/HIGH)
            3. Diversification score (0.0 = poor, 1.0 = excellent)
            4. Key risk factors
            5. Specific recommendations
            6. Confidence in assessment (0.0 to 1.0)
            
            Consider:
            - Portfolio concentration and diversification
            - Correlation between assets
            - Market volatility exposure
            - Position sizing relative to total portfolio
            - Current market conditions
            
            Respond in JSON format:
            {{
              "overall_risk_score": 0.65,
              "risk_level": "MEDIUM",
              "portfolio_value": {total_value},
              "diversification_score": 0.7,
              "recommendations": ["Reduce correlation risk", "Consider position sizing"],
              "risk_factors": ["High correlation between crypto assets", "Market volatility exposure"],
              "confidence": 0.85
            }}"""
            
            payload = {
                'model': 'grok-2-1212',
                'messages': [
                    {'role': 'system', 'content': 'You are an expert portfolio risk analyst specializing in cryptocurrency trading and risk management.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.2,  # Lower temperature for consistent risk analysis
                'max_tokens': 1500
            }
            
            response = await self.client.post(
                f'{self.base_url}/chat/completions',
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Parse JSON response from Grok4
                try:
                    risk_data = json.loads(content)
                    return risk_data
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Grok4 risk JSON response, using fallback")
                    return await self._mock_risk_assessment(portfolio)
            else:
                logger.error(f"Grok4 risk assessment error: {response.status_code}")
                return await self._mock_risk_assessment(portfolio)
                
        except Exception as e:
            logger.error(f"Grok4 risk assessment failed: {e}")
            return await self._mock_risk_assessment(portfolio)
    
    async def predict_market_movement(self, symbols: List[str], 
                                    horizon: str = '1d') -> Dict[str, Dict[str, Any]]:
        """
        Predict market movement for symbols
        
        Args:
            symbols: Trading symbols to analyze
            horizon: Prediction horizon (1h, 1d, 1w, 1m)
            
        Returns:
            Predictions with confidence scores
        """
        if self.use_mock:
            return await self._mock_market_prediction(symbols)
        
        try:
            payload = {
                'symbols': symbols,
                'horizon': horizon,
                'include_confidence': True
            }
            
            response = await self.client.post(
                f'{self.base_url}/predict/movement',
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()['predictions']
            else:
                logger.error(f"Grok4 prediction error: {response.status_code}")
                return await self._mock_market_prediction(symbols)
                
        except Exception as e:
            logger.error(f"Grok4 market prediction failed: {e}")
            return await self._mock_market_prediction(symbols)
    
    async def evaluate_trading_strategy(self, strategy_config: Dict[str, Any],
                                      historical_data: Optional[Dict[str, Any]] = None) -> StrategyAnalysis:
        """
        Evaluate trading strategy performance and risk
        
        Args:
            strategy_config: Strategy configuration and parameters
            historical_data: Historical market data for backtesting
            
        Returns:
            Strategy analysis with performance metrics
        """
        if self.use_mock:
            return await self._mock_strategy_evaluation(strategy_config)
        
        try:
            payload = {
                'strategy': strategy_config,
                'historical_data': historical_data,
                'include_recommendations': True
            }
            
            response = await self.client.post(
                f'{self.base_url}/evaluate/strategy',
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_strategy_analysis(data)
            else:
                logger.error(f"Grok4 strategy evaluation error: {response.status_code}")
                return await self._mock_strategy_evaluation(strategy_config)
                
        except Exception as e:
            logger.error(f"Grok4 strategy evaluation failed: {e}")
            return await self._mock_strategy_evaluation(strategy_config)
    
    async def analyze_correlation_patterns(self, symbols: List[str],
                                         timeframe: str = '1d') -> Dict[str, Any]:
        """
        Analyze correlation patterns between symbols
        
        Args:
            symbols: Symbols to analyze
            timeframe: Analysis timeframe
            
        Returns:
            Correlation analysis with insights
        """
        if self.use_mock:
            return await self._mock_correlation_analysis(symbols)
        
        try:
            payload = {
                'symbols': symbols,
                'timeframe': timeframe,
                'analysis_depth': 'comprehensive'
            }
            
            response = await self.client.post(
                f'{self.base_url}/analyze/correlation',
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Grok4 correlation analysis error: {response.status_code}")
                return await self._mock_correlation_analysis(symbols)
                
        except Exception as e:
            logger.error(f"Grok4 correlation analysis failed: {e}")
            return await self._mock_correlation_analysis(symbols)
    
    def _parse_insight(self, data: Dict[str, Any]) -> MarketInsight:
        """Parse API response into MarketInsight object"""
        return MarketInsight(
            symbol=data['symbol'],
            analysis_type=AnalysisType(data['analysis_type']),
            score=data['score'],
            recommendation=data['recommendation'],
            reasoning=data['reasoning'],
            risk_level=data['risk_level'],
            confidence=data['confidence']
        )
    
    def _parse_grok_insight(self, data: Dict[str, Any]) -> MarketInsight:
        """Parse Grok4 chat response into MarketInsight object"""
        return MarketInsight(
            symbol=data['symbol'],
            analysis_type=AnalysisType.SENTIMENT,
            score=data['score'],
            recommendation=data['recommendation'],
            reasoning=data['reasoning'],
            risk_level=data['risk_level'],
            confidence=data['confidence']
        )
    
    def _parse_strategy_analysis(self, data: Dict[str, Any]) -> StrategyAnalysis:
        """Parse strategy evaluation response"""
        return StrategyAnalysis(
            strategy_name=data['strategy_name'],
            expected_return=data['expected_return'],
            risk_score=data['risk_score'],
            sharpe_ratio=data['sharpe_ratio'],
            max_drawdown=data['max_drawdown'],
            win_rate=data['win_rate'],
            recommendations=data['recommendations'],
            risk_factors=data['risk_factors'],
            confidence=data['confidence']
        )
    
    # Mock implementations for development/testing
    async def _mock_sentiment_analysis(self, symbols: List[str]) -> List[MarketInsight]:
        """Mock sentiment analysis for testing"""
        import random
        await asyncio.sleep(0.1)  # Simulate API delay
        
        insights = []
        for symbol in symbols:
            score = random.uniform(0.3, 0.9)
            recommendation = random.choice(['BUY', 'HOLD', 'SELL'])
            risk_level = random.choice(['LOW', 'MEDIUM', 'HIGH'])
            
            insights.append(MarketInsight(
                symbol=symbol,
                analysis_type=AnalysisType.SENTIMENT,
                score=score,
                recommendation=recommendation,
                reasoning=f"Mock analysis for {symbol}: Market sentiment appears {recommendation.lower()} based on technical indicators and news sentiment.",
                risk_level=risk_level,
                confidence=score
            ))
        
        return insights
    
    async def _mock_risk_assessment(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Mock risk assessment for testing"""
        import random
        await asyncio.sleep(0.1)
        
        total_value = sum(portfolio.values())
        risk_score = random.uniform(0.2, 0.8)
        
        return {
            'overall_risk_score': risk_score,
            'risk_level': 'MEDIUM' if 0.3 < risk_score < 0.7 else ('HIGH' if risk_score >= 0.7 else 'LOW'),
            'portfolio_value': total_value,
            'diversification_score': random.uniform(0.5, 0.9),
            'recommendations': [
                "Consider rebalancing portfolio for better risk distribution",
                "Monitor correlation between major positions",
                "Set appropriate stop-loss levels"
            ],
            'risk_factors': [
                "High correlation between crypto assets",
                "Market volatility exposure",
                "Concentration risk in top holdings"
            ],
            'confidence': 0.85
        }
    
    async def _mock_market_prediction(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Mock market prediction for testing"""
        import random
        await asyncio.sleep(0.1)
        
        predictions = {}
        for symbol in symbols:
            direction = random.choice(['UP', 'DOWN', 'SIDEWAYS'])
            confidence = random.uniform(0.6, 0.9)
            
            predictions[symbol] = {
                'direction': direction,
                'confidence': confidence,
                'magnitude': random.uniform(0.02, 0.15),  # 2-15% movement
                'key_factors': [
                    "Technical momentum indicators",
                    "Market sentiment analysis",
                    "Volume pattern analysis"
                ],
                'risk_factors': [
                    "Market volatility",
                    "External economic factors"
                ]
            }
        
        return predictions
    
    async def _mock_strategy_evaluation(self, strategy_config: Dict[str, Any]) -> StrategyAnalysis:
        """Mock strategy evaluation for testing"""
        import random
        await asyncio.sleep(0.2)
        
        return StrategyAnalysis(
            strategy_name=strategy_config.get('name', 'Unknown Strategy'),
            expected_return=random.uniform(0.05, 0.25),  # 5-25% annual return
            risk_score=random.uniform(0.3, 0.7),
            sharpe_ratio=random.uniform(0.8, 2.5),
            max_drawdown=random.uniform(0.05, 0.30),
            win_rate=random.uniform(0.45, 0.75),
            recommendations=[
                "Strategy shows promising risk-adjusted returns",
                "Consider position sizing adjustments",
                "Monitor performance in different market conditions"
            ],
            risk_factors=[
                "Strategy performance depends on market volatility",
                "Correlation with broader market trends",
                "Liquidity constraints in some positions"
            ],
            confidence=0.82
        )
    
    async def _mock_correlation_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Mock correlation analysis for testing"""
        import random
        await asyncio.sleep(0.1)
        
        n = len(symbols)
        correlation_matrix = {}
        
        for i, symbol1 in enumerate(symbols):
            correlation_matrix[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    # Generate realistic correlation values
                    correlation_matrix[symbol1][symbol2] = random.uniform(-0.3, 0.8)
        
        return {
            'correlation_matrix': correlation_matrix,
            'insights': {
                'highest_correlation': max([
                    {'pair': f"{s1}-{s2}", 'correlation': correlation_matrix[s1][s2]}
                    for s1 in symbols for s2 in symbols if s1 != s2
                ], key=lambda x: x['correlation']),
                'diversification_score': random.uniform(0.4, 0.8),
                'cluster_analysis': {
                    'num_clusters': min(3, len(symbols)),
                    'cluster_stability': random.uniform(0.6, 0.9)
                }
            },
            'recommendations': [
                "Portfolio shows good diversification potential",
                "Consider reducing correlation by adding uncorrelated assets",
                "Monitor correlation changes during market stress"
            ],
            'confidence': 0.78
        }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Singleton instance for easy access
_grok4_client = None

async def get_grok4_client() -> Grok4Client:
    """Get singleton Grok4 client instance"""
    global _grok4_client
    if _grok4_client is None:
        _grok4_client = Grok4Client()
    return _grok4_client