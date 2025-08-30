"""
Grok4 AI Client for Market Analysis and Trading Insights
Provides intelligent market analysis, sentiment scoring, and trading recommendations
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class Grok4Error(Exception):
    """Base exception for Grok4 client errors"""

    pass


class Grok4APIError(Grok4Error):
    """API request failed with error status"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Grok4 API error ({status_code}): {message}")


class Grok4ParseError(Grok4Error):
    """Failed to parse Grok4 response"""

    pass


class Grok4ConfigError(Grok4Error):
    """Configuration error (missing API key, etc)"""

    pass


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
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK4_API_KEY")
        self.base_url = base_url or os.getenv("GROK4_BASE_URL", "https://api.x.ai/v1")

        # Require API key for real AI
        if not self.api_key:
            raise Grok4ConfigError(
                "XAI_API_KEY or GROK4_API_KEY is required for real AI intelligence - no mock mode available"
            )

        # Configure HTTP client with best practices
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "CryptoTrading/1.0",
            },
            http2=True,  # Enable HTTP/2 for better performance
            follow_redirects=True,
        )

        # Cache for API responses
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    async def analyze_market_sentiment(
        self, symbols: List[str], timeframe: str = "1d"
    ) -> List[MarketInsight]:
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
                "model": "grok-4-0709",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency market analyst with deep knowledge of trading patterns, market sentiment, and risk assessment.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,  # Lower temperature for more consistent analysis
                "max_tokens": 2000,
            }

            response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON response from Grok4 (handle markdown code blocks)
                try:
                    # Try direct parsing first
                    analysis_data = json.loads(content)
                    return [self._parse_grok_insight(item) for item in analysis_data["insights"]]
                except json.JSONDecodeError:
                    # Try extracting JSON from markdown code blocks
                    try:
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_part = content[start:end]
                            analysis_data = json.loads(json_part)
                            return [
                                self._parse_grok_insight(item) for item in analysis_data["insights"]
                            ]
                        else:
                            raise ValueError("No JSON found in response")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse Grok4 JSON response: {content[:500]}...")
                        raise ValueError("Grok4 returned invalid JSON format")
            else:
                logger.error(f"Grok4 API error: {response.status_code} - {response.text}")
                raise RuntimeError(f"Grok4 API failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Grok4 sentiment analysis failed: {e}")
            raise

    async def assess_trading_risk(
        self, portfolio: Dict[str, float], market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess trading risk for portfolio and market conditions

        Args:
            portfolio: Current portfolio positions
            market_conditions: Market state information

        Returns:
            Risk assessment with recommendations
        """
        try:
            # Use real Grok4 chat API for risk assessment
            portfolio_summary = ", ".join([f"{k}: ${v:,.2f}" for k, v in portfolio.items()])
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
                "model": "grok-4-0709",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert portfolio risk analyst specializing in cryptocurrency trading and risk management.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,  # Lower temperature for consistent risk analysis
                "max_tokens": 1500,
            }

            response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON response from Grok4 (handle markdown code blocks)
                try:
                    # Try direct parsing first
                    risk_data = json.loads(content)
                    return risk_data
                except json.JSONDecodeError:
                    # Try extracting JSON from markdown code blocks
                    try:
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_part = content[start:end]
                            risk_data = json.loads(json_part)
                            return risk_data
                        else:
                            raise ValueError("No JSON found in response")
                    except json.JSONDecodeError:
                        logger.error(
                            f"Failed to parse Grok4 risk JSON response: {content[:500]}..."
                        )
                        raise ValueError("Grok4 returned invalid JSON format")
            else:
                logger.error(f"Grok4 risk assessment error: {response.status_code}")
                raise RuntimeError(f"Grok4 API failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Grok4 risk assessment failed: {e}")
            raise

    async def predict_market_movement(
        self, symbols: List[str], horizon: str = "1d"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Predict market movement for symbols

        Args:
            symbols: Trading symbols to analyze
            horizon: Prediction horizon (1h, 1d, 1w, 1m)

        Returns:
            Predictions with confidence scores
        """
        try:
            # Use real Grok4 chat API for market predictions
            prompt = f"""Predict market movement for these cryptocurrency symbols: {', '.join(symbols)}
            
            Prediction horizon: {horizon}
            
            For each symbol, analyze:
            - Current price trends and momentum
            - Volume patterns and market activity
            - Technical indicators and chart patterns
            - Market sentiment and news impact
            - Risk factors and potential catalysts
            
            Provide predictions with:
            1. Direction (UP/DOWN/SIDEWAYS)
            2. Confidence (0.0 to 1.0)
            3. Expected magnitude (percentage change)
            4. Key supporting factors
            5. Risk factors to monitor
            
            Respond in JSON format:
            {{
              "predictions": {{
                "BTC": {{
                  "direction": "UP",
                  "confidence": 0.75,
                  "magnitude": 0.08,
                  "key_factors": ["Institutional adoption", "Technical breakout"],
                  "risk_factors": ["Market volatility", "Regulatory uncertainty"]
                }}
              }}
            }}"""

            payload = {
                "model": "grok-4-0709",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency market analyst specializing in price prediction and trend analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1500,
            }

            response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                try:
                    # Try direct parsing first
                    prediction_data = json.loads(content)
                    return prediction_data["predictions"]
                except json.JSONDecodeError:
                    # Try extracting JSON from markdown code blocks
                    try:
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_part = content[start:end]
                            prediction_data = json.loads(json_part)
                            return prediction_data["predictions"]
                        else:
                            raise ValueError("No JSON found in response")
                    except json.JSONDecodeError:
                        logger.error(
                            f"Failed to parse Grok4 prediction JSON response: {content[:500]}..."
                        )
                        raise ValueError("Grok4 returned invalid JSON format")
            else:
                logger.error(f"Grok4 prediction error: {response.status_code}")
                raise RuntimeError(f"Grok4 API failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Grok4 market prediction failed: {e}")
            raise

    async def evaluate_trading_strategy(
        self, strategy_config: Dict[str, Any], historical_data: Optional[Dict[str, Any]] = None
    ) -> StrategyAnalysis:
        """
        Evaluate trading strategy performance and risk

        Args:
            strategy_config: Strategy configuration and parameters
            historical_data: Historical market data for backtesting

        Returns:
            Strategy analysis with performance metrics
        """
        try:
            # Use real Grok4 chat API for strategy evaluation
            strategy_summary = json.dumps(strategy_config, indent=2)

            prompt = f"""Evaluate this cryptocurrency trading strategy:
            
            Strategy Configuration:
            {strategy_summary}
            
            Historical Data Available: {"Yes" if historical_data else "No"}
            
            Please provide a comprehensive analysis including:
            1. Expected annual return (as decimal, e.g., 0.15 for 15%)
            2. Risk score (0.0 = very low risk, 1.0 = very high risk)
            3. Estimated Sharpe ratio
            4. Expected maximum drawdown (as decimal)
            5. Estimated win rate (as decimal)
            6. Specific recommendations for improvement
            7. Key risk factors to monitor
            8. Confidence in your evaluation (0.0 to 1.0)
            
            Respond in JSON format:
            {{
              "strategy_name": "{strategy_config.get('name', 'Trading Strategy')}",
              "expected_return": 0.15,
              "risk_score": 0.4,
              "sharpe_ratio": 1.2,
              "max_drawdown": 0.12,
              "win_rate": 0.65,
              "recommendations": ["Improve risk management", "Optimize position sizing"],
              "risk_factors": ["Market volatility exposure", "Correlation risk"],
              "confidence": 0.8
            }}"""

            payload = {
                "model": "grok-4-0709",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert quantitative analyst specializing in cryptocurrency trading strategy evaluation and risk assessment.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1500,
            }

            response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                try:
                    # Try direct parsing first
                    strategy_data = json.loads(content)
                    return self._parse_strategy_analysis(strategy_data)
                except json.JSONDecodeError:
                    # Try extracting JSON from markdown code blocks
                    try:
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_part = content[start:end]
                            strategy_data = json.loads(json_part)
                            return self._parse_strategy_analysis(strategy_data)
                        else:
                            raise ValueError("No JSON found in response")
                    except json.JSONDecodeError:
                        logger.error(
                            f"Failed to parse Grok4 strategy JSON response: {content[:500]}..."
                        )
                        raise ValueError("Grok4 returned invalid JSON format")
            else:
                logger.error(f"Grok4 strategy evaluation error: {response.status_code}")
                raise RuntimeError(f"Grok4 API failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Grok4 strategy evaluation failed: {e}")
            raise

    async def analyze_correlation_patterns(
        self, symbols: List[str], timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Analyze correlation patterns between symbols

        Args:
            symbols: Symbols to analyze
            timeframe: Analysis timeframe

        Returns:
            Correlation analysis with insights
        """
        try:
            # Use real Grok4 chat API for correlation analysis
            prompt = f"""Analyze correlation patterns between these cryptocurrency symbols: {', '.join(symbols)}
            
            Timeframe: {timeframe}
            
            Please provide a comprehensive correlation analysis including:
            1. Correlation matrix between all symbol pairs
            2. Identification of highest and lowest correlations
            3. Diversification score for the portfolio
            4. Market behavior clustering analysis
            5. Recommendations for portfolio diversification
            6. Risk factors related to correlation
            7. Confidence in your analysis
            
            Respond in JSON format:
            {{
              "correlation_matrix": {{
                "BTC": {{"BTC": 1.0, "ETH": 0.75}},
                "ETH": {{"BTC": 0.75, "ETH": 1.0}}
              }},
              "insights": {{
                "highest_correlation": {{"pair": "BTC-ETH", "correlation": 0.75}},
                "diversification_score": 0.65,
                "cluster_analysis": {{
                  "num_clusters": 2,
                  "cluster_stability": 0.8
                }}
              }},
              "recommendations": ["Consider adding uncorrelated assets", "Monitor correlation during market stress"],
              "confidence": 0.85
            }}"""

            payload = {
                "model": "grok-4-0709",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert quantitative analyst specializing in cryptocurrency correlation analysis and portfolio diversification.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1500,
            }

            response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                try:
                    # Try direct parsing first
                    correlation_data = json.loads(content)
                    return correlation_data
                except json.JSONDecodeError:
                    # Try extracting JSON from markdown code blocks
                    try:
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_part = content[start:end]
                            correlation_data = json.loads(json_part)
                            return correlation_data
                        else:
                            raise ValueError("No JSON found in response")
                    except json.JSONDecodeError:
                        logger.error(
                            f"Failed to parse Grok4 correlation JSON response: {content[:500]}..."
                        )
                        raise ValueError("Grok4 returned invalid JSON format")
            else:
                logger.error(f"Grok4 correlation analysis error: {response.status_code}")
                raise RuntimeError(f"Grok4 API failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Grok4 correlation analysis failed: {e}")
            raise

    def _parse_insight(self, data: Dict[str, Any]) -> MarketInsight:
        """Parse API response into MarketInsight object"""
        return MarketInsight(
            symbol=data["symbol"],
            analysis_type=AnalysisType(data["analysis_type"]),
            score=data["score"],
            recommendation=data["recommendation"],
            reasoning=data["reasoning"],
            risk_level=data["risk_level"],
            confidence=data["confidence"],
        )

    def _parse_grok_insight(self, data: Dict[str, Any]) -> MarketInsight:
        """Parse Grok4 chat response into MarketInsight object"""
        return MarketInsight(
            symbol=data["symbol"],
            analysis_type=AnalysisType.SENTIMENT,
            score=data["score"],
            recommendation=data["recommendation"],
            reasoning=data["reasoning"],
            risk_level=data["risk_level"],
            confidence=data["confidence"],
        )

    def _parse_strategy_analysis(self, data: Dict[str, Any]) -> StrategyAnalysis:
        """Parse strategy evaluation response"""
        return StrategyAnalysis(
            strategy_name=data["strategy_name"],
            expected_return=data["expected_return"],
            risk_score=data["risk_score"],
            sharpe_ratio=data["sharpe_ratio"],
            max_drawdown=data["max_drawdown"],
            win_rate=data["win_rate"],
            recommendations=data["recommendations"],
            risk_factors=data["risk_factors"],
            confidence=data["confidence"],
        )

    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensure cleanup"""
        await self.close()


# Singleton instance with thread-safe initialization
import asyncio
from typing import Optional

_grok4_client: Optional[Grok4Client] = None
_grok4_lock = asyncio.Lock()


async def get_grok4_client() -> Grok4Client:
    """
    Get singleton Grok4 client instance with thread-safe initialization.

    Returns:
        Grok4Client: Singleton instance of the Grok4 client

    Raises:
        ValueError: If GROK4_API_KEY is not set
    """
    global _grok4_client

    if _grok4_client is None:
        async with _grok4_lock:
            # Double-check pattern to prevent race conditions
            if _grok4_client is None:
                _grok4_client = Grok4Client()
                logger.info("Grok4Client singleton instance created")

    return _grok4_client


async def close_grok4_client():
    """
    Close and cleanup the singleton Grok4 client instance.
    Should be called during application shutdown.
    """
    global _grok4_client

    async with _grok4_lock:
        if _grok4_client is not None:
            await _grok4_client.close()
            _grok4_client = None
            logger.info("Grok4Client singleton instance closed")


class Grok4ClientFactory:
    """
    Factory for creating Grok4Client instances with dependency injection support.
    """

    @staticmethod
    def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Grok4Client:
        """
        Create a new Grok4Client instance.

        Args:
            api_key: Optional API key override
            base_url: Optional base URL override

        Returns:
            Grok4Client: New client instance
        """
        return Grok4Client(api_key=api_key, base_url=base_url)

    @staticmethod
    async def create_managed_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Create a Grok4Client instance for use with async context manager.

        Usage:
            async with Grok4ClientFactory.create_managed_client() as client:
                result = await client.analyze_market_sentiment(['BTC', 'ETH'])
        """
        return Grok4Client(api_key=api_key, base_url=base_url)
