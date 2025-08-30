"""
AI Gateway Client - Enhanced wrapper around Grok4Client
Provides backwards compatibility while leveraging real AI intelligence
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .grok4_client import Grok4Client, get_grok4_client

logger = logging.getLogger(__name__)


class AIGatewayClient:
    """
    Enhanced AI Gateway Client that wraps Grok4Client for real AI intelligence.
    Provides backwards compatibility while upgrading to advanced capabilities.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI Gateway Client with Grok4 backend

        Args:
            api_key: Optional API key for Grok4 (uses env vars if not provided)
        """
        self.api_key = api_key
        self._grok4_client: Optional[Grok4Client] = None
        self._loop = None

    def _get_loop(self):
        """Get or create event loop for async operations"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create new one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    def _run_async(self, coro):
        """Run async coroutine in sync context"""
        loop = self._get_loop()
        if loop.is_running():
            # If loop is already running, we need to use a different approach
            # This shouldn't happen in normal usage but let's handle it
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    async def _get_grok4_client(self) -> Grok4Client:
        """Get Grok4 client instance"""
        if self._grok4_client is None:
            if self.api_key:
                self._grok4_client = Grok4Client(api_key=self.api_key)
            else:
                self._grok4_client = await get_grok4_client()
        return self._grok4_client

    def analyze_market(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using Grok4's advanced sentiment analysis

        Args:
            data: Market data containing symbol and analysis parameters

        Returns:
            Enhanced market analysis with AI insights
        """

        async def _analyze():
            try:
                client = await self._get_grok4_client()
                symbol = data.get("symbol", "BTC")

                # Use Grok4's advanced market sentiment analysis
                insights = await client.analyze_market_sentiment(
                    symbols=[symbol], timeframe=data.get("timeframe", "1d")
                )

                # Also get market predictions for enhanced analysis
                predictions = await client.predict_market_movement(
                    symbols=[symbol], horizon=data.get("horizon", "1d")
                )

                # Combine insights and predictions into comprehensive analysis
                if insights:
                    insight = insights[0]  # First symbol
                    prediction = predictions.get(symbol, {})

                    return {
                        "symbol": symbol,
                        "recommendation": insight.recommendation,
                        "sentiment_score": insight.score,
                        "confidence": insight.confidence,
                        "risk_level": insight.risk_level,
                        "reasoning": insight.reasoning,
                        "prediction": {
                            "direction": prediction.get("direction"),
                            "magnitude": prediction.get("magnitude"),
                            "confidence": prediction.get("confidence"),
                            "key_factors": prediction.get("key_factors", []),
                            "risk_factors": prediction.get("risk_factors", []),
                        },
                        "analysis_type": "grok4_enhanced",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                else:
                    return {
                        "symbol": symbol,
                        "error": "Failed to get market insights",
                        "analysis_type": "grok4_enhanced",
                        "timestamp": datetime.utcnow().isoformat(),
                    }

            except Exception as e:
                logger.error(f"Grok4 market analysis failed: {e}")
                return {
                    "symbol": data.get("symbol", "Unknown"),
                    "error": str(e),
                    "analysis_type": "grok4_enhanced",
                    "timestamp": datetime.utcnow().isoformat(),
                }

        return self._run_async(_analyze())

    def generate_trading_strategy(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate advanced trading strategy using Grok4's strategy evaluation

        Args:
            user_profile: User profile with preferences and constraints

        Returns:
            Comprehensive trading strategy with risk assessment
        """

        async def _generate_strategy():
            try:
                client = await self._get_grok4_client()

                # Create strategy configuration from user profile
                strategy_config = {
                    "name": f"Personalized Strategy for {user_profile.get('user_id', 'User')}",
                    "risk_tolerance": user_profile.get("risk_tolerance", "medium"),
                    "investment_horizon": user_profile.get("investment_horizon", "1y"),
                    "preferred_assets": user_profile.get("preferred_assets", ["BTC", "ETH"]),
                    "max_position_size": user_profile.get("max_position_size", 0.1),
                    "stop_loss": user_profile.get("stop_loss", 0.05),
                    "take_profit": user_profile.get("take_profit", 0.15),
                    "portfolio_value": user_profile.get("portfolio_value", 10000),
                }

                # Get strategy evaluation from Grok4
                strategy_analysis = await client.evaluate_trading_strategy(strategy_config)

                # Get risk assessment for the profile
                portfolio = {
                    asset: strategy_config["portfolio_value"]
                    / len(strategy_config["preferred_assets"])
                    for asset in strategy_config["preferred_assets"]
                }

                risk_assessment = await client.assess_trading_risk(
                    portfolio=portfolio, market_conditions=user_profile.get("market_conditions")
                )

                return {
                    "strategy_name": strategy_analysis.strategy_name,
                    "user_id": user_profile.get("user_id"),
                    "configuration": strategy_config,
                    "performance_metrics": {
                        "expected_return": strategy_analysis.expected_return,
                        "risk_score": strategy_analysis.risk_score,
                        "sharpe_ratio": strategy_analysis.sharpe_ratio,
                        "max_drawdown": strategy_analysis.max_drawdown,
                        "win_rate": strategy_analysis.win_rate,
                    },
                    "recommendations": strategy_analysis.recommendations,
                    "risk_factors": strategy_analysis.risk_factors,
                    "risk_assessment": risk_assessment,
                    "confidence": strategy_analysis.confidence,
                    "strategy_type": "grok4_optimized",
                    "created_at": datetime.utcnow().isoformat(),
                }

            except Exception as e:
                logger.error(f"Grok4 strategy generation failed: {e}")
                return {
                    "user_id": user_profile.get("user_id"),
                    "error": str(e),
                    "strategy_type": "grok4_optimized",
                    "created_at": datetime.utcnow().isoformat(),
                }

        return self._run_async(_generate_strategy())

    def analyze_news_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze news sentiment using Grok4's advanced sentiment analysis

        Args:
            news_items: List of news articles with content

        Returns:
            Comprehensive sentiment analysis with market impact
        """

        async def _analyze_sentiment():
            try:
                client = await self._get_grok4_client()

                # Extract symbols mentioned in news
                symbols = set()
                news_content = []

                for item in news_items:
                    content = (
                        item.get("content", "")
                        or item.get("title", "")
                        or item.get("description", "")
                    )
                    news_content.append(content)

                    # Extract common crypto symbols from content
                    content_upper = content.upper()
                    common_symbols = ["BTC", "ETH", "ADA", "SOL", "DOT", "LINK", "MATIC", "AVAX"]
                    for symbol in common_symbols:
                        if symbol in content_upper:
                            symbols.add(symbol)

                # If no symbols found, default to major ones
                if not symbols:
                    symbols = {"BTC", "ETH"}

                symbols_list = list(symbols)

                # Get sentiment analysis for affected symbols
                sentiment_insights = await client.analyze_market_sentiment(symbols_list)

                # Aggregate sentiment scores
                total_sentiment = 0
                total_confidence = 0
                recommendations = {}
                risk_levels = {}

                for insight in sentiment_insights:
                    total_sentiment += insight.score
                    total_confidence += insight.confidence
                    recommendations[insight.symbol] = insight.recommendation
                    risk_levels[insight.symbol] = insight.risk_level

                avg_sentiment = (
                    total_sentiment / len(sentiment_insights) if sentiment_insights else 0.5
                )
                avg_confidence = (
                    total_confidence / len(sentiment_insights) if sentiment_insights else 0.5
                )

                # Determine overall market sentiment
                if avg_sentiment >= 0.7:
                    overall_sentiment = "BULLISH"
                elif avg_sentiment <= 0.3:
                    overall_sentiment = "BEARISH"
                else:
                    overall_sentiment = "NEUTRAL"

                return {
                    "overall_sentiment": overall_sentiment,
                    "sentiment_score": avg_sentiment,
                    "confidence": avg_confidence,
                    "affected_symbols": symbols_list,
                    "symbol_analysis": {
                        symbol: {
                            "recommendation": recommendations.get(symbol),
                            "risk_level": risk_levels.get(symbol),
                            "sentiment_score": next(
                                (
                                    insight.score
                                    for insight in sentiment_insights
                                    if insight.symbol == symbol
                                ),
                                avg_sentiment,
                            ),
                        }
                        for symbol in symbols_list
                    },
                    "news_count": len(news_items),
                    "analysis_method": "grok4_enhanced",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            except Exception as e:
                logger.error(f"Grok4 news sentiment analysis failed: {e}")
                return {
                    "overall_sentiment": "UNKNOWN",
                    "sentiment_score": 0.5,
                    "confidence": 0.0,
                    "error": str(e),
                    "news_count": len(news_items),
                    "analysis_method": "grok4_enhanced",
                    "timestamp": datetime.utcnow().isoformat(),
                }

        return self._run_async(_analyze_sentiment())

    # Enhanced methods that expose Grok4's advanced capabilities

    def predict_market_movements(self, symbols: List[str], horizon: str = "1d") -> Dict[str, Any]:
        """
        Predict market movements using Grok4's prediction engine

        Args:
            symbols: Symbols to analyze
            horizon: Prediction horizon

        Returns:
            Market movement predictions
        """

        async def _predict():
            try:
                client = await self._get_grok4_client()
                return await client.predict_market_movement(symbols, horizon)
            except Exception as e:
                logger.error(f"Grok4 market prediction failed: {e}")
                return {"error": str(e)}

        return self._run_async(_predict())

    def analyze_correlations(self, symbols: List[str], timeframe: str = "1d") -> Dict[str, Any]:
        """
        Analyze correlation patterns using Grok4's correlation analysis

        Args:
            symbols: Symbols to analyze
            timeframe: Analysis timeframe

        Returns:
            Correlation analysis results
        """

        async def _analyze_correlations():
            try:
                client = await self._get_grok4_client()
                return await client.analyze_correlation_patterns(symbols, timeframe)
            except Exception as e:
                logger.error(f"Grok4 correlation analysis failed: {e}")
                return {"error": str(e)}

        return self._run_async(_analyze_correlations())

    def close(self):
        """Close the client and cleanup resources"""
        if self._grok4_client:
            try:
                loop = self._get_loop()
                if not loop.is_closed():
                    loop.run_until_complete(self._grok4_client.close())
            except Exception as e:
                logger.warning(f"Error closing Grok4 client: {e}")
            finally:
                self._grok4_client = None

        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None
