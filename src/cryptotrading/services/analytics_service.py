"""
Real Analytics Service - No mock data
Calculates real market analytics and insights
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..infrastructure.monitoring import get_business_metrics, get_logger, trace_context
from .market_service import MarketDataService
from .technical_indicators import TechnicalIndicatorsService

logger = get_logger("services.analytics")


class AnalyticsService:
    """Real market analytics service"""

    def __init__(self):
        self.market_service = MarketDataService()
        self.indicators_service = TechnicalIndicatorsService()
        self.business_metrics = get_business_metrics()

    async def get_market_analytics(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market analytics for a symbol"""
        start_time = time.time()

        with trace_context(f"analytics_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("service", "analytics")

                logger.info(f"Computing analytics for {symbol}")

                # Get historical data for analysis
                historical_data = await self.market_service.get_historical_data(symbol, 365)

                if not historical_data or not historical_data.get("data"):
                    raise ValueError(f"No historical data available for {symbol}")

                price_data = historical_data["data"]

                # Calculate technical indicators
                technical_analysis = self.indicators_service.calculate_all_indicators(price_data)

                # Calculate volatility metrics
                volatility_metrics = self._calculate_volatility_metrics(price_data)

                # Get trend analysis
                trend_analysis = self.indicators_service.get_trend_analysis(price_data)

                # Calculate support/resistance levels
                support_resistance = technical_analysis.get("indicators", {}).get(
                    "support_resistance", {}
                )

                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(price_data)

                # Get real-time price for current analysis
                current_price_data = await self.market_service.get_realtime_price(symbol)
                current_price = current_price_data.get("price", 0)

                analytics = {
                    "symbol": symbol.upper(),
                    "current_price": current_price,
                    "volatility": volatility_metrics,
                    "trend_analysis": trend_analysis,
                    "support_resistance": support_resistance,
                    "technical_indicators": technical_analysis.get("current", {}),
                    "performance_metrics": performance_metrics,
                    "data_period": {
                        "start_date": historical_data.get("start_date"),
                        "end_date": historical_data.get("end_date"),
                        "data_points": len(price_data),
                    },
                    "computed_at": datetime.utcnow().isoformat(),
                }

                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_data_processing(
                    source="analytics",
                    symbol=symbol,
                    records_processed=len(price_data),
                    success=True,
                    duration_ms=duration_ms,
                )

                span.set_attribute("success", "true")
                span.set_attribute("data_points", len(price_data))

                return analytics

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_data_processing(
                    source="analytics",
                    symbol=symbol,
                    records_processed=0,
                    success=False,
                    duration_ms=duration_ms,
                )

                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))

                logger.error(f"Analytics calculation failed for {symbol}: {e}")
                raise

    def _calculate_volatility_metrics(self, price_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate real volatility metrics"""
        if not price_data:
            return {}

        # Extract closing prices
        closes = [float(item.get("Close", item.get("price", 0))) for item in price_data]
        closes = [p for p in closes if p > 0]  # Filter out zeros

        if len(closes) < 2:
            return {}

        # Calculate returns
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]

        # Calculate different period volatilities
        volatility_metrics = {}

        # Daily volatility (last 30 days)
        if len(returns) >= 30:
            daily_returns = returns[-30:]
            volatility_metrics["daily"] = float(np.std(daily_returns))

        # Weekly volatility (last 12 weeks)
        if len(returns) >= 84:  # 12 weeks * 7 days
            weekly_returns = returns[-84:]
            weekly_vol = np.std(weekly_returns) * np.sqrt(7)  # Scale to weekly
            volatility_metrics["weekly"] = float(weekly_vol)

        # Monthly volatility (last 6 months)
        if len(returns) >= 180:  # ~6 months
            monthly_returns = returns[-180:]
            monthly_vol = np.std(monthly_returns) * np.sqrt(30)  # Scale to monthly
            volatility_metrics["monthly"] = float(monthly_vol)

        # Annualized volatility
        if len(returns) >= 252:  # 1 year of trading days
            annual_returns = returns[-252:]
            annual_vol = np.std(annual_returns) * np.sqrt(252)
            volatility_metrics["annualized"] = float(annual_vol)

        return volatility_metrics

    def _calculate_performance_metrics(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate real performance metrics"""
        if not price_data:
            return {}

        # Extract prices and dates
        prices = []
        dates = []

        for item in price_data:
            price = float(item.get("Close", item.get("price", 0)))
            if price > 0:
                prices.append(price)
                # Try to get date from various fields
                date_str = item.get("Date", item.get("date", item.get("timestamp", "")))
                if date_str:
                    try:
                        if isinstance(date_str, str):
                            date = pd.to_datetime(date_str)
                        else:
                            date = date_str
                        dates.append(date)
                    except:
                        dates.append(datetime.now())
                else:
                    dates.append(datetime.now())

        if len(prices) < 2:
            return {}

        # Calculate returns
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]

        performance = {}

        # Basic performance metrics
        performance["total_return"] = float((prices[-1] - prices[0]) / prices[0])
        performance["max_price"] = float(max(prices))
        performance["min_price"] = float(min(prices))
        performance["price_range"] = float(max(prices) - min(prices))

        # Calculate different period returns
        current_price = prices[-1]

        if len(prices) >= 7:  # 1 week
            week_ago_price = prices[-7]
            performance["return_1w"] = float((current_price - week_ago_price) / week_ago_price)

        if len(prices) >= 30:  # 1 month
            month_ago_price = prices[-30]
            performance["return_1m"] = float((current_price - month_ago_price) / month_ago_price)

        if len(prices) >= 90:  # 3 months
            quarter_ago_price = prices[-90]
            performance["return_3m"] = float(
                (current_price - quarter_ago_price) / quarter_ago_price
            )

        if len(prices) >= 252:  # 1 year
            year_ago_price = prices[-252]
            performance["return_1y"] = float((current_price - year_ago_price) / year_ago_price)

        # Risk metrics
        if returns:
            performance["avg_daily_return"] = float(np.mean(returns))
            performance["std_daily_return"] = float(np.std(returns))

            # Sharpe ratio (assuming 0% risk-free rate)
            if np.std(returns) > 0:
                performance["sharpe_ratio"] = float(
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )

            # Maximum drawdown
            peak = prices[0]
            max_drawdown = 0
            for price in prices:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            performance["max_drawdown"] = float(max_drawdown)

        return performance

    async def get_correlation_analysis(self, symbols: List[str], days: int = 90) -> Dict[str, Any]:
        """Calculate correlation matrix between symbols"""
        start_time = time.time()

        with trace_context("correlation_analysis") as span:
            try:
                span.set_attribute("symbols_count", len(symbols))
                span.set_attribute("days", days)

                logger.info(f"Computing correlation analysis for {len(symbols)} symbols")

                # Get historical data for all symbols
                symbol_data = {}
                for symbol in symbols:
                    try:
                        data = await self.market_service.get_historical_data(symbol, days)
                        if data and data.get("data"):
                            symbol_data[symbol] = data["data"]
                    except Exception as e:
                        logger.warning(f"Failed to get data for {symbol}: {e}")
                        continue

                if len(symbol_data) < 2:
                    raise ValueError("Need at least 2 symbols with valid data for correlation")

                # Create price matrix
                price_matrix = {}
                min_length = min(len(data) for data in symbol_data.values())

                for symbol, data in symbol_data.items():
                    prices = [
                        float(item.get("Close", item.get("price", 0)))
                        for item in data[-min_length:]
                    ]
                    prices = [p for p in prices if p > 0]
                    if len(prices) == min_length:
                        price_matrix[symbol] = prices

                # Calculate correlation matrix
                correlation_matrix = {}
                for symbol1 in price_matrix:
                    correlation_matrix[symbol1] = {}
                    for symbol2 in price_matrix:
                        if symbol1 == symbol2:
                            correlation_matrix[symbol1][symbol2] = 1.0
                        else:
                            corr = np.corrcoef(price_matrix[symbol1], price_matrix[symbol2])[0, 1]
                            correlation_matrix[symbol1][symbol2] = (
                                float(corr) if not np.isnan(corr) else 0.0
                            )

                duration_ms = (time.time() - start_time) * 1000

                span.set_attribute("success", "true")
                span.set_attribute("symbols_analyzed", len(price_matrix))

                return {
                    "correlation_matrix": correlation_matrix,
                    "symbols_analyzed": list(price_matrix.keys()),
                    "data_period_days": days,
                    "data_points": min_length,
                    "computed_at": datetime.utcnow().isoformat(),
                    "computation_time_ms": round(duration_ms, 2),
                }

            except Exception as e:
                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))

                logger.error(f"Correlation analysis failed: {e}")
                raise

    async def get_market_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market summary with real metrics"""
        start_time = time.time()

        try:
            logger.info(f"Computing market summary for {len(symbols)} symbols")

            # Get overview data
            overview_data = await self.market_service.get_market_overview(symbols)

            if not overview_data or not overview_data.get("data"):
                raise ValueError("No market data available for summary")

            market_data = overview_data["data"]

            # Calculate real market metrics
            prices = []
            volumes = []
            changes = []

            for symbol, data in market_data.items():
                price = data.get("price", 0)
                volume = data.get("volume", 0)
                change = data.get("change_24h", 0)

                if price > 0:
                    prices.append(price)
                if volume > 0:
                    volumes.append(volume)
                if change != 0:
                    changes.append(change)

            summary = {
                "total_symbols": len(symbols),
                "symbols_with_data": len(market_data),
                "market_metrics": {
                    "avg_price_change_24h": float(np.mean(changes)) if changes else 0,
                    "total_volume_24h": float(sum(volumes)) if volumes else 0,
                    "price_volatility": float(np.std(changes)) if len(changes) > 1 else 0,
                    "bullish_count": len([c for c in changes if c > 0]),
                    "bearish_count": len([c for c in changes if c < 0]),
                },
                "top_performers": self._get_top_performers(market_data, "change_24h", 5),
                "top_volume": self._get_top_performers(market_data, "volume", 5),
                "computed_at": datetime.utcnow().isoformat(),
            }

            duration_ms = (time.time() - start_time) * 1000
            summary["computation_time_ms"] = round(duration_ms, 2)

            return summary

        except Exception as e:
            logger.error(f"Market summary failed: {e}")
            raise

    def _get_top_performers(
        self, market_data: Dict[str, Any], metric: str, count: int
    ) -> List[Dict[str, Any]]:
        """Get top performers by metric"""
        performers = []

        for symbol, data in market_data.items():
            value = data.get(metric, 0)
            if value != 0:
                performers.append(
                    {"symbol": symbol, "value": float(value), "price": data.get("price", 0)}
                )

        # Sort by value descending
        def get_value(item):
            return item["value"]

        performers.sort(key=get_value, reverse=True)

        return performers[:count]
