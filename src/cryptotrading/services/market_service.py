"""
Market Data Service - Business Logic for Market Operations
Extracted from app.py for better modularity
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..data.historical.yahoo_finance import YahooFinanceClient
from ..data.providers.real_only_provider import RealOnlyDataProvider
from ..infrastructure.monitoring import get_business_metrics, get_logger, trace_context

logger = get_logger("services.market")


class MarketDataService:
    """Service for market data operations"""

    def __init__(self):
        self.yahoo_client = YahooFinanceClient()
        self.real_provider = RealOnlyDataProvider()
        self.business_metrics = get_business_metrics()

    async def get_realtime_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time price data for a symbol"""
        start_time = time.time()

        with trace_context(f"market_data_realtime_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("service", "market_data")

                logger.info(f"Fetching real-time price for {symbol}")

                # Try Yahoo Finance first
                data = self.yahoo_client.get_realtime_price(symbol)

                if data is None:
                    # Fallback to real provider
                    data = await self.real_provider.get_real_time_price(symbol)

                if data is None:
                    raise ValueError(f"No price data available for {symbol}")

                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_api_request("market_data", "GET", 200, duration_ms)

                span.set_attribute("price", str(data.get("price", 0)))
                span.set_attribute("success", "true")

                return data

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_api_request("market_data", "GET", 500, duration_ms)

                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))

                logger.error(f"Failed to fetch real-time price for {symbol}: {e}")
                raise

    async def get_market_overview(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market overview for multiple symbols"""
        start_time = time.time()

        with trace_context("market_overview") as span:
            try:
                span.set_attribute("symbols_count", len(symbols))

                results = {}
                for symbol in symbols:
                    try:
                        price_data = await self.real_provider.get_real_time_price(symbol)
                        if price_data:
                            results[symbol] = price_data
                    except Exception as e:
                        logger.warning(f"Failed to get price for {symbol}: {e}")
                        continue

                if not results:
                    raise ValueError("No market data available for any symbols")

                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_data_processing(
                    source="market_overview",
                    symbol=",".join(symbols),
                    records_processed=len(results),
                    success=True,
                    duration_ms=duration_ms,
                )

                return {
                    "symbols": list(results.keys()),
                    "data": results,
                    "timestamp": datetime.now().isoformat(),
                    "source": "real_provider",
                }

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_data_processing(
                    source="market_overview",
                    symbol=",".join(symbols),
                    records_processed=0,
                    success=False,
                    duration_ms=duration_ms,
                )
                logger.error(f"Market overview failed: {e}")
                raise

    async def get_historical_data(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get historical market data"""
        start_time = time.time()

        with trace_context(f"historical_data_{symbol}") as span:
            try:
                span.set_attribute("symbol", symbol)
                span.set_attribute("days", days)

                logger.info(f"Fetching {days} days of historical data for {symbol}")

                # Calculate date range
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

                # Download historical data
                df = self.yahoo_client.download_data(symbol, start_date, end_date, save=False)

                if df is None or df.empty:
                    raise ValueError(f"No historical data available for {symbol}")

                # Convert to JSON-serializable format
                import numpy as np
                import pandas as pd

                df_clean = df.reset_index()

                records = []
                for _, row in df_clean.iterrows():
                    record = {}
                    for col, value in row.items():
                        if pd.isna(value):
                            record[col] = None
                        elif isinstance(value, (pd.Timestamp, np.datetime64)):
                            record[col] = pd.to_datetime(value).strftime("%Y-%m-%d")
                        elif isinstance(value, np.integer):
                            record[col] = int(value)
                        elif isinstance(value, np.floating):
                            record[col] = float(value)
                        else:
                            record[col] = value
                    records.append(record)

                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_data_processing(
                    source="yahoo_finance",
                    symbol=symbol,
                    records_processed=len(df),
                    success=True,
                    duration_ms=duration_ms,
                )

                span.set_attribute("records_count", len(df))
                span.set_attribute("success", "true")

                return {
                    "symbol": symbol,
                    "days": days,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data": records,
                    "count": len(df),
                }

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.business_metrics.track_data_processing(
                    source="yahoo_finance",
                    symbol=symbol,
                    records_processed=0,
                    success=False,
                    duration_ms=duration_ms,
                )

                span.set_attribute("error", "true")
                span.set_attribute("error_message", str(e))

                logger.error(f"Failed to fetch historical data for {symbol}: {e}")
                raise
