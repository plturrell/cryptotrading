"""
Real Data Only Provider - NO MOCK DATA ALLOWED
Simple, direct, no fake data
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


class RealOnlyDataProvider:
    """Only real data - fails if data unavailable rather than faking it"""
    
    def __init__(self):
        logger.info("RealOnlyDataProvider initialized - NO MOCK DATA")
    
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time price or fail"""
        try:
            ticker = f"{symbol}-USD" if symbol in ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL'] else symbol
            data = yf.Ticker(ticker).info
            
            if 'regularMarketPrice' in data:
                return {
                    'symbol': symbol,
                    'price': data['regularMarketPrice'],
                    'volume': data.get('volume', 0),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'yahoo_finance'
                }
            else:
                raise ValueError(f"No price data for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            raise
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get real historical data or fail"""
        try:
            ticker = f"{symbol}-USD" if symbol in ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL'] else symbol
            data = yf.download(ticker, period=period, progress=False)
            
            if data.empty:
                raise ValueError(f"No historical data for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real market data for multiple symbols or fail"""
        results = {}
        
        for symbol in symbols:
            try:
                price = await self.get_real_time_price(symbol)
                results[symbol] = price
            except Exception as e:
                logger.error(f"Skipping {symbol}: {e}")
                # Don't include failed symbols
        
        if not results:
            raise ValueError("No market data available for any symbols")
        
        return results


# Single global instance
_provider = RealOnlyDataProvider()

def get_real_data_provider() -> RealOnlyDataProvider:
    """Get the real data provider - no mocks"""
    return _provider