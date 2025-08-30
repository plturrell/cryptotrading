"""
Unified data provider with environment-aware real/mock switching
Handles both local development and Vercel production environments
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import random

from ..historical.yahoo_finance import YahooFinanceClient
from ...core.config.environment import get_data_source_config, is_vercel, get_feature_flags

logger = logging.getLogger(__name__)

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time price data"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get historical price data"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data"""
        pass

class MockDataProvider(DataProvider):
    """Mock data provider for development and testing"""
    
    def __init__(self):
        self.base_prices = {
            'BTC-USD': 45000,
            'ETH-USD': 3000,
            'ADA-USD': 0.5,
            'DOT-USD': 25,
            'LINK-USD': 15,
            'SOL-USD': 100,
            'AVAX-USD': 35,
            'MATIC-USD': 1.2
        }
        logger.info("MockDataProvider initialized for development")
    
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """Generate mock real-time price data"""
        base_price = self.base_prices.get(symbol, 100)
        
        # Add realistic price movement
        price_change = random.uniform(-0.05, 0.05)  # ±5% random movement
        current_price = base_price * (1 + price_change)
        
        volume_24h = random.randint(1000000, 10000000)
        
        return {
            'symbol': symbol,
            'price': round(current_price, 2),
            'price_change_24h': round(price_change * 100, 2),
            'volume_24h': volume_24h,
            'market_cap': current_price * random.randint(10000000, 100000000),
            'timestamp': datetime.now().isoformat(),
            'source': 'mock',
            'bid': round(current_price * 0.999, 2),
            'ask': round(current_price * 1.001, 2),
            'spread': round(current_price * 0.002, 4)
        }
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Generate mock historical data"""
        base_price = self.base_prices.get(symbol, 100)
        
        # Generate mock historical data
        days = 365 if period == "1y" else 30 if period == "1mo" else 7
        historical_data = []
        
        current_price = base_price
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            
            # Random walk price movement
            daily_change = random.uniform(-0.03, 0.03)  # ±3% daily movement
            current_price *= (1 + daily_change)
            
            high = current_price * random.uniform(1.01, 1.05)
            low = current_price * random.uniform(0.95, 0.99)
            volume = random.randint(500000, 5000000)
            
            historical_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(current_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(current_price, 2),
                'volume': volume
            })
        
        return {
            'symbol': symbol,
            'period': period,
            'data': historical_data,
            'source': 'mock'
        }
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive mock market data"""
        price_data = await self.get_real_time_price(symbol)
        
        # Add technical indicators (mock values)
        technical_indicators = {
            'rsi_14': random.uniform(30, 70),
            'macd_signal': random.uniform(-50, 50),
            'bollinger_upper': price_data['price'] * 1.02,
            'bollinger_lower': price_data['price'] * 0.98,
            'sma_20': price_data['price'] * random.uniform(0.98, 1.02),
            'ema_12': price_data['price'] * random.uniform(0.99, 1.01),
            'volume_sma': random.randint(2000000, 8000000)
        }
        
        return {
            **price_data,
            'technical_indicators': technical_indicators,
            'market_sentiment': random.choice(['bullish', 'bearish', 'neutral']),
            'fear_greed_index': random.randint(20, 80)
        }

class RealDataProvider(DataProvider):
    """Real data provider using Yahoo Finance and other APIs"""
    
    def __init__(self):
        self.yahoo_client = YahooFinanceClient()
        self.config = get_data_source_config()
        logger.info("RealDataProvider initialized with real APIs")
    
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time price from Yahoo Finance"""
        try:
            # Get current price from Yahoo Finance
            data = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.yahoo_client.get_current_price, 
                symbol
            )
            
            if data:
                return {
                    'symbol': symbol,
                    'price': data.get('price', 0),
                    'price_change_24h': data.get('change_percent', 0),
                    'volume_24h': data.get('volume', 0),
                    'market_cap': data.get('market_cap', 0),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'yahoo_finance',
                    'bid': data.get('bid', data.get('price', 0)),
                    'ask': data.get('ask', data.get('price', 0)),
                    'spread': abs(data.get('ask', 0) - data.get('bid', 0))
                }
            else:
                raise Exception("No data received from Yahoo Finance")
                
        except Exception as e:
            logger.error(f"Real-time price fetch failed for {symbol}: {e}")
            # NO FALLBACK TO MOCK DATA - Return error instead
            return {
                "error": f"Real-time price unavailable for {symbol}: {str(e)}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": None
            }
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get historical data from Yahoo Finance"""
        try:
            # Convert period to date range
            end_date = datetime.now()
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "1w":
                start_date = end_date - timedelta(days=7)
            else:
                start_date = end_date - timedelta(days=365)
            
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.yahoo_client.get_historical_data,
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if data is not None and not data.empty:
                historical_data = []
                for date, row in data.iterrows():
                    historical_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'open': float(row.get('Open', 0)),
                        'high': float(row.get('High', 0)),
                        'low': float(row.get('Low', 0)),
                        'close': float(row.get('Close', 0)),
                        'volume': int(row.get('Volume', 0))
                    })
                
                return {
                    'symbol': symbol,
                    'period': period,
                    'data': historical_data,
                    'source': 'yahoo_finance'
                }
            else:
                raise Exception("No historical data received")
                
        except Exception as e:
            logger.error(f"Historical data fetch failed for {symbol}: {e}")
            # NO FALLBACK TO MOCK DATA - Return error instead
            return {
                "error": f"Historical data unavailable for {symbol}: {str(e)}",
                "symbol": symbol,
                "period": period,
                "timestamp": datetime.now().isoformat(),
                "data": []
            }
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data"""
        try:
            # Get real-time price data
            price_data = await self.get_real_time_price(symbol)
            
            # Get recent historical data for technical indicators
            historical = await self.get_historical_data(symbol, "1mo")
            
            # Calculate basic technical indicators
            if historical['data']:
                closes = [float(d['close']) for d in historical['data'][-20:]]  # Last 20 days
                volumes = [int(d['volume']) for d in historical['data'][-20:]]
                
                technical_indicators = {
                    'sma_20': sum(closes) / len(closes) if closes else price_data['price'],
                    'volume_sma': sum(volumes) / len(volumes) if volumes else price_data['volume_24h'],
                    'price_change_20d': ((closes[-1] - closes[0]) / closes[0] * 100) if len(closes) > 1 else 0
                }
            else:
                technical_indicators = {}
            
            return {
                **price_data,
                'technical_indicators': technical_indicators,
                'historical_period': '1mo'
            }
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            # NO FALLBACK TO MOCK DATA - Return error instead
            return {
                "error": f"Market data unavailable for {symbol}: {str(e)}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": None,
                "indicators": {},
                "volume": None
            }

class UnifiedDataProvider:
    """Unified data provider - production-ready with real APIs only"""
    
    def __init__(self):
        self.config = get_data_source_config()
        self.flags = get_feature_flags()
        
        # Always use real data provider - no mock fallback
        self.provider = RealDataProvider()
        logger.info("UnifiedDataProvider using real APIs only - no mock data")
    
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time price data"""
        return await self.provider.get_real_time_price(symbol)
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get historical price data"""
        return await self.provider.get_historical_data(symbol, period)
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data"""
        return await self.provider.get_market_data(symbol)
    
    async def get_multiple_symbols(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for multiple symbols efficiently"""
        # Use parallel execution if available
        if self.flags.enable_parallel_processing and len(symbols) > 1:
            tasks = [self.get_market_data(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to get data for {symbols[i]}: {result}")
                    data[symbols[i]] = None
                else:
                    data[symbols[i]] = result
            
            return data
        else:
            # Sequential execution
            data = {}
            for symbol in symbols:
                try:
                    data[symbol] = await self.get_market_data(symbol)
                except Exception as e:
                    logger.error(f"Failed to get data for {symbol}: {e}")
                    data[symbol] = None
            
            return data
    
    def is_using_real_apis(self) -> bool:
        """Check if using real APIs"""
        return isinstance(self.provider, RealDataProvider)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider"""
        return {
            'provider_type': 'real' if self.is_using_real_apis() else 'mock',
            'environment': 'vercel' if is_vercel() else 'local',
            'config': self.config,
            'parallel_processing': self.flags.enable_parallel_processing,
            'max_workers': self.flags.max_workers
        }

# Global provider instance
_global_provider: Optional[UnifiedDataProvider] = None

def get_unified_data_provider() -> UnifiedDataProvider:
    """Get global unified data provider instance"""
    global _global_provider
    if _global_provider is None:
        _global_provider = UnifiedDataProvider()
    return _global_provider