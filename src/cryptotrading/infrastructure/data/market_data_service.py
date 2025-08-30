"""
Market Data Service - Database Integration
Provides market data functionality with real database integration
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
import sqlite3
# Use direct database queries instead of SQLAlchemy model due to schema mismatch
from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Market data point structure"""
    symbol: str
    price: float
    price_change_24h: float
    price_change_percentage_24h: float
    market_cap: float
    volume_24h: float
    last_updated: datetime

class MarketDataService:
    """Market data service with real database integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db = None
        
    async def __aenter__(self):
        self.db = UnifiedDatabase()
        await self.db.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            # Clean up database connections if needed
            pass
    
    async def get_market_data_dict(self, symbol: str) -> Dict[str, Any]:
        """Get market data as dictionary from database"""
        try:
            # Query database directly using the actual schema
            market_data = await self.db.get_latest_market_data(symbol.upper())
            
            if market_data:
                # Use close price as the current price
                price = float(market_data.get('close', 0))
                high = float(market_data.get('high', 0))
                low = float(market_data.get('low', 0))
                volume = float(market_data.get('volume', 0))
                
                # Calculate 24h change (simplified - would need historical data for accurate calculation)
                change_24h = 0.0  # Would need previous day's close price
                change_percent_24h = 0.0  # Would need previous day's close price
                
                return {
                    "symbol": market_data.get('symbol', symbol.upper()),
                    "price": price,
                    "volume_24h": volume,
                    "high_24h": high,
                    "low_24h": low,
                    "change_24h": change_24h,
                    "change_percent_24h": change_percent_24h,
                    "market_cap": 0.0,  # Not available in current schema
                    "last_updated": market_data.get('timestamp', datetime.utcnow().isoformat())
                }
            else:
                # Return error when no database record found - no placeholder data
                self.logger.error(f"No market data found for symbol {symbol} in database")
                raise ValueError(f"Market data not available for {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            raise e
    
    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate market data has required fields"""
        required_fields = ['symbol', 'price', 'volume_24h', 'high_24h', 'low_24h']
        return all(field in data for field in required_fields)
    
    async def get_multiple_symbols(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for multiple symbols from database"""
        result = {}
        try:
            # Get data for each symbol individually (can be optimized later)
            for symbol in symbols:
                result[symbol] = await self.get_market_data_dict(symbol)
                        
        except Exception as e:
            self.logger.error(f"Error fetching multiple market data: {e}")
            # Fallback to individual queries
            for symbol in symbols:
                result[symbol] = await self.get_market_data_dict(symbol)
                
        return result

    async def store_market_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Store market data to database"""
        try:
            # Store using the actual database schema (OHLC format)
            market_data_record = {
                'symbol': symbol.upper(),
                'source': 'api',
                'open': float(data.get('price', 0)),  # Use current price as open for now
                'high': float(data.get('high_24h', data.get('price', 0))),
                'low': float(data.get('low_24h', data.get('price', 0))),
                'close': float(data.get('price', 0)),
                'volume': float(data.get('volume_24h', 0)),
                'timestamp': datetime.fromisoformat(data['last_updated'].replace('Z', '+00:00')) if 'last_updated' in data else datetime.utcnow()
            }
            
            success = await self.db.store_market_data(symbol.upper(), market_data_record)
            
            if success:
                self.logger.info(f"Stored market data for {symbol}")
                return True
            else:
                self.logger.error(f"Failed to store market data for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing market data for {symbol}: {e}")
            return False

    async def get_trending_coins(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending coins from database"""
        try:
            # Query database directly for trending coins
            cursor = self.db.db_conn.cursor()
            try:
                cursor.execute("""
                    SELECT symbol, close as price, volume, high, low, timestamp
                    FROM market_data 
                    WHERE volume > 0
                    ORDER BY volume DESC, timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                
                return [
                    {
                        "symbol": row['symbol'] if hasattr(row, 'keys') else row[0],
                        "price": float(row['price'] if hasattr(row, 'keys') else row[1]),
                        "volume_24h": float(row['volume'] if hasattr(row, 'keys') else row[2]),
                        "change_percent_24h": 0.0,  # Would need historical data
                        "market_cap": 0.0,  # Not available in current schema
                        "last_updated": (row['timestamp'] if hasattr(row, 'keys') else row[5]) or datetime.utcnow().isoformat()
                    }
                    for row in rows
                ]
            finally:
                cursor.close()
                
        except Exception as e:
            self.logger.error(f"Error fetching trending coins: {e}")
            return []

# Global service instance
_market_data_service = None

async def get_market_data_service() -> MarketDataService:
    """Get or create market data service instance"""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service