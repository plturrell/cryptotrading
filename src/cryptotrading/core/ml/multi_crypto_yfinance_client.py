"""
Multi-Cryptocurrency Yahoo Finance Client
Handles all 10 main crypto trading pairs with unified interface
"""

import yfinance as yf
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MultiCryptoYFinanceClient:
    """Yahoo Finance client for top 10 crypto trading pairs"""
    
    # Top 10 crypto trading pairs with Yahoo Finance symbols
    SUPPORTED_PAIRS = {
        'BTC-USD': {'name': 'Bitcoin', 'symbol': 'BTC-USD'},
        'ETH-USD': {'name': 'Ethereum', 'symbol': 'ETH-USD'},
        'SOL-USD': {'name': 'Solana', 'symbol': 'SOL-USD'},
        'BNB-USD': {'name': 'BNB', 'symbol': 'BNB-USD'},
        'XRP-USD': {'name': 'XRP', 'symbol': 'XRP-USD'},
        'ADA-USD': {'name': 'Cardano', 'symbol': 'ADA-USD'},
        'DOGE-USD': {'name': 'Dogecoin', 'symbol': 'DOGE-USD'},
        'MATIC-USD': {'name': 'Polygon', 'symbol': 'MATIC-USD'}
    }
    
    def __init__(self):
        self.tickers = {}
        self._ticker_info_cache = {}
        logger.info(f"Multi-crypto YFinance client initialized for {len(self.SUPPORTED_PAIRS)} pairs")
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create ticker instance for symbol"""
        if symbol not in self.tickers:
            self.tickers[symbol] = yf.Ticker(symbol)
        return self.tickers[symbol]
    
    def normalize_symbol(self, symbol: str) -> Optional[str]:
        """Normalize symbol to Yahoo Finance format"""
        symbol_upper = symbol.upper()
        
        # Direct match
        if symbol_upper in self.SUPPORTED_PAIRS:
            return symbol_upper
        
        # Common format conversions
        conversions = {
            'BTC': 'BTC-USD', 'BTCUSDT': 'BTC-USD',
            'ETH': 'ETH-USD', 'ETHUSDT': 'ETH-USD',
            'SOL': 'SOL-USD', 'SOLUSDT': 'SOL-USD',
            'BNB': 'BNB-USD', 'BNBUSDT': 'BNB-USD',
            'XRP': 'XRP-USD', 'XRPUSDT': 'XRP-USD',
            'ADA': 'ADA-USD', 'ADAUSDT': 'ADA-USD',
            'DOGE': 'DOGE-USD', 'DOGEUSDT': 'DOGE-USD',
            'MATIC': 'MATIC-USD', 'MATICUSDT': 'MATIC-USD'
        }
        
        return conversions.get(symbol_upper)
    
    def get_historical_data(
        self, 
        symbol: str,
        days_back: int = 365,
        interval: str = '1d',
        prepost: bool = False,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data for any supported crypto pair
        
        Args:
            symbol: Crypto symbol (BTC, ETH, BTC-USD, etc.)
            days_back: Number of days of historical data
            interval: Data interval (1d, 1h, 5m, etc.)
            prepost: Include pre/post market data
            auto_adjust: Auto adjust for splits/dividends
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            if not normalized_symbol:
                raise ValueError(f"Symbol {symbol} not supported. Supported: {list(self.SUPPORTED_PAIRS.keys())}")
            
            ticker = self.get_ticker(normalized_symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Fetching {normalized_symbol} data from {start_date} to {end_date}")
            
            # Download historical data
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost,
                actions=True,
                raise_errors=True
            )
            
            if hist_data.empty:
                logger.warning(f"No data returned for {normalized_symbol}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(hist_data)} records for {normalized_symbol}")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    def get_ticker_info(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get detailed ticker information
        
        Args:
            symbol: Crypto symbol
            force_refresh: Force refresh of cached info
            
        Returns:
            Dictionary with ticker metadata
        """
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            if not normalized_symbol:
                raise ValueError(f"Symbol {symbol} not supported")
            
            if normalized_symbol not in self._ticker_info_cache or force_refresh:
                ticker = self.get_ticker(normalized_symbol)
                self._ticker_info_cache[normalized_symbol] = ticker.info
                logger.info(f"Fetched ticker info for {normalized_symbol}")
            
            return self._ticker_info_cache[normalized_symbol]
            
        except Exception as e:
            logger.error(f"Error fetching ticker info for {symbol}: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            info = self.get_ticker_info(symbol)
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            
            if price is None:
                # Fallback to latest historical close
                hist = self.get_historical_data(symbol, days_back=5)
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            return price
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data for symbol"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            if not normalized_symbol:
                return {"error": f"Symbol {symbol} not supported"}
            
            info = self.get_ticker_info(symbol)
            
            # Get recent historical data for calculations
            hist = self.get_historical_data(symbol, days_back=30)
            
            if hist.empty:
                return {"error": "No historical data available"}
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_24h = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            # Volume metrics
            volume_24h = hist['Volume'].iloc[-1]
            avg_volume_30d = hist['Volume'].mean()
            
            # Price metrics
            high_30d = hist['High'].max()
            low_30d = hist['Low'].min()
            
            return {
                "symbol": normalized_symbol,
                "name": self.SUPPORTED_PAIRS[normalized_symbol]["name"],
                "current_price": float(current_price),
                "previous_close": float(prev_close),
                "change_24h": float(change_24h),
                "volume_24h": float(volume_24h),
                "avg_volume_30d": float(avg_volume_30d),
                "high_30d": float(high_30d),
                "low_30d": float(low_30d),
                "market_cap": info.get('marketCap'),
                "currency": info.get('currency', 'USD'),
                "exchange": info.get('exchange', 'CCC'),
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_multiple_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.get_market_data(symbol)
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        return results
    
    def get_data_for_analysis(self, symbol: str, days_back: int = 365) -> Dict[str, Any]:
        """
        Get crypto data formatted for AI analysis
        
        Returns:
            Dictionary with data ready for analysis
        """
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            if not normalized_symbol:
                return {"error": f"Symbol {symbol} not supported"}
            
            hist = self.get_historical_data(symbol, days_back=days_back)
            
            if hist.empty:
                return {"error": "No data available"}
            
            # Reset index to include date
            hist_with_date = hist.reset_index()
            
            # Format dates as strings for JSON serialization
            hist_with_date['Date'] = hist_with_date['Date'].dt.strftime('%Y-%m-%d')
            
            # Get recent metrics
            recent_data = hist.tail(30)
            
            return {
                "symbol": normalized_symbol,
                "name": self.SUPPORTED_PAIRS[normalized_symbol]["name"],
                "data": hist_with_date.to_dict(orient='records'),
                "summary": {
                    "total_records": len(hist),
                    "date_range": {
                        "start": hist_with_date['Date'].iloc[0],
                        "end": hist_with_date['Date'].iloc[-1]
                    },
                    "price_range": {
                        "min": float(hist['Low'].min()),
                        "max": float(hist['High'].max()),
                        "current": float(hist['Close'].iloc[-1])
                    },
                    "volume_stats": {
                        "mean": float(hist['Volume'].mean()),
                        "max": float(hist['Volume'].max()),
                        "recent": float(hist['Volume'].iloc[-1])
                    },
                    "recent_trend": {
                        "sma_10": float(recent_data['Close'].rolling(10).mean().iloc[-1]),
                        "sma_20": float(recent_data['Close'].rolling(20).mean().iloc[-1]),
                        "price_change_30d": float((hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30] * 100) if len(hist) >= 30 else 0
                    }
                },
                "columns": list(hist_with_date.columns),
                "source": "Yahoo Finance",
                "interval": "1d"
            }
            
        except Exception as e:
            logger.error(f"Error preparing data for analysis: {e}")
            return {"error": str(e)}
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality of historical data"""
        try:
            total_rows = len(df)
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            completeness = 1 - (missing_counts.sum() / (total_rows * len(df.columns)))
            
            # Check for zero/negative prices
            price_cols = ['Open', 'High', 'Low', 'Close']
            invalid_prices = 0
            for col in price_cols:
                if col in df.columns:
                    invalid_prices += (df[col] <= 0).sum()
            
            # Check volume validity
            invalid_volume = (df['Volume'] < 0).sum() if 'Volume' in df.columns else 0
            
            # Check date consistency
            if df.index.name == 'Date' or 'Date' in df.columns:
                dates = pd.to_datetime(df.index if df.index.name == 'Date' else df['Date'])
                date_gaps = (dates.diff() > pd.Timedelta(days=1)).sum()
            else:
                date_gaps = 0
            
            accuracy = 1 - ((invalid_prices + invalid_volume) / (total_rows * 5))
            consistency = 1 - (date_gaps / total_rows) if total_rows > 0 else 1
            
            return {
                "total_rows": total_rows,
                "completeness": round(completeness, 3),
                "accuracy": round(accuracy, 3),
                "consistency": round(consistency, 3),
                "missing_values": missing_counts.to_dict(),
                "invalid_prices": invalid_prices,
                "invalid_volume": invalid_volume,
                "date_gaps": date_gaps,
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return {"error": str(e)}


# Singleton instance
_multi_crypto_client = None

def get_multi_crypto_client() -> MultiCryptoYFinanceClient:
    """Get singleton multi-crypto YFinance client instance"""
    global _multi_crypto_client
    if _multi_crypto_client is None:
        _multi_crypto_client = MultiCryptoYFinanceClient()
    return _multi_crypto_client