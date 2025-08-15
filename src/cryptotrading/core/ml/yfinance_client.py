"""
Yahoo Finance Client for ETH Data
Handles all Yahoo Finance API interactions
"""

import yfinance as yf
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YFinanceClient:
    """Yahoo Finance client focused on ETH data retrieval"""
    
    def __init__(self):
        self.symbol = 'ETH-USD'
        self.ticker = None
        self._ticker_info = None
        logger.info("YFinance client initialized for ETH-USD")
    
    def get_ticker(self) -> yf.Ticker:
        """Get or create ticker instance"""
        if self.ticker is None:
            self.ticker = yf.Ticker(self.symbol)
        return self.ticker
    
    def get_historical_data(
        self, 
        days_back: int = 365,
        interval: str = '1d',
        prepost: bool = False,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Get historical ETH data from Yahoo Finance
        
        Args:
            days_back: Number of days of historical data
            interval: Data interval (1d, 1h, 5m, etc.)
            prepost: Include pre/post market data
            auto_adjust: Auto adjust for splits/dividends
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = self.get_ticker()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Fetching ETH data from {start_date} to {end_date}")
            
            # Download historical data
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost,
                actions=True,  # Include dividends and stock splits
                raise_errors=True
            )
            
            if hist_data.empty:
                logger.warning("No data returned from Yahoo Finance")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(hist_data)} records for ETH-USD")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def get_ticker_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get detailed ticker information
        
        Args:
            force_refresh: Force refresh of cached info
            
        Returns:
            Dictionary with ticker metadata
        """
        try:
            if self._ticker_info is None or force_refresh:
                ticker = self.get_ticker()
                self._ticker_info = ticker.info
                logger.info("Fetched ticker info for ETH-USD")
            
            return self._ticker_info
            
        except Exception as e:
            logger.error(f"Error fetching ticker info: {e}")
            return {}
    
    def get_current_price(self) -> Optional[float]:
        """Get current ETH price"""
        try:
            info = self.get_ticker_info()
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            
            if price is None:
                # Fallback to latest historical close
                hist = self.get_historical_data(days_back=5)
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            return price
            
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return None
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data for ETH"""
        try:
            info = self.get_ticker_info()
            
            # Get recent historical data for calculations
            hist = self.get_historical_data(days_back=30)
            
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
                "symbol": self.symbol,
                "name": info.get('longName', 'Ethereum USD'),
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
            logger.error(f"Error fetching market data: {e}")
            return {"error": str(e)}
    
    def get_data_for_analysis(self, days_back: int = 365) -> Dict[str, Any]:
        """
        Get ETH data formatted for AI analysis
        
        Returns:
            Dictionary with data ready for analysis
        """
        try:
            hist = self.get_historical_data(days_back=days_back)
            
            if hist.empty:
                return {"error": "No data available"}
            
            # Reset index to include date
            hist_with_date = hist.reset_index()
            
            # Format dates as strings for JSON serialization
            hist_with_date['Date'] = hist_with_date['Date'].dt.strftime('%Y-%m-%d')
            
            # Get recent metrics
            recent_data = hist.tail(30)
            
            return {
                "symbol": self.symbol,
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
        """
        Validate data quality of historical data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
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
_yfinance_client = None

def get_yfinance_client() -> YFinanceClient:
    """Get singleton YFinance client instance"""
    global _yfinance_client
    if _yfinance_client is None:
        _yfinance_client = YFinanceClient()
    return _yfinance_client