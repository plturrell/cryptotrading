"""
Yahoo Finance client for crypto historical data
Free access to major crypto pairs
Python integration via yfinance
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os
from pathlib import Path
import numpy as np
import logging
from ...utils import rate_limiter

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not installed. Run: pip install yfinance")

class YahooFinanceClient:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or "data/historical/yahoo")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Common crypto symbols on Yahoo Finance
        self.crypto_symbols = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "BNB": "BNB-USD",
            "XRP": "XRP-USD",
            "ADA": "ADA-USD",
            "DOGE": "DOGE-USD",
            "DOT": "DOT11419-USD",
            "MATIC": "MATIC-USD",
            "SOL": "SOL-USD",
            "SHIB": "SHIB-USD",
            "LTC": "LTC-USD",
            "AVAX": "AVAX-USD",
            "LINK": "LINK-USD",
            "UNI": "UNI3-USD",
            "ATOM": "ATOM-USD"
        }
    
    def download_data(self, symbol: str, start_date: str = None, 
                     end_date: str = None, interval: str = "1d",
                     save: bool = True) -> Optional[pd.DataFrame]:
        """Download historical data for a crypto symbol
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC' or 'BTC-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        if yf is None:
            logger.error("yfinance not available. Install with: pip install yfinance")
            return None
        
        # Rate limiting
        rate_limiter.wait_if_needed("yahoo")
        
        # Convert symbol if needed
        if symbol in self.crypto_symbols:
            yahoo_symbol = self.crypto_symbols[symbol]
        elif not symbol.endswith('-USD'):
            yahoo_symbol = f"{symbol}-USD"
        else:
            yahoo_symbol = symbol
        
        # Default date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            # Default to 2 years of data
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        try:
            # Download data
            logger.info(f"Downloading {yahoo_symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {yahoo_symbol}")
                return None
            
            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Save if requested
            if save:
                filename = f"{yahoo_symbol}_{interval}_{start_date}_{end_date}.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"Saved to {filepath}")
            
            # Record successful call
            rate_limiter.record_call("yahoo")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {yahoo_symbol}: {e}")
            return None
    
    def download_multiple(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Download multiple symbols"""
        results = {}
        
        for symbol in symbols:
            logger.info(f"Downloading {symbol}...")
            df = self.download_data(symbol, **kwargs)
            if df is not None:
                results[symbol] = df
                logger.info(f"✓ Downloaded {len(df)} rows for {symbol}")
        
        return results
    
    def download_all_major_cryptos(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download all major cryptocurrencies"""
        return self.download_multiple(list(self.crypto_symbols.keys()), **kwargs)
    
    def get_realtime_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time price data"""
        if yf is None:
            return None
        
        # Convert symbol
        if symbol in self.crypto_symbols:
            yahoo_symbol = self.crypto_symbols[symbol]
        else:
            yahoo_symbol = f"{symbol}-USD" if not symbol.endswith('-USD') else symbol
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "price": info.get("regularMarketPrice", info.get("price")),
                "previous_close": info.get("previousClose"),
                "open": info.get("open"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting realtime price for {symbol}: {e}")
            return None
    
    def load_cached_data(self, symbol: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Load previously downloaded data"""
        # Find most recent file for this symbol
        pattern = f"{symbol}*{interval}*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if files:
            # Sort by modification time and get most recent
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            try:
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
        
        return None
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare Yahoo Finance data for model training"""
        training_df = df.copy()
        
        # Ensure we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in training_df.columns for col in required_cols):
            logger.error("Missing required OHLCV columns")
            return training_df
        
        # Technical indicators
        # RSI
        training_df['rsi'] = self._calculate_rsi(training_df['close'])
        
        # MACD
        exp1 = training_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = training_df['close'].ewm(span=26, adjust=False).mean()
        training_df['macd'] = exp1 - exp2
        training_df['signal'] = training_df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        training_df['bb_middle'] = training_df['close'].rolling(window=20).mean()
        bb_std = training_df['close'].rolling(window=20).std()
        training_df['bb_upper'] = training_df['bb_middle'] + (bb_std * 2)
        training_df['bb_lower'] = training_df['bb_middle'] - (bb_std * 2)
        
        # Price features
        training_df['returns'] = training_df['close'].pct_change()
        training_df['log_returns'] = (training_df['close'] / training_df['close'].shift(1)).apply(
            lambda x: np.log(x) if x > 0 else 0
        )
        
        # Remove NaN rows
        training_df = training_df.dropna()
        
        return training_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi