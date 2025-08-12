"""
Bitget Historical Data client
Free CSV/Excel downloads (1 per coin per day)
Multiple timeframes available
"""

import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, date
import os
from pathlib import Path
import json
from ..utils import rate_limiter

class BitgetHistoricalClient:
    def __init__(self, data_dir: Optional[str] = None):
        self.base_url = "https://www.bitget.com"
        self.api_base = "https://api.bitget.com/api/v2/public"
        self.data_dir = Path(data_dir or "data/historical/bitget")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track daily downloads per coin
        self.download_tracker_file = self.data_dir / "download_tracker.json"
        self.download_tracker = self._load_tracker()
        
        # Available timeframes
        self.timeframes = {
            "1min": "1m",
            "5min": "5m", 
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h",
            "4hour": "4h",
            "1day": "1d",
            "1week": "1w",
            "1month": "1M"
        }
    
    def _load_tracker(self) -> Dict[str, Dict[str, Any]]:
        """Load download tracker"""
        if self.download_tracker_file.exists():
            try:
                with open(self.download_tracker_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_tracker(self):
        """Save download tracker"""
        with open(self.download_tracker_file, 'w') as f:
            json.dump(self.download_tracker, f)
    
    def _check_daily_limit(self, symbol: str) -> bool:
        """Check if we can download this symbol today"""
        today = date.today().isoformat()
        
        if symbol in self.download_tracker:
            last_download = self.download_tracker[symbol].get('date')
            if last_download == today:
                print(f"Daily limit reached for {symbol}. Last download: {last_download}")
                return False
        
        return True
    
    def _record_download(self, symbol: str):
        """Record a download"""
        self.download_tracker[symbol] = {
            'date': date.today().isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        self._save_tracker()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from Bitget"""
        try:
            # Get spot symbols
            response = requests.get(f"{self.api_base}/market/tickers", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                symbols = [item['symbol'] for item in data['data']]
                return symbols
        except Exception as e:
            print(f"Error fetching symbols: {e}")
        
        # Return common symbols as fallback
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "MATICUSDT", "DOTUSDT", "SHIBUSDT", "AVAXUSDT"
        ]
    
    def download_historical_data(self, symbol: str, timeframe: str = "1day",
                               limit: int = 1000, save: bool = True) -> Optional[pd.DataFrame]:
        """Download historical data for a symbol
        
        Note: Bitget allows 1 download per coin per day
        """
        # Check daily limit
        if not self._check_daily_limit(symbol):
            # Try to load cached data instead
            cached = self.load_cached_data(symbol, timeframe)
            if cached is not None:
                print(f"Returning cached data for {symbol}")
                return cached
            return None
        
        # Rate limiting
        rate_limiter.wait_if_needed("bitget")
        
        # Convert timeframe
        tf = self.timeframes.get(timeframe, "1d")
        
        try:
            # Bitget public klines endpoint
            params = {
                "symbol": symbol,
                "interval": tf,
                "limit": min(limit, 1000)  # Max 1000
            }
            
            response = requests.get(
                f"{self.api_base}/market/candles",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') == '00000' and data.get('data'):
                # Convert to DataFrame
                # Bitget returns: [timestamp, open, high, low, close, volume]
                df = pd.DataFrame(
                    data['data'],
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Convert price columns to float
                price_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in price_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Sort by time
                df.sort_index(inplace=True)
                
                # Save if requested
                if save:
                    filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
                    filepath = self.data_dir / filename
                    df.to_csv(filepath)
                    print(f"Saved to {filepath}")
                
                # Record download
                self._record_download(symbol)
                rate_limiter.record_call("bitget")
                
                return df
            else:
                print(f"No data returned for {symbol}: {data.get('msg', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {symbol}: {e}")
        except Exception as e:
            print(f"Error processing data: {e}")
        
        return None
    
    def download_multiple(self, symbols: List[str], timeframe: str = "1day") -> Dict[str, pd.DataFrame]:
        """Download multiple symbols (respecting daily limits)"""
        results = {}
        
        for symbol in symbols:
            print(f"Attempting to download {symbol}...")
            df = self.download_historical_data(symbol, timeframe)
            if df is not None:
                results[symbol] = df
                print(f"✓ Downloaded {len(df)} rows for {symbol}")
            else:
                print(f"✗ Skipped {symbol} (daily limit or error)")
        
        return results
    
    def load_cached_data(self, symbol: str, timeframe: str = "1day") -> Optional[pd.DataFrame]:
        """Load previously downloaded data"""
        # Find files for this symbol and timeframe
        pattern = f"{symbol}_{timeframe}_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if files:
            # Get most recent file
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            try:
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                print(f"Error loading cached data: {e}")
        
        return None
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get current download status"""
        today = date.today().isoformat()
        
        downloaded_today = [
            symbol for symbol, info in self.download_tracker.items()
            if info.get('date') == today
        ]
        
        return {
            "date": today,
            "downloaded_today": downloaded_today,
            "count": len(downloaded_today),
            "tracker": self.download_tracker
        }
    
    def reset_daily_limits(self):
        """Reset daily download limits (use with caution)"""
        self.download_tracker = {}
        self._save_tracker()
        print("Daily download limits reset")
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare Bitget data for model training"""
        training_df = df.copy()
        
        # Add basic technical indicators
        # Moving averages
        training_df['sma_10'] = training_df['close'].rolling(window=10).mean()
        training_df['sma_20'] = training_df['close'].rolling(window=20).mean()
        training_df['sma_50'] = training_df['close'].rolling(window=50).mean()
        
        # EMA
        training_df['ema_12'] = training_df['close'].ewm(span=12, adjust=False).mean()
        training_df['ema_26'] = training_df['close'].ewm(span=26, adjust=False).mean()
        
        # Volume indicators
        training_df['volume_sma'] = training_df['volume'].rolling(window=10).mean()
        training_df['volume_ratio'] = training_df['volume'] / training_df['volume_sma']
        
        # Price changes
        training_df['returns'] = training_df['close'].pct_change()
        training_df['high_low_ratio'] = training_df['high'] / training_df['low']
        training_df['close_open_ratio'] = training_df['close'] / training_df['open']
        
        # Remove NaN rows
        training_df = training_df.dropna()
        
        return training_df