"""
CryptoDataDownload client for free historical crypto data
Coverage: 20+ major exchanges
Format: Standardized CSV files
"""

import requests
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
from pathlib import Path
import numpy as np
from ..utils import rate_limiter

class CryptoDataDownloadClient:
    def __init__(self, data_dir: Optional[str] = None):
        self.base_url = "https://www.cryptodatadownload.com/cdd"
        self.data_dir = Path(data_dir or "data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Available exchanges and their formats
        self.exchanges = {
            "Binance": "binance",
            "Bitstamp": "bitstamp", 
            "Gemini": "gemini",
            "Bitfinex": "bitfinex",
            "Kraken": "kraken",
            "Coinbase": "coinbase",
            "Poloniex": "poloniex"
        }
        
        # Common trading pairs
        self.common_pairs = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
            "XRPUSDT", "DOTUSDT", "UNIUSDT", "LINKUSDT", "MATICUSDT"
        ]
    
    def _construct_url(self, exchange: str, pair: str, timeframe: str = "d") -> str:
        """Construct download URL
        Timeframe: d (daily), h (hourly), m (minute)
        """
        # Format: exchange_pair_timeframe.csv
        filename = f"{exchange}_{pair}_{timeframe}.csv"
        return f"{self.base_url}/{filename}"
    
    def download_data(self, exchange: str, pair: str, timeframe: str = "d", 
                     save: bool = True) -> Optional[pd.DataFrame]:
        """Download historical data for a trading pair"""
        # Rate limiting
        rate_limiter.wait_if_needed("cryptodatadownload")
        
        if exchange not in self.exchanges.values():
            print(f"Exchange {exchange} not supported. Available: {list(self.exchanges.values())}")
            return None
        
        url = self._construct_url(exchange, pair, timeframe)
        
        try:
            # Download CSV
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw CSV if requested
            if save:
                filename = f"{exchange}_{pair}_{timeframe}.csv"
                filepath = self.data_dir / filename
                with open(filepath, 'w') as f:
                    f.write(response.text)
                print(f"Saved to {filepath}")
            
            # Parse CSV
            # CryptoDataDownload CSVs have 2 header rows
            lines = response.text.strip().split('\n')
            if len(lines) > 2:
                # Skip first row (description)
                csv_data = '\n'.join(lines[1:])
                
                # Read into pandas
                from io import StringIO
                df = pd.read_csv(StringIO(csv_data))
                
                # Standardize column names
                df.columns = df.columns.str.strip()
                
                # Convert date column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # Record successful call
                rate_limiter.record_call("cryptodatadownload")
                
                return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {exchange} {pair}: {e}")
        except Exception as e:
            print(f"Error processing data: {e}")
        
        return None
    
    def download_multiple(self, pairs: List[Dict[str, str]], 
                         timeframe: str = "d") -> Dict[str, pd.DataFrame]:
        """Download multiple trading pairs
        pairs: List of dicts with 'exchange' and 'pair' keys
        """
        results = {}
        
        for item in pairs:
            exchange = item.get('exchange')
            pair = item.get('pair')
            
            if exchange and pair:
                key = f"{exchange}_{pair}"
                print(f"Downloading {key}...")
                df = self.download_data(exchange, pair, timeframe)
                if df is not None:
                    results[key] = df
                    print(f"✓ Downloaded {len(df)} rows for {key}")
        
        return results
    
    def download_all_binance(self, timeframe: str = "d") -> Dict[str, pd.DataFrame]:
        """Download all common Binance pairs"""
        pairs = [{"exchange": "binance", "pair": pair} for pair in self.common_pairs]
        return self.download_multiple(pairs, timeframe)
    
    def get_available_data(self) -> Dict[str, List[str]]:
        """Get list of available data (cached locally)"""
        available = {}
        
        for csv_file in self.data_dir.glob("*.csv"):
            parts = csv_file.stem.split('_')
            if len(parts) >= 3:
                exchange = parts[0]
                pair = parts[1]
                
                if exchange not in available:
                    available[exchange] = []
                available[exchange].append(pair)
        
        return available
    
    def load_cached_data(self, exchange: str, pair: str, 
                        timeframe: str = "d") -> Optional[pd.DataFrame]:
        """Load previously downloaded data"""
        filename = f"{exchange}_{pair}_{timeframe}.csv"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            try:
                # Read CSV, skip first row if it's descriptive
                df = pd.read_csv(filepath, skiprows=1)
                
                # Standardize
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                return df
            except Exception as e:
                print(f"Error loading cached data: {e}")
        
        return None
    
    def prepare_training_data(self, df: pd.DataFrame, 
                            features: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare data for model training"""
        if features is None:
            # Default OHLCV features
            features = ['open', 'high', 'low', 'close', 'volume']
        
        # Ensure features exist
        available_features = [f for f in features if f in df.columns]
        
        # Clean data
        training_df = df[available_features].copy()
        training_df = training_df.dropna()
        
        # Add technical indicators
        if 'close' in training_df.columns:
            # Simple Moving Averages
            training_df['sma_7'] = training_df['close'].rolling(window=7).mean()
            training_df['sma_30'] = training_df['close'].rolling(window=30).mean()
            
            # Price changes
            training_df['returns'] = training_df['close'].pct_change()
            training_df['log_returns'] = (training_df['close'] / training_df['close'].shift(1)).apply(lambda x: np.log(x) if x > 0 else 0)
            
            # Volume indicators
            if 'volume' in training_df.columns:
                training_df['volume_sma'] = training_df['volume'].rolling(window=7).mean()
                training_df['volume_ratio'] = training_df['volume'] / training_df['volume_sma']
        
        # Remove NaN rows created by indicators
        training_df = training_df.dropna()
        
        return training_df