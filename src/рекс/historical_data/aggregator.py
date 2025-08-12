"""
Historical Data Aggregator for model training
Combines data from multiple free sources
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .cryptodatadownload import CryptoDataDownloadClient
from .yahoo_finance import YahooFinanceClient
from .bitget import BitgetHistoricalClient
from ..database import get_db

class HistoricalDataAggregator:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or "data/historical/aggregated")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize clients
        self.clients = {
            "cryptodatadownload": CryptoDataDownloadClient(),
            "yahoo": YahooFinanceClient(),
            "bitget": BitgetHistoricalClient()
        }
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.db = get_db()
    
    def download_all_sources(self, symbol: str, start_date: str = None,
                           end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Download data from all available sources"""
        results = {}
        
        # Prepare parameters
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # CryptoDataDownload (for major exchanges)
        exchanges_to_try = ["binance", "coinbase", "kraken"]
        for exchange in exchanges_to_try:
            try:
                df = self.clients["cryptodatadownload"].download_data(
                    exchange, symbol.replace("-", "").upper(), "d"
                )
                if df is not None:
                    results[f"cdd_{exchange}"] = df
                    break
            except:
                continue
        
        # Yahoo Finance
        try:
            df = self.clients["yahoo"].download_data(
                symbol, start_date, end_date, "1d"
            )
            if df is not None:
                results["yahoo"] = df
        except:
            pass
        
        # Bitget (if not already downloaded today)
        try:
            df = self.clients["bitget"].download_historical_data(
                symbol.replace("-", "").upper() + "USDT", "1day"
            )
            if df is not None:
                results["bitget"] = df
        except:
            pass
        
        return results
    
    def merge_data_sources(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data from multiple sources into unified dataset"""
        if not data_dict:
            return pd.DataFrame()
        
        merged_df = None
        
        for source, df in data_dict.items():
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
            
            # Add source column
            df['source'] = source
            
            # Merge
            if merged_df is None:
                merged_df = df
            else:
                # Combine data, preferring non-null values
                merged_df = pd.concat([merged_df, df])
        
        # Remove duplicates, keeping first (most reliable source)
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        # Sort by date
        merged_df.sort_index(inplace=True)
        
        return merged_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across sources"""
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'Date': 'date',
            'Timestamp': 'date'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def create_training_dataset(self, symbol: str, features: List[str] = None,
                              lookback_days: int = 365) -> pd.DataFrame:
        """Create comprehensive training dataset with technical indicators"""
        # Download from all sources
        print(f"Downloading {symbol} data from all sources...")
        all_data = self.download_all_sources(symbol)
        
        if not all_data:
            print("No data available from any source")
            return pd.DataFrame()
        
        # Merge data
        print(f"Merging data from {len(all_data)} sources...")
        merged_df = self.merge_data_sources(all_data)
        
        # Add comprehensive technical indicators
        print("Adding technical indicators...")
        training_df = self.add_all_indicators(merged_df)
        
        # Feature selection
        if features:
            available_features = [f for f in features if f in training_df.columns]
            training_df = training_df[available_features]
        
        # Save processed dataset
        filename = f"{symbol}_training_data_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        training_df.to_csv(filepath)
        print(f"Saved training dataset to {filepath}")
        
        return training_df
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators for ML training"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['close_open_spread'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_28'] = self._calculate_rsi(df['close'], 28)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_middle = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
            df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
            df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / df[f'bb_width_{period}']
        
        # Volume indicators
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        df['on_balance_volume'] = (df['volume'] * (~df['returns'].isna()) * np.sign(df['returns'])).cumsum()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Remove initial NaN rows
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """List all available training datasets"""
        datasets = []
        
        for csv_file in self.data_dir.glob("*_training_data_*.csv"):
            parts = csv_file.stem.split('_')
            if len(parts) >= 4:
                symbol = parts[0]
                date_str = parts[-1]
                
                # Get file info
                stat = csv_file.stat()
                
                datasets.append({
                    "symbol": symbol,
                    "date_created": date_str,
                    "file_path": str(csv_file),
                    "size_mb": stat.st_size / 1024 / 1024,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return sorted(datasets, key=lambda x: x['modified'], reverse=True)
    
    def load_training_dataset(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load most recent training dataset for a symbol"""
        pattern = f"{symbol}_training_data_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if files:
            # Get most recent
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            try:
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                print(f"Error loading dataset: {e}")
        
        return None
    
    def prepare_ml_features(self, df: pd.DataFrame, 
                          target_col: str = 'returns',
                          prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML models"""
        # Create target variable (future returns)
        df[f'target_{prediction_horizon}d'] = df[target_col].shift(-prediction_horizon)
        
        # Feature columns (exclude target and metadata)
        exclude_cols = ['source', 'target', 'adj_close'] + \
                      [col for col in df.columns if 'target_' in col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create feature matrix and target
        X = df[feature_cols].dropna()
        y = df[f'target_{prediction_horizon}d'].dropna()
        
        # Align indices
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        return X, y