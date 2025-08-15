"""
CBOE (Chicago Board Options Exchange) client for volatility data
Free access to VIX and volatility indices via CSV downloads
No API key required
"""

import pandas as pd
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
from ..utils import rate_limiter

logger = logging.getLogger(__name__)

class CBOEClient:
    """CBOE client for volatility indices that influence crypto markets"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or "data/historical/cboe")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
        
        # Available CBOE volatility indices
        self.volatility_indices = {
            "VIX": "CBOE Volatility Index",
            "VIX9D": "CBOE 9-Day Volatility Index",
            "VIX3M": "CBOE 3-Month Volatility Index",
            "VIX6M": "CBOE 6-Month Volatility Index",
            "SKEW": "CBOE SKEW Index",
            "VXN": "CBOE NASDAQ-100 Volatility Index",
            "RVX": "CBOE Russell 2000 Volatility Index",
            "VXD": "CBOE DJIA Volatility Index",
            "VVIX": "CBOE VIX of VIX Index"
        }
    
    def download_volatility_data(self, indicator: str = "VIX", save: bool = True) -> Optional[pd.DataFrame]:
        """
        Download CBOE volatility data
        
        Args:
            indicator: Volatility indicator (VIX, VIX3M, SKEW, etc.)
            save: Whether to save data to disk
        """
        if indicator not in self.volatility_indices:
            logger.error(f"Unknown CBOE indicator: {indicator}. Available: {list(self.volatility_indices.keys())}")
            return None
        
        # Rate limiting (CBOE doesn't have strict limits, but be respectful)
        rate_limiter.wait_if_needed("cboe")
        
        try:
            logger.info(f"Downloading CBOE {indicator} data...")
            
            # Build URL
            url = f"{self.base_url}/{indicator}_History.csv"
            
            # Download CSV data
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            if df.empty:
                logger.warning(f"Empty dataset for CBOE {indicator}")
                return None
            
            # Clean and process data
            # CBOE CSV format: DATE,OPEN,HIGH,LOW,CLOSE
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.set_index('DATE')
            
            # Standardize column names
            column_mapping = {
                'OPEN': f"{indicator}_Open",
                'HIGH': f"{indicator}_High", 
                'LOW': f"{indicator}_Low",
                'CLOSE': f"{indicator}_Close"
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date (most recent first in CBOE data)
            df = df.sort_index()
            
            # Save if requested
            if save:
                filename = f"{indicator}_history_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"Saved CBOE data to {filepath}")
            
            # Record successful call
            rate_limiter.record_call("cboe")
            
            logger.info(f"✓ Downloaded {len(df)} observations for CBOE {indicator}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CBOE request failed for {indicator}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing CBOE data for {indicator}: {e}")
            return None
    
    def download_multiple_indices(self, indicators: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Download multiple CBOE volatility indices"""
        results = {}
        
        for indicator in indicators:
            logger.info(f"Downloading CBOE {indicator}...")
            df = self.download_volatility_data(indicator, **kwargs)
            if df is not None:
                results[indicator] = df
                logger.info(f"✓ Downloaded {len(df)} observations for {indicator}")
            else:
                logger.warning(f"✗ Failed to download {indicator}")
        
        return results
    
    def download_all_volatility_data(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download all available CBOE volatility indices"""
        logger.info("Downloading all CBOE volatility data...")
        return self.download_multiple_indices(list(self.volatility_indices.keys()), **kwargs)
    
    def get_vix_term_structure(self, **kwargs) -> pd.DataFrame:
        """
        Get VIX term structure data (VIX, VIX9D, VIX3M, VIX6M)
        
        Returns combined DataFrame with VIX term structure
        """
        term_structure_indices = ["VIX9D", "VIX", "VIX3M", "VIX6M"]
        
        data = self.download_multiple_indices(term_structure_indices, **kwargs)
        
        if not data:
            logger.error("Failed to download VIX term structure data")
            return pd.DataFrame()
        
        # Combine into single DataFrame with close prices
        combined_df = pd.DataFrame()
        for indicator, df in data.items():
            close_col = f"{indicator}_Close"
            if close_col in df.columns:
                combined_df[indicator] = df[close_col]
        
        if combined_df.empty:
            return combined_df
        
        # Calculate term structure metrics
        try:
            # VIX contango/backwardation
            if "VIX" in combined_df.columns and "VIX3M" in combined_df.columns:
                combined_df["VIX_CONTANGO"] = combined_df["VIX3M"] - combined_df["VIX"]
                combined_df["VIX_CONTANGO_RATIO"] = combined_df["VIX3M"] / combined_df["VIX"]
            
            # Short-term vs long-term volatility
            if "VIX9D" in combined_df.columns and "VIX" in combined_df.columns:
                combined_df["SHORT_TERM_PREMIUM"] = combined_df["VIX"] - combined_df["VIX9D"]
            
            logger.info(f"✓ Built VIX term structure with {len(combined_df)} observations")
            
        except Exception as e:
            logger.error(f"Error calculating term structure metrics: {e}")
        
        return combined_df
    
    def load_cached_data(self, indicator: str) -> Optional[pd.DataFrame]:
        """Load previously downloaded CBOE data"""
        pattern = f"{indicator}_history_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if files:
            # Sort by modification time and get most recent
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            try:
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                logger.error(f"Error loading cached CBOE data: {e}")
        
        return None
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare CBOE volatility data for model training"""
        training_df = df.copy()
        
        # Calculate volatility-specific indicators
        for col in training_df.columns:
            if 'Close' in col and training_df[col].dtype in ['float64', 'int64']:
                base_name = col.replace('_Close', '')
                
                # Volatility moving averages
                training_df[f"{base_name}_MA_10"] = training_df[col].rolling(window=10).mean()
                training_df[f"{base_name}_MA_30"] = training_df[col].rolling(window=30).mean()
                
                # Volatility percentiles
                training_df[f"{base_name}_PERCENTILE_30"] = training_df[col].rolling(window=30).rank(pct=True)
                training_df[f"{base_name}_PERCENTILE_252"] = training_df[col].rolling(window=252).rank(pct=True)
                
                # Volatility regime indicators
                ma_30 = training_df[col].rolling(window=30).mean()
                training_df[f"{base_name}_REGIME"] = (training_df[col] > ma_30).astype(int)
                
                # Volatility spikes (above 95th percentile)
                rolling_95th = training_df[col].rolling(window=252).quantile(0.95)
                training_df[f"{base_name}_SPIKE"] = (training_df[col] > rolling_95th).astype(int)
        
        # Remove NaN rows
        training_df = training_df.dropna()
        
        return training_df
