"""
FRED (Federal Reserve Economic Data) client for macroeconomic data
Free access to Federal Reserve economic indicators
Python integration via requests
"""

import pandas as pd
import requests
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
from ..utils import rate_limiter

logger = logging.getLogger(__name__)

class FREDClient:
    """FRED API client for macroeconomic data that influences crypto markets"""
    
    def __init__(self, api_key: Optional[str] = None, data_dir: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.data_dir = Path(data_dir or "data/historical/fred")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.stlouisfed.org/fred"
        
        if not self.api_key:
            logger.warning("FRED API key not found. Set FRED_API_KEY environment variable.")
        
        # Key economic series for crypto trading
        self.crypto_relevant_series = {
            # Treasury Data
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "T10Y2Y": "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
            "T10YIE": "10-Year Breakeven Inflation Rate",
            "DGS2": "2-Year Treasury Constant Maturity Rate",
            "DGS30": "30-Year Treasury Constant Maturity Rate",
            
            # Federal Reserve Metrics
            "WALCL": "All Federal Reserve Banks - Total Assets",
            "RRPONTSYD": "Overnight Reverse Repurchase Agreements - Treasury Securities Sold by the Federal Reserve",
            "WTREGEN": "Treasury General Account",
            "EFFR": "Effective Federal Funds Rate",
            
            # Money Supply & Liquidity
            "MANMM101USA189N": "M1 Money Stock for United States",
            "M2SL": "M2 Money Stock",
            "BOGMBASE": "Monetary Base - Total",
            
            # Inflation & Economic Indicators
            "CPIAUCSL": "Consumer Price Index for All Urban Consumers - All Items",
            "CPILFESL": "Consumer Price Index for All Urban Consumers - All Items Less Food and Energy",
            "UNRATE": "Unemployment Rate",
            "GDP": "Gross Domestic Product",
            
            # Dollar Strength
            "DTWEXBGS": "Trade Weighted U.S. Dollar Index - Broad, Goods and Services",
            "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
            
            # Market Stress Indicators
            "TEDRATE": "TED Spread",
            "T5YIE": "5-Year Breakeven Inflation Rate",
            "T5YIFR": "5-Year, 5-Year Forward Inflation Expectation Rate"
        }
    
    def get_series_data(self, series_id: str, start_date: str = None, 
                       end_date: str = None, frequency: str = None,
                       save: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical data for a FRED series
        
        Args:
            series_id: FRED series identifier (e.g., 'DGS10')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (d/w/m/q/a)
            save: Whether to save data to disk
        """
        if not self.api_key:
            logger.error("FRED API key required")
            return None
        
        # Rate limiting for FRED (120 requests per minute)
        rate_limiter.wait_if_needed("fred")
        
        # Default date range (2 years)
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        # Build request parameters
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }
        
        if frequency:
            params["frequency"] = frequency
        
        try:
            logger.info(f"Downloading FRED series {series_id} from {start_date} to {end_date}")
            
            # Make API request
            url = f"{self.base_url}/series/observations"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "observations" not in data:
                logger.warning(f"No observations found for series {series_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data["observations"])
            
            if df.empty:
                logger.warning(f"Empty dataset for series {series_id}")
                return None
            
            # Clean and process data
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Handle missing values (FRED uses '.' for missing data)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Rename value column to series_id for clarity
            df = df.rename(columns={'value': series_id})
            df = df[[series_id]]  # Keep only the data column
            
            # Forward fill missing values for daily data
            if frequency == 'd' or not frequency:
                df = df.fillna(method='ffill')
            
            # Save if requested
            if save:
                filename = f"{series_id}_{start_date}_{end_date}.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"Saved FRED data to {filepath}")
            
            # Record successful call
            rate_limiter.record_call("fred")
            
            logger.info(f"✓ Downloaded {len(df)} observations for {series_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed for {series_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing FRED data for {series_id}: {e}")
            return None
    
    def get_multiple_series(self, series_ids: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Download multiple FRED series"""
        results = {}
        
        for series_id in series_ids:
            logger.info(f"Downloading FRED series {series_id}...")
            df = self.get_series_data(series_id, **kwargs)
            if df is not None:
                results[series_id] = df
                logger.info(f"✓ Downloaded {len(df)} observations for {series_id}")
            else:
                logger.warning(f"✗ Failed to download {series_id}")
        
        return results
    
    def get_crypto_relevant_data(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download all crypto-relevant economic indicators"""
        logger.info("Downloading crypto-relevant FRED economic data...")
        return self.get_multiple_series(list(self.crypto_relevant_series.keys()), **kwargs)
    
    def get_liquidity_metrics(self, **kwargs) -> pd.DataFrame:
        """
        Calculate comprehensive liquidity metrics from FRED data
        
        Returns combined DataFrame with calculated liquidity indicators
        """
        # Core liquidity series
        liquidity_series = ["WALCL", "RRPONTSYD", "WTREGEN", "M2SL", "BOGMBASE"]
        
        data = self.get_multiple_series(liquidity_series, **kwargs)
        
        if not data:
            logger.error("Failed to download liquidity data")
            return pd.DataFrame()
        
        # Combine into single DataFrame
        combined_df = pd.DataFrame()
        for series_id, df in data.items():
            if not df.empty:
                combined_df[series_id] = df[series_id]
        
        if combined_df.empty:
            return combined_df
        
        # Calculate derived liquidity metrics
        try:
            # Net liquidity (Fed balance sheet minus reverse repo minus TGA)
            if all(col in combined_df.columns for col in ["WALCL", "RRPONTSYD", "WTREGEN"]):
                combined_df["NET_LIQUIDITY"] = (
                    combined_df["WALCL"] - 
                    combined_df["RRPONTSYD"] - 
                    combined_df["WTREGEN"]
                )
            
            # M2 velocity proxy (if we had GDP data)
            if "M2SL" in combined_df.columns:
                combined_df["M2_GROWTH"] = combined_df["M2SL"].pct_change(periods=252)  # YoY growth
            
            # Monetary base growth
            if "BOGMBASE" in combined_df.columns:
                combined_df["BASE_GROWTH"] = combined_df["BOGMBASE"].pct_change(periods=252)
            
            logger.info(f"✓ Calculated liquidity metrics with {len(combined_df)} observations")
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
        
        return combined_df
    
    def load_cached_data(self, series_id: str) -> Optional[pd.DataFrame]:
        """Load previously downloaded FRED data"""
        pattern = f"{series_id}_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if files:
            # Sort by modification time and get most recent
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            try:
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                logger.error(f"Error loading cached FRED data: {e}")
        
        return None
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare FRED data for model training"""
        training_df = df.copy()
        
        # Calculate technical indicators for economic data
        for col in training_df.columns:
            if training_df[col].dtype in ['float64', 'int64']:
                # Moving averages
                training_df[f"{col}_MA_30"] = training_df[col].rolling(window=30).mean()
                training_df[f"{col}_MA_90"] = training_df[col].rolling(window=90).mean()
                
                # Rate of change
                training_df[f"{col}_ROC_30"] = training_df[col].pct_change(periods=30)
                training_df[f"{col}_ROC_90"] = training_df[col].pct_change(periods=90)
                
                # Z-score (standardization)
                rolling_mean = training_df[col].rolling(window=252).mean()
                rolling_std = training_df[col].rolling(window=252).std()
                training_df[f"{col}_ZSCORE"] = (training_df[col] - rolling_mean) / rolling_std
        
        # Remove NaN rows
        training_df = training_df.dropna()
        
        return training_df
