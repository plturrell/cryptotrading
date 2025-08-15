"""
DeFiLlama client for on-chain DeFi metrics
Free access to TVL, stablecoin, and protocol data
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

class DeFiLlamaClient:
    """DeFiLlama client for on-chain DeFi metrics that influence crypto markets"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or "data/historical/defillama")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.llama.fi"
        
        # Key DeFi protocols for crypto analysis
        self.key_protocols = {
            "uniswap": "Uniswap DEX",
            "aave": "Aave Lending",
            "compound": "Compound Lending",
            "makerdao": "MakerDAO",
            "curve": "Curve Finance",
            "convex-finance": "Convex Finance",
            "lido": "Lido Staking",
            "pancakeswap": "PancakeSwap",
            "justlend": "JustLend",
            "gmx": "GMX Perpetuals"
        }
        
        # Major stablecoins
        self.stablecoins = [
            "USDT", "USDC", "BUSD", "DAI", "FRAX", 
            "TUSD", "USDP", "LUSD", "sUSD", "GUSD"
        ]
    
    def get_total_tvl_history(self, save: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical Total Value Locked (TVL) across all DeFi protocols
        """
        # Rate limiting (DeFiLlama allows ~100 requests per minute)
        rate_limiter.wait_if_needed("defillama")
        
        try:
            logger.info("Downloading DeFiLlama total TVL history...")
            
            # Get TVL history
            url = f"{self.base_url}/tvl"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning("No TVL data found")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning("Empty TVL dataset")
                return None
            
            # Process data
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df = df.set_index('date')
            df = df.rename(columns={'totalLiquidityUSD': 'TOTAL_TVL_USD'})
            
            # Convert to numeric
            df['TOTAL_TVL_USD'] = pd.to_numeric(df['TOTAL_TVL_USD'], errors='coerce')
            
            # Save if requested
            if save:
                filename = f"total_tvl_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"Saved DeFiLlama TVL data to {filepath}")
            
            # Record successful call
            rate_limiter.record_call("defillama")
            
            logger.info(f"✓ Downloaded {len(df)} TVL observations")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"DeFiLlama TVL request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing DeFiLlama TVL data: {e}")
            return None
    
    def get_stablecoin_data(self, save: bool = True) -> Optional[pd.DataFrame]:
        """
        Get current stablecoin market caps and historical data
        """
        rate_limiter.wait_if_needed("defillama")
        
        try:
            logger.info("Downloading DeFiLlama stablecoin data...")
            
            # Get current stablecoin data
            url = f"{self.base_url}/stablecoins"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'peggedAssets' not in data:
                logger.warning("No stablecoin data found")
                return None
            
            # Convert to DataFrame
            stablecoins_data = []
            for stablecoin in data['peggedAssets']:
                stablecoins_data.append({
                    'symbol': stablecoin.get('symbol', ''),
                    'name': stablecoin.get('name', ''),
                    'circulating': stablecoin.get('circulating', {}).get('peggedUSD', 0),
                    'chains': len(stablecoin.get('chainCirculating', {})),
                    'price': stablecoin.get('price', 1.0)
                })
            
            df = pd.DataFrame(stablecoins_data)
            
            if df.empty:
                logger.warning("Empty stablecoin dataset")
                return None
            
            # Add timestamp
            df['timestamp'] = datetime.now()
            df = df.set_index('timestamp')
            
            # Convert to numeric
            df['circulating'] = pd.to_numeric(df['circulating'], errors='coerce')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Save if requested
            if save:
                filename = f"stablecoins_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"Saved stablecoin data to {filepath}")
            
            rate_limiter.record_call("defillama")
            
            logger.info(f"✓ Downloaded {len(df)} stablecoin entries")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"DeFiLlama stablecoin request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing stablecoin data: {e}")
            return None
    
    def get_protocol_tvl(self, protocol: str, save: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical TVL for a specific protocol
        
        Args:
            protocol: Protocol slug (e.g., 'uniswap', 'aave')
        """
        rate_limiter.wait_if_needed("defillama")
        
        try:
            logger.info(f"Downloading DeFiLlama protocol data for {protocol}...")
            
            url = f"{self.base_url}/protocol/{protocol}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'tvl' not in data:
                logger.warning(f"No TVL data found for protocol {protocol}")
                return None
            
            # Convert TVL history to DataFrame
            tvl_data = data['tvl']
            df = pd.DataFrame(tvl_data)
            
            if df.empty:
                logger.warning(f"Empty dataset for protocol {protocol}")
                return None
            
            # Process data
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df = df.set_index('date')
            df = df.rename(columns={'totalLiquidityUSD': f'{protocol.upper()}_TVL_USD'})
            
            # Convert to numeric
            for col in df.columns:
                if col != 'date':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Save if requested
            if save:
                filename = f"protocol_{protocol}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"Saved protocol data to {filepath}")
            
            rate_limiter.record_call("defillama")
            
            logger.info(f"✓ Downloaded {len(df)} observations for {protocol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"DeFiLlama protocol request failed for {protocol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing protocol data for {protocol}: {e}")
            return None
    
    def get_multiple_protocols(self, protocols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Download multiple protocol TVL data"""
        results = {}
        
        for protocol in protocols:
            logger.info(f"Downloading protocol {protocol}...")
            df = self.get_protocol_tvl(protocol, **kwargs)
            if df is not None:
                results[protocol] = df
                logger.info(f"✓ Downloaded {len(df)} observations for {protocol}")
            else:
                logger.warning(f"✗ Failed to download {protocol}")
        
        return results
    
    def get_key_protocols_data(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download all key DeFi protocols data"""
        logger.info("Downloading key DeFi protocols data...")
        return self.get_multiple_protocols(list(self.key_protocols.keys()), **kwargs)
    
    def get_defi_metrics_summary(self, **kwargs) -> pd.DataFrame:
        """
        Get comprehensive DeFi metrics summary
        
        Returns combined DataFrame with key DeFi indicators
        """
        # Get total TVL
        total_tvl = self.get_total_tvl_history(**kwargs)
        
        # Get key protocols
        protocols_data = self.get_multiple_protocols(
            ["uniswap", "aave", "makerdao", "curve"], **kwargs
        )
        
        if not total_tvl or not protocols_data:
            logger.error("Failed to download DeFi metrics")
            return pd.DataFrame()
        
        # Combine into single DataFrame
        combined_df = total_tvl.copy()
        
        for protocol, df in protocols_data.items():
            if not df.empty:
                # Get the TVL column (should be the main numeric column)
                tvl_col = [col for col in df.columns if 'TVL' in col.upper()]
                if tvl_col:
                    combined_df[f"{protocol.upper()}_TVL"] = df[tvl_col[0]]
        
        if combined_df.empty:
            return combined_df
        
        # Calculate derived metrics
        try:
            # Protocol dominance (as % of total TVL)
            for protocol in ["uniswap", "aave", "makerdao", "curve"]:
                protocol_col = f"{protocol.upper()}_TVL"
                if protocol_col in combined_df.columns:
                    combined_df[f"{protocol.upper()}_DOMINANCE"] = (
                        combined_df[protocol_col] / combined_df["TOTAL_TVL_USD"] * 100
                    )
            
            # TVL growth rates
            combined_df["TVL_GROWTH_7D"] = combined_df["TOTAL_TVL_USD"].pct_change(periods=7)
            combined_df["TVL_GROWTH_30D"] = combined_df["TOTAL_TVL_USD"].pct_change(periods=30)
            
            logger.info(f"✓ Built DeFi metrics summary with {len(combined_df)} observations")
            
        except Exception as e:
            logger.error(f"Error calculating DeFi metrics: {e}")
        
        return combined_df
    
    def load_cached_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """Load previously downloaded DeFiLlama data"""
        pattern = f"{data_type}_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if files:
            # Sort by modification time and get most recent
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            try:
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                logger.error(f"Error loading cached DeFiLlama data: {e}")
        
        return None
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DeFiLlama data for model training"""
        training_df = df.copy()
        
        # Calculate DeFi-specific indicators
        for col in training_df.columns:
            if training_df[col].dtype in ['float64', 'int64']:
                # TVL moving averages
                training_df[f"{col}_MA_7"] = training_df[col].rolling(window=7).mean()
                training_df[f"{col}_MA_30"] = training_df[col].rolling(window=30).mean()
                
                # TVL growth rates
                training_df[f"{col}_GROWTH_1D"] = training_df[col].pct_change(periods=1)
                training_df[f"{col}_GROWTH_7D"] = training_df[col].pct_change(periods=7)
                training_df[f"{col}_GROWTH_30D"] = training_df[col].pct_change(periods=30)
                
                # TVL volatility
                training_df[f"{col}_VOLATILITY_30D"] = training_df[col].pct_change().rolling(window=30).std()
                
                # TVL momentum
                training_df[f"{col}_MOMENTUM"] = (
                    training_df[f"{col}_MA_7"] / training_df[f"{col}_MA_30"]
                ).fillna(1.0)
        
        # Remove NaN rows
        training_df = training_df.dropna()
        
        return training_df
