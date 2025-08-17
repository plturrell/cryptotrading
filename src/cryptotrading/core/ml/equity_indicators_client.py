"""
Equity Indicators Yahoo Finance Client
Loads equity and traditional market indicators that predict cryptocurrency movements
"""

import yfinance as yf
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EquityIndicatorsClient:
    """Yahoo Finance client for equity indicators that predict crypto movements"""
    
    # Primary equity indicators that correlate with crypto
    EQUITY_INDICATORS = {
        # Tier 1: High Correlation Indices
        'SPY': {'name': 'S&P 500 ETF', 'correlation': 0.48, 'category': 'market_index'},
        'QQQ': {'name': 'NASDAQ-100 ETF', 'correlation': 0.48, 'category': 'tech_index'},
        
        # Tech Stocks (Individual Predictors)
        'AAPL': {'name': 'Apple Inc', 'correlation': 0.45, 'category': 'tech_stock'},
        'MSFT': {'name': 'Microsoft Corp', 'correlation': 0.43, 'category': 'tech_stock'},
        'NVDA': {'name': 'NVIDIA Corp', 'correlation': 0.52, 'category': 'tech_stock'},
        
        # Currency & Volatility
        'DX-Y.NYB': {'name': 'US Dollar Index', 'correlation': -0.35, 'category': 'currency'},
        '^VIX': {'name': 'CBOE Volatility Index (via Yahoo Finance)', 'correlation': 0.49, 'category': 'volatility'},
        
        # Bond Markets
        '^TNX': {'name': '10-Year Treasury Yield', 'correlation': -0.25, 'category': 'bonds'},
        '^IRX': {'name': '3-Month Treasury Bill', 'correlation': -0.20, 'category': 'bonds'},
        
        # Crypto-Adjacent Stocks
        'COIN': {'name': 'Coinbase Global Inc', 'correlation': 0.75, 'category': 'crypto_stock'},
        'MSTR': {'name': 'MicroStrategy Inc', 'correlation': 0.65, 'category': 'crypto_stock'},
        'MARA': {'name': 'Marathon Digital Holdings', 'correlation': 0.60, 'category': 'crypto_stock'},
        
        # Commodities
        'GLD': {'name': 'SPDR Gold Trust', 'correlation': 0.05, 'category': 'commodity'},
        'SLV': {'name': 'iShares Silver Trust', 'correlation': 0.15, 'category': 'commodity'},
        
        # Sector ETFs
        'XLK': {'name': 'Technology Select Sector', 'correlation': 0.50, 'category': 'sector_etf'},
        'IWM': {'name': 'Russell 2000 ETF', 'correlation': 0.35, 'category': 'sector_etf'},
        
        # International
        '^FTSE': {'name': 'FTSE 100 Index', 'correlation': 0.30, 'category': 'international'},
        '^N225': {'name': 'Nikkei 225', 'correlation': 0.25, 'category': 'international'}
    }
    
    # Crypto pair specific predictors
    PAIR_PREDICTORS = {
        'BTC': ['SPY', 'DX-Y.NYB', 'MSTR', '^TNX', 'GLD'],
        'ETH': ['QQQ', '^VIX', 'COIN', 'XLK', 'NVDA'],
        'SOL': ['QQQ', 'NVDA', 'XLK', 'IWM', 'AAPL'],
        'BNB': ['^VIX', 'COIN', 'XLK', 'SPY', 'MARA'],
        'XRP': ['IWM', 'XLK', 'SPY', '^VIX', 'COIN'],
        'ADA': ['IWM', 'XLK', 'QQQ', 'NVDA', 'SPY'],
        'DOGE': ['IWM', 'MSTR', 'COIN', 'SPY', 'QQQ'],
        'MATIC': ['QQQ', 'XLK', 'NVDA', 'ETH-USD', 'SPY']
    }
    
    def __init__(self):
        self.tickers = {}
        self._ticker_info_cache = {}
        logger.info(f"Equity indicators client initialized with {len(self.EQUITY_INDICATORS)} indicators")
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create ticker instance for symbol"""
        if symbol not in self.tickers:
            self.tickers[symbol] = yf.Ticker(symbol)
        return self.tickers[symbol]
    
    def get_equity_data(
        self,
        symbol: str,
        days_back: int = 365,
        interval: str = '1d',
        prepost: bool = False,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data for equity indicator
        
        Args:
            symbol: Equity symbol (SPY, QQQ, AAPL, etc.)
            days_back: Number of days of historical data
            interval: Data interval (1d, 1h, 5m, etc.)
            prepost: Include pre/post market data
            auto_adjust: Auto adjust for splits/dividends
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if symbol not in self.EQUITY_INDICATORS:
                raise ValueError(f"Symbol {symbol} not in supported equity indicators: {list(self.EQUITY_INDICATORS.keys())}")
            
            ticker = self.get_ticker(symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Fetching {symbol} equity data from {start_date} to {end_date}")
            
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
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(hist_data)} records for {symbol}")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching equity data for {symbol}: {e}")
            raise
    
    def get_multiple_equity_data(
        self,
        symbols: List[str],
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple equity indicators"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_equity_data(symbol, days_back, interval)
                if not data.empty:
                    results[symbol] = data
                    logger.info(f"✓ Loaded {len(data)} records for {symbol}")
                else:
                    logger.warning(f"✗ No data for {symbol}")
            except Exception as e:
                logger.error(f"✗ Error loading {symbol}: {e}")
        
        return results
    
    def get_predictors_for_crypto(
        self,
        crypto_symbol: str,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get equity predictors specific to a crypto symbol"""
        try:
            # Remove common suffixes to get base symbol
            base_symbol = crypto_symbol.upper().replace('-USD', '').replace('USDT', '').replace('-USDT', '')
            
            if base_symbol not in self.PAIR_PREDICTORS:
                raise ValueError(f"Crypto symbol {base_symbol} not supported. Available: {list(self.PAIR_PREDICTORS.keys())}")
            
            predictor_symbols = self.PAIR_PREDICTORS[base_symbol]
            logger.info(f"Loading {len(predictor_symbols)} predictors for {crypto_symbol}: {predictor_symbols}")
            
            return self.get_multiple_equity_data(predictor_symbols, days_back, interval)
            
        except Exception as e:
            logger.error(f"Error getting predictors for {crypto_symbol}: {e}")
            return {}
    
    def get_all_tier1_indicators(
        self,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get all Tier 1 high correlation indicators"""
        tier1_symbols = [
            'SPY', 'QQQ',           # Indices
            'AAPL', 'MSFT', 'NVDA', # Tech stocks
            'DX-Y.NYB', '^VIX',     # Currency/Volatility
            'COIN', 'MSTR'          # Crypto stocks
        ]
        
        logger.info(f"Loading all Tier 1 indicators: {tier1_symbols}")
        return self.get_multiple_equity_data(tier1_symbols, days_back, interval)
    
    def get_indicator_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about a specific equity indicator"""
        if symbol not in self.EQUITY_INDICATORS:
            return {"error": f"Symbol {symbol} not found"}
        
        info = self.EQUITY_INDICATORS[symbol].copy()
        
        try:
            # Get current market data
            ticker = self.get_ticker(symbol)
            ticker_info = ticker.info
            
            info.update({
                "current_price": ticker_info.get('regularMarketPrice'),
                "previous_close": ticker_info.get('previousClose'),
                "volume": ticker_info.get('volume'),
                "market_cap": ticker_info.get('marketCap'),
                "currency": ticker_info.get('currency', 'USD'),
                "exchange": ticker_info.get('exchange'),
                "last_update": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"Could not get current info for {symbol}: {e}")
            info["note"] = "Current market data unavailable"
        
        return info
    
    def get_all_indicators_info(self) -> List[Dict[str, Any]]:
        """Get information about all supported equity indicators"""
        indicators_info = []
        
        for symbol in self.EQUITY_INDICATORS.keys():
            try:
                info = self.get_indicator_info(symbol)
                indicators_info.append({
                    "symbol": symbol,
                    **info
                })
            except Exception as e:
                logger.warning(f"Error getting info for {symbol}: {e}")
                indicators_info.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return indicators_info
    
    def calculate_correlation_matrix(
        self,
        crypto_data: pd.DataFrame,
        equity_data: Dict[str, pd.DataFrame],
        crypto_symbol: str = 'BTC'
    ) -> pd.DataFrame:
        """Calculate correlation matrix between crypto and equity indicators"""
        try:
            # Prepare data for correlation
            correlation_data = {}
            
            # Add crypto data
            if 'Close' in crypto_data.columns:
                correlation_data[f'{crypto_symbol}_Close'] = crypto_data['Close']
                correlation_data[f'{crypto_symbol}_Returns'] = crypto_data['Close'].pct_change()
            
            # Add equity data
            for symbol, data in equity_data.items():
                if not data.empty and 'Close' in data.columns:
                    # Align dates
                    aligned_data = data.reindex(crypto_data.index)
                    correlation_data[f'{symbol}_Close'] = aligned_data['Close']
                    correlation_data[f'{symbol}_Returns'] = aligned_data['Close'].pct_change()
            
            # Create DataFrame and calculate correlations
            corr_df = pd.DataFrame(correlation_data)
            correlation_matrix = corr_df.corr()
            
            logger.info(f"Calculated correlation matrix with {len(correlation_data)} series")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()


# Singleton instance
_equity_indicators_client = None

def get_equity_indicators_client() -> EquityIndicatorsClient:
    """Get singleton equity indicators client instance"""
    global _equity_indicators_client
    if _equity_indicators_client is None:
        _equity_indicators_client = EquityIndicatorsClient()
    return _equity_indicators_client