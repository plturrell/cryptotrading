"""
FX Rates Yahoo Finance Client
Loads foreign exchange pairs that provide early trading signals for cryptocurrency movements
"""

import yfinance as yf
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FXRatesClient:
    """Yahoo Finance client for FX rates that predict crypto movements"""
    
    # Primary FX pairs that provide early crypto signals
    FX_PAIRS = {
        # Tier 1: Primary Early Warning Signals
        'USDJPY=X': {
            'name': 'USD/JPY', 
            'tier': 1, 
            'category': 'carry_trade',
            'crypto_correlation': 0.65,
            'signal_strength': 'very_high',
            'best_for': ['BTC', 'ETH', 'SOL', 'BNB'],
            'mechanism': 'Carry trade flows, risk-off/on indicator',
            'critical_levels': [150.00, 145.00, 155.00]
        },
        'USDCNH=X': {
            'name': 'USD/CNH', 
            'tier': 1, 
            'category': 'capital_flight',
            'crypto_correlation': 0.80,
            'signal_strength': 'very_high',
            'best_for': ['BTC', 'ETH', 'SOL'],
            'mechanism': 'Capital flight from China, hard asset allocation',
            'critical_levels': [7.30, 7.35, 7.25]
        },
        'USDKRW=X': {
            'name': 'USD/KRW', 
            'tier': 1, 
            'category': 'regional_sentiment',
            'crypto_correlation': 0.70,
            'signal_strength': 'high',
            'best_for': ['XRP', 'ADA', 'DOGE'],
            'mechanism': 'Kimchi premium, regional altcoin demand',
            'critical_levels': [1300, 1350, 1250]
        },
        
        # Tier 2: Risk Sentiment Indicators
        'EURUSD=X': {
            'name': 'EUR/USD', 
            'tier': 2, 
            'category': 'risk_sentiment',
            'crypto_correlation': 0.77,
            'signal_strength': 'high',
            'best_for': ['ETH', 'BTC', 'MATIC'],
            'mechanism': 'Global risk appetite, DXY inverse',
            'critical_levels': [1.0447, 1.0500, 1.0300]
        },
        'GBPUSD=X': {
            'name': 'GBP/USD', 
            'tier': 2, 
            'category': 'volatility_catalyst',
            'crypto_correlation': 0.75,
            'signal_strength': 'high',
            'best_for': ['ETH', 'BTC', 'SOL'],
            'mechanism': 'High beta, volatility leader',
            'critical_levels': [1.27, 1.32, 1.25]
        },
        
        # Tier 3: Safe Haven vs Risk Signals
        'USDCHF=X': {
            'name': 'USD/CHF', 
            'tier': 3, 
            'category': 'safe_haven',
            'crypto_correlation': -0.60,
            'signal_strength': 'medium',
            'best_for': ['BTC', 'ETH'],
            'mechanism': 'Safe haven proxy, inverse risk indicator',
            'critical_levels': [0.90, 0.95, 0.88]
        },
        'AUDUSD=X': {
            'name': 'AUD/USD', 
            'tier': 3, 
            'category': 'commodity_proxy',
            'crypto_correlation': 0.68,
            'signal_strength': 'medium',
            'best_for': ['BTC', 'SOL', 'ADA'],
            'mechanism': 'Commodity sensitivity, China proxy',
            'critical_levels': [0.64, 0.68, 0.62]
        },
        'NZDUSD=X': {
            'name': 'NZD/USD', 
            'tier': 3, 
            'category': 'commodity_proxy',
            'crypto_correlation': 0.65,
            'signal_strength': 'medium',
            'best_for': ['BTC', 'ETH'],
            'mechanism': 'Risk appetite gauge, commodity sensitivity',
            'critical_levels': [0.58, 0.62, 0.56]
        },
        
        # Tier 4: Cross-Currency Early Signals
        'EURJPY=X': {
            'name': 'EUR/JPY', 
            'tier': 4, 
            'category': 'cross_currency',
            'crypto_correlation': 0.85,
            'signal_strength': 'very_high',
            'best_for': ['ETH', 'BTC', 'SOL'],
            'mechanism': 'Pure risk indicator, carry trade proxy',
            'critical_levels': [160.00, 165.00, 155.00]
        },
        'GBPJPY=X': {
            'name': 'GBP/JPY', 
            'tier': 4, 
            'category': 'cross_currency',
            'crypto_correlation': 0.82,
            'signal_strength': 'high',
            'best_for': ['ETH', 'BTC', 'high_beta_alts'],
            'mechanism': 'Volatility amplifier, extreme move indicator',
            'critical_levels': [193.00, 195.00, 190.00]
        },
        
        # Additional Important Pairs
        'USDCAD=X': {
            'name': 'USD/CAD', 
            'tier': 4, 
            'category': 'commodity_proxy',
            'crypto_correlation': -0.55,
            'signal_strength': 'medium',
            'best_for': ['BTC', 'ETH'],
            'mechanism': 'Oil proxy, North American flows',
            'critical_levels': [1.35, 1.40, 1.30]
        },
        'CHFJPY=X': {
            'name': 'CHF/JPY', 
            'tier': 4, 
            'category': 'cross_currency',
            'crypto_correlation': 0.70,
            'signal_strength': 'medium',
            'best_for': ['BTC', 'ETH'],
            'mechanism': 'Safe haven cross, risk sentiment',
            'critical_levels': [170.00, 175.00, 165.00]
        }
    }
    
    # Crypto pair specific FX predictors
    CRYPTO_FX_PREDICTORS = {
        'BTC': ['USDJPY=X', 'USDCNH=X', 'EURUSD=X', 'EURJPY=X'],
        'ETH': ['EURJPY=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X'],
        'SOL': ['USDKRW=X', 'AUDUSD=X', 'GBPJPY=X', 'EURUSD=X'],
        'BNB': ['USDJPY=X', 'USDCNH=X', 'EURUSD=X', 'AUDUSD=X'],
        'XRP': ['USDKRW=X', 'USDJPY=X', 'EURJPY=X', 'GBPUSD=X'],
        'ADA': ['USDKRW=X', 'AUDUSD=X', 'GBPJPY=X', 'EURUSD=X'],
        'DOGE': ['USDKRW=X', 'GBPJPY=X', 'AUDUSD=X', 'USDJPY=X'],
        'MATIC': ['EURUSD=X', 'EURJPY=X', 'GBPUSD=X', 'USDJPY=X']
    }
    
    # Session timing for FX signals
    SESSION_WEIGHTS = {
        'asian': {  # 19:00-04:00 UTC
            'primary': ['USDJPY=X', 'USDCNH=X', 'USDKRW=X', 'AUDUSD=X'],
            'weight': 0.6
        },
        'european': {  # 07:00-16:00 UTC
            'primary': ['EURUSD=X', 'GBPUSD=X', 'USDCHF=X', 'EURJPY=X'],
            'weight': 0.5
        },
        'us': {  # 13:00-22:00 UTC
            'primary': ['EURUSD=X', 'GBPUSD=X', 'USDCAD=X', 'USDJPY=X'],
            'weight': 0.7
        }
    }
    
    def __init__(self):
        self.tickers = {}
        self._ticker_info_cache = {}
        logger.info(f"FX rates client initialized with {len(self.FX_PAIRS)} currency pairs")
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create ticker instance for FX pair"""
        if symbol not in self.tickers:
            self.tickers[symbol] = yf.Ticker(symbol)
        return self.tickers[symbol]
    
    def get_fx_data(
        self,
        symbol: str,
        days_back: int = 365,
        interval: str = '1d',
        prepost: bool = False,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data for FX pair
        
        Args:
            symbol: FX symbol (EURUSD=X, USDJPY=X, etc.)
            days_back: Number of days of historical data
            interval: Data interval (1d, 1h, 5m, etc.)
            prepost: Include pre/post market data
            auto_adjust: Auto adjust for splits/dividends
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            if symbol not in self.FX_PAIRS:
                raise ValueError(f"FX pair {symbol} not in supported pairs: {list(self.FX_PAIRS.keys())}")
            
            ticker = self.get_ticker(symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Fetching {symbol} FX data from {start_date} to {end_date}")
            
            # Download historical data
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost,
                actions=False,  # FX pairs don't have corporate actions
                raise_errors=True
            )
            
            if hist_data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(hist_data)} records for {symbol}")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching FX data for {symbol}: {e}")
            raise
    
    def get_multiple_fx_data(
        self,
        symbols: List[str],
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple FX pairs"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_fx_data(symbol, days_back, interval)
                if not data.empty:
                    results[symbol] = data
                    logger.info(f"✓ Loaded {len(data)} records for {symbol}")
                else:
                    logger.warning(f"✗ No data for {symbol}")
            except Exception as e:
                logger.error(f"✗ Error loading {symbol}: {e}")
        
        return results
    
    def get_fx_predictors_for_crypto(
        self,
        crypto_symbol: str,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get FX predictors specific to a crypto symbol"""
        try:
            # Remove common suffixes to get base symbol
            base_symbol = crypto_symbol.upper().replace('-USD', '').replace('USDT', '').replace('-USDT', '')
            
            if base_symbol not in self.CRYPTO_FX_PREDICTORS:
                raise ValueError(f"Crypto symbol {base_symbol} not supported. Available: {list(self.CRYPTO_FX_PREDICTORS.keys())}")
            
            predictor_symbols = self.CRYPTO_FX_PREDICTORS[base_symbol]
            logger.info(f"Loading {len(predictor_symbols)} FX predictors for {crypto_symbol}: {predictor_symbols}")
            
            return self.get_multiple_fx_data(predictor_symbols, days_back, interval)
            
        except Exception as e:
            logger.error(f"Error getting FX predictors for {crypto_symbol}: {e}")
            return {}
    
    def get_tier1_fx_pairs(
        self,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get all Tier 1 FX pairs (highest predictive power)"""
        tier1_symbols = [
            symbol for symbol, info in self.FX_PAIRS.items() 
            if info['tier'] == 1
        ]
        
        logger.info(f"Loading Tier 1 FX pairs: {tier1_symbols}")
        return self.get_multiple_fx_data(tier1_symbols, days_back, interval)
    
    def get_session_fx_pairs(
        self,
        session: str,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get FX pairs optimized for specific trading session"""
        if session.lower() not in self.SESSION_WEIGHTS:
            raise ValueError(f"Session must be one of: {list(self.SESSION_WEIGHTS.keys())}")
        
        session_pairs = self.SESSION_WEIGHTS[session.lower()]['primary']
        logger.info(f"Loading {session.title()} session FX pairs: {session_pairs}")
        
        return self.get_multiple_fx_data(session_pairs, days_back, interval)
    
    def get_fx_pair_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about a specific FX pair"""
        if symbol not in self.FX_PAIRS:
            return {"error": f"FX pair {symbol} not found"}
        
        info = self.FX_PAIRS[symbol].copy()
        
        try:
            # Get current market data
            ticker = self.get_ticker(symbol)
            ticker_info = ticker.info
            
            # Get recent price data
            recent_data = self.get_fx_data(symbol, days_back=5)
            if not recent_data.empty:
                current_rate = recent_data['Close'].iloc[-1]
                prev_close = recent_data['Close'].iloc[-2] if len(recent_data) > 1 else current_rate
                change_24h = ((current_rate - prev_close) / prev_close * 100) if prev_close > 0 else 0
                
                info.update({
                    "current_rate": float(current_rate),
                    "previous_close": float(prev_close),
                    "change_24h": float(change_24h),
                    "last_update": datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.warning(f"Could not get current info for {symbol}: {e}")
            info["note"] = "Current market data unavailable"
        
        return info
    
    def get_all_fx_pairs_info(self) -> List[Dict[str, Any]]:
        """Get information about all supported FX pairs"""
        fx_pairs_info = []
        
        for symbol in self.FX_PAIRS.keys():
            try:
                info = self.get_fx_pair_info(symbol)
                fx_pairs_info.append({
                    "symbol": symbol,
                    **info
                })
            except Exception as e:
                logger.warning(f"Error getting info for {symbol}: {e}")
                fx_pairs_info.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return fx_pairs_info
    
    def get_early_warning_signals(
        self,
        days_back: int = 5,
        threshold_pct: float = 2.0
    ) -> Dict[str, Any]:
        """Get early warning signals from key FX pairs"""
        try:
            # Key pairs for early warning
            warning_pairs = ['USDJPY=X', 'EURJPY=X', 'USDCNH=X', 'USDKRW=X']
            
            signals = {}
            alerts = []
            
            for symbol in warning_pairs:
                try:
                    data = self.get_fx_data(symbol, days_back=days_back)
                    if not data.empty and len(data) >= 2:
                        current = data['Close'].iloc[-1]
                        previous = data['Close'].iloc[-2]
                        change_pct = ((current - previous) / previous * 100)
                        
                        pair_info = self.FX_PAIRS[symbol]
                        
                        signals[symbol] = {
                            "name": pair_info['name'],
                            "current_rate": float(current),
                            "change_24h": float(change_pct),
                            "signal_strength": pair_info['signal_strength'],
                            "crypto_correlation": pair_info['crypto_correlation'],
                            "mechanism": pair_info['mechanism']
                        }
                        
                        # Check for alert conditions
                        if abs(change_pct) >= threshold_pct:
                            alert_type = "CRYPTO_RISK" if change_pct < -threshold_pct else "CRYPTO_OPPORTUNITY"
                            alerts.append({
                                "fx_pair": pair_info['name'],
                                "alert_type": alert_type,
                                "change_pct": float(change_pct),
                                "impact": f"{'Negative' if change_pct < 0 else 'Positive'} for {', '.join(pair_info['best_for'])}",
                                "mechanism": pair_info['mechanism']
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing {symbol} for early warning: {e}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "signals": signals,
                "alerts": alerts,
                "alert_count": len(alerts),
                "summary": f"Monitoring {len(signals)} key FX pairs for crypto signals"
            }
            
        except Exception as e:
            logger.error(f"Error generating early warning signals: {e}")
            return {"error": str(e)}
    
    def calculate_fx_crypto_correlation(
        self,
        fx_data: pd.DataFrame,
        crypto_data: pd.DataFrame,
        fx_symbol: str
    ) -> Dict[str, float]:
        """Calculate correlation between FX pair and crypto"""
        try:
            # Align data by date
            fx_returns = fx_data['Close'].pct_change().dropna()
            crypto_returns = crypto_data['Close'].pct_change().dropna()
            
            aligned_data = pd.concat([fx_returns, crypto_returns], axis=1, keys=['FX', 'CRYPTO']).dropna()
            
            if len(aligned_data) < 10:
                return {"error": "Insufficient data for correlation"}
            
            # Calculate correlations
            correlation_returns = aligned_data['FX'].corr(aligned_data['CRYPTO'])
            
            # Price level correlation
            aligned_prices = pd.concat([fx_data['Close'], crypto_data['Close']], axis=1, keys=['FX', 'CRYPTO']).dropna()
            correlation_prices = aligned_prices['FX'].corr(aligned_prices['CRYPTO']) if len(aligned_prices) >= 10 else None
            
            pair_info = self.FX_PAIRS.get(fx_symbol, {})
            
            return {
                "fx_pair": pair_info.get('name', fx_symbol),
                "correlation_returns": float(correlation_returns),
                "correlation_prices": float(correlation_prices) if correlation_prices is not None else None,
                "expected_correlation": pair_info.get('crypto_correlation', 0),
                "data_points": len(aligned_data),
                "signal_strength": pair_info.get('signal_strength', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error calculating FX-crypto correlation: {e}")
            return {"error": str(e)}


# Singleton instance
_fx_rates_client = None

def get_fx_rates_client() -> FXRatesClient:
    """Get singleton FX rates client instance"""
    global _fx_rates_client
    if _fx_rates_client is None:
        _fx_rates_client = FXRatesClient()
    return _fx_rates_client