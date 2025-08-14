"""
Comprehensive Metrics Yahoo Finance Client
Loads comprehensive financial data from Yahoo Finance for cryptocurrency prediction
"""

import yfinance as yf
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ComprehensiveMetricsClient:
    """Yahoo Finance client for comprehensive financial metrics that predict crypto movements"""
    
    # Comprehensive Yahoo Finance indicators organized by category
    COMPREHENSIVE_INDICATORS = {
        # Volatility & Fear Metrics
        '^VIX': {
            'name': 'CBOE Volatility Index', 
            'category': 'volatility',
            'crypto_correlation': -0.75,
            'predictive_power': 'very_high',
            'signal': 'VIX >25 = crypto selloff warning',
            'weight': 0.20
        },
        '^VVIX': {
            'name': 'VIX Volatility Index',
            'category': 'volatility', 
            'crypto_correlation': -0.65,
            'predictive_power': 'high',
            'signal': 'Volatility of volatility',
            'weight': 0.05
        },
        '^VIX9D': {
            'name': '9-Day VIX',
            'category': 'volatility',
            'crypto_correlation': -0.70,
            'predictive_power': 'high', 
            'signal': 'Short-term fear gauge',
            'weight': 0.10
        },
        '^OVX': {
            'name': 'Oil Volatility Index',
            'category': 'volatility',
            'crypto_correlation': -0.45,
            'predictive_power': 'medium',
            'signal': 'Energy volatility proxy',
            'weight': 0.03
        },
        '^GVZ': {
            'name': 'Gold Volatility Index', 
            'category': 'volatility',
            'crypto_correlation': 0.35,
            'predictive_power': 'medium',
            'signal': 'Safe haven volatility',
            'weight': 0.05
        },
        
        # Treasury Yields & Interest Rates
        '^TNX': {
            'name': '10-Year Treasury Yield',
            'category': 'bonds',
            'crypto_correlation': -0.60,
            'predictive_power': 'very_high',
            'signal': 'Risk-free rate benchmark',
            'weight': 0.15
        },
        '^TYX': {
            'name': '30-Year Treasury Yield',
            'category': 'bonds',
            'crypto_correlation': -0.55,
            'predictive_power': 'high',
            'signal': 'Long-term bond sentiment',
            'weight': 0.08
        },
        '^FVX': {
            'name': '5-Year Treasury Yield',
            'category': 'bonds',
            'crypto_correlation': -0.58,
            'predictive_power': 'high',
            'signal': 'Medium-term rates',
            'weight': 0.07
        },
        '^IRX': {
            'name': '3-Month Treasury Bill',
            'category': 'bonds',
            'crypto_correlation': -0.50,
            'predictive_power': 'medium',
            'signal': 'Short-term rate expectations',
            'weight': 0.05
        },
        
        # Major Market Indices
        '^GSPC': {
            'name': 'S&P 500 Index',
            'category': 'market_index',
            'crypto_correlation': 0.85,
            'predictive_power': 'very_high',
            'signal': 'Broad market sentiment',
            'weight': 0.15
        },
        '^IXIC': {
            'name': 'NASDAQ Composite',
            'category': 'market_index',
            'crypto_correlation': 0.88,
            'predictive_power': 'very_high',
            'signal': 'Tech sector proxy',
            'weight': 0.12
        },
        '^DJI': {
            'name': 'Dow Jones Industrial',
            'category': 'market_index',
            'crypto_correlation': 0.75,
            'predictive_power': 'high',
            'signal': 'Industrial sentiment',
            'weight': 0.08
        },
        '^RUT': {
            'name': 'Russell 2000 Index',
            'category': 'market_index',
            'crypto_correlation': 0.82,
            'predictive_power': 'high',
            'signal': 'Small cap risk appetite',
            'weight': 0.10
        },
        
        # International Indices
        '^N225': {
            'name': 'Nikkei 225',
            'category': 'international',
            'crypto_correlation': 0.65,
            'predictive_power': 'medium',
            'signal': 'Asian risk appetite',
            'weight': 0.05
        },
        '^FTSE': {
            'name': 'FTSE 100',
            'category': 'international',
            'crypto_correlation': 0.70,
            'predictive_power': 'medium',
            'signal': 'European sentiment',
            'weight': 0.05
        },
        '^GDAXI': {
            'name': 'DAX Index',
            'category': 'international',
            'crypto_correlation': 0.72,
            'predictive_power': 'medium',
            'signal': 'German economic health',
            'weight': 0.04
        },
        
        # Commodities Futures
        'GC=F': {
            'name': 'Gold Futures',
            'category': 'commodity',
            'crypto_correlation': 0.25,
            'predictive_power': 'medium',
            'signal': 'Safe haven competition',
            'weight': 0.10
        },
        'SI=F': {
            'name': 'Silver Futures',
            'category': 'commodity',
            'crypto_correlation': 0.45,
            'predictive_power': 'medium',
            'signal': 'Industrial metal proxy',
            'weight': 0.05
        },
        'HG=F': {
            'name': 'Copper Futures',
            'category': 'commodity',
            'crypto_correlation': 0.55,
            'predictive_power': 'medium',
            'signal': 'Economic growth indicator',
            'weight': 0.04
        },
        'CL=F': {
            'name': 'Crude Oil Futures',
            'category': 'commodity',
            'crypto_correlation': 0.60,
            'predictive_power': 'medium',
            'signal': 'Risk appetite gauge',
            'weight': 0.06
        },
        'NG=F': {
            'name': 'Natural Gas Futures',
            'category': 'commodity',
            'crypto_correlation': 0.35,
            'predictive_power': 'low',
            'signal': 'Energy transition indicator',
            'weight': 0.02
        },
        
        # Sector ETFs
        'XLK': {
            'name': 'Technology Select Sector',
            'category': 'sector_etf',
            'crypto_correlation': 0.90,
            'predictive_power': 'very_high',
            'signal': 'Tech sector flows',
            'weight': 0.12
        },
        'XLF': {
            'name': 'Financial Select Sector',
            'category': 'sector_etf',
            'crypto_correlation': 0.75,
            'predictive_power': 'high',
            'signal': 'Banking sentiment',
            'weight': 0.06
        },
        'XLE': {
            'name': 'Energy Select Sector',
            'category': 'sector_etf',
            'crypto_correlation': 0.50,
            'predictive_power': 'medium',
            'signal': 'Energy sector correlation',
            'weight': 0.04
        },
        'XLRE': {
            'name': 'Real Estate Select Sector',
            'category': 'sector_etf',
            'crypto_correlation': -0.30,
            'predictive_power': 'medium',
            'signal': 'Real estate vs crypto competition',
            'weight': 0.03
        },
        'QQQ': {
            'name': 'Invesco QQQ Trust',
            'category': 'sector_etf',
            'crypto_correlation': 0.88,
            'predictive_power': 'very_high',
            'signal': 'NASDAQ 100 proxy',
            'weight': 0.10
        },
        
        # Dollar Index
        'DX-Y.NYB': {
            'name': 'US Dollar Index',
            'category': 'currency',
            'crypto_correlation': -0.80,
            'predictive_power': 'very_high',
            'signal': 'Dollar strength inverse',
            'weight': 0.15
        },
        
        # Fixed Income ETFs (High Priority - Institutional Usage)
        'TIP': {
            'name': 'iShares TIPS Bond ETF',
            'category': 'fixed_income_etf',
            'crypto_correlation': 0.35,
            'predictive_power': 'high',
            'signal': 'Inflation protection - Two Sigma validated',
            'weight': 0.08,
            'institutional_usage': 'Two Sigma (0.76 beta to Bitcoin)'
        },
        'TLT': {
            'name': 'iShares 20+ Year Treasury ETF',
            'category': 'fixed_income_etf', 
            'crypto_correlation': -0.65,
            'predictive_power': 'high',
            'signal': 'Long-term rate sensitivity',
            'weight': 0.07
        },
        'SHY': {
            'name': 'iShares 1-3 Year Treasury ETF',
            'category': 'fixed_income_etf',
            'crypto_correlation': -0.45,
            'predictive_power': 'medium',
            'signal': 'Short-term rate positioning',
            'weight': 0.04
        },
        'LQD': {
            'name': 'iShares Investment Grade Corporate Bond ETF',
            'category': 'fixed_income_etf',
            'crypto_correlation': -0.55,
            'predictive_power': 'high',
            'signal': 'Credit spread analysis',
            'weight': 0.06
        },
        'HYG': {
            'name': 'iShares High Yield Corporate Bond ETF',
            'category': 'fixed_income_etf',
            'crypto_correlation': 0.70,
            'predictive_power': 'high',
            'signal': 'Risk appetite indicator',
            'weight': 0.08
        },
        
        # Currency ETFs (Professional Usage)
        'UUP': {
            'name': 'Invesco DB USD Bull ETF',
            'category': 'currency_etf',
            'crypto_correlation': -0.75,
            'predictive_power': 'high',
            'signal': 'Alternative DXY tracking',
            'weight': 0.06
        },
        'FXE': {
            'name': 'Invesco CurrencyShares Euro ETF',
            'category': 'currency_etf',
            'crypto_correlation': 0.65,
            'predictive_power': 'medium',
            'signal': 'EUR/USD exposure',
            'weight': 0.04
        },
        'FXY': {
            'name': 'Invesco CurrencyShares Japanese Yen ETF',
            'category': 'currency_etf',
            'crypto_correlation': -0.50,
            'predictive_power': 'medium',
            'signal': 'JPY carry trade indicator',
            'weight': 0.04
        },
        
        # Additional Sector ETFs (Institutional Rotation)
        'XLF': {
            'name': 'Financial Select Sector SPDR Fund',
            'category': 'sector_etf',
            'crypto_correlation': 0.75,
            'predictive_power': 'high',
            'signal': 'Rate sensitivity proxy',
            'weight': 0.06
        },
        'XLE': {
            'name': 'Energy Select Sector SPDR Fund',
            'category': 'sector_etf',
            'crypto_correlation': 0.50,
            'predictive_power': 'medium',
            'signal': 'Inflation hedge correlation',
            'weight': 0.04
        },
        'XLU': {
            'name': 'Utilities Select Sector SPDR Fund',
            'category': 'sector_etf',
            'crypto_correlation': -0.40,
            'predictive_power': 'medium',
            'signal': 'Defensive positioning indicator',
            'weight': 0.03
        },
        'IYR': {
            'name': 'iShares Real Estate ETF',
            'category': 'sector_etf',
            'crypto_correlation': -0.30,
            'predictive_power': 'medium',
            'signal': 'Alternative inflation hedge',
            'weight': 0.03
        },
        
        # International Exposure (Professional Global Analysis)
        'EEM': {
            'name': 'iShares MSCI Emerging Markets ETF',
            'category': 'international_etf',
            'crypto_correlation': 0.80,
            'predictive_power': 'high',
            'signal': 'Risk-on/risk-off sentiment',
            'weight': 0.07
        },
        'EFA': {
            'name': 'iShares MSCI EAFE ETF',
            'category': 'international_etf',
            'crypto_correlation': 0.70,
            'predictive_power': 'medium',
            'signal': 'Global risk sentiment',
            'weight': 0.05
        },
        
        # Additional Volatility (Professional Trading)
        'VIXY': {
            'name': 'ProShares VIX Short-Term Futures ETF',
            'category': 'volatility',
            'crypto_correlation': -0.80,
            'predictive_power': 'very_high',
            'signal': 'Volatility trading instrument',
            'weight': 0.08
        },
        '^SKEW': {
            'name': 'CBOE Skew Index',
            'category': 'volatility',
            'crypto_correlation': -0.55,
            'predictive_power': 'medium',
            'signal': 'Tail risk measurement',
            'weight': 0.04
        }
    }
    
    # Crypto-specific comprehensive predictors
    CRYPTO_COMPREHENSIVE_PREDICTORS = {
        'BTC': ['^VIX', '^TNX', '^GSPC', 'DX-Y.NYB', 'GC=F', '^IXIC', 'XLK'],
        'ETH': ['^IXIC', 'XLK', '^VIX', 'QQQ', '^TNX', '^GSPC', 'XLF'],
        'SOL': ['QQQ', 'XLK', '^IXIC', '^RUT', '^VIX', '^TNX', 'HG=F'],
        'BNB': ['^VIX', '^GSPC', 'XLK', 'DX-Y.NYB', '^TNX', 'QQQ', '^IXIC'],
        'XRP': ['^RUT', 'XLF', '^VIX', '^GSPC', 'DX-Y.NYB', '^TNX', 'QQQ'],
        'ADA': ['^RUT', 'XLK', 'QQQ', '^IXIC', '^VIX', '^TNX', 'HG=F'],
        'DOGE': ['^RUT', '^VIX', '^GSPC', 'QQQ', 'DX-Y.NYB', '^TNX', 'XLK'],
        'MATIC': ['QQQ', 'XLK', '^IXIC', '^VIX', '^TNX', '^GSPC', 'HG=F']
    }
    
    # Predictive power tiers
    PREDICTIVE_TIERS = {
        'very_high': ['^VIX', '^TNX', '^GSPC', '^IXIC', 'DX-Y.NYB', 'XLK'],
        'high': ['^RUT', '^TYX', '^FVX', '^DJI', 'QQQ', 'XLF', '^VIX9D'],
        'medium': ['GC=F', 'CL=F', '^N225', '^FTSE', '^GDAXI', 'HG=F', 'SI=F']
    }
    
    def __init__(self):
        self.tickers = {}
        self._ticker_info_cache = {}
        logger.info(f"Comprehensive metrics client initialized with {len(self.COMPREHENSIVE_INDICATORS)} indicators")
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create ticker instance for symbol"""
        if symbol not in self.tickers:
            self.tickers[symbol] = yf.Ticker(symbol)
        return self.tickers[symbol]
    
    def get_comprehensive_data(
        self,
        symbol: str,
        days_back: int = 365,
        interval: str = '1d',
        prepost: bool = False,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data for comprehensive financial indicator
        
        Args:
            symbol: Financial indicator symbol (^VIX, ^TNX, ^GSPC, etc.)
            days_back: Number of days of historical data
            interval: Data interval (1d, 1h, 5m, etc.)
            prepost: Include pre/post market data
            auto_adjust: Auto adjust for splits/dividends
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            if symbol not in self.COMPREHENSIVE_INDICATORS:
                raise ValueError(f"Symbol {symbol} not in comprehensive indicators: {list(self.COMPREHENSIVE_INDICATORS.keys())}")
            
            ticker = self.get_ticker(symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Fetching {symbol} comprehensive data from {start_date} to {end_date}")
            
            # Download historical data
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost,
                actions=False,  # Most indicators don't have corporate actions
                raise_errors=True
            )
            
            if hist_data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(hist_data)} records for {symbol}")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive data for {symbol}: {e}")
            raise
    
    def get_multiple_comprehensive_data(
        self,
        symbols: List[str],
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple comprehensive indicators"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_comprehensive_data(symbol, days_back, interval)
                if not data.empty:
                    results[symbol] = data
                    logger.info(f"✓ Loaded {len(data)} records for {symbol}")
                else:
                    logger.warning(f"✗ No data for {symbol}")
            except Exception as e:
                logger.error(f"✗ Error loading {symbol}: {e}")
        
        return results
    
    def get_comprehensive_predictors_for_crypto(
        self,
        crypto_symbol: str,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get comprehensive predictors specific to a crypto symbol"""
        try:
            # Remove common suffixes to get base symbol
            base_symbol = crypto_symbol.upper().replace('-USD', '').replace('USDT', '').replace('-USDT', '')
            
            if base_symbol not in self.CRYPTO_COMPREHENSIVE_PREDICTORS:
                raise ValueError(f"Crypto symbol {base_symbol} not supported. Available: {list(self.CRYPTO_COMPREHENSIVE_PREDICTORS.keys())}")
            
            predictor_symbols = self.CRYPTO_COMPREHENSIVE_PREDICTORS[base_symbol]
            logger.info(f"Loading {len(predictor_symbols)} comprehensive predictors for {crypto_symbol}: {predictor_symbols}")
            
            return self.get_multiple_comprehensive_data(predictor_symbols, days_back, interval)
            
        except Exception as e:
            logger.error(f"Error getting comprehensive predictors for {crypto_symbol}: {e}")
            return {}
    
    def get_tier_indicators(
        self,
        tier: str,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get indicators by predictive power tier"""
        if tier.lower() not in self.PREDICTIVE_TIERS:
            raise ValueError(f"Tier must be one of: {list(self.PREDICTIVE_TIERS.keys())}")
        
        tier_symbols = self.PREDICTIVE_TIERS[tier.lower()]
        logger.info(f"Loading {tier} tier indicators: {tier_symbols}")
        
        return self.get_multiple_comprehensive_data(tier_symbols, days_back, interval)
    
    def get_category_indicators(
        self,
        category: str,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get indicators by category"""
        category_symbols = [
            symbol for symbol, info in self.COMPREHENSIVE_INDICATORS.items() 
            if info['category'] == category.lower()
        ]
        
        if not category_symbols:
            available_categories = list(set(info['category'] for info in self.COMPREHENSIVE_INDICATORS.values()))
            raise ValueError(f"Category {category} not found. Available: {available_categories}")
        
        logger.info(f"Loading {category} category indicators: {category_symbols}")
        return self.get_multiple_comprehensive_data(category_symbols, days_back, interval)
    
    def get_indicator_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about a specific comprehensive indicator"""
        if symbol not in self.COMPREHENSIVE_INDICATORS:
            return {"error": f"Symbol {symbol} not found"}
        
        info = self.COMPREHENSIVE_INDICATORS[symbol].copy()
        
        try:
            # Get recent price data for current values
            recent_data = self.get_comprehensive_data(symbol, days_back=5)
            if not recent_data.empty:
                current_value = recent_data['Close'].iloc[-1]
                prev_close = recent_data['Close'].iloc[-2] if len(recent_data) > 1 else current_value
                change_24h = ((current_value - prev_close) / prev_close * 100) if prev_close > 0 else 0
                
                info.update({
                    "current_value": float(current_value),
                    "previous_close": float(prev_close),
                    "change_24h": float(change_24h),
                    "last_update": datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.warning(f"Could not get current info for {symbol}: {e}")
            info["note"] = "Current market data unavailable"
        
        return info
    
    def get_all_indicators_info(self) -> List[Dict[str, Any]]:
        """Get information about all comprehensive indicators"""
        indicators_info = []
        
        for symbol in self.COMPREHENSIVE_INDICATORS.keys():
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
    
    def calculate_predictive_score(
        self,
        crypto_data: pd.DataFrame,
        comprehensive_data: Dict[str, pd.DataFrame],
        crypto_symbol: str = 'BTC'
    ) -> Dict[str, float]:
        """Calculate weighted predictive score for crypto movements"""
        try:
            predictive_scores = {}
            
            # Get crypto returns
            crypto_returns = crypto_data['Close'].pct_change().dropna()
            
            total_weighted_correlation = 0
            total_weight = 0
            
            for symbol, data in comprehensive_data.items():
                if not data.empty and symbol in self.COMPREHENSIVE_INDICATORS:
                    # Calculate correlation
                    indicator_returns = data['Close'].pct_change().dropna()
                    
                    # Align data
                    aligned_data = pd.concat([crypto_returns, indicator_returns], axis=1, keys=['crypto', 'indicator']).dropna()
                    
                    if len(aligned_data) >= 10:
                        correlation = aligned_data['crypto'].corr(aligned_data['indicator'])
                        weight = self.COMPREHENSIVE_INDICATORS[symbol]['weight']
                        
                        predictive_scores[symbol] = {
                            "correlation": float(correlation),
                            "weight": weight,
                            "weighted_score": float(abs(correlation) * weight),
                            "data_points": len(aligned_data),
                            "expected_correlation": self.COMPREHENSIVE_INDICATORS[symbol]['crypto_correlation']
                        }
                        
                        total_weighted_correlation += abs(correlation) * weight
                        total_weight += weight
            
            # Calculate overall predictive power
            overall_score = total_weighted_correlation / total_weight if total_weight > 0 else 0
            
            return {
                "crypto_symbol": crypto_symbol,
                "overall_predictive_score": float(overall_score),
                "total_weight": float(total_weight),
                "indicators_analyzed": len(predictive_scores),
                "indicator_scores": predictive_scores,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating predictive score: {e}")
            return {"error": str(e)}
    
    def get_alert_conditions(self, days_back: int = 5) -> Dict[str, Any]:
        """Get alert conditions from key comprehensive indicators"""
        try:
            # Key alert indicators
            alert_indicators = ['^VIX', '^TNX', '^GSPC', 'DX-Y.NYB', 'GC=F']
            
            alerts = []
            indicator_status = {}
            
            for symbol in alert_indicators:
                try:
                    data = self.get_comprehensive_data(symbol, days_back=days_back)
                    if not data.empty and len(data) >= 2:
                        current = data['Close'].iloc[-1]
                        previous = data['Close'].iloc[-2]
                        change_pct = ((current - previous) / previous * 100)
                        
                        indicator_info = self.COMPREHENSIVE_INDICATORS[symbol]
                        
                        # Define alert thresholds
                        threshold_map = {
                            '^VIX': {'high': 25, 'critical': 30},
                            '^TNX': {'change': 0.15},  # 15 basis points
                            '^GSPC': {'change': 2.0},  # 2% change
                            'DX-Y.NYB': {'change': 1.5},  # 1.5% change
                            'GC=F': {'change': 2.0}  # 2% change
                        }
                        
                        indicator_status[symbol] = {
                            "name": indicator_info['name'],
                            "current_value": float(current),
                            "change_24h": float(change_pct),
                            "signal": indicator_info['signal'],
                            "predictive_power": indicator_info['predictive_power']
                        }
                        
                        # Check alert conditions
                        if symbol == '^VIX':
                            if current >= threshold_map[symbol]['critical']:
                                alerts.append({
                                    "indicator": indicator_info['name'],
                                    "alert_type": "CRITICAL_CRYPTO_RISK",
                                    "value": float(current),
                                    "threshold": threshold_map[symbol]['critical'],
                                    "message": f"VIX at {current:.1f} - Major crypto selloff warning"
                                })
                            elif current >= threshold_map[symbol]['high']:
                                alerts.append({
                                    "indicator": indicator_info['name'],
                                    "alert_type": "HIGH_CRYPTO_RISK",
                                    "value": float(current),
                                    "threshold": threshold_map[symbol]['high'],
                                    "message": f"VIX at {current:.1f} - Elevated crypto risk"
                                })
                        else:
                            if abs(change_pct) >= threshold_map[symbol]['change']:
                                alert_type = "CRYPTO_OPPORTUNITY" if (
                                    (symbol in ['^GSPC', 'GC=F'] and change_pct > 0) or
                                    (symbol in ['^TNX', 'DX-Y.NYB'] and change_pct < 0)
                                ) else "CRYPTO_RISK"
                                
                                alerts.append({
                                    "indicator": indicator_info['name'],
                                    "alert_type": alert_type,
                                    "change_pct": float(change_pct),
                                    "threshold": threshold_map[symbol]['change'],
                                    "message": f"{indicator_info['name']} moved {change_pct:+.1f}% - {alert_type.lower().replace('_', ' ')}"
                                })
                        
                except Exception as e:
                    logger.error(f"Error processing alert for {symbol}: {e}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "alerts": alerts,
                "alert_count": len(alerts),
                "indicators_monitored": indicator_status,
                "summary": f"Monitoring {len(alert_indicators)} key indicators, {len(alerts)} alerts triggered"
            }
            
        except Exception as e:
            logger.error(f"Error getting alert conditions: {e}")
            return {"error": str(e)}


# Singleton instance
_comprehensive_metrics_client = None

def get_comprehensive_metrics_client() -> ComprehensiveMetricsClient:
    """Get singleton comprehensive metrics client instance"""
    global _comprehensive_metrics_client
    if _comprehensive_metrics_client is None:
        _comprehensive_metrics_client = ComprehensiveMetricsClient()
    return _comprehensive_metrics_client