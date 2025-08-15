"""
Comprehensive Indicators Yahoo Finance Client
Loads institutional-grade financial indicators from Yahoo Finance for cryptocurrency prediction
Based on strategies from Two Sigma, Deribit, Jump Trading, and Galaxy Digital
"""

import yfinance as yf
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class ComprehensiveIndicatorsClient:
    """Yahoo Finance client for comprehensive financial indicators that predict crypto movements"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the comprehensive indicators client
        
        Args:
            config_path: Optional path to enhanced_indicators.yaml config file
        """
        self.tickers = {}
        self._ticker_info_cache = {}
        
        # Load configuration if provided
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.COMPREHENSIVE_INDICATORS = self._build_indicators_from_config()
        else:
            # Use built-in indicators
            self.COMPREHENSIVE_INDICATORS = self._get_default_indicators()
        
        # Build crypto predictor mappings
        self.CRYPTO_COMPREHENSIVE_PREDICTORS = self._build_crypto_predictors()
        
        # Build predictive tiers
        self.PREDICTIVE_TIERS = self._build_predictive_tiers()
        
        logger.info(f"Comprehensive indicators client initialized with {len(self.COMPREHENSIVE_INDICATORS)} indicators")
    
    def _get_default_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Get default comprehensive indicators"""
        return {
            # Volatility & Fear Metrics
            '^VIX': {
                'name': 'CBOE Volatility Index', 
                'category': 'volatility',
                'crypto_correlation': -0.75,
                'predictive_power': 'very_high',
                'signal': 'VIX >25 = crypto selloff warning',
                'weight': 0.20,
                'institutional_usage': 'Deribit DVOL constructed similarly'
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
            '^SKEW': {
                'name': 'CBOE Skew Index',
                'category': 'volatility',
                'crypto_correlation': -0.55,
                'predictive_power': 'medium',
                'signal': 'Tail risk measurement',
                'weight': 0.04
            },
            
            # Treasury Yields & Interest Rates
            '^TNX': {
                'name': '10-Year Treasury Yield',
                'category': 'rates',
                'crypto_correlation': -0.60,
                'predictive_power': 'very_high',
                'signal': 'Risk-free rate benchmark',
                'weight': 0.15,
                'institutional_usage': 'Deribit explicitly analyzes in market reports'
            },
            '^TYX': {
                'name': '30-Year Treasury Yield',
                'category': 'rates',
                'crypto_correlation': -0.55,
                'predictive_power': 'high',
                'signal': 'Long-term bond sentiment',
                'weight': 0.08
            },
            '^FVX': {
                'name': '5-Year Treasury Yield',
                'category': 'rates',
                'crypto_correlation': -0.58,
                'predictive_power': 'high',
                'signal': 'Medium-term rates',
                'weight': 0.07
            },
            '^IRX': {
                'name': '3-Month Treasury Bill',
                'category': 'rates',
                'crypto_correlation': -0.50,
                'predictive_power': 'medium',
                'signal': 'Short-term rate expectations',
                'weight': 0.05
            },
            
            # Major Market Indices
            '^GSPC': {
                'name': 'S&P 500 Index',
                'category': 'equity',
                'crypto_correlation': 0.85,
                'predictive_power': 'very_high',
                'signal': 'Broad market sentiment',
                'weight': 0.15,
                'correlation_note': 'BTC correlation reached 60% during COVID crisis'
            },
            '^IXIC': {
                'name': 'NASDAQ Composite',
                'category': 'equity',
                'crypto_correlation': 0.88,
                'predictive_power': 'very_high',
                'signal': 'Tech sector proxy',
                'weight': 0.12
            },
            '^DJI': {
                'name': 'Dow Jones Industrial',
                'category': 'equity',
                'crypto_correlation': 0.75,
                'predictive_power': 'high',
                'signal': 'Industrial sentiment',
                'weight': 0.08
            },
            '^RUT': {
                'name': 'Russell 2000 Index',
                'category': 'equity',
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
                'weight': 0.10,
                'correlation_note': 'Galaxy Digital compares BTC to gold\'s $17.8T market'
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
                'category': 'sector',
                'crypto_correlation': 0.90,
                'predictive_power': 'very_high',
                'signal': 'Tech sector flows',
                'weight': 0.12,
                'correlation_note': 'Crypto often exhibits tech-stock behavior patterns'
            },
            'XLF': {
                'name': 'Financial Select Sector',
                'category': 'sector',
                'crypto_correlation': 0.75,
                'predictive_power': 'high',
                'signal': 'Banking sentiment',
                'weight': 0.06
            },
            'XLE': {
                'name': 'Energy Select Sector',
                'category': 'sector',
                'crypto_correlation': 0.50,
                'predictive_power': 'medium',
                'signal': 'Energy sector correlation',
                'weight': 0.04
            },
            'XLU': {
                'name': 'Utilities Select Sector',
                'category': 'sector',
                'crypto_correlation': -0.40,
                'predictive_power': 'medium',
                'signal': 'Defensive positioning indicator',
                'weight': 0.03
            },
            'XLRE': {
                'name': 'Real Estate Select Sector',
                'category': 'sector',
                'crypto_correlation': -0.30,
                'predictive_power': 'medium',
                'signal': 'Real estate vs crypto competition',
                'weight': 0.03
            },
            'QQQ': {
                'name': 'Invesco QQQ Trust',
                'category': 'sector',
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
                'weight': 0.15,
                'correlation_note': 'CoinGlass confirms inverse relationship with crypto'
            },
            
            # Fixed Income ETFs (High Priority - Institutional Usage)
            'TIP': {
                'name': 'iShares TIPS Bond ETF',
                'category': 'fixed_income',
                'crypto_correlation': 0.35,
                'predictive_power': 'high',
                'signal': 'Inflation protection',
                'weight': 0.08,
                'institutional_usage': 'Two Sigma: 0.76 beta to Bitcoin vs 0.09 for gold',
                'correlation_note': 'Two Sigma found 15% BTC correlation to 10Y breakevens'
            },
            'TLT': {
                'name': 'iShares 20+ Year Treasury ETF',
                'category': 'fixed_income', 
                'crypto_correlation': -0.65,
                'predictive_power': 'high',
                'signal': 'Long-term rate sensitivity',
                'weight': 0.07
            },
            'SHY': {
                'name': 'iShares 1-3 Year Treasury ETF',
                'category': 'fixed_income',
                'crypto_correlation': -0.45,
                'predictive_power': 'medium',
                'signal': 'Short-term rate positioning',
                'weight': 0.04
            },
            'LQD': {
                'name': 'iShares Investment Grade Corporate Bond ETF',
                'category': 'fixed_income',
                'crypto_correlation': -0.55,
                'predictive_power': 'high',
                'signal': 'Credit spread analysis',
                'weight': 0.06
            },
            'HYG': {
                'name': 'iShares High Yield Corporate Bond ETF',
                'category': 'fixed_income',
                'crypto_correlation': 0.70,
                'predictive_power': 'high',
                'signal': 'Risk appetite indicator',
                'weight': 0.08
            },
            
            # Currency ETFs (Professional Usage)
            'UUP': {
                'name': 'Invesco DB USD Bull ETF',
                'category': 'currency',
                'crypto_correlation': -0.75,
                'predictive_power': 'high',
                'signal': 'Alternative DXY tracking',
                'weight': 0.06
            },
            'FXE': {
                'name': 'Invesco CurrencyShares Euro ETF',
                'category': 'currency',
                'crypto_correlation': 0.65,
                'predictive_power': 'medium',
                'signal': 'EUR/USD exposure',
                'weight': 0.04
            },
            'FXY': {
                'name': 'Invesco CurrencyShares Japanese Yen ETF',
                'category': 'currency',
                'crypto_correlation': -0.50,
                'predictive_power': 'medium',
                'signal': 'JPY carry trade indicator',
                'weight': 0.04
            },
            
            # Additional Sector ETFs
            'IYR': {
                'name': 'iShares Real Estate ETF',
                'category': 'sector',
                'crypto_correlation': -0.30,
                'predictive_power': 'medium',
                'signal': 'Alternative inflation hedge',
                'weight': 0.03
            },
            
            # International Exposure (Professional Global Analysis)
            'EEM': {
                'name': 'iShares MSCI Emerging Markets ETF',
                'category': 'international',
                'crypto_correlation': 0.80,
                'predictive_power': 'high',
                'signal': 'Risk-on/risk-off sentiment',
                'weight': 0.07,
                'correlation_note': 'Two Sigma found negative EM exposure in BTC'
            },
            'EFA': {
                'name': 'iShares MSCI EAFE ETF',
                'category': 'international',
                'crypto_correlation': 0.70,
                'predictive_power': 'medium',
                'signal': 'Global risk sentiment',
                'weight': 0.05
            },
            'VGK': {
                'name': 'iShares MSCI Europe ETF',
                'category': 'international',
                'crypto_correlation': 0.65,
                'predictive_power': 'medium',
                'signal': 'European correlation',
                'weight': 0.04
            },
            'EWJ': {
                'name': 'iShares MSCI Japan ETF',
                'category': 'international',
                'crypto_correlation': 0.60,
                'predictive_power': 'medium',
                'signal': 'Asian session correlation patterns',
                'weight': 0.04
            },
            'ASHR': {
                'name': 'Xtrackers Harvest CSI 300 China A-Shares ETF',
                'category': 'international',
                'crypto_correlation': 0.55,
                'predictive_power': 'medium',
                'signal': 'China exposure',
                'weight': 0.03
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
            
            # Commodity ETFs
            'DJP': {
                'name': 'iPath Bloomberg Commodity Index ETN',
                'category': 'commodity',
                'crypto_correlation': 0.50,
                'predictive_power': 'medium',
                'signal': 'Broad commodity exposure',
                'weight': 0.04
            },
            'PDBC': {
                'name': 'Invesco Optimum Yield Diversified Commodity ETF',
                'category': 'commodity',
                'crypto_correlation': 0.48,
                'predictive_power': 'medium',
                'signal': 'Alternative to DJP',
                'weight': 0.04
            },
            'USO': {
                'name': 'United States Oil Fund',
                'category': 'commodity',
                'crypto_correlation': 0.55,
                'predictive_power': 'medium',
                'signal': 'Energy correlation analysis',
                'weight': 0.04
            },
            'UNG': {
                'name': 'United States Natural Gas Fund',
                'category': 'commodity',
                'crypto_correlation': 0.30,
                'predictive_power': 'low',
                'signal': 'Energy complex',
                'weight': 0.02
            },
            
            # Credit & Liquidity Indicators
            'EMB': {
                'name': 'iShares J.P. Morgan USD Emerging Markets Bond ETF',
                'category': 'credit',
                'crypto_correlation': 0.65,
                'predictive_power': 'medium',
                'signal': 'EM risk appetite',
                'weight': 0.05
            },
            'BKLN': {
                'name': 'Invesco Senior Loan ETF',
                'category': 'credit',
                'crypto_correlation': 0.50,
                'predictive_power': 'medium',
                'signal': 'Floating rate exposure',
                'weight': 0.03
            },
            'MBB': {
                'name': 'iShares MBS ETF',
                'category': 'credit',
                'crypto_correlation': -0.40,
                'predictive_power': 'medium',
                'signal': 'Housing/liquidity proxy',
                'weight': 0.03
            },
            
            # Currency Pairs
            'USDCNY=X': {
                'name': 'USD/CNY Exchange Rate',
                'category': 'currency',
                'crypto_correlation': -0.60,
                'predictive_power': 'medium',
                'signal': 'China capital flow',
                'weight': 0.04
            }
        }
    
    def _build_indicators_from_config(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive indicators dictionary from config file"""
        indicators = {}
        
        # Map categories to default weights and predictive power
        category_weights = {
            'volatility': {'weight': 0.20, 'power': 'very_high'},
            'rates': {'weight': 0.15, 'power': 'very_high'},
            'fixed_income': {'weight': 0.10, 'power': 'high'},
            'currency': {'weight': 0.15, 'power': 'very_high'},
            'sector': {'weight': 0.12, 'power': 'high'},
            'commodity': {'weight': 0.10, 'power': 'medium'},
            'international': {'weight': 0.08, 'power': 'medium'},
            'equity': {'weight': 0.10, 'power': 'very_high'},
            'credit': {'weight': 0.05, 'power': 'medium'}
        }
        
        # Process each category in config
        for category_group, items in self.config.items():
            if category_group in ['weighting_model', 'regime_detection', 'correlation_windows', 'data_quality_requirements']:
                continue
                
            if isinstance(items, dict):
                for subcategory, tickers in items.items():
                    if isinstance(tickers, list):
                        for ticker_info in tickers:
                            symbol = ticker_info['symbol']
                            category = ticker_info.get('category', 'other')
                            
                            # Get default weights and power for category
                            cat_defaults = category_weights.get(category, {'weight': 0.05, 'power': 'medium'})
                            
                            indicators[symbol] = {
                                'name': ticker_info['name'],
                                'category': category,
                                'description': ticker_info.get('description', ''),
                                'crypto_correlation': ticker_info.get('crypto_correlation', 0.0),
                                'predictive_power': ticker_info.get('predictive_power', cat_defaults['power']),
                                'signal': ticker_info.get('description', ''),
                                'weight': ticker_info.get('weight', cat_defaults['weight']),
                                'institutional_usage': ticker_info.get('institutional_usage', ''),
                                'correlation_note': ticker_info.get('correlation_note', '')
                            }
        
        return indicators
    
    def _build_crypto_predictors(self) -> Dict[str, List[str]]:
        """Build crypto-specific predictor mappings"""
        # Enhanced predictors based on institutional usage
        return {
            'BTC': ['^VIX', '^TNX', '^GSPC', 'DX-Y.NYB', 'GC=F', '^IXIC', 'XLK', 'TIP', 'TLT', 'UUP', 'EEM', 'HYG'],
            'ETH': ['^IXIC', 'XLK', '^VIX', 'QQQ', '^TNX', '^GSPC', 'XLF', 'TIP', 'LQD', 'EFA', '^VIX9D'],
            'SOL': ['QQQ', 'XLK', '^IXIC', '^RUT', '^VIX', '^TNX', 'HG=F', 'EEM', 'VIXY', 'IYR'],
            'BNB': ['^VIX', '^GSPC', 'XLK', 'DX-Y.NYB', '^TNX', 'QQQ', '^IXIC', 'ASHR', 'FXE'],
            'XRP': ['^RUT', 'XLF', '^VIX', '^GSPC', 'DX-Y.NYB', '^TNX', 'QQQ', 'EMB', 'LQD'],
            'ADA': ['^RUT', 'XLK', 'QQQ', '^IXIC', '^VIX', '^TNX', 'HG=F', 'EEM', 'TIP'],
            'DOGE': ['^RUT', '^VIX', '^GSPC', 'QQQ', 'DX-Y.NYB', '^TNX', 'XLK', 'VIXY', '^SKEW'],
            'MATIC': ['QQQ', 'XLK', '^IXIC', '^VIX', '^TNX', '^GSPC', 'HG=F', 'EEM', 'VGK']
        }
    
    def _build_predictive_tiers(self) -> Dict[str, List[str]]:
        """Build predictive power tiers"""
        tiers = {'very_high': [], 'high': [], 'medium': [], 'low': []}
        
        for symbol, info in self.COMPREHENSIVE_INDICATORS.items():
            power = info.get('predictive_power', 'medium')
            if power in tiers:
                tiers[power].append(symbol)
        
        return tiers
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create ticker instance for symbol"""
        if symbol not in self.tickers:
            self.tickers[symbol] = yf.Ticker(symbol)
        return self.tickers[symbol]
    
    def validate_ticker_availability(self, symbol: str) -> Dict[str, Any]:
        """
        Validate if a ticker is available on Yahoo Finance
        
        Returns:
            Dict with availability status and data quality info
        """
        try:
            ticker = self.get_ticker(symbol)
            
            # Try to get recent data
            test_data = ticker.history(period='5d')
            
            if test_data.empty:
                return {
                    'symbol': symbol,
                    'available': False,
                    'error': 'No data returned'
                }
            
            # Get info about the ticker
            info = ticker.info if hasattr(ticker, 'info') else {}
            
            # Calculate data availability
            hist_1y = ticker.history(period='1y')
            hist_max = ticker.history(period='max')
            
            return {
                'symbol': symbol,
                'available': True,
                'name': info.get('longName', self.COMPREHENSIVE_INDICATORS.get(symbol, {}).get('name', '')),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'data_points_1y': len(hist_1y),
                'data_points_total': len(hist_max),
                'oldest_date': hist_max.index[0].strftime('%Y-%m-%d') if not hist_max.empty else None,
                'latest_date': hist_max.index[-1].strftime('%Y-%m-%d') if not hist_max.empty else None,
                'has_volume': 'Volume' in test_data.columns and test_data['Volume'].sum() > 0
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'available': False,
                'error': str(e)
            }
    
    def validate_all_tickers(self) -> Dict[str, List[Dict[str, Any]]]:
        """Validate availability of all configured tickers"""
        results = {'available': [], 'unavailable': [], 'errors': []}
        
        for symbol in self.COMPREHENSIVE_INDICATORS.keys():
            validation = self.validate_ticker_availability(symbol)
            
            if validation['available']:
                results['available'].append(validation)
            elif 'error' in validation:
                results['errors'].append(validation)
            else:
                results['unavailable'].append(validation)
            
            logger.info(f"Validated {symbol}: {'✓' if validation['available'] else '✗'}")
        
        return results
    
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
                # Check if it's a valid ticker anyway
                validation = self.validate_ticker_availability(symbol)
                if not validation['available']:
                    raise ValueError(f"Symbol {symbol} not available or not in comprehensive indicators")
            
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
    
    def get_regime_indicators(
        self,
        regime: str,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get indicators for specific market regime (risk_on or risk_off)"""
        # Import here to avoid circular dependency
        from .professional_trading_config import ProfessionalTradingConfig, MarketRegime
        
        try:
            market_regime = MarketRegime(regime.lower())
        except ValueError:
            raise ValueError(f"Regime must be one of: {[r.value for r in MarketRegime]}")
        
        regime_symbols = ProfessionalTradingConfig.get_regime_indicators(market_regime)
        logger.info(f"Loading {regime} regime indicators: {regime_symbols}")
        
        return self.get_multiple_comprehensive_data(regime_symbols, days_back, interval)
    
    def get_institutional_indicators(
        self,
        days_back: int = 365,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get indicators specifically mentioned in institutional usage"""
        institutional_symbols = [
            symbol for symbol, info in self.COMPREHENSIVE_INDICATORS.items()
            if info.get('institutional_usage', '') != ''
        ]
        
        logger.info(f"Loading institutional indicators: {institutional_symbols}")
        return self.get_multiple_comprehensive_data(institutional_symbols, days_back, interval)
    
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


# Maintain backward compatibility
ComprehensiveMetricsClient = ComprehensiveIndicatorsClient