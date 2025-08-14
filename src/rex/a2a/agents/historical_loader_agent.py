"""
Historical Data Loader Agent powered by Strand Agents
Handles bulk loading of historical crypto data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from strands import tool
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging

from .memory_strands_agent import MemoryStrandsAgent
from ...ml.yfinance_client import get_yfinance_client
from ...ml.multi_crypto_yfinance_client import get_multi_crypto_client
from ...ml.equity_indicators_client import get_equity_indicators_client
from ...ml.fx_rates_client import get_fx_rates_client
from ..protocols import MessageType

# Import observability
from ...observability import (
    get_logger, get_tracer, get_error_tracker, get_business_metrics,
    observable_agent_method, track_errors, ErrorSeverity, ErrorCategory,
    trace_context
)

logger = get_logger(__name__)
tracer = get_tracer()
error_tracker = get_error_tracker()
business_metrics = get_business_metrics()

class HistoricalLoaderAgent(MemoryStrandsAgent):
    def __init__(self, model_provider: str = "grok4", private_key: str = None, use_blockchain: bool = False):
        self.yf_client = get_yfinance_client()  # ETH-specific client
        self.multi_crypto_client = get_multi_crypto_client()  # Multi-crypto client
        self.equity_client = get_equity_indicators_client()  # Equity indicators client
        self.fx_client = get_fx_rates_client()  # FX rates client
        
        # Initialize A2A agent (simplified - no blockchain for now to avoid circular imports)
        super().__init__(
            agent_id='historical-loader-001',
            agent_type='historical_loader',
            capabilities=['data_loading', 'historical_data', 'technical_indicators', 'bulk_processing', 'multi_crypto', 'equity_indicators', 'fx_rates'],
            model_provider=model_provider
        )
        
    @observable_agent_method("historical-loader-001", "process_request") 
    @track_errors(severity=ErrorSeverity.HIGH, category=ErrorCategory.API_ERROR)
    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process agent request with full observability"""
        with trace_context("historical_loader_process_request"):
            logger.info("Processing historical loader request", extra={'request': request[:200]})
            
            # Use the agent model to process the request
            response = self.agent.run([{"role": "user", "content": request}])
            
            logger.info("Historical loader request completed")
            return {"success": True, "response": response}

    def _create_tools(self):
        """Create historical loader specific tools"""
        @tool
        def load_symbol_data(symbol: str, days_back: int = 365, interval: str = '1d', prepost: bool = False, auto_adjust: bool = True, include_indicators: bool = False) -> Dict[str, Any]:
            """Load historical data for top 10 crypto trading pairs from Yahoo Finance
            
            Args:
                symbol: Crypto symbol (BTC, ETH, BTC-USD, etc.)
                days_back: Number of days to load (any integer, default: 365)
                interval: Data granularity: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (default: 1d)
                prepost: Include pre/post market data (default: False)
                auto_adjust: Auto adjust for splits/dividends (default: True)
                include_indicators: Whether to include technical indicators (default: False)
            """
            with trace_context(f"load_symbol_data_{symbol}"):
                start_time = datetime.now()
                
                try:
                    # Support top 10 crypto trading pairs
                    supported_symbols = {
                        'BTC': 'BTC-USD', 'BTC-USD': 'BTC-USD', 'BTCUSDT': 'BTC-USD',
                        'ETH': 'ETH-USD', 'ETH-USD': 'ETH-USD', 'ETHUSDT': 'ETH-USD', 
                        'SOL': 'SOL-USD', 'SOL-USD': 'SOL-USD', 'SOLUSDT': 'SOL-USD',
                        'BNB': 'BNB-USD', 'BNB-USD': 'BNB-USD', 'BNBUSDT': 'BNB-USD',
                        'XRP': 'XRP-USD', 'XRP-USD': 'XRP-USD', 'XRPUSDT': 'XRP-USD',
                        'ADA': 'ADA-USD', 'ADA-USD': 'ADA-USD', 'ADAUSDT': 'ADA-USD',
                        'DOGE': 'DOGE-USD', 'DOGE-USD': 'DOGE-USD', 'DOGEUSDT': 'DOGE-USD',
                        'MATIC': 'MATIC-USD', 'MATIC-USD': 'MATIC-USD', 'MATICUSDT': 'MATIC-USD'
                    }
                
                    yahoo_symbol = supported_symbols.get(symbol.upper())
                    if not yahoo_symbol:
                        return {
                            "success": False,
                            "error": f"Symbol {symbol} not supported. Supported: {list(set(supported_symbols.values()))}",
                            "records_count": 0
                        }
                    
                    logger.info(f"Loading {days_back} days of Yahoo Finance data for {yahoo_symbol}")
                    
                    # Use appropriate client with all parameters
                    if yahoo_symbol == 'ETH-USD':
                        # ETH-specific client - use its own method with available parameters
                        hist_data = self.yf_client.get_historical_data(days_back=days_back, interval=interval, prepost=prepost, auto_adjust=auto_adjust)
                        data_package = self.yf_client.get_data_for_analysis(days_back=days_back)
                    else:
                        # Multi-crypto client - use its method with all parameters  
                        hist_data = self.multi_crypto_client.get_historical_data(yahoo_symbol, days_back=days_back, interval=interval, prepost=prepost, auto_adjust=auto_adjust)
                        data_package = self.multi_crypto_client.get_data_for_analysis(yahoo_symbol, days_back=days_back)
                    
                    if "error" in data_package:
                        return {
                            "success": False,
                            "error": data_package["error"],
                            "records_count": 0
                        }
                
                # Validate data quality using appropriate client
                import pandas as pd
                df = pd.DataFrame(data_package["data"])
                if yahoo_symbol == 'ETH-USD':
                    quality_metrics = self.yf_client.validate_data_quality(df)
                else:
                    quality_metrics = self.multi_crypto_client.validate_data_quality(df)
                
                # Prepare enriched data for database agent
                data_dict = {
                    "symbol": yahoo_symbol,
                    "data": data_package["data"],
                    "columns": data_package["columns"],
                    "records_count": data_package["summary"]["total_records"],
                    "date_range": data_package["summary"]["date_range"],
                    "sources": ["yahoo_finance"],
                    "source_info": {
                        "provider": "Yahoo Finance",
                        "api": "yfinance",
                        "interval": data_package["interval"],
                        "currency": "USD"
                    },
                    "quality_metrics": quality_metrics,
                    "summary_stats": data_package["summary"]
                }
                
                    logger.info(f"Successfully loaded {data_dict['records_count']} records for {yahoo_symbol}")
                    
                    # Track business metrics
                    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                    business_metrics.track_data_processing(
                        source="yahoo_finance",
                        symbol=yahoo_symbol,
                        records_processed=data_dict['records_count'],
                        success=True,
                        duration_ms=duration_ms
                    )
                    
                    return {
                        "success": True,
                        "data": data_dict,
                        "message": f"Loaded {data_dict['records_count']} records for {yahoo_symbol} from Yahoo Finance with {quality_metrics['completeness']*100:.1f}% completeness"
                    }
                    
                except Exception as e:
                    logger.error(f"Error loading Yahoo Finance data: {e}", error=e)
                    error_tracker.track_error(e, severity=ErrorSeverity.HIGH, category=ErrorCategory.API_ERROR)
                    
                    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                    business_metrics.track_data_processing(
                        source="yahoo_finance",
                        symbol=symbol,
                        records_processed=0,
                        success=False,
                        duration_ms=duration_ms
                    )
                    
                    return {
                        "success": False,
                        "error": str(e),
                        "records_count": 0
                    }

        @tool
        def load_multiple_symbols(symbols: List[str], days_back: int = 365, interval: str = '1d', prepost: bool = False, auto_adjust: bool = True) -> Dict[str, Any]:
            """Load historical data for multiple supported crypto symbols from Yahoo Finance
            
            Args:
                symbols: List of crypto symbols to load
                days_back: Number of days to load (any integer, default: 365)
                interval: Data granularity: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (default: 1d)
                prepost: Include pre/post market data (default: False)
                auto_adjust: Auto adjust for splits/dividends (default: True)
            """
            results = {}
            total_records = 0
            successful_symbols = []
            
            # Process each symbol
            for symbol in symbols:
                try:
                    result = load_symbol_data(symbol, days_back, interval, prepost, auto_adjust, False)
                    results[symbol] = result
                    
                    if result["success"]:
                        successful_symbols.append(symbol)
                        total_records += result["data"]["records_count"]
                        
                except Exception as e:
                    logger.error(f"Error loading {symbol}: {e}")
                    results[symbol] = {
                        "success": False,
                        "error": str(e),
                        "records_count": 0
                    }
            
            return {
                "symbols_processed": len(symbols),
                "symbols_successful": len(successful_symbols),
                "total_records": total_records,
                "results": results,
                "successful_symbols": successful_symbols,
                "summary": f"Loaded Yahoo Finance data for {len(successful_symbols)}/{len(symbols)} symbols: {total_records} total records"
            }

        @tool
        def get_available_datasets() -> List[Dict[str, Any]]:
            """Get available Yahoo Finance data info for all supported crypto pairs"""
            try:
                datasets = []
                
                # Get data for all supported pairs
                for symbol_key, symbol_info in self.multi_crypto_client.SUPPORTED_PAIRS.items():
                    try:
                        market_data = self.multi_crypto_client.get_market_data(symbol_key)
                        
                        if "error" not in market_data:
                            datasets.append({
                                "symbol": symbol_key,
                                "name": symbol_info["name"],
                                "source": "Yahoo Finance",
                                "current_price": market_data.get("current_price"),
                                "volume_24h": market_data.get("volume_24h"),
                                "change_24h": market_data.get("change_24h"),
                                "last_update": market_data.get("last_update"),
                                "available_history": "Multiple years of daily OHLCV data",
                                "data_quality": "High - Direct from Yahoo Finance API"
                            })
                    except Exception as e:
                        logger.warning(f"Could not get market data for {symbol_key}: {e}")
                        datasets.append({
                            "symbol": symbol_key,
                            "name": symbol_info["name"],
                            "source": "Yahoo Finance",
                            "available_history": "Multiple years of daily OHLCV data",
                            "data_quality": "High - Direct from Yahoo Finance API",
                            "note": "Current market data unavailable"
                        })
                
                logger.info(f"Retrieved dataset info for {len(datasets)} crypto pairs")
                return datasets
                
            except Exception as e:
                logger.error(f"Error getting dataset info: {e}")
                return []

        @tool
        def create_training_dataset(symbol: str, days_back: int = 365, interval: str = '1d', features: List[str] = None) -> Dict[str, Any]:
            """Create training dataset from Yahoo Finance data for supported crypto pairs
            
            Args:
                symbol: Crypto symbol to create dataset for
                days_back: Number of days to load (any integer, default: 365)  
                interval: Data granularity: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (default: 1d)
                features: List of specific features to include (optional)
            """
            try:
                # First load the raw data with dynamic parameters
                raw_result = load_symbol_data(symbol, days_back, interval, False, True, False)
                
                if not raw_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to load {symbol} data: {raw_result.get('error')}"
                    }
                
                # Convert back to DataFrame for feature creation
                import pandas as pd
                df = pd.DataFrame(raw_result["data"]["data"])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # Select features if specified
                if features:
                    available_features = [f for f in features if f in df.columns]
                    if available_features:
                        df = df[available_features]
                
                return {
                    "success": True,
                    "symbol": raw_result["data"]["symbol"],
                    "records_count": len(df),
                    "features_count": len(df.columns),
                    "features": list(df.columns),
                    "date_range": {
                        "start": str(df.index.min()),
                        "end": str(df.index.max())
                    },
                    "message": f"Created Yahoo Finance training dataset for {raw_result['data']['symbol']} with {len(df)} records and {len(df.columns)} features"
                }
                
            except Exception as e:
                logger.error(f"Error creating training dataset for {symbol}: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @tool
        def load_equity_indicators(symbols: List[str], days_back: int = 365, interval: str = '1d') -> Dict[str, Any]:
            """Load equity indicators that predict crypto movements from Yahoo Finance
            
            Args:
                symbols: List of equity symbols (SPY, QQQ, AAPL, NVDA, etc.)
                days_back: Number of days to load (default: 365)
                interval: Data granularity: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (default: 1d)
            
            Available Indicators:
                - Indices: SPY, QQQ (S&P 500, NASDAQ)
                - Tech Stocks: AAPL, MSFT, NVDA
                - Crypto Stocks: COIN, MSTR, MARA
                - Volatility: ^VIX
                - Currency: DX-Y.NYB (Dollar Index)
                - Bonds: ^TNX, ^IRX (Treasury yields)
                - Commodities: GLD, SLV
            """
            try:
                logger.info(f"Loading {len(symbols)} equity indicators for {days_back} days")
                
                results = {}
                successful_symbols = []
                total_records = 0
                
                for symbol in symbols:
                    try:
                        equity_data = self.equity_client.get_equity_data(
                            symbol=symbol,
                            days_back=days_back,
                            interval=interval
                        )
                        
                        if not equity_data.empty:
                            # Get indicator info
                            indicator_info = self.equity_client.get_indicator_info(symbol)
                            
                            # Format data
                            equity_data_reset = equity_data.reset_index()
                            equity_data_reset['Date'] = equity_data_reset['Date'].dt.strftime('%Y-%m-%d')
                            
                            results[symbol] = {
                                "success": True,
                                "symbol": symbol,
                                "name": indicator_info.get('name', symbol),
                                "category": indicator_info.get('category', 'unknown'),
                                "correlation": indicator_info.get('correlation', 0),
                                "data": equity_data_reset.to_dict(orient='records'),
                                "records_count": len(equity_data),
                                "date_range": {
                                    "start": equity_data_reset['Date'].iloc[0],
                                    "end": equity_data_reset['Date'].iloc[-1]
                                },
                                "current_price": indicator_info.get('current_price'),
                                "summary": {
                                    "latest_close": float(equity_data['Close'].iloc[-1]),
                                    "price_range": {
                                        "min": float(equity_data['Low'].min()),
                                        "max": float(equity_data['High'].max())
                                    }
                                }
                            }
                            successful_symbols.append(symbol)
                            total_records += len(equity_data)
                            logger.info(f"✓ Loaded {len(equity_data)} records for {symbol}")
                        else:
                            results[symbol] = {
                                "success": False,
                                "error": f"No data returned for {symbol}",
                                "records_count": 0
                            }
                            logger.warning(f"✗ No data for {symbol}")
                            
                    except Exception as e:
                        results[symbol] = {
                            "success": False,
                            "error": str(e),
                            "records_count": 0
                        }
                        logger.error(f"✗ Error loading {symbol}: {e}")
                
                return {
                    "success": True,
                    "symbols_processed": len(symbols),
                    "symbols_successful": len(successful_symbols),
                    "total_records": total_records,
                    "successful_symbols": successful_symbols,
                    "results": results,
                    "summary": f"Loaded {len(successful_symbols)}/{len(symbols)} equity indicators: {total_records:,} total records"
                }
                
            except Exception as e:
                logger.error(f"Error loading equity indicators: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "symbols_processed": 0,
                    "symbols_successful": 0,
                    "total_records": 0
                }

        @tool
        def load_crypto_predictors(crypto_symbol: str, days_back: int = 365, interval: str = '1d') -> Dict[str, Any]:
            """Load equity indicators that specifically predict a crypto symbol's movements
            
            Args:
                crypto_symbol: Crypto symbol (BTC, ETH, SOL, etc.)
                days_back: Number of days to load (default: 365)
                interval: Data granularity (default: 1d)
            
            Predictors by Crypto:
                - BTC: SPY, DX-Y.NYB, MSTR, ^TNX, GLD
                - ETH: QQQ, ^VIX, COIN, XLK, NVDA
                - SOL: QQQ, NVDA, XLK, IWM, AAPL
                - Others: Optimized predictor sets
            """
            try:
                logger.info(f"Loading equity predictors for {crypto_symbol}")
                
                predictor_data = self.equity_client.get_predictors_for_crypto(
                    crypto_symbol=crypto_symbol,
                    days_back=days_back,
                    interval=interval
                )
                
                if not predictor_data:
                    return {
                        "success": False,
                        "error": f"No predictors found for {crypto_symbol}",
                        "crypto_symbol": crypto_symbol
                    }
                
                # Format results similar to load_equity_indicators
                results = {}
                total_records = 0
                
                for symbol, data in predictor_data.items():
                    if not data.empty:
                        indicator_info = self.equity_client.get_indicator_info(symbol)
                        
                        data_reset = data.reset_index()
                        data_reset['Date'] = data_reset['Date'].dt.strftime('%Y-%m-%d')
                        
                        results[symbol] = {
                            "success": True,
                            "symbol": symbol,
                            "name": indicator_info.get('name', symbol),
                            "category": indicator_info.get('category', 'unknown'),
                            "correlation": indicator_info.get('correlation', 0),
                            "data": data_reset.to_dict(orient='records'),
                            "records_count": len(data),
                            "summary": {
                                "latest_close": float(data['Close'].iloc[-1]),
                                "price_change": float((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100)
                            }
                        }
                        total_records += len(data)
                
                return {
                    "success": True,
                    "crypto_symbol": crypto_symbol,
                    "predictors_loaded": len(results),
                    "total_records": total_records,
                    "results": results,
                    "summary": f"Loaded {len(results)} equity predictors for {crypto_symbol}: {total_records:,} records"
                }
                
            except Exception as e:
                logger.error(f"Error loading crypto predictors for {crypto_symbol}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "crypto_symbol": crypto_symbol
                }

        @tool
        def get_equity_indicators_list() -> List[Dict[str, Any]]:
            """Get list of all available equity indicators with their correlations and categories"""
            try:
                indicators_info = self.equity_client.get_all_indicators_info()
                
                # Group by category
                categorized = {}
                for indicator in indicators_info:
                    category = indicator.get('category', 'unknown')
                    if category not in categorized:
                        categorized[category] = []
                    categorized[category].append(indicator)
                
                logger.info(f"Retrieved info for {len(indicators_info)} equity indicators")
                
                return {
                    "total_indicators": len(indicators_info),
                    "categories": list(categorized.keys()),
                    "indicators_by_category": categorized,
                    "all_indicators": indicators_info
                }
                
            except Exception as e:
                logger.error(f"Error getting equity indicators list: {e}")
                return {"error": str(e)}

        @tool
        def load_tier1_indicators(days_back: int = 365, interval: str = '1d') -> Dict[str, Any]:
            """Load all Tier 1 high-correlation equity indicators (most predictive)
            
            Tier 1 Indicators:
                - SPY, QQQ (Market indices)
                - AAPL, MSFT, NVDA (Tech leaders)  
                - COIN, MSTR (Crypto stocks)
                - ^VIX, DX-Y.NYB (Risk indicators)
            """
            try:
                tier1_data = self.equity_client.get_all_tier1_indicators(days_back, interval)
                
                if not tier1_data:
                    return {
                        "success": False,
                        "error": "No Tier 1 indicator data loaded"
                    }
                
                # Format similar to other equity loading functions
                results = {}
                total_records = 0
                
                for symbol, data in tier1_data.items():
                    if not data.empty:
                        indicator_info = self.equity_client.get_indicator_info(symbol)
                        
                        data_reset = data.reset_index()
                        data_reset['Date'] = data_reset['Date'].dt.strftime('%Y-%m-%d')
                        
                        results[symbol] = {
                            "success": True,
                            "symbol": symbol,
                            "name": indicator_info.get('name', symbol),
                            "category": indicator_info.get('category', 'tier1'),
                            "correlation": indicator_info.get('correlation', 0),
                            "data": data_reset.to_dict(orient='records'),
                            "records_count": len(data)
                        }
                        total_records += len(data)
                
                return {
                    "success": True,
                    "tier": "Tier 1 - High Correlation",
                    "indicators_loaded": len(results),
                    "total_records": total_records,
                    "results": results,
                    "summary": f"Loaded {len(results)} Tier 1 equity indicators: {total_records:,} records"
                }
                
            except Exception as e:
                logger.error(f"Error loading Tier 1 indicators: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @tool
        def load_fx_rates(symbols: List[str], days_back: int = 365, interval: str = '1d') -> Dict[str, Any]:
            """Load FX rates that provide early trading signals for crypto movements
            
            Args:
                symbols: List of FX symbols (USDJPY=X, EURUSD=X, GBPUSD=X, etc.)
                days_back: Number of days to load (default: 365)
                interval: Data granularity: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (default: 1d)
            
            Available FX Pairs:
                - Tier 1: USDJPY=X, USDCNH=X, USDKRW=X (early warning signals)
                - Tier 2: EURUSD=X, GBPUSD=X (risk sentiment)
                - Tier 3: USDCHF=X, AUDUSD=X, NZDUSD=X (safe haven/commodity)
                - Crosses: EURJPY=X, GBPJPY=X (volatility amplifiers)
            """
            try:
                logger.info(f"Loading {len(symbols)} FX pairs for {days_back} days")
                
                results = {}
                successful_symbols = []
                total_records = 0
                
                for symbol in symbols:
                    try:
                        fx_data = self.fx_client.get_fx_data(
                            symbol=symbol,
                            days_back=days_back,
                            interval=interval
                        )
                        
                        if not fx_data.empty:
                            # Get FX pair info
                            pair_info = self.fx_client.get_fx_pair_info(symbol)
                            
                            # Format data
                            fx_data_reset = fx_data.reset_index()
                            fx_data_reset['Date'] = fx_data_reset['Date'].dt.strftime('%Y-%m-%d')
                            
                            results[symbol] = {
                                "success": True,
                                "symbol": symbol,
                                "name": pair_info.get('name', symbol),
                                "tier": pair_info.get('tier', 0),
                                "category": pair_info.get('category', 'unknown'),
                                "crypto_correlation": pair_info.get('crypto_correlation', 0),
                                "signal_strength": pair_info.get('signal_strength', 'unknown'),
                                "best_for": pair_info.get('best_for', []),
                                "data": fx_data_reset.to_dict(orient='records'),
                                "records_count": len(fx_data),
                                "date_range": {
                                    "start": fx_data_reset['Date'].iloc[0],
                                    "end": fx_data_reset['Date'].iloc[-1]
                                },
                                "current_rate": pair_info.get('current_rate'),
                                "summary": {
                                    "latest_close": float(fx_data['Close'].iloc[-1]),
                                    "rate_range": {
                                        "min": float(fx_data['Low'].min()),
                                        "max": float(fx_data['High'].max())
                                    },
                                    "mechanism": pair_info.get('mechanism', 'Unknown mechanism')
                                }
                            }
                            successful_symbols.append(symbol)
                            total_records += len(fx_data)
                            logger.info(f"✓ Loaded {len(fx_data)} records for {symbol}")
                        else:
                            results[symbol] = {
                                "success": False,
                                "error": f"No data returned for {symbol}",
                                "records_count": 0
                            }
                            logger.warning(f"✗ No data for {symbol}")
                            
                    except Exception as e:
                        results[symbol] = {
                            "success": False,
                            "error": str(e),
                            "records_count": 0
                        }
                        logger.error(f"✗ Error loading {symbol}: {e}")
                
                return {
                    "success": True,
                    "symbols_processed": len(symbols),
                    "symbols_successful": len(successful_symbols),
                    "total_records": total_records,
                    "successful_symbols": successful_symbols,
                    "results": results,
                    "summary": f"Loaded {len(successful_symbols)}/{len(symbols)} FX pairs: {total_records:,} total records"
                }
                
            except Exception as e:
                logger.error(f"Error loading FX rates: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "symbols_processed": 0,
                    "symbols_successful": 0,
                    "total_records": 0
                }

        @tool
        def load_crypto_fx_predictors(crypto_symbol: str, days_back: int = 365, interval: str = '1d') -> Dict[str, Any]:
            """Load FX pairs that specifically predict a crypto symbol's movements
            
            Args:
                crypto_symbol: Crypto symbol (BTC, ETH, SOL, etc.)
                days_back: Number of days to load (default: 365)
                interval: Data granularity (default: 1d)
            
            FX Predictors by Crypto:
                - BTC: USDJPY=X, USDCNH=X, EURUSD=X, EURJPY=X
                - ETH: EURJPY=X, USDJPY=X, GBPUSD=X, USDCHF=X
                - SOL: USDKRW=X, AUDUSD=X, GBPJPY=X, EURUSD=X
                - Others: Optimized FX predictor sets
            """
            try:
                logger.info(f"Loading FX predictors for {crypto_symbol}")
                
                predictor_data = self.fx_client.get_fx_predictors_for_crypto(
                    crypto_symbol=crypto_symbol,
                    days_back=days_back,
                    interval=interval
                )
                
                if not predictor_data:
                    return {
                        "success": False,
                        "error": f"No FX predictors found for {crypto_symbol}",
                        "crypto_symbol": crypto_symbol
                    }
                
                # Format results similar to load_fx_rates
                results = {}
                total_records = 0
                
                for symbol, data in predictor_data.items():
                    if not data.empty:
                        pair_info = self.fx_client.get_fx_pair_info(symbol)
                        
                        data_reset = data.reset_index()
                        data_reset['Date'] = data_reset['Date'].dt.strftime('%Y-%m-%d')
                        
                        results[symbol] = {
                            "success": True,
                            "symbol": symbol,
                            "name": pair_info.get('name', symbol),
                            "tier": pair_info.get('tier', 0),
                            "crypto_correlation": pair_info.get('crypto_correlation', 0),
                            "signal_strength": pair_info.get('signal_strength', 'unknown'),
                            "data": data_reset.to_dict(orient='records'),
                            "records_count": len(data),
                            "summary": {
                                "latest_close": float(data['Close'].iloc[-1]),
                                "rate_change": float((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100),
                                "mechanism": pair_info.get('mechanism', 'Unknown mechanism')
                            }
                        }
                        total_records += len(data)
                
                return {
                    "success": True,
                    "crypto_symbol": crypto_symbol,
                    "predictors_loaded": len(results),
                    "total_records": total_records,
                    "results": results,
                    "summary": f"Loaded {len(results)} FX predictors for {crypto_symbol}: {total_records:,} records"
                }
                
            except Exception as e:
                logger.error(f"Error loading FX predictors for {crypto_symbol}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "crypto_symbol": crypto_symbol
                }

        @tool
        def get_fx_early_warning_signals(threshold_pct: float = 2.0) -> Dict[str, Any]:
            """Get early warning signals from key FX pairs for crypto movements
            
            Args:
                threshold_pct: Change percentage threshold for alerts (default: 2.0%)
            
            Monitors key FX pairs:
                - USD/JPY: Carry trade flows
                - EUR/JPY: Risk sentiment  
                - USD/CNH: Capital flight
                - USD/KRW: Regional demand
            """
            try:
                early_signals = self.fx_client.get_early_warning_signals(
                    days_back=5, 
                    threshold_pct=threshold_pct
                )
                
                if "error" in early_signals:
                    return {
                        "success": False,
                        "error": early_signals["error"]
                    }
                
                return {
                    "success": True,
                    "timestamp": early_signals["timestamp"],
                    "threshold_pct": threshold_pct,
                    "signals_monitored": len(early_signals["signals"]),
                    "alerts_triggered": early_signals["alert_count"],
                    "signals": early_signals["signals"],
                    "alerts": early_signals["alerts"],
                    "summary": early_signals["summary"]
                }
                
            except Exception as e:
                logger.error(f"Error getting FX early warning signals: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @tool
        def load_tier1_fx_pairs(days_back: int = 365, interval: str = '1d') -> Dict[str, Any]:
            """Load all Tier 1 FX pairs (highest crypto predictive power)
            
            Tier 1 FX Pairs:
                - USD/JPY: Carry trade indicator (correlation: 0.65)
                - USD/CNH: Capital flight proxy (correlation: 0.80)  
                - USD/KRW: Regional sentiment gauge (correlation: 0.70)
            """
            try:
                tier1_data = self.fx_client.get_tier1_fx_pairs(days_back, interval)
                
                if not tier1_data:
                    return {
                        "success": False,
                        "error": "No Tier 1 FX pairs data loaded"
                    }
                
                # Format similar to other FX loading functions
                results = {}
                total_records = 0
                
                for symbol, data in tier1_data.items():
                    if not data.empty:
                        pair_info = self.fx_client.get_fx_pair_info(symbol)
                        
                        data_reset = data.reset_index()
                        data_reset['Date'] = data_reset['Date'].dt.strftime('%Y-%m-%d')
                        
                        results[symbol] = {
                            "success": True,
                            "symbol": symbol,
                            "name": pair_info.get('name', symbol),
                            "tier": 1,
                            "crypto_correlation": pair_info.get('crypto_correlation', 0),
                            "signal_strength": pair_info.get('signal_strength', 'very_high'),
                            "data": data_reset.to_dict(orient='records'),
                            "records_count": len(data)
                        }
                        total_records += len(data)
                
                return {
                    "success": True,
                    "tier": "Tier 1 - Highest Predictive Power",
                    "pairs_loaded": len(results),
                    "total_records": total_records,
                    "results": results,
                    "summary": f"Loaded {len(results)} Tier 1 FX pairs: {total_records:,} records"
                }
                
            except Exception as e:
                logger.error(f"Error loading Tier 1 FX pairs: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @tool
        def get_fx_pairs_list() -> Dict[str, Any]:
            """Get list of all available FX pairs with their predictive characteristics"""
            try:
                fx_pairs_info = self.fx_client.get_all_fx_pairs_info()
                
                # Group by tier
                by_tier = {}
                for pair in fx_pairs_info:
                    tier = pair.get('tier', 0)
                    if tier not in by_tier:
                        by_tier[tier] = []
                    by_tier[tier].append(pair)
                
                # Group by category
                by_category = {}
                for pair in fx_pairs_info:
                    category = pair.get('category', 'unknown')
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(pair)
                
                logger.info(f"Retrieved info for {len(fx_pairs_info)} FX pairs")
                
                return {
                    "total_pairs": len(fx_pairs_info),
                    "tiers": list(by_tier.keys()),
                    "categories": list(by_category.keys()),
                    "pairs_by_tier": by_tier,
                    "pairs_by_category": by_category,
                    "all_pairs": fx_pairs_info
                }
                
            except Exception as e:
                logger.error(f"Error getting FX pairs list: {e}")
                return {"error": str(e)}

        # Return tools for base class to use
        return [
            load_symbol_data,
            load_multiple_symbols, 
            get_available_datasets,
            create_training_dataset,
            load_equity_indicators,
            load_crypto_predictors,
            get_equity_indicators_list,
            load_tier1_indicators,
            load_fx_rates,
            load_crypto_fx_predictors,
            get_fx_early_warning_signals,
            load_tier1_fx_pairs,
            get_fx_pairs_list
        ]

    def _message_to_prompt(self, message):
        """Convert A2A message to natural language prompt for historical loader"""
        message_type = message.message_type.value
        payload = message.payload
        
        if message_type == 'DATA_LOAD_REQUEST':
            symbol = payload.get('symbol', 'ETH')
            days_back = payload.get('days_back', 365)
            return f"Load historical data for {symbol} for the last {days_back} days"
        elif message_type == 'BULK_LOAD_REQUEST':
            symbols = payload.get('symbols', [])
            days_back = payload.get('days_back', 365)
            return f"Load historical data for symbols {symbols} for the last {days_back} days"
        else:
            # Check if we have the method in our inheritance hierarchy
            if hasattr(super(), '_message_to_prompt'):
                return super()._message_to_prompt(message)
            else:
                return f"Process {message_type} with data: {payload}"
    
    # A2A Protocol specific message handlers
    async def _handle_data_load_request(self, message):
        """Handle A2A data load request"""
        payload = message.payload
        symbol = payload.get('symbol', 'ETH')
        days_back = payload.get('days_back', 365)
        
        # Process the request using the agent's tools
        prompt = f"Load historical data for {symbol} for the last {days_back} days"
        response = await self.process_request(prompt)
        
        return self.format_success_response(
            data={"response": response, "symbol": symbol, "days_back": days_back},
            message=f"Historical data loaded for {symbol}"
        )

    async def load_historical_data(self, symbol: str, days_back: int = 365) -> Dict[str, Any]:
        """Load historical data for a symbol (async wrapper)"""
        try:
            # Use the agent to process the request
            prompt = f"Load historical data for {symbol} for the last {days_back} days"
            result = self.agent(prompt)
            
            if hasattr(result, 'message'):
                return {"success": True, "response": str(result.message)}
            
            return {"success": True, "response": str(result)}
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return {"success": False, "error": str(e)}
    
    def load_data_for_symbols(self, symbols: List[str], days_back: int = 365) -> Dict[str, Any]:
        """Direct method for programmatic data loading"""
        results = {}
        
        for symbol in symbols:
            try:
                # Download data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                raw_data = self.aggregator.download_all_sources(
                    symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if raw_data:
                    merged_df = self.aggregator.merge_data_sources(raw_data)
                    processed_df = self.aggregator.add_all_indicators(merged_df)
                    
                    results[symbol] = {
                        "success": True,
                        "data": processed_df,
                        "records_count": len(processed_df),
                        "sources": list(raw_data.keys())
                    }
                else:
                    results[symbol] = {
                        "success": False,
                        "error": f"No data available for {symbol}",
                        "data": None
                    }
                    
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                results[symbol] = {
                    "success": False,
                    "error": str(e),
                    "data": None
                }
        
        return results

# Global instance - create on demand to avoid async issues during import
historical_loader_agent = None

def get_historical_loader_agent():
    """Get or create the historical loader agent instance"""
    global historical_loader_agent
    if historical_loader_agent is None:
        historical_loader_agent = HistoricalLoaderAgent()
    return historical_loader_agent