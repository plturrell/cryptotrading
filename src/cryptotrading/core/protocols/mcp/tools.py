"""
MCP Tools Implementation
Defines tool structures and execution for MCP protocol
"""
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolContent:
    """Content block for tool results"""
    type: str
    text: Optional[str] = None
    data: Optional[str] = None
    mimeType: Optional[str] = None


@dataclass
class ToolResult:
    """Result from tool execution"""
    content: List[ToolContent]
    isError: bool = False
    
    @classmethod
    def text_result(cls, text: str, is_error: bool = False) -> 'ToolResult':
        """Create a text result"""
        return cls(
            content=[ToolContent(type="text", text=text)],
            isError=is_error
        )
    
    @classmethod
    def data_result(cls, data: str, mime_type: str, is_error: bool = False) -> 'ToolResult':
        """Create a data result"""
        return cls(
            content=[ToolContent(type="resource", data=data, mimeType=mime_type)],
            isError=is_error
        )
    
    @classmethod
    def json_result(cls, data: Dict[str, Any], is_error: bool = False) -> 'ToolResult':
        """Create a JSON result"""
        return cls(
            content=[ToolContent(
                type="resource", 
                data=json.dumps(data, indent=2),
                mimeType="application/json"
            )],
            isError=is_error
        )
    
    @classmethod
    def error_result(cls, message: str) -> 'ToolResult':
        """Create an error result"""
        return cls(
            content=[ToolContent(type="text", text=f"Error: {message}")],
            isError=True
        )


@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to MCP format"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()) if self.parameters else []
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments"""
        if not self.function:
            return ToolResult.error_result(f"No function defined for tool '{self.name}'")
        
        try:
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(**arguments)
            else:
                result = self.function(**arguments)
            
            # Convert result to ToolResult if needed
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, dict):
                return ToolResult.json_result(result)
            elif isinstance(result, str):
                return ToolResult.text_result(result)
            else:
                return ToolResult.text_result(str(result))
                
        except Exception as e:
            return ToolResult.error_result(str(e))


class CryptoTradingTools:
    """Crypto trading specific tools for MCP server - Real historical data integration"""
    
    @staticmethod
    def get_yahoo_finance_data_tool() -> MCPTool:
        """Get Yahoo Finance crypto historical data tool"""
        return MCPTool(
            name="get_yahoo_finance_data",
            description="Download historical cryptocurrency data from Yahoo Finance",
            parameters={
                "symbol": {
                    "type": "string",
                    "description": "Cryptocurrency symbol (e.g., BTC, ETH, BTC-USD)"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD), defaults to 2 years ago",
                    "default": None
                },
                "end_date": {
                    "type": "string", 
                    "description": "End date (YYYY-MM-DD), defaults to today",
                    "default": None
                },
                "interval": {
                    "type": "string",
                    "description": "Data interval: 1d, 1h, 5m, etc.",
                    "default": "1d"
                },
                "prepare_for_training": {
                    "type": "boolean",
                    "description": "Add technical indicators for ML training",
                    "default": False
                }
            },
            function=CryptoTradingTools._get_yahoo_finance_data
        )
    
    @staticmethod
    def get_fred_economic_data_tool() -> MCPTool:
        """Get FRED economic data tool"""
        return MCPTool(
            name="get_fred_economic_data",
            description="Download economic indicators from Federal Reserve (FRED) database",
            parameters={
                "series_id": {
                    "type": "string",
                    "description": "FRED series ID (e.g., DGS10, WALCL, M2SL)"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD), defaults to 2 years ago",
                    "default": None
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD), defaults to today", 
                    "default": None
                },
                "frequency": {
                    "type": "string",
                    "description": "Data frequency: d, w, m, q, a",
                    "default": None
                }
            },
            function=CryptoTradingTools._get_fred_economic_data
        )
    
    @staticmethod
    def get_crypto_relevant_indicators_tool() -> MCPTool:
        """Get crypto-relevant economic indicators tool"""
        return MCPTool(
            name="get_crypto_relevant_indicators",
            description="Download all crypto-relevant economic indicators from FRED",
            parameters={
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD), defaults to 2 years ago",
                    "default": None
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD), defaults to today",
                    "default": None
                },
                "include_liquidity_metrics": {
                    "type": "boolean", 
                    "description": "Calculate derived liquidity metrics",
                    "default": True
                }
            },
            function=CryptoTradingTools._get_crypto_relevant_indicators
        )
    
    @staticmethod
    def get_comprehensive_trading_dataset_tool() -> MCPTool:
        """Get comprehensive crypto trading dataset tool"""
        return MCPTool(
            name="get_comprehensive_trading_dataset",
            description="Load complete dataset with crypto prices and economic indicators for trading analysis",
            parameters={
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Crypto symbols to include",
                    "default": ["BTC", "ETH", "BNB", "XRP", "ADA"]
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD), defaults to 2 years ago",
                    "default": None
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD), defaults to today",
                    "default": None
                },
                "align_data": {
                    "type": "boolean",
                    "description": "Temporally align all data sources",
                    "default": True
                }
            },
            function=CryptoTradingTools._get_comprehensive_trading_dataset
        )
    
    # Tool implementation methods - Real historical data integration
    @staticmethod
    async def _get_yahoo_finance_data(symbol: str, start_date: str = None, 
                                    end_date: str = None, interval: str = "1d",
                                    prepare_for_training: bool = False) -> Dict[str, Any]:
        """Implementation for Yahoo Finance data tool"""
        try:
            from ....data.historical.yahoo_finance import YahooFinanceClient
            
            # Initialize Yahoo Finance client
            yahoo_client = YahooFinanceClient()
            
            # Download data
            df = yahoo_client.download_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                save=True
            )
            
            if df is None or df.empty:
                return {
                    "status": "error",
                    "error": f"No data found for symbol {symbol}",
                    "symbol": symbol
                }
            
            # Prepare training data if requested
            if prepare_for_training:
                df = yahoo_client.prepare_training_data(df)
            
            # Return summary and data
            return {
                "status": "success",
                "symbol": symbol,
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                "data_points": len(df),
                "columns": list(df.columns),
                "latest_price": float(df['close'].iloc[-1]) if 'close' in df.columns else None,
                "date_range": {
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat()
                },
                "with_indicators": prepare_for_training,
                "sample_data": df.head(5).to_dict('records') if len(df) > 0 else []
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "symbol": symbol
            }
    
    @staticmethod
    async def _get_fred_economic_data(series_id: str, start_date: str = None,
                                    end_date: str = None, frequency: str = None) -> Dict[str, Any]:
        """Implementation for FRED economic data tool"""
        try:
            from ....data.historical.fred_client import FREDClient
            
            # Initialize FRED client
            fred_client = FREDClient()
            
            # Download data
            df = fred_client.get_series_data(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                save=True
            )
            
            if df is None or df.empty:
                return {
                    "status": "error",
                    "error": f"No data found for FRED series {series_id}",
                    "series_id": series_id
                }
            
            # Get series description if available
            series_description = fred_client.crypto_relevant_series.get(
                series_id, f"FRED Series {series_id}"
            )
            
            return {
                "status": "success",
                "series_id": series_id,
                "description": series_description,
                "frequency": frequency,
                "start_date": start_date,
                "end_date": end_date,
                "data_points": len(df),
                "latest_value": float(df[series_id].iloc[-1]) if series_id in df.columns else None,
                "date_range": {
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat()
                },
                "sample_data": df.head(5).to_dict('records') if len(df) > 0 else []
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "series_id": series_id
            }
    
    @staticmethod
    async def _get_crypto_relevant_indicators(start_date: str = None, end_date: str = None,
                                            include_liquidity_metrics: bool = True) -> Dict[str, Any]:
        """Implementation for crypto-relevant economic indicators tool"""
        try:
            from ....data.historical.fred_client import FREDClient
            
            # Initialize FRED client
            fred_client = FREDClient()
            
            # Download all crypto-relevant data
            data_dict = fred_client.get_crypto_relevant_data(
                start_date=start_date,
                end_date=end_date,
                save=True
            )
            
            result = {
                "status": "success",
                "start_date": start_date,
                "end_date": end_date,
                "series_downloaded": len(data_dict),
                "series_data": {}
            }
            
            # Process each series
            for series_id, df in data_dict.items():
                if df is not None and not df.empty:
                    result["series_data"][series_id] = {
                        "description": fred_client.crypto_relevant_series.get(series_id, series_id),
                        "data_points": len(df),
                        "latest_value": float(df[series_id].iloc[-1]) if series_id in df.columns else None,
                        "date_range": {
                            "start": df.index.min().isoformat(),
                            "end": df.index.max().isoformat()
                        }
                    }
            
            # Calculate liquidity metrics if requested
            if include_liquidity_metrics:
                liquidity_df = fred_client.get_liquidity_metrics(
                    start_date=start_date,
                    end_date=end_date,
                    save=True
                )
                
                if liquidity_df is not None and not liquidity_df.empty:
                    result["liquidity_metrics"] = {
                        "data_points": len(liquidity_df),
                        "columns": list(liquidity_df.columns),
                        "latest_values": liquidity_df.iloc[-1].to_dict() if len(liquidity_df) > 0 else {}
                    }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    async def _get_comprehensive_trading_dataset(symbols: List[str] = None, start_date: str = None,
                                               end_date: str = None, align_data: bool = True) -> Dict[str, Any]:
        """Implementation for comprehensive trading dataset tool"""
        try:
            from ....data.historical.yahoo_finance import YahooFinanceClient
            from ....data.historical.fred_client import FREDClient
            
            # Use default symbols if none provided
            if symbols is None:
                symbols = ["BTC", "ETH", "BNB", "XRP", "ADA"]
            
            # Initialize clients directly
            yahoo_client = YahooFinanceClient()
            fred_client = FREDClient()
            
            results = {
                "status": "success",
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "yahoo_data": {},
                "fred_data": {},
                "errors": []
            }
            
            # Load Yahoo Finance data
            logger.info(f"Loading Yahoo Finance data for {symbols}")
            for symbol in symbols:
                try:
                    df = yahoo_client.download_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        save=True
                    )
                    if df is not None and not df.empty:
                        results["yahoo_data"][symbol] = {
                            "data_points": len(df),
                            "latest_price": float(df['close'].iloc[-1]),
                            "date_range": {
                                "start": df.index.min().isoformat(),
                                "end": df.index.max().isoformat()
                            }
                        }
                    else:
                        results["errors"].append(f"No Yahoo data for {symbol}")
                except Exception as e:
                    results["errors"].append(f"Yahoo error for {symbol}: {str(e)}")
            
            # Load FRED data (if API key available)
            fred_series = ["DGS10", "T10Y2Y", "WALCL", "M2SL", "EFFR"]
            logger.info(f"Loading FRED data for {fred_series}")
            for series_id in fred_series:
                try:
                    df = fred_client.get_series_data(
                        series_id=series_id,
                        start_date=start_date,
                        end_date=end_date,
                        save=True
                    )
                    if df is not None and not df.empty:
                        results["fred_data"][series_id] = {
                            "data_points": len(df),
                            "latest_value": float(df[series_id].iloc[-1]),
                            "description": fred_client.crypto_relevant_series.get(series_id, series_id)
                        }
                    else:
                        results["errors"].append(f"No FRED data for {series_id}")
                except Exception as e:
                    results["errors"].append(f"FRED error for {series_id}: {str(e)}")
            
            # Summary
            results["summary"] = {
                "yahoo_symbols_loaded": len(results["yahoo_data"]),
                "fred_series_loaded": len(results["fred_data"]),
                "total_errors": len(results["errors"])
            }
            
            if not results["yahoo_data"] and not results["fred_data"]:
                results["status"] = "error"
                results["error"] = "No data could be loaded"
            
            return results
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "symbols": symbols or []
            }