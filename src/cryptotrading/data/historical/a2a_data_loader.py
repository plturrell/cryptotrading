"""
Comprehensive A2A Historical Data Loader
Integrates FRED and Yahoo Finance data sources
Uses strand framework for agent-based data acquisition
"""

import pandas as pd
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
from dataclasses import dataclass

from ...strands.agent import Agent
from ...strands.models.model import Model
from ...strands.types.tools import ToolSpec

from .yahoo_finance import YahooFinanceClient
from .fred_client import FREDClient

logger = logging.getLogger(__name__)

@dataclass
class DataLoadRequest:
    """Request for historical data loading"""
    sources: List[str]  # ['yahoo', 'fred']
    symbols: Optional[List[str]] = None  # For Yahoo Finance
    fred_series: Optional[List[str]] = None  # For FRED
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    align_data: bool = True  # Temporally align all data sources
    save_data: bool = True

class A2AHistoricalDataLoader:
    """
    Agent-to-Agent Historical Data Loader
    Orchestrates multiple data sources using strand framework
    """
    
    def __init__(self, data_dir: Optional[str] = None, model: Optional[Model] = None):
        self.data_dir = Path(data_dir or "data/historical/a2a")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data source clients
        self.yahoo_client = YahooFinanceClient(data_dir=str(self.data_dir / "yahoo"))
        self.fred_client = FREDClient(data_dir=str(self.data_dir / "fred"))
        
        # Create strand agent with data loading tools
        self.agent = Agent(tools=self._create_tools(), model=model)
        
        # Default configurations
        self.default_crypto_symbols = ["BTC", "ETH", "BNB", "XRP", "ADA"]
        self.default_fred_series = [
            "DGS10", "T10Y2Y", "WALCL", "RRPONTSYD", "M2SL", "EFFR"
        ]
    
    def _create_tools(self) -> List[ToolSpec]:
        """Create strand tools for data loading subskills"""
        
        def load_yahoo_data(symbols: List[str], start_date: str = None, 
                          end_date: str = None) -> Dict[str, Any]:
            """Load Yahoo Finance crypto data"""
            try:
                logger.info(f"Loading Yahoo Finance data for {symbols}")
                results = self.yahoo_client.download_multiple(
                    symbols, start_date=start_date, end_date=end_date
                )
                return {
                    "status": "success",
                    "source": "yahoo_finance",
                    "data_count": sum(len(df) for df in results.values()),
                    "symbols": list(results.keys()),
                    "data": {symbol: df.to_dict('records') for symbol, df in results.items()}
                }
            except Exception as e:
                logger.error(f"Yahoo Finance loading failed: {e}")
                return {"status": "error", "source": "yahoo_finance", "error": str(e)}
        
        def load_fred_data(series_ids: List[str], start_date: str = None,
                          end_date: str = None) -> Dict[str, Any]:
            """Load FRED economic data"""
            try:
                logger.info(f"Loading FRED data for {series_ids}")
                results = self.fred_client.get_multiple_series(
                    series_ids, start_date=start_date, end_date=end_date
                )
                return {
                    "status": "success",
                    "source": "fred",
                    "data_count": sum(len(df) for df in results.values()),
                    "series": list(results.keys()),
                    "data": {series: df.to_dict('records') for series, df in results.items()}
                }
            except Exception as e:
                logger.error(f"FRED loading failed: {e}")
                return {"status": "error", "source": "fred", "error": str(e)}
        
        
        
        def align_temporal_data(data_sources: Dict[str, Any], 
                              frequency: str = "D") -> Dict[str, Any]:
            """Align data from multiple sources temporally"""
            try:
                logger.info("Aligning temporal data across sources")
                
                aligned_data = {}
                all_dataframes = []
                
                # Collect all DataFrames
                for source, source_data in data_sources.items():
                    if source_data.get("status") == "success" and "data" in source_data:
                        data_dict = source_data["data"]
                        
                        if isinstance(data_dict, dict):
                            for key, records in data_dict.items():
                                if records:  # Check if records exist
                                    df = pd.DataFrame(records)
                                    if 'date' in df.columns or df.index.name == 'date':
                                        if 'date' in df.columns:
                                            df['date'] = pd.to_datetime(df['date'])
                                            df = df.set_index('date')
                                        all_dataframes.append((f"{source}_{key}", df))
                
                if not all_dataframes:
                    return {"status": "error", "error": "No valid dataframes to align"}
                
                # Find common date range
                start_dates = []
                end_dates = []
                
                for name, df in all_dataframes:
                    if not df.empty:
                        start_dates.append(df.index.min())
                        end_dates.append(df.index.max())
                
                if not start_dates:
                    return {"status": "error", "error": "No valid date ranges found"}
                
                common_start = max(start_dates)
                common_end = min(end_dates)
                
                # Create common date range
                date_range = pd.date_range(start=common_start, end=common_end, freq=frequency)
                
                # Align all data to common date range
                for name, df in all_dataframes:
                    if not df.empty:
                        # Resample to common frequency and forward fill
                        aligned_df = df.reindex(date_range, method='ffill')
                        aligned_data[name] = aligned_df.to_dict('records')
                
                return {
                    "status": "success",
                    "aligned_data": aligned_data,
                    "date_range": {
                        "start": common_start.isoformat(),
                        "end": common_end.isoformat(),
                        "frequency": frequency,
                        "total_periods": len(date_range)
                    }
                }
                
            except Exception as e:
                logger.error(f"Data alignment failed: {e}")
                return {"status": "error", "error": str(e)}
        
        # Register strand tools
        self.tools = [
            ToolSpec(
                name="load_yahoo_data",
                description="Load Yahoo Finance crypto data",
                parameters={"symbols": "List[str]", "period": "str"},
                function=load_yahoo_data
            ),
            ToolSpec(
                name="load_fred_data", 
                description="Load FRED economic data",
                parameters={"series": "List[str]"},
                function=load_fred_data
            ),
            ToolSpec(
                name="align_temporal_data",
                description="Align data from multiple sources temporally",
                parameters={"data_sources": "Dict[str, Any]", "frequency": "str"},
                function=align_temporal_data
            )
        ]
    
    async def load_comprehensive_data(self, request: DataLoadRequest) -> Dict[str, Any]:
        """
        Load comprehensive historical data using strand agent
        
        Args:
            request: Data loading request specification
        """
        logger.info("Starting comprehensive A2A historical data loading...")
        
        # Build agent prompt
        prompt_parts = [
            "Load comprehensive historical data for crypto trading analysis.",
            f"Data sources requested: {', '.join(request.sources)}",
        ]
        
        if request.start_date:
            prompt_parts.append(f"Start date: {request.start_date}")
        if request.end_date:
            prompt_parts.append(f"End date: {request.end_date}")
        
        # Add specific requests for each source
        tool_calls = []
        
        if "yahoo" in request.sources:
            symbols = request.symbols or self.default_crypto_symbols
            prompt_parts.append(f"Yahoo Finance symbols: {', '.join(symbols)}")
            tool_calls.append(f"load_yahoo_data({symbols}, '{request.start_date}', '{request.end_date}')")
        
        if "fred" in request.sources:
            series = request.fred_series or self.default_fred_series
            prompt_parts.append(f"FRED series: {', '.join(series)}")
            tool_calls.append(f"load_fred_data({series}, '{request.start_date}', '{request.end_date}')")
        
        if "cboe" in request.sources:
            indices = request.cboe_indices or self.default_cboe_indices
            prompt_parts.append(f"CBOE indices: {', '.join(indices)}")
            tool_calls.append(f"load_cboe_data({indices})")
        
        if "defillama" in request.sources:
            protocols = request.defillama_protocols or self.default_defillama_protocols
            prompt_parts.append(f"DeFiLlama protocols: {', '.join(protocols)}")
            tool_calls.append(f"load_defillama_data({protocols})")
        
        prompt_parts.extend([
            "",
            "Execute the following data loading operations:",
            *tool_calls
        ])
        
        if request.align_data:
            prompt_parts.append("Then align all temporal data to daily frequency.")
        
        prompt = "\n".join(prompt_parts)
        
        # Execute via strand agent
        try:
            result = await self.agent.process_async(prompt)
            
            # Extract results from agent response
            agent_response = str(result)
            
            # Save comprehensive results
            if request.save_data:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = self.data_dir / f"comprehensive_load_{timestamp}.json"
                
                import json
                with open(result_file, 'w') as f:
                    json.dump({
                        "request": {
                            "sources": request.sources,
                            "symbols": request.symbols,
                            "fred_series": request.fred_series,
                            "start_date": request.start_date,
                            "end_date": request.end_date,
                            "align_data": request.align_data
                        },
                        "agent_response": agent_response,
                        "timestamp": timestamp
                    }, f, indent=2)
                
                logger.info(f"Saved comprehensive results to {result_file}")
            
            return {
                "status": "success",
                "agent_response": agent_response,
                "sources_loaded": request.sources,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive data loading failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "sources_requested": request.sources
            }
    
    def load_crypto_trading_dataset(self, start_date: str = None, 
                                  end_date: str = None) -> Dict[str, Any]:
        """
        Load complete crypto trading dataset with all relevant indicators
        
        Convenience method for comprehensive crypto trading analysis
        """
        # Default to 2 years of data
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        request = DataLoadRequest(
            sources=["yahoo", "fred"],
            symbols=["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "MATIC", "DOT"],
            fred_series=[
                "DGS10", "T10Y2Y", "T10YIE", "WALCL", "RRPONTSYD", "WTREGEN",
                "M2SL", "EFFR", "CPIAUCSL", "UNRATE", "DTWEXBGS"
            ],
            start_date=start_date,
            end_date=end_date,
            align_data=True,
            save_data=True
        )
        
        # Run synchronously for convenience
        return asyncio.run(self.load_comprehensive_data(request))
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available cached data"""
        summary = {
            "yahoo_finance": {},
            "fred": {},
            "total_files": 0
        }
        
        # Count files in each data directory
        for source in ["yahoo", "fred"]:
            source_dir = self.data_dir / source
            if source_dir.exists():
                files = list(source_dir.glob("*.csv"))
                summary[source] = {
                    "file_count": len(files),
                    "latest_file": max(files, key=lambda f: f.stat().st_mtime).name if files else None,
                    "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024)
                }
                summary["total_files"] += len(files)
        
        return summary
