"""
Historical Data Loader Agent powered by Strand Agents
Handles bulk loading of historical crypto data
"""

from strands import Agent, tool
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging

from ...historical_data.aggregator import HistoricalDataAggregator
from ..registry.registry import agent_registry

logger = logging.getLogger(__name__)

class HistoricalLoaderAgent:
    def __init__(self, model_provider: str = "anthropic"):
        self.aggregator = HistoricalDataAggregator()
        
        # Define tools for the Strand agent
        @tool
        def load_symbol_data(symbol: str, days_back: int = 365, include_indicators: bool = True) -> Dict[str, Any]:
            """Load historical data for a crypto symbol with technical indicators"""
            try:
                logger.info(f"Loading {days_back} days of data for {symbol}")
                
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                # Download from all sources
                raw_data = self.aggregator.download_all_sources(
                    symbol, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if not raw_data:
                    return {
                        "success": False,
                        "error": f"No data available for {symbol}",
                        "records_count": 0
                    }
                
                # Merge and process data
                merged_df = self.aggregator.merge_data_sources(raw_data)
                
                if include_indicators:
                    processed_df = self.aggregator.add_all_indicators(merged_df)
                else:
                    processed_df = merged_df
                
                # Prepare data for transfer to database agent
                data_dict = {
                    "symbol": symbol,
                    "data": processed_df.to_dict(orient='records'),
                    "columns": list(processed_df.columns),
                    "index": [str(idx) for idx in processed_df.index],
                    "records_count": len(processed_df),
                    "date_range": {
                        "start": str(processed_df.index.min()),
                        "end": str(processed_df.index.max())
                    },
                    "sources": list(raw_data.keys())
                }
                
                logger.info(f"Successfully loaded {len(processed_df)} records for {symbol}")
                
                return {
                    "success": True,
                    "data": data_dict,
                    "message": f"Loaded {len(processed_df)} records from {len(raw_data)} sources"
                }
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "records_count": 0
                }

        @tool
        def load_multiple_symbols(symbols: List[str], days_back: int = 365) -> Dict[str, Any]:
            """Load historical data for multiple crypto symbols"""
            results = {}
            total_records = 0
            
            for symbol in symbols:
                result = load_symbol_data(symbol, days_back, True)
                results[symbol] = result
                if result["success"]:
                    total_records += result["data"]["records_count"]
            
            successful_symbols = [s for s, r in results.items() if r["success"]]
            
            return {
                "symbols_processed": len(symbols),
                "symbols_successful": len(successful_symbols),
                "total_records": total_records,
                "results": results,
                "summary": f"Loaded data for {len(successful_symbols)}/{len(symbols)} symbols"
            }

        @tool
        def get_available_datasets() -> List[Dict[str, Any]]:
            """Get list of all available cached datasets"""
            return self.aggregator.get_available_datasets()

        @tool
        def create_training_dataset(symbol: str, features: List[str] = None) -> Dict[str, Any]:
            """Create a comprehensive training dataset with all indicators"""
            try:
                df = self.aggregator.create_training_dataset(symbol, features)
                
                if df.empty:
                    return {
                        "success": False,
                        "error": f"No training data created for {symbol}"
                    }
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "records_count": len(df),
                    "features_count": len(df.columns),
                    "date_range": {
                        "start": str(df.index.min()),
                        "end": str(df.index.max())
                    },
                    "message": f"Created training dataset with {len(df)} records and {len(df.columns)} features"
                }
                
            except Exception as e:
                logger.error(f"Error creating training dataset for {symbol}: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        # Create Strand agent with tools
        self.agent = Agent(
            tools=[
                load_symbol_data,
                load_multiple_symbols, 
                get_available_datasets,
                create_training_dataset
            ]
        )
        
        # Register with A2A registry
        agent_registry.register_agent(
            'historical-loader-001',
            'historical_loader',
            ['data_loading', 'historical_data', 'technical_indicators', 'bulk_processing'],
            {'version': '1.0', 'model_provider': model_provider}
        )

    async def process_request(self, request: str) -> str:
        """Process natural language requests for data loading"""
        try:
            response = await self.agent(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return f"Error: {str(e)}"

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

# Global instance
historical_loader_agent = HistoricalLoaderAgent()