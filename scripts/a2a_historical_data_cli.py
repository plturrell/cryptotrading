#!/usr/bin/env python3
"""
A2A Historical Data CLI - Load and manage historical data from multiple sources
Real implementation connecting to Yahoo Finance and FRED APIs
"""

import os
import sys
import asyncio
import json
import click
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set environment variables for development
os.environ['ENVIRONMENT'] = 'development'
os.environ['SKIP_DB_INIT'] = 'true'

try:
    from src.cryptotrading.infrastructure.mcp.historical_data_mcp_tools import HistoricalDataMCPTools
    from src.cryptotrading.data.historical.a2a_data_loader import A2AHistoricalDataLoader
    from src.cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
    from src.cryptotrading.data.historical.fred_client import FREDClient
    REAL_IMPLEMENTATION = True
except ImportError as e:
    print(f"⚠️ Using fallback implementation: {e}")
    REAL_IMPLEMENTATION = False

class HistoricalDataAgent:
    """Real Historical Data Agent implementation"""
    
    def __init__(self):
        self.agent_id = "historical_data_agent"
        self.capabilities = [
            'fetch_historical_data', 'store_historical_data', 
            'analyze_data_quality', 'get_data_summary'
        ]
        
        if REAL_IMPLEMENTATION:
            self.mcp_tools = HistoricalDataMCPTools()
            self.data_loader = A2AHistoricalDataLoader()
            self.yahoo_client = YahooFinanceClient()
            self.fred_client = FREDClient()
        
    async def fetch_historical_data(self, symbols: List[str], start_date: str = None, 
                                  end_date: str = None, source: str = "yahoo") -> Dict[str, Any]:
        """Fetch historical data from specified source"""
        if not REAL_IMPLEMENTATION:
            return self._mock_fetch_data(symbols, start_date, end_date, source)
            
        try:
            if source == "yahoo":
                return await self._fetch_yahoo_data(symbols, start_date, end_date)
            elif source == "fred":
                return await self._fetch_fred_data(symbols, start_date, end_date)
            else:
                return {"error": f"Unsupported data source: {source}"}
        except Exception as e:
            return {"error": f"Data fetch failed: {str(e)}"}
    
    async def _fetch_yahoo_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance"""
        results = {}
        for symbol in symbols:
            try:
                data = await self.yahoo_client.get_historical_data(
                    symbol, start_date=start_date, end_date=end_date
                )
                results[symbol] = {
                    "data": data.to_dict() if hasattr(data, 'to_dict') else data,
                    "records": len(data) if hasattr(data, '__len__') else 0,
                    "source": "yahoo_finance"
                }
            except Exception as e:
                results[symbol] = {"error": str(e)}
        
        return {
            "success": True,
            "symbols": symbols,
            "source": "yahoo_finance",
            "start_date": start_date,
            "end_date": end_date,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fetch_fred_data(self, series_ids: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch data from FRED"""
        results = {}
        for series_id in series_ids:
            try:
                data = await self.fred_client.get_series_data(
                    series_id, start_date=start_date, end_date=end_date
                )
                results[series_id] = {
                    "data": data.to_dict() if hasattr(data, 'to_dict') else data,
                    "records": len(data) if hasattr(data, '__len__') else 0,
                    "source": "fred"
                }
            except Exception as e:
                results[series_id] = {"error": str(e)}
        
        return {
            "success": True,
            "series_ids": series_ids,
            "source": "fred",
            "start_date": start_date,
            "end_date": end_date,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_fetch_data(self, symbols: List[str], start_date: str, end_date: str, source: str) -> Dict[str, Any]:
        """Mock data fetch for testing"""
        return {
            "success": True,
            "symbols": symbols,
            "source": source,
            "start_date": start_date,
            "end_date": end_date,
            "records_fetched": 1000,
            "mock": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def store_historical_data(self, data: Dict[str, Any], storage_path: str = None) -> Dict[str, Any]:
        """Store historical data to persistent storage"""
        if not REAL_IMPLEMENTATION:
            return {
                "success": True,
                "storage_path": storage_path or "mock/storage/path.csv",
                "mock": True,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            result = await self.data_loader.store_data(data, storage_path)
            return {
                "success": True,
                "storage_path": result.get("path"),
                "records_stored": result.get("record_count", 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Storage failed: {str(e)}"}
    
    async def analyze_data_quality(self, data_path: str) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        if not REAL_IMPLEMENTATION:
            return {
                "data_path": data_path,
                "quality_score": 95.5,
                "completeness": 98.2,
                "consistency": 94.1,
                "accuracy": 96.8,
                "missing_values": 47,
                "outliers_detected": 12,
                "mock": True,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            analysis = await self.data_loader.analyze_quality(data_path)
            return {
                "data_path": data_path,
                "quality_score": analysis.get("overall_score"),
                "completeness": analysis.get("completeness"),
                "consistency": analysis.get("consistency"),  
                "accuracy": analysis.get("accuracy"),
                "missing_values": analysis.get("missing_count"),
                "outliers_detected": analysis.get("outlier_count"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Quality analysis failed: {str(e)}"}
    
    async def get_data_summary(self, data_path: str = None, source: str = None) -> Dict[str, Any]:
        """Get summary of available historical data"""
        if not REAL_IMPLEMENTATION:
            return {
                "total_datasets": 156,
                "sources": ["yahoo_finance", "fred", "coinbase"],
                "symbols_available": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"],
                "date_range": {
                    "earliest": "2010-07-17",
                    "latest": "2024-01-15"
                },
                "total_records": 2485672,
                "mock": True,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            summary = await self.data_loader.get_summary(data_path, source)
            return {
                "total_datasets": summary.get("dataset_count"),
                "sources": summary.get("sources", []),
                "symbols_available": summary.get("symbols", []),
                "date_range": summary.get("date_range", {}),
                "total_records": summary.get("total_records"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Summary generation failed: {str(e)}"}

# Global agent instance
agent = HistoricalDataAgent()

def async_command(f):
    """Decorator to run async commands"""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """A2A Historical Data CLI - Load and manage historical data"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if not REAL_IMPLEMENTATION:
        click.echo("⚠️ Running in fallback mode - some features may be limited")

@cli.command()
@click.argument('symbols', nargs=-1, required=True)
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--source', default='yahoo', type=click.Choice(['yahoo', 'fred']), 
              help='Data source')
@click.pass_context
@async_command
async def fetch(ctx, symbols, start_date, end_date, source):
    """Fetch historical data from specified source"""
    try:
        result = await agent.fetch_historical_data(
            list(symbols), start_date, end_date, source
        )
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo(f"📈 Historical Data Fetch - {source.upper()}")
        click.echo("=" * 50)
        click.echo(f"Symbols: {', '.join(symbols)}")
        click.echo(f"Date Range: {start_date or 'default'} to {end_date or 'latest'}")
        click.echo(f"Source: {result.get('source', source)}")
        
        if 'results' in result:
            total_records = 0
            successful = 0
            for symbol, data in result['results'].items():
                if 'error' not in data:
                    successful += 1
                    records = data.get('records', 0)
                    total_records += records
                    click.echo(f"  ✅ {symbol}: {records} records")
                else:
                    click.echo(f"  ❌ {symbol}: {data['error']}")
            
            click.echo(f"\nSummary: {successful}/{len(symbols)} successful, {total_records} total records")
        
        if result.get('mock'):
            click.echo("🔄 Mock data - enable real implementation for live data")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error fetching data: {e}", err=True)

@cli.command()
@click.argument('data-file', type=click.Path(exists=True))
@click.option('--storage-path', help='Custom storage path')
@click.pass_context
@async_command  
async def store(ctx, data_file, storage_path):
    """Store historical data to persistent storage"""
    try:
        # Load data from file
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        result = await agent.store_historical_data(data, storage_path)
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("💾 Data Storage Complete")
        click.echo("=" * 50)
        click.echo(f"Input: {data_file}")
        click.echo(f"Storage Path: {result.get('storage_path')}")
        click.echo(f"Records Stored: {result.get('records_stored', 'N/A')}")
        
        if result.get('mock'):
            click.echo("🔄 Mock storage - enable real implementation for persistent storage")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error storing data: {e}", err=True)

@cli.command()
@click.argument('data-path', type=click.Path())
@click.pass_context
@async_command
async def quality(ctx, data_path):
    """Analyze data quality metrics"""
    try:
        result = await agent.analyze_data_quality(data_path)
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("🔍 Data Quality Analysis")
        click.echo("=" * 50)
        click.echo(f"Data Path: {result.get('data_path')}")
        click.echo(f"Overall Quality Score: {result.get('quality_score'):.1f}/100")
        click.echo(f"Completeness: {result.get('completeness'):.1f}%")
        click.echo(f"Consistency: {result.get('consistency'):.1f}%")
        click.echo(f"Accuracy: {result.get('accuracy'):.1f}%")
        click.echo(f"Missing Values: {result.get('missing_values')}")
        click.echo(f"Outliers Detected: {result.get('outliers_detected')}")
        
        # Quality indicators
        score = result.get('quality_score', 0)
        if score >= 90:
            click.echo("✅ Excellent data quality")
        elif score >= 75:
            click.echo("⚠️ Good data quality with minor issues")
        else:
            click.echo("❌ Poor data quality - review recommended")
        
        if result.get('mock'):
            click.echo("🔄 Mock analysis - enable real implementation for actual metrics")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error analyzing data quality: {e}", err=True)

@cli.command()
@click.option('--data-path', help='Specific data path to summarize')
@click.option('--source', help='Filter by data source')
@click.pass_context
@async_command
async def summary(ctx, data_path, source):
    """Get summary of available historical data"""
    try:
        result = await agent.get_data_summary(data_path, source)
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("📊 Historical Data Summary")
        click.echo("=" * 50)
        click.echo(f"Total Datasets: {result.get('total_datasets')}")
        click.echo(f"Available Sources: {', '.join(result.get('sources', []))}")
        
        symbols = result.get('symbols_available', [])
        if symbols:
            click.echo(f"Symbols Available: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
            click.echo(f"Total Symbols: {len(symbols)}")
        
        date_range = result.get('date_range', {})
        if date_range:
            click.echo(f"Date Range: {date_range.get('earliest')} to {date_range.get('latest')}")
        
        click.echo(f"Total Records: {result.get('total_records'):,}")
        
        if result.get('mock'):
            click.echo("🔄 Mock summary - enable real implementation for live data")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error generating summary: {e}", err=True)

@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    click.echo("🔧 Historical Data Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")

@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    click.echo("🏥 Historical Data Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo(f"Implementation: {'Real' if REAL_IMPLEMENTATION else 'Fallback'}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")

if __name__ == '__main__':
    cli()