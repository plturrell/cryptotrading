#!/usr/bin/env python3
"""
A2A Database Agent CLI - Database operations and management
Comprehensive CLI interface for Database Agent with all capabilities
"""

import asyncio
import os
import sys
from datetime import datetime

import click

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set environment variables for CLI
os.environ["ENVIRONMENT"] = "development"
os.environ["SKIP_DB_INIT"] = "true"

try:
    from cryptotrading.core.agents.database_agent import DatabaseAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback minimal Database agent for CLI testing...")

    class FallbackDatabaseAgent:
        """Minimal Database agent for CLI testing when imports fail"""

        def __init__(self):
            self.agent_id = "database_agent"
            self.capabilities = [
                "data_storage",
                "data_retrieval",
                "bulk_insert",
                "ai_analysis_storage",
                "portfolio_management",
                "trade_history",
                "database_health",
                "query_optimization",
                "data_cleanup",
            ]

        async def store_data(self, table, data, metadata=None):
            """Mock data storage"""
            return {
                "table": table,
                "records_stored": len(data) if isinstance(data, list) else 1,
                "metadata": metadata or {},
                "storage_id": f"store_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
            }

        async def retrieve_data(self, table, filters=None, limit=100):
            """Mock data retrieval"""
            return {
                "table": table,
                "filters": filters or {},
                "records_found": 42,
                "limit": limit,
                "data": [{"id": i, "value": f"sample_{i}"} for i in range(min(5, limit))],
                "timestamp": datetime.now().isoformat(),
            }

        async def bulk_insert(self, table, records, batch_size=1000):
            """Mock bulk insert"""
            return {
                "table": table,
                "total_records": len(records) if isinstance(records, list) else records,
                "batch_size": batch_size,
                "batches_processed": max(
                    1, (len(records) if isinstance(records, list) else records) // batch_size
                ),
                "success_rate": 0.98,
                "timestamp": datetime.now().isoformat(),
            }

        async def store_ai_analysis(self, analysis_type, symbol, results):
            """Mock AI analysis storage"""
            return {
                "analysis_type": analysis_type,
                "symbol": symbol,
                "analysis_id": f"ai_{analysis_type}_{symbol}_{datetime.now().strftime('%Y%m%d')}",
                "results_size": len(str(results)),
                "stored": True,
                "timestamp": datetime.now().isoformat(),
            }

        async def manage_portfolio(self, action, portfolio_data):
            """Mock portfolio management"""
            return {
                "action": action,
                "portfolio_id": portfolio_data.get("id", "default"),
                "assets_count": len(portfolio_data.get("assets", [])),
                "total_value": portfolio_data.get("total_value", 100000),
                "status": "updated",
                "timestamp": datetime.now().isoformat(),
            }

        async def get_trade_history(self, symbol=None, start_date=None, end_date=None):
            """Mock trade history retrieval"""
            return {
                "symbol": symbol or "ALL",
                "start_date": start_date,
                "end_date": end_date,
                "trades_found": 156,
                "total_volume": 2500000.0,
                "profit_loss": 12500.0,
                "timestamp": datetime.now().isoformat(),
            }

        async def check_database_health(self):
            """Mock database health check"""
            return {
                "status": "healthy",
                "connections": {"active": 5, "max": 100},
                "storage": {"used": "2.5GB", "available": "47.5GB"},
                "performance": {"avg_query_time": "15ms", "slow_queries": 2},
                "last_backup": "2025-08-30T06:00:00Z",
                "timestamp": datetime.now().isoformat(),
            }

        async def optimize_queries(self, table=None):
            """Mock query optimization"""
            return {
                "table": table or "ALL",
                "indexes_analyzed": 12,
                "optimizations_applied": 5,
                "performance_improvement": "25%",
                "recommendations": ["Add index on timestamp", "Partition large tables"],
                "timestamp": datetime.now().isoformat(),
            }

        async def cleanup_data(self, older_than_days=30, dry_run=True):
            """Mock data cleanup"""
            return {
                "older_than_days": older_than_days,
                "dry_run": dry_run,
                "records_to_delete": 1250,
                "space_to_free": "150MB",
                "tables_affected": ["trades", "market_data", "logs"],
                "timestamp": datetime.now().isoformat(),
            }


# Global agent instance
agent = None


def get_agent():
    """Get or create agent instance"""
    global agent
    if agent is None:
        try:
            agent = DatabaseAgent()
        except:
            agent = FallbackDatabaseAgent()
    return agent


def async_command(f):
    """Decorator to run async commands"""

    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """A2A Database Agent CLI - Database operations and management"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("table")
@click.argument("data")
@click.option("--metadata", help="JSON metadata string")
@click.pass_context
@async_command
async def store(ctx, table, data, metadata):
    """Store data in database table"""
    agent = get_agent()

    try:
        import json

        data_obj = json.loads(data) if data.startswith("[") or data.startswith("{") else data
        metadata_obj = json.loads(metadata) if metadata else None

        result = await agent.store_data(table, data_obj, metadata_obj)

        click.echo(f"💾 Data Storage - {table}")
        click.echo("=" * 50)
        click.echo(f"Records Stored: {result['records_stored']}")
        click.echo(f"Storage ID: {result['storage_id']}")

        if ctx.obj["verbose"]:
            click.echo(f"Metadata: {result['metadata']}")
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error storing data: {e}", err=True)


@cli.command()
@click.argument("table")
@click.option("--filters", help="JSON filter string")
@click.option("--limit", default=100, help="Maximum records to retrieve")
@click.pass_context
@async_command
async def retrieve(ctx, table, filters, limit):
    """Retrieve data from database table"""
    agent = get_agent()

    try:
        import json

        filters_obj = json.loads(filters) if filters else None

        result = await agent.retrieve_data(table, filters_obj, limit)

        click.echo(f"🔍 Data Retrieval - {table}")
        click.echo("=" * 50)
        click.echo(f"Records Found: {result['records_found']}")
        click.echo(f"Limit: {result['limit']}")

        if ctx.obj["verbose"]:
            click.echo(f"Filters: {result['filters']}")
            click.echo(f"Sample Data: {result['data'][:2]}")
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error retrieving data: {e}", err=True)


@cli.command()
@click.argument("table")
@click.argument("records", type=int)
@click.option("--batch-size", default=1000, help="Batch size for processing")
@click.pass_context
@async_command
async def bulk(ctx, table, records, batch_size):
    """Perform bulk insert operation"""
    agent = get_agent()

    try:
        result = await agent.bulk_insert(table, records, batch_size)

        click.echo(f"📦 Bulk Insert - {table}")
        click.echo("=" * 50)
        click.echo(f"Total Records: {result['total_records']}")
        click.echo(f"Batch Size: {result['batch_size']}")
        click.echo(f"Batches Processed: {result['batches_processed']}")
        click.echo(f"Success Rate: {result['success_rate']:.1%}")

        if ctx.obj["verbose"]:
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error in bulk insert: {e}", err=True)


@cli.command()
@click.argument("analysis-type")
@click.argument("symbol")
@click.argument("results")
@click.pass_context
@async_command
async def ai_store(ctx, analysis_type, symbol, results):
    """Store AI analysis results"""
    agent = get_agent()

    try:
        import json

        results_obj = json.loads(results) if results.startswith("{") else results

        result = await agent.store_ai_analysis(analysis_type, symbol, results_obj)

        click.echo(f"🤖 AI Analysis Storage")
        click.echo("=" * 50)
        click.echo(f"Analysis Type: {result['analysis_type']}")
        click.echo(f"Symbol: {result['symbol']}")
        click.echo(f"Analysis ID: {result['analysis_id']}")
        click.echo(f"Results Size: {result['results_size']} bytes")

        if ctx.obj["verbose"]:
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error storing AI analysis: {e}", err=True)


@cli.command()
@click.argument("action")
@click.argument("portfolio-data")
@click.pass_context
@async_command
async def portfolio(ctx, action, portfolio_data):
    """Manage portfolio data"""
    agent = get_agent()

    try:
        import json

        portfolio_obj = json.loads(portfolio_data)

        result = await agent.manage_portfolio(action, portfolio_obj)

        click.echo(f"💼 Portfolio Management")
        click.echo("=" * 50)
        click.echo(f"Action: {result['action']}")
        click.echo(f"Portfolio ID: {result['portfolio_id']}")
        click.echo(f"Assets Count: {result['assets_count']}")
        click.echo(f"Total Value: ${result['total_value']:,.2f}")
        click.echo(f"Status: {result['status']}")

        if ctx.obj["verbose"]:
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error managing portfolio: {e}", err=True)


@cli.command()
@click.option("--symbol", help="Filter by symbol")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.pass_context
@async_command
async def trades(ctx, symbol, start_date, end_date):
    """Get trade history"""
    agent = get_agent()

    try:
        result = await agent.get_trade_history(symbol, start_date, end_date)

        click.echo(f"📈 Trade History")
        click.echo("=" * 50)
        click.echo(f"Symbol: {result['symbol']}")
        click.echo(f"Trades Found: {result['trades_found']}")
        click.echo(f"Total Volume: ${result['total_volume']:,.2f}")
        click.echo(f"P&L: ${result['profit_loss']:,.2f}")

        if ctx.obj["verbose"]:
            click.echo(f"Date Range: {result['start_date']} to {result['end_date']}")
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error retrieving trade history: {e}", err=True)


@cli.command()
@click.pass_context
@async_command
async def health(ctx):
    """Check database health"""
    agent = get_agent()

    try:
        result = await agent.check_database_health()

        click.echo(f"🏥 Database Health Check")
        click.echo("=" * 50)
        click.echo(f"Status: {result['status'].upper()}")
        click.echo(
            f"Active Connections: {result['connections']['active']}/{result['connections']['max']}"
        )
        click.echo(
            f"Storage: {result['storage']['used']} used, {result['storage']['available']} available"
        )
        click.echo(f"Avg Query Time: {result['performance']['avg_query_time']}")
        click.echo(f"Slow Queries: {result['performance']['slow_queries']}")
        click.echo(f"Last Backup: {result['last_backup']}")

        if ctx.obj["verbose"]:
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error checking database health: {e}", err=True)


@cli.command()
@click.option("--table", help="Specific table to optimize")
@click.pass_context
@async_command
async def optimize(ctx, table):
    """Optimize database queries"""
    agent = get_agent()

    try:
        result = await agent.optimize_queries(table)

        click.echo(f"⚡ Query Optimization")
        click.echo("=" * 50)
        click.echo(f"Target: {result['table']}")
        click.echo(f"Indexes Analyzed: {result['indexes_analyzed']}")
        click.echo(f"Optimizations Applied: {result['optimizations_applied']}")
        click.echo(f"Performance Improvement: {result['performance_improvement']}")

        click.echo("\nRecommendations:")
        for rec in result["recommendations"]:
            click.echo(f"  • {rec}")

        if ctx.obj["verbose"]:
            click.echo(f"\nTimestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error optimizing queries: {e}", err=True)


@cli.command()
@click.option("--days", default=30, help="Delete data older than N days")
@click.option("--execute", is_flag=True, help="Execute cleanup (default is dry run)")
@click.pass_context
@async_command
async def cleanup(ctx, days, execute):
    """Clean up old database data"""
    agent = get_agent()

    try:
        result = await agent.cleanup_data(days, not execute)

        click.echo(f"🧹 Data Cleanup")
        click.echo("=" * 50)
        click.echo(f"Mode: {'DRY RUN' if result['dry_run'] else 'EXECUTE'}")
        click.echo(f"Older Than: {result['older_than_days']} days")
        click.echo(f"Records to Delete: {result['records_to_delete']}")
        click.echo(f"Space to Free: {result['space_to_free']}")
        click.echo(f"Tables Affected: {', '.join(result['tables_affected'])}")

        if ctx.obj["verbose"]:
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error in data cleanup: {e}", err=True)


@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    agent = get_agent()

    click.echo("🔧 Database Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")


@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    agent = get_agent()

    click.echo("🏥 Database Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    cli()
