#!/usr/bin/env python3
"""
A2A S3 Storage CLI - AWS S3 storage management and data archiving
Real implementation with S3 operations, agent logging, and data management
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set environment variables for development
os.environ["ENVIRONMENT"] = "development"
os.environ["SKIP_DB_INIT"] = "true"

try:
    import boto3

    from src.cryptotrading.core.agents.s3_logging_mixin import S3LoggingMixin
    from src.cryptotrading.infrastructure.mcp.s3_storage_mcp_tools import S3StorageMCPTools
    from src.cryptotrading.infrastructure.storage.s3_client import S3StorageClient

    REAL_IMPLEMENTATION = True
except ImportError as e:
    print(f"⚠️ Using fallback implementation: {e}")
    REAL_IMPLEMENTATION = False


class S3StorageAgent:
    """S3 Storage Agent with comprehensive S3 operations"""

    def __init__(self):
        self.agent_id = "s3_storage_agent"
        self.capabilities = [
            "log_agent_activity",
            "store_agent_data",
            "store_calculation_result",
            "store_market_analysis",
            "backup_agent_state",
            "retrieve_agent_logs",
            "get_storage_stats",
        ]

        if REAL_IMPLEMENTATION:
            self.mcp_tools = S3StorageMCPTools()
            self.s3_client = S3StorageClient()
            self.logging_mixin = S3LoggingMixin()

        # Mock S3 bucket structure
        self.bucket_name = "cryptotrading-a2a-storage"
        self.folder_structure = {
            "agent-logs/": "Agent activity and error logs",
            "agent-data/": "Agent state and configuration data",
            "calculations/": "MCTS and ML calculation results",
            "market-analysis/": "Technical analysis and market data",
            "backups/": "Agent state backups and snapshots",
            "reports/": "Generated reports and summaries",
        }

    async def log_agent_activity(
        self,
        agent_id: str,
        activity_type: str,
        activity_data: Dict[str, Any],
        log_level: str = "info",
    ) -> Dict[str, Any]:
        """Log agent activity to S3"""
        if not REAL_IMPLEMENTATION:
            return self._mock_log_activity(agent_id, activity_type, activity_data, log_level)

        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "activity_type": activity_type,
                "log_level": log_level,
                "data": activity_data,
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            }

            # Store in S3
            s3_key = f"agent-logs/{agent_id}/{datetime.now().strftime('%Y/%m/%d')}/{log_level}_{datetime.now().strftime('%H%M%S')}.json"

            result = await self.s3_client.upload_json(
                bucket=self.bucket_name, key=s3_key, data=log_entry
            )

            return {
                "success": True,
                "agent_id": agent_id,
                "activity_type": activity_type,
                "log_level": log_level,
                "s3_location": f"s3://{self.bucket_name}/{s3_key}",
                "upload_size": result.get("size", 0),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Activity logging failed: {str(e)}"}

    def _mock_log_activity(
        self, agent_id: str, activity_type: str, activity_data: Dict[str, Any], log_level: str
    ) -> Dict[str, Any]:
        """Mock activity logging"""
        import random

        s3_key = f"agent-logs/{agent_id}/{datetime.now().strftime('%Y/%m/%d')}/{log_level}_{datetime.now().strftime('%H%M%S')}.json"

        return {
            "success": True,
            "agent_id": agent_id,
            "activity_type": activity_type,
            "log_level": log_level,
            "s3_location": f"s3://{self.bucket_name}/{s3_key}",
            "upload_size": random.randint(500, 5000),
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def store_agent_data(
        self, agent_id: str, data_type: str, data: Dict[str, Any], metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Store agent data to S3"""
        if not REAL_IMPLEMENTATION:
            return self._mock_store_data(agent_id, data_type, data, metadata)

        try:
            storage_entry = {
                "agent_id": agent_id,
                "data_type": data_type,
                "data": data,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
            }

            s3_key = (
                f"agent-data/{agent_id}/{data_type}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            result = await self.s3_client.upload_json(
                bucket=self.bucket_name, key=s3_key, data=storage_entry
            )

            return {
                "success": True,
                "agent_id": agent_id,
                "data_type": data_type,
                "s3_location": f"s3://{self.bucket_name}/{s3_key}",
                "upload_size": result.get("size", 0),
                "object_key": s3_key,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Data storage failed: {str(e)}"}

    def _mock_store_data(
        self, agent_id: str, data_type: str, data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock data storage"""
        import random

        s3_key = (
            f"agent-data/{agent_id}/{data_type}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "data_type": data_type,
            "s3_location": f"s3://{self.bucket_name}/{s3_key}",
            "upload_size": random.randint(1000, 50000),
            "object_key": s3_key,
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def store_calculation_result(
        self, calculation_type: str, result_data: Dict[str, Any], agent_id: str = None
    ) -> Dict[str, Any]:
        """Store calculation results to S3"""
        if not REAL_IMPLEMENTATION:
            return self._mock_store_calculation(calculation_type, result_data, agent_id)

        try:
            calc_entry = {
                "calculation_type": calculation_type,
                "agent_id": agent_id or "unknown",
                "result_data": result_data,
                "computation_timestamp": datetime.now().isoformat(),
                "storage_version": "1.0",
            }

            s3_key = f"calculations/{calculation_type}/{datetime.now().strftime('%Y/%m')}/calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            result = await self.s3_client.upload_json(
                bucket=self.bucket_name, key=s3_key, data=calc_entry
            )

            return {
                "success": True,
                "calculation_type": calculation_type,
                "agent_id": agent_id,
                "s3_location": f"s3://{self.bucket_name}/{s3_key}",
                "upload_size": result.get("size", 0),
                "object_key": s3_key,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Calculation storage failed: {str(e)}"}

    def _mock_store_calculation(
        self, calculation_type: str, result_data: Dict[str, Any], agent_id: str
    ) -> Dict[str, Any]:
        """Mock calculation storage"""
        import random

        s3_key = f"calculations/{calculation_type}/{datetime.now().strftime('%Y/%m')}/calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        return {
            "success": True,
            "calculation_type": calculation_type,
            "agent_id": agent_id or "unknown",
            "s3_location": f"s3://{self.bucket_name}/{s3_key}",
            "upload_size": random.randint(2000, 100000),
            "object_key": s3_key,
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def store_market_analysis(
        self, analysis_type: str, symbol: str, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store market analysis results to S3"""
        if not REAL_IMPLEMENTATION:
            return self._mock_store_analysis(analysis_type, symbol, analysis_data)

        try:
            analysis_entry = {
                "analysis_type": analysis_type,
                "symbol": symbol,
                "analysis_data": analysis_data,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_version": "1.0",
            }

            s3_key = f"market-analysis/{symbol}/{analysis_type}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            result = await self.s3_client.upload_json(
                bucket=self.bucket_name, key=s3_key, data=analysis_entry
            )

            return {
                "success": True,
                "analysis_type": analysis_type,
                "symbol": symbol,
                "s3_location": f"s3://{self.bucket_name}/{s3_key}",
                "upload_size": result.get("size", 0),
                "object_key": s3_key,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Market analysis storage failed: {str(e)}"}

    def _mock_store_analysis(
        self, analysis_type: str, symbol: str, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock analysis storage"""
        import random

        s3_key = f"market-analysis/{symbol}/{analysis_type}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        return {
            "success": True,
            "analysis_type": analysis_type,
            "symbol": symbol,
            "s3_location": f"s3://{self.bucket_name}/{s3_key}",
            "upload_size": random.randint(5000, 200000),
            "object_key": s3_key,
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def backup_agent_state(
        self, agent_id: str, state_data: Dict[str, Any], backup_type: str = "scheduled"
    ) -> Dict[str, Any]:
        """Backup agent state to S3"""
        if not REAL_IMPLEMENTATION:
            return self._mock_backup_state(agent_id, state_data, backup_type)

        try:
            backup_entry = {
                "agent_id": agent_id,
                "backup_type": backup_type,
                "state_data": state_data,
                "backup_timestamp": datetime.now().isoformat(),
                "backup_version": "1.0",
            }

            s3_key = f"backups/{agent_id}/{backup_type}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            result = await self.s3_client.upload_json(
                bucket=self.bucket_name, key=s3_key, data=backup_entry
            )

            return {
                "success": True,
                "agent_id": agent_id,
                "backup_type": backup_type,
                "s3_location": f"s3://{self.bucket_name}/{s3_key}",
                "backup_size": result.get("size", 0),
                "object_key": s3_key,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Agent backup failed: {str(e)}"}

    def _mock_backup_state(
        self, agent_id: str, state_data: Dict[str, Any], backup_type: str
    ) -> Dict[str, Any]:
        """Mock state backup"""
        import random

        s3_key = f"backups/{agent_id}/{backup_type}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        return {
            "success": True,
            "agent_id": agent_id,
            "backup_type": backup_type,
            "s3_location": f"s3://{self.bucket_name}/{s3_key}",
            "backup_size": random.randint(10000, 500000),
            "object_key": s3_key,
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def retrieve_agent_logs(
        self,
        agent_id: str,
        start_date: str = None,
        end_date: str = None,
        log_level: str = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Retrieve agent logs from S3"""
        if not REAL_IMPLEMENTATION:
            return self._mock_retrieve_logs(agent_id, start_date, end_date, log_level, limit)

        try:
            # Build S3 prefix based on parameters
            prefix = f"agent-logs/{agent_id}/"
            if start_date:
                # Add date filtering to prefix
                prefix += start_date.replace("-", "/")

            result = await self.s3_client.list_objects(
                bucket=self.bucket_name, prefix=prefix, limit=limit
            )

            logs = []
            for obj in result.get("objects", []):
                try:
                    log_data = await self.s3_client.download_json(
                        bucket=self.bucket_name, key=obj["key"]
                    )

                    # Filter by log level if specified
                    if not log_level or log_data.get("log_level") == log_level:
                        logs.append(log_data)

                except Exception as e:
                    continue

            return {
                "success": True,
                "agent_id": agent_id,
                "logs_retrieved": len(logs),
                "logs": logs,
                "filters": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "log_level": log_level,
                    "limit": limit,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Log retrieval failed: {str(e)}"}

    def _mock_retrieve_logs(
        self, agent_id: str, start_date: str, end_date: str, log_level: str, limit: int
    ) -> Dict[str, Any]:
        """Mock log retrieval"""
        import random

        # Generate mock logs
        logs = []
        log_types = ["info", "warning", "error", "debug"]
        activities = ["calculation", "data_load", "analysis", "backup", "communication"]

        for i in range(min(limit, random.randint(5, 50))):
            log_entry = {
                "timestamp": (datetime.now() - timedelta(hours=random.randint(0, 72))).isoformat(),
                "agent_id": agent_id,
                "activity_type": random.choice(activities),
                "log_level": log_level or random.choice(log_types),
                "data": {
                    "message": f"Mock log entry {i+1}",
                    "details": {"status": "success", "duration": random.randint(100, 5000)},
                },
            }
            logs.append(log_entry)

        return {
            "success": True,
            "agent_id": agent_id,
            "logs_retrieved": len(logs),
            "logs": logs,
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "log_level": log_level,
                "limit": limit,
            },
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_storage_stats(self, time_period: str = "24h") -> Dict[str, Any]:
        """Get S3 storage statistics"""
        if not REAL_IMPLEMENTATION:
            return self._mock_storage_stats(time_period)

        try:
            stats = await self.s3_client.get_bucket_stats(self.bucket_name)

            return {
                "success": True,
                "bucket_name": self.bucket_name,
                "time_period": time_period,
                "storage_stats": {
                    "total_objects": stats.get("object_count", 0),
                    "total_size_bytes": stats.get("total_size", 0),
                    "total_size_mb": stats.get("total_size", 0) / (1024 * 1024),
                    "folders": stats.get("folder_stats", {}),
                },
                "recent_activity": stats.get("recent_activity", {}),
                "cost_estimate": stats.get("cost_estimate", {}),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Storage stats retrieval failed: {str(e)}"}

    def _mock_storage_stats(self, time_period: str) -> Dict[str, Any]:
        """Mock storage statistics"""
        import random

        total_objects = random.randint(1000, 10000)
        total_size_mb = random.randint(500, 5000)

        folder_stats = {}
        for folder, description in self.folder_structure.items():
            folder_stats[folder] = {
                "objects": random.randint(50, total_objects // 4),
                "size_mb": random.randint(10, total_size_mb // 4),
                "description": description,
            }

        return {
            "success": True,
            "bucket_name": self.bucket_name,
            "time_period": time_period,
            "storage_stats": {
                "total_objects": total_objects,
                "total_size_bytes": total_size_mb * 1024 * 1024,
                "total_size_mb": total_size_mb,
                "folders": folder_stats,
            },
            "recent_activity": {
                "uploads_24h": random.randint(50, 500),
                "downloads_24h": random.randint(20, 200),
                "deletes_24h": random.randint(0, 10),
            },
            "cost_estimate": {
                "monthly_storage_cost": round(total_size_mb * 0.023 / 1024, 2),
                "monthly_request_cost": round(random.uniform(1.0, 10.0), 2),
                "total_estimated_monthly": round(random.uniform(5.0, 50.0), 2),
            },
            "mock": True,
            "timestamp": datetime.now().isoformat(),
        }


# Global agent instance
agent = S3StorageAgent()


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
    """A2A S3 Storage CLI - AWS S3 storage management"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if not REAL_IMPLEMENTATION:
        click.echo("⚠️ Running in fallback mode - using mock S3 operations")


@cli.command()
@click.argument("agent-id")
@click.argument("activity-type")
@click.option("--data", help="JSON string with activity data")
@click.option(
    "--level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Log level",
)
@click.pass_context
@async_command
async def log(ctx, agent_id, activity_type, data, level):
    """Log agent activity to S3"""
    try:
        activity_data = json.loads(data) if data else {"message": f"{activity_type} activity"}

        result = await agent.log_agent_activity(agent_id, activity_type, activity_data, level)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"📝 Activity Logged - {level.upper()}")
        click.echo("=" * 40)
        click.echo(f"Agent: {result.get('agent_id')}")
        click.echo(f"Activity: {result.get('activity_type')}")
        click.echo(f"Level: {result.get('log_level')}")
        click.echo(f"S3 Location: {result.get('s3_location')}")
        click.echo(f"Upload Size: {result.get('upload_size')} bytes")

        if result.get("mock"):
            click.echo("🔄 Mock logging - enable real implementation for actual S3 storage")

        if ctx.obj["verbose"]:
            click.echo(f"Timestamp: {result.get('timestamp')}")

    except json.JSONDecodeError:
        click.echo("❌ Invalid JSON in --data parameter", err=True)
    except Exception as e:
        click.echo(f"Error logging activity: {e}", err=True)


@cli.command()
@click.argument("agent-id")
@click.argument("data-type")
@click.argument("data-file", type=click.Path(exists=True))
@click.option("--metadata", help="JSON string with metadata")
@click.pass_context
@async_command
async def store(ctx, agent_id, data_type, data_file, metadata):
    """Store agent data to S3"""
    try:
        # Load data from file
        with open(data_file, "r") as f:
            if data_file.endswith(".json"):
                data = json.load(f)
            else:
                data = {"content": f.read(), "file_type": "text"}

        metadata_dict = json.loads(metadata) if metadata else {}

        result = await agent.store_agent_data(agent_id, data_type, data, metadata_dict)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo("💾 Agent Data Stored")
        click.echo("=" * 40)
        click.echo(f"Agent: {result.get('agent_id')}")
        click.echo(f"Data Type: {result.get('data_type')}")
        click.echo(f"S3 Location: {result.get('s3_location')}")
        click.echo(f"Upload Size: {result.get('upload_size')} bytes")

        if result.get("mock"):
            click.echo("🔄 Mock storage - enable real implementation for actual S3 storage")

        if ctx.obj["verbose"]:
            click.echo(f"Object Key: {result.get('object_key')}")
            click.echo(f"Timestamp: {result.get('timestamp')}")

    except json.JSONDecodeError:
        click.echo("❌ Invalid JSON in file or metadata", err=True)
    except Exception as e:
        click.echo(f"Error storing data: {e}", err=True)


@cli.command()
@click.argument("calculation-type")
@click.argument("result-file", type=click.Path(exists=True))
@click.option("--agent-id", help="Agent that performed the calculation")
@click.pass_context
@async_command
async def calculation(ctx, calculation_type, result_file, agent_id):
    """Store calculation results to S3"""
    try:
        with open(result_file, "r") as f:
            result_data = json.load(f)

        result = await agent.store_calculation_result(calculation_type, result_data, agent_id)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo("🧮 Calculation Result Stored")
        click.echo("=" * 40)
        click.echo(f"Type: {result.get('calculation_type')}")
        click.echo(f"Agent: {result.get('agent_id', 'N/A')}")
        click.echo(f"S3 Location: {result.get('s3_location')}")
        click.echo(f"Upload Size: {result.get('upload_size')} bytes")

        if result.get("mock"):
            click.echo("🔄 Mock storage - enable real implementation for actual S3 storage")

        if ctx.obj["verbose"]:
            click.echo(f"Object Key: {result.get('object_key')}")
            click.echo(f"Timestamp: {result.get('timestamp')}")

    except json.JSONDecodeError:
        click.echo("❌ Invalid JSON in result file", err=True)
    except Exception as e:
        click.echo(f"Error storing calculation: {e}", err=True)


@cli.command()
@click.argument("analysis-type")
@click.argument("symbol")
@click.argument("analysis-file", type=click.Path(exists=True))
@click.pass_context
@async_command
async def analysis(ctx, analysis_type, symbol, analysis_file):
    """Store market analysis to S3"""
    try:
        with open(analysis_file, "r") as f:
            analysis_data = json.load(f)

        result = await agent.store_market_analysis(analysis_type, symbol, analysis_data)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo("📊 Market Analysis Stored")
        click.echo("=" * 40)
        click.echo(f"Type: {result.get('analysis_type')}")
        click.echo(f"Symbol: {result.get('symbol')}")
        click.echo(f"S3 Location: {result.get('s3_location')}")
        click.echo(f"Upload Size: {result.get('upload_size')} bytes")

        if result.get("mock"):
            click.echo("🔄 Mock storage - enable real implementation for actual S3 storage")

        if ctx.obj["verbose"]:
            click.echo(f"Object Key: {result.get('object_key')}")
            click.echo(f"Timestamp: {result.get('timestamp')}")

    except json.JSONDecodeError:
        click.echo("❌ Invalid JSON in analysis file", err=True)
    except Exception as e:
        click.echo(f"Error storing analysis: {e}", err=True)


@cli.command()
@click.argument("agent-id")
@click.argument("state-file", type=click.Path(exists=True))
@click.option(
    "--backup-type",
    default="manual",
    type=click.Choice(["manual", "scheduled", "emergency"]),
    help="Type of backup",
)
@click.pass_context
@async_command
async def backup(ctx, agent_id, state_file, backup_type):
    """Backup agent state to S3"""
    try:
        with open(state_file, "r") as f:
            state_data = json.load(f)

        result = await agent.backup_agent_state(agent_id, state_data, backup_type)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo("💾 Agent State Backed Up")
        click.echo("=" * 40)
        click.echo(f"Agent: {result.get('agent_id')}")
        click.echo(f"Backup Type: {result.get('backup_type')}")
        click.echo(f"S3 Location: {result.get('s3_location')}")
        click.echo(f"Backup Size: {result.get('backup_size')} bytes")

        if result.get("mock"):
            click.echo("🔄 Mock backup - enable real implementation for actual S3 storage")

        if ctx.obj["verbose"]:
            click.echo(f"Object Key: {result.get('object_key')}")
            click.echo(f"Timestamp: {result.get('timestamp')}")

    except json.JSONDecodeError:
        click.echo("❌ Invalid JSON in state file", err=True)
    except Exception as e:
        click.echo(f"Error backing up state: {e}", err=True)


@cli.command()
@click.argument("agent-id")
@click.option("--start-date", help="Start date for log retrieval (YYYY-MM-DD)")
@click.option("--end-date", help="End date for log retrieval (YYYY-MM-DD)")
@click.option(
    "--level", type=click.Choice(["debug", "info", "warning", "error"]), help="Filter by log level"
)
@click.option("--limit", default=100, help="Maximum number of logs to retrieve")
@click.option("--output", help="Save logs to file")
@click.pass_context
@async_command
async def logs(ctx, agent_id, start_date, end_date, level, limit, output):
    """Retrieve agent logs from S3"""
    try:
        result = await agent.retrieve_agent_logs(agent_id, start_date, end_date, level, limit)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"📜 Agent Logs Retrieved - {agent_id}")
        click.echo("=" * 50)
        click.echo(f"Logs Found: {result.get('logs_retrieved')}")

        filters = result.get("filters", {})
        if any(filters.values()):
            click.echo("Filters Applied:")
            for key, value in filters.items():
                if value:
                    click.echo(f"  {key.replace('_', ' ').title()}: {value}")
        click.echo()

        logs = result.get("logs", [])[:10]  # Show first 10 logs
        if logs:
            click.echo("📋 Recent Logs:")
            for log in logs:
                level_emoji = {"error": "🔴", "warning": "🟡", "info": "🟢", "debug": "🔵"}.get(
                    log.get("log_level", "info"), "⚪"
                )
                click.echo(
                    f"  {level_emoji} {log.get('timestamp')} - {log.get('activity_type')} ({log.get('log_level')})"
                )
                if ctx.obj["verbose"]:
                    data = log.get("data", {})
                    if data.get("message"):
                        click.echo(f"      {data['message']}")

            if result.get("logs_retrieved", 0) > 10:
                click.echo(f"  ... and {result.get('logs_retrieved') - 10} more logs")

        if output:
            with open(output, "w") as f:
                json.dump(result.get("logs", []), f, indent=2)
            click.echo(f"\n💾 Logs saved to: {output}")

        if result.get("mock"):
            click.echo("\n🔄 Mock retrieval - enable real implementation for actual S3 logs")

        if ctx.obj["verbose"]:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")

    except Exception as e:
        click.echo(f"Error retrieving logs: {e}", err=True)


@cli.command()
@click.option(
    "--period",
    default="24h",
    type=click.Choice(["1h", "24h", "7d", "30d"]),
    help="Time period for statistics",
)
@click.pass_context
@async_command
async def stats(ctx, period):
    """Get S3 storage statistics"""
    try:
        result = await agent.get_storage_stats(period)

        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return

        click.echo(f"📊 S3 Storage Statistics - {period}")
        click.echo("=" * 50)
        click.echo(f"Bucket: {result.get('bucket_name')}")

        storage_stats = result.get("storage_stats", {})
        if storage_stats:
            click.echo(f"Total Objects: {storage_stats.get('total_objects', 0):,}")
            click.echo(f"Total Size: {storage_stats.get('total_size_mb', 0):.1f} MB")
            click.echo()

            folders = storage_stats.get("folders", {})
            if folders:
                click.echo("📁 Folder Breakdown:")
                for folder, stats in folders.items():
                    click.echo(f"  {folder}")
                    click.echo(f"    Objects: {stats.get('objects', 0):,}")
                    click.echo(f"    Size: {stats.get('size_mb', 0):.1f} MB")
                    if ctx.obj["verbose"]:
                        click.echo(f"    Description: {stats.get('description', 'N/A')}")
                    click.echo()

        recent_activity = result.get("recent_activity", {})
        if recent_activity:
            click.echo("🔄 Recent Activity (24h):")
            click.echo(f"  Uploads: {recent_activity.get('uploads_24h', 0):,}")
            click.echo(f"  Downloads: {recent_activity.get('downloads_24h', 0):,}")
            click.echo(f"  Deletes: {recent_activity.get('deletes_24h', 0):,}")
            click.echo()

        if ctx.obj["verbose"]:
            cost_estimate = result.get("cost_estimate", {})
            if cost_estimate:
                click.echo("💰 Cost Estimate:")
                click.echo(f"  Storage: ${cost_estimate.get('monthly_storage_cost', 0):.2f}/month")
                click.echo(f"  Requests: ${cost_estimate.get('monthly_request_cost', 0):.2f}/month")
                click.echo(f"  Total: ${cost_estimate.get('total_estimated_monthly', 0):.2f}/month")
                click.echo()

        if result.get("mock"):
            click.echo("🔄 Mock statistics - enable real implementation for actual S3 data")

        click.echo(f"Timestamp: {result.get('timestamp')}")

    except Exception as e:
        click.echo(f"Error retrieving statistics: {e}", err=True)


@cli.command()
@click.pass_context
def structure(ctx):
    """Show S3 bucket folder structure"""
    click.echo(f"📁 S3 Bucket Structure - {agent.bucket_name}")
    click.echo("=" * 60)

    for folder, description in agent.folder_structure.items():
        click.echo(f"📂 {folder}")
        click.echo(f"   {description}")
        click.echo()


@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    click.echo("🔧 S3 Storage Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")


@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    click.echo("🏥 S3 Storage Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo(f"Bucket: {agent.bucket_name}")
    click.echo(f"Folders: {len(agent.folder_structure)}")
    click.echo(f"Implementation: {'Real' if REAL_IMPLEMENTATION else 'Fallback'}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    cli()
