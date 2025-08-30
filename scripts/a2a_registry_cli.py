#!/usr/bin/env python3
"""
CLI for A2A Registry and Discovery - Agent registration and discovery system
Provides command-line access to A2A agent registry, discovery, and health monitoring
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import click

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from cryptotrading.core.protocols.a2a.a2a_protocol import AgentStatus
    from cryptotrading.infrastructure.registry.a2a_registry_v2 import (
        AgentHealthStatus,
        EnhancedA2ARegistry,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback minimal registry for CLI testing...")

    class FallbackA2ARegistry:
        """Minimal registry for CLI testing when imports fail"""

        def __init__(self):
            self.agents = {
                "data-loader-agent": {
                    "agent_id": "data-loader-agent",
                    "agent_type": "data_loader",
                    "capabilities": ["yahoo_finance", "fred_data", "data_alignment"],
                    "status": "active",
                    "last_heartbeat": datetime.now().isoformat(),
                    "endpoint": "http://localhost:8001",
                },
                "ml-training-agent": {
                    "agent_id": "ml-training-agent",
                    "agent_type": "ml_trainer",
                    "capabilities": ["model_training", "hyperparameter_tuning", "validation"],
                    "status": "active",
                    "last_heartbeat": (datetime.now() - timedelta(minutes=5)).isoformat(),
                    "endpoint": "http://localhost:8002",
                },
                "inference-agent": {
                    "agent_id": "inference-agent",
                    "agent_type": "inference",
                    "capabilities": ["batch_inference", "streaming_inference", "model_serving"],
                    "status": "inactive",
                    "last_heartbeat": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "endpoint": "http://localhost:8003",
                },
            }

        async def register_agent(self, agent_info: Dict[str, Any]) -> Dict[str, Any]:
            """Mock agent registration"""
            agent_id = agent_info.get("agent_id")
            self.agents[agent_id] = {
                **agent_info,
                "registered_at": datetime.now().isoformat(),
                "status": "active",
            }
            return {
                "status": "registered",
                "agent_id": agent_id,
                "registration_time": datetime.now().isoformat(),
            }

        async def discover_agents(
            self, filter_type: str = None, filter_value: str = None
        ) -> List[Dict[str, Any]]:
            """Mock agent discovery"""
            agents = list(self.agents.values())

            if filter_type == "capability" and filter_value:
                agents = [a for a in agents if filter_value in a.get("capabilities", [])]
            elif filter_type == "agent_type" and filter_value:
                agents = [a for a in agents if a.get("agent_type") == filter_value]
            elif filter_type == "status" and filter_value:
                agents = [a for a in agents if a.get("status") == filter_value]

            return agents

        async def get_agent_health(self, agent_id: str = None) -> Dict[str, Any]:
            """Mock agent health status"""
            if agent_id:
                agent = self.agents.get(agent_id)
                if not agent:
                    return {"error": f"Agent {agent_id} not found"}

                last_heartbeat = datetime.fromisoformat(agent["last_heartbeat"])
                minutes_since = (datetime.now() - last_heartbeat).total_seconds() / 60

                return {
                    "agent_id": agent_id,
                    "status": agent["status"],
                    "last_heartbeat": agent["last_heartbeat"],
                    "minutes_since_heartbeat": round(minutes_since, 1),
                    "health_score": max(0, 100 - minutes_since * 2),
                    "response_time_ms": 45.2 if agent["status"] == "active" else None,
                }
            else:
                # Return health for all agents
                health_data = {}
                for aid, agent in self.agents.items():
                    health_data[aid] = await self.get_agent_health(aid)
                return health_data

        async def unregister_agent(self, agent_id: str) -> Dict[str, Any]:
            """Mock agent unregistration"""
            if agent_id in self.agents:
                del self.agents[agent_id]
                return {"status": "unregistered", "agent_id": agent_id}
            else:
                return {"status": "not_found", "agent_id": agent_id}


def async_command(f):
    """Decorator to run async functions in click commands"""

    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """A2A Registry CLI - Agent registration and discovery"""
    ctx.ensure_object(dict)

    # Set environment for CLI mode
    os.environ["ENVIRONMENT"] = "development"
    os.environ["SKIP_DB_INIT"] = "true"

    # Initialize A2A registry
    try:
        registry = EnhancedA2ARegistry()
    except:
        if verbose:
            print("Using fallback registry due to import/initialization issues")
        registry = FallbackA2ARegistry()

    ctx.obj["registry"] = registry
    ctx.obj["verbose"] = verbose


@cli.command("register")
@click.argument("agent_id")
@click.option(
    "--agent-type", "-t", required=True, help="Agent type (data_loader, ml_trainer, etc.)"
)
@click.option("--capabilities", "-c", multiple=True, help="Agent capabilities")
@click.option("--endpoint", "-e", help="Agent endpoint URL")
@click.option("--metadata", "-m", help="Additional metadata as JSON")
@click.pass_context
@async_command
async def register_agent(ctx, agent_id, agent_type, capabilities, endpoint, metadata):
    """Register a new A2A agent"""
    registry = ctx.obj["registry"]
    verbose = ctx.obj["verbose"]

    try:
        agent_info = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": list(capabilities),
            "endpoint": endpoint,
        }

        if metadata:
            agent_info["metadata"] = json.loads(metadata)

        result = await registry.register_agent(agent_info)

        if verbose:
            print(json.dumps(result, indent=2))
        else:
            print(f"✅ Agent registered successfully:")
            print(f"Agent ID: {result['agent_id']}")
            print(f"Status: {result['status']}")
            print(f"Registration time: {result['registration_time']}")

    except json.JSONDecodeError:
        print("Error: Invalid JSON in metadata")
    except Exception as e:
        print(f"Error registering agent: {e}")


@cli.command("discover")
@click.option(
    "--filter-type",
    "-f",
    type=click.Choice(["capability", "agent_type", "status"]),
    help="Filter type for discovery",
)
@click.option("--filter-value", "-v", help="Filter value")
@click.pass_context
@async_command
async def discover_agents(ctx, filter_type, filter_value):
    """Discover A2A agents"""
    registry = ctx.obj["registry"]
    verbose = ctx.obj["verbose"]

    try:
        agents = await registry.discover_agents(filter_type, filter_value)

        if verbose:
            print(json.dumps(agents, indent=2))
        else:
            filter_desc = f" (filtered by {filter_type}={filter_value})" if filter_type else ""
            print(f"Discovered A2A Agents{filter_desc}:")

            if not agents:
                print("No agents found matching criteria")
                return

            for agent in agents:
                status_emoji = "🟢" if agent.get("status") == "active" else "🔴"
                print(f"\n{status_emoji} {agent['agent_id']}")
                print(f"  Type: {agent.get('agent_type', 'unknown')}")
                print(f"  Status: {agent.get('status', 'unknown')}")

                capabilities = agent.get("capabilities", [])
                if capabilities:
                    print(f"  Capabilities: {', '.join(capabilities)}")

                endpoint = agent.get("endpoint")
                if endpoint:
                    print(f"  Endpoint: {endpoint}")

                last_heartbeat = agent.get("last_heartbeat")
                if last_heartbeat:
                    print(f"  Last heartbeat: {last_heartbeat}")

    except Exception as e:
        print(f"Error discovering agents: {e}")


@cli.command("health")
@click.argument("agent_id", required=False)
@click.pass_context
@async_command
async def agent_health(ctx, agent_id):
    """Get agent health status"""
    registry = ctx.obj["registry"]
    verbose = ctx.obj["verbose"]

    try:
        health_data = await registry.get_agent_health(agent_id)

        if verbose:
            print(json.dumps(health_data, indent=2))
        else:
            if agent_id:
                # Single agent health
                if "error" in health_data:
                    print(f"❌ {health_data['error']}")
                    return

                health_score = health_data.get("health_score", 0)
                health_emoji = "🟢" if health_score > 80 else "🟡" if health_score > 50 else "🔴"

                print(f"Agent Health: {agent_id}")
                print(f"{health_emoji} Status: {health_data.get('status', 'unknown')}")
                print(f"Health Score: {health_score:.1f}/100")
                print(f"Last Heartbeat: {health_data.get('last_heartbeat', 'N/A')}")
                print(
                    f"Minutes Since Heartbeat: {health_data.get('minutes_since_heartbeat', 'N/A')}"
                )

                response_time = health_data.get("response_time_ms")
                if response_time:
                    print(f"Response Time: {response_time}ms")
            else:
                # All agents health
                print("A2A Agent Health Status:")
                for aid, health in health_data.items():
                    if isinstance(health, dict) and "error" not in health:
                        health_score = health.get("health_score", 0)
                        health_emoji = (
                            "🟢" if health_score > 80 else "🟡" if health_score > 50 else "🔴"
                        )
                        print(
                            f"  {health_emoji} {aid}: {health_score:.1f}/100 ({health.get('status', 'unknown')})"
                        )

    except Exception as e:
        print(f"Error getting health status: {e}")


@cli.command("unregister")
@click.argument("agent_id")
@click.pass_context
@async_command
async def unregister_agent(ctx, agent_id):
    """Unregister an A2A agent"""
    registry = ctx.obj["registry"]
    verbose = ctx.obj["verbose"]

    try:
        result = await registry.unregister_agent(agent_id)

        if verbose:
            print(json.dumps(result, indent=2))
        else:
            if result["status"] == "unregistered":
                print(f"✅ Agent {agent_id} unregistered successfully")
            elif result["status"] == "not_found":
                print(f"❌ Agent {agent_id} not found")
            else:
                print(f"Status: {result['status']}")

    except Exception as e:
        print(f"Error unregistering agent: {e}")


@cli.command("list")
@click.option(
    "--status-filter",
    type=click.Choice(["active", "inactive", "error"]),
    help="Filter by agent status",
)
@click.pass_context
@async_command
async def list_agents(ctx, status_filter):
    """List all registered A2A agents"""
    registry = ctx.obj["registry"]
    verbose = ctx.obj["verbose"]

    try:
        agents = (
            await registry.discover_agents("status", status_filter)
            if status_filter
            else await registry.discover_agents()
        )

        if verbose:
            print(json.dumps(agents, indent=2))
        else:
            filter_desc = f" ({status_filter} only)" if status_filter else ""
            print(f"Registered A2A Agents{filter_desc}:")

            if not agents:
                print("No agents registered")
                return

            # Group by type
            by_type = {}
            for agent in agents:
                agent_type = agent.get("agent_type", "unknown")
                if agent_type not in by_type:
                    by_type[agent_type] = []
                by_type[agent_type].append(agent)

            for agent_type, type_agents in by_type.items():
                print(f"\n📋 {agent_type.replace('_', ' ').title()} ({len(type_agents)}):")
                for agent in type_agents:
                    status_emoji = "🟢" if agent.get("status") == "active" else "🔴"
                    print(f"  {status_emoji} {agent['agent_id']}")

                    capabilities = agent.get("capabilities", [])
                    if capabilities:
                        print(
                            f"    Capabilities: {', '.join(capabilities[:3])}{'...' if len(capabilities) > 3 else ''}"
                        )

    except Exception as e:
        print(f"Error listing agents: {e}")


@cli.command("monitor")
@click.option("--interval", "-i", default=30, help="Monitoring interval in seconds")
@click.pass_context
@async_command
async def monitor_agents(ctx, interval):
    """Monitor A2A agent health in real-time"""
    registry = ctx.obj["registry"]

    print(f"🔄 Monitoring A2A agent health (interval: {interval}s)")
    print("Press Ctrl+C to stop monitoring\n")

    try:
        for i in range(10):  # Demo: 10 iterations
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Health check...")

            health_data = await registry.get_agent_health()

            active_count = 0
            inactive_count = 0

            for agent_id, health in health_data.items():
                if isinstance(health, dict) and "error" not in health:
                    status = health.get("status", "unknown")
                    health_score = health.get("health_score", 0)

                    if status == "active":
                        active_count += 1
                        if health_score < 70:
                            print(f"  ⚠️  {agent_id}: Health declining ({health_score:.1f}/100)")
                    else:
                        inactive_count += 1
                        print(f"  🔴 {agent_id}: Inactive")

            print(f"  📊 Active: {active_count}, Inactive: {inactive_count}")

            if i < 9:  # Don't sleep on last iteration
                await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
    except Exception as e:
        print(f"Error during monitoring: {e}")


if __name__ == "__main__":
    cli()
