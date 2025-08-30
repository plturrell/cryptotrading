#!/usr/bin/env python3
"""
CLI for A2A Communication Protocol - Agent-to-agent messaging and coordination
Provides command-line access to A2A message sending, workflow coordination, and protocol management
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
    from cryptotrading.core.agents.real_a2a_communication import (
        A2ACommunicationManager,
        A2AEndpoint,
    )
    from cryptotrading.core.protocols.a2a.a2a_message_types_v2 import A2AMessage
    from cryptotrading.core.protocols.a2a.a2a_protocol_v2 import EnhancedMessageType
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback minimal communication manager for CLI testing...")

    class FallbackA2ACommunicationManager:
        """Minimal communication manager for CLI testing when imports fail"""

        def __init__(self):
            self.active_connections = {
                "data-loader-agent": {"endpoint": "http://localhost:8001", "status": "connected"},
                "ml-training-agent": {"endpoint": "http://localhost:8002", "status": "connected"},
                "inference-agent": {"endpoint": "http://localhost:8003", "status": "disconnected"},
            }
            self.message_history = []

        async def send_message(
            self, recipient_id: str, message_type: str, payload: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Mock message sending"""
            message = {
                "message_id": f"msg_{int(datetime.now().timestamp())}",
                "sender_id": "cli-client",
                "recipient_id": recipient_id,
                "message_type": message_type,
                "payload": payload,
                "timestamp": datetime.now().isoformat(),
                "status": "delivered" if recipient_id in self.active_connections else "failed",
            }

            self.message_history.append(message)

            return {
                "status": message["status"],
                "message_id": message["message_id"],
                "delivery_time_ms": 45.2 if message["status"] == "delivered" else None,
                "error": None
                if message["status"] == "delivered"
                else f"Agent {recipient_id} not reachable",
            }

        async def broadcast_message(
            self, message_type: str, payload: Dict[str, Any], agent_filter: str = None
        ) -> Dict[str, Any]:
            """Mock message broadcasting"""
            recipients = list(self.active_connections.keys())
            if agent_filter:
                recipients = [r for r in recipients if agent_filter in r]

            results = {}
            for recipient in recipients:
                result = await self.send_message(recipient, message_type, payload)
                results[recipient] = result

            delivered = sum(1 for r in results.values() if r["status"] == "delivered")
            failed = len(results) - delivered

            return {
                "broadcast_id": f"broadcast_{int(datetime.now().timestamp())}",
                "recipients": len(recipients),
                "delivered": delivered,
                "failed": failed,
                "results": results,
            }

        async def start_workflow(
            self, workflow_type: str, participants: List[str], config: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Mock workflow coordination"""
            workflow_id = f"workflow_{int(datetime.now().timestamp())}"

            # Send workflow start messages to all participants
            for participant in participants:
                await self.send_message(
                    participant,
                    "workflow_start",
                    {
                        "workflow_id": workflow_id,
                        "workflow_type": workflow_type,
                        "config": config,
                        "role": "participant",
                    },
                )

            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "participants": participants,
                "status": "initiated",
                "estimated_duration": "5-10 minutes",
            }

        async def get_connection_status(self) -> Dict[str, Any]:
            """Get A2A connection status"""
            connected = sum(
                1 for c in self.active_connections.values() if c["status"] == "connected"
            )
            total = len(self.active_connections)

            return {
                "total_agents": total,
                "connected": connected,
                "disconnected": total - connected,
                "connections": self.active_connections,
                "last_updated": datetime.now().isoformat(),
            }

        async def get_message_history(self, limit: int = 10) -> List[Dict[str, Any]]:
            """Get recent message history"""
            return self.message_history[-limit:]


def async_command(f):
    """Decorator to run async functions in click commands"""

    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.group()
@click.option("--sender-id", default="cli-client", help="Sender agent ID")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, sender_id, verbose):
    """A2A Communication CLI - Agent-to-agent messaging and coordination"""
    ctx.ensure_object(dict)

    # Set environment for CLI mode
    os.environ["ENVIRONMENT"] = "development"
    os.environ["SKIP_DB_INIT"] = "true"

    # Initialize A2A communication manager
    try:
        comm_manager = A2ACommunicationManager(agent_id=sender_id)
    except:
        if verbose:
            print("Using fallback communication manager due to import/initialization issues")
        comm_manager = FallbackA2ACommunicationManager()

    ctx.obj["comm_manager"] = comm_manager
    ctx.obj["sender_id"] = sender_id
    ctx.obj["verbose"] = verbose


@cli.command("send")
@click.argument("recipient_id")
@click.argument("message_type")
@click.option("--payload", "-p", help="Message payload as JSON string")
@click.pass_context
@async_command
async def send_message(ctx, recipient_id, message_type, payload):
    """Send A2A message to specific agent"""
    comm_manager = ctx.obj["comm_manager"]
    verbose = ctx.obj["verbose"]

    try:
        payload_dict = json.loads(payload) if payload else {}

        print(f"📤 Sending {message_type} message to {recipient_id}...")
        result = await comm_manager.send_message(recipient_id, message_type, payload_dict)

        if verbose:
            print(json.dumps(result, indent=2))
        else:
            if result["status"] == "delivered":
                print(f"✅ Message delivered successfully")
                print(f"Message ID: {result['message_id']}")
                print(f"Delivery time: {result.get('delivery_time_ms', 'N/A')}ms")
            else:
                print(f"❌ Message delivery failed")
                print(f"Error: {result.get('error', 'Unknown error')}")

    except json.JSONDecodeError:
        print("Error: Invalid JSON in payload")
    except Exception as e:
        print(f"Error sending message: {e}")


@cli.command("broadcast")
@click.argument("message_type")
@click.option("--payload", "-p", help="Message payload as JSON string")
@click.option("--filter", "-f", help="Agent filter (substring match)")
@click.pass_context
@async_command
async def broadcast_message(ctx, message_type, payload, filter):
    """Broadcast A2A message to multiple agents"""
    comm_manager = ctx.obj["comm_manager"]
    verbose = ctx.obj["verbose"]

    try:
        payload_dict = json.loads(payload) if payload else {}

        filter_desc = f" (filtered by '{filter}')" if filter else ""
        print(f"📡 Broadcasting {message_type} message{filter_desc}...")

        result = await comm_manager.broadcast_message(message_type, payload_dict, filter)

        if verbose:
            print(json.dumps(result, indent=2))
        else:
            print(f"✅ Broadcast completed:")
            print(f"Broadcast ID: {result['broadcast_id']}")
            print(f"Recipients: {result['recipients']}")
            print(f"Delivered: {result['delivered']}")
            print(f"Failed: {result['failed']}")

            if result["failed"] > 0:
                print("\nFailed deliveries:")
                for recipient, res in result["results"].items():
                    if res["status"] != "delivered":
                        print(f"  ❌ {recipient}: {res.get('error', 'Unknown error')}")

    except json.JSONDecodeError:
        print("Error: Invalid JSON in payload")
    except Exception as e:
        print(f"Error broadcasting message: {e}")


@cli.command("workflow")
@click.argument("workflow_type")
@click.argument("participants", nargs=-1, required=True)
@click.option("--config", "-c", help="Workflow configuration as JSON string")
@click.pass_context
@async_command
async def start_workflow(ctx, workflow_type, participants, config):
    """Start A2A workflow coordination"""
    comm_manager = ctx.obj["comm_manager"]
    verbose = ctx.obj["verbose"]

    try:
        config_dict = json.loads(config) if config else {}

        print(f"🔄 Starting {workflow_type} workflow...")
        print(f"Participants: {', '.join(participants)}")

        result = await comm_manager.start_workflow(workflow_type, list(participants), config_dict)

        if verbose:
            print(json.dumps(result, indent=2))
        else:
            print(f"✅ Workflow initiated:")
            print(f"Workflow ID: {result['workflow_id']}")
            print(f"Type: {result['workflow_type']}")
            print(f"Participants: {len(result['participants'])}")
            print(f"Status: {result['status']}")
            print(f"Estimated duration: {result.get('estimated_duration', 'Unknown')}")

    except json.JSONDecodeError:
        print("Error: Invalid JSON in config")
    except Exception as e:
        print(f"Error starting workflow: {e}")


@cli.command("status")
@click.pass_context
@async_command
async def connection_status(ctx):
    """Get A2A connection status"""
    comm_manager = ctx.obj["comm_manager"]
    verbose = ctx.obj["verbose"]

    try:
        status = await comm_manager.get_connection_status()

        if verbose:
            print(json.dumps(status, indent=2))
        else:
            print("A2A Connection Status:")
            print(f"Total agents: {status['total_agents']}")
            print(f"Connected: {status['connected']} 🟢")
            print(f"Disconnected: {status['disconnected']} 🔴")
            print(f"Last updated: {status['last_updated']}")

            connections = status.get("connections", {})
            if connections:
                print("\nAgent Connections:")
                for agent_id, conn_info in connections.items():
                    status_emoji = "🟢" if conn_info["status"] == "connected" else "🔴"
                    print(f"  {status_emoji} {agent_id}: {conn_info['status']}")
                    print(f"    Endpoint: {conn_info.get('endpoint', 'N/A')}")

    except Exception as e:
        print(f"Error getting connection status: {e}")


@cli.command("history")
@click.option("--limit", "-l", default=10, help="Number of recent messages to show")
@click.pass_context
@async_command
async def message_history(ctx, limit):
    """Get recent A2A message history"""
    comm_manager = ctx.obj["comm_manager"]
    verbose = ctx.obj["verbose"]

    try:
        history = await comm_manager.get_message_history(limit)

        if verbose:
            print(json.dumps(history, indent=2))
        else:
            print(f"Recent A2A Messages (last {len(history)}):")

            if not history:
                print("No messages in history")
                return

            for msg in reversed(history):  # Show most recent first
                status_emoji = "✅" if msg["status"] == "delivered" else "❌"
                print(f"\n{status_emoji} {msg['message_id']}")
                print(f"  From: {msg['sender_id']} → To: {msg['recipient_id']}")
                print(f"  Type: {msg['message_type']}")
                print(f"  Time: {msg['timestamp']}")
                print(f"  Status: {msg['status']}")

                if msg.get("payload"):
                    payload_preview = str(msg["payload"])[:100]
                    if len(str(msg["payload"])) > 100:
                        payload_preview += "..."
                    print(f"  Payload: {payload_preview}")

    except Exception as e:
        print(f"Error getting message history: {e}")


@cli.command("ping")
@click.argument("agent_id")
@click.pass_context
@async_command
async def ping_agent(ctx, agent_id):
    """Ping specific A2A agent"""
    comm_manager = ctx.obj["comm_manager"]

    try:
        print(f"🏓 Pinging {agent_id}...")

        start_time = datetime.now()
        result = await comm_manager.send_message(
            agent_id, "ping", {"timestamp": start_time.isoformat()}
        )
        end_time = datetime.now()

        if result["status"] == "delivered":
            response_time = (end_time - start_time).total_seconds() * 1000
            print(f"✅ Pong from {agent_id}")
            print(f"Response time: {response_time:.1f}ms")
        else:
            print(f"❌ Ping failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Error pinging agent: {e}")


@cli.command("data-request")
@click.argument("target_agent")
@click.option("--symbols", "-s", multiple=True, help="Data symbols to request")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.pass_context
@async_command
async def request_data(ctx, target_agent, symbols, start_date, end_date):
    """Request data from A2A data loader agent (convenience command)"""
    comm_manager = ctx.obj["comm_manager"]

    try:
        payload = {
            "request_type": "historical_data",
            "symbols": list(symbols) if symbols else ["BTC", "ETH"],
            "start_date": start_date or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
            "sources": ["yahoo", "fred"],
        }

        print(f"📊 Requesting data from {target_agent}...")
        print(f"Symbols: {', '.join(payload['symbols'])}")
        print(f"Date range: {payload['start_date']} to {payload['end_date']}")

        result = await comm_manager.send_message(target_agent, "data_request", payload)

        if result["status"] == "delivered":
            print(f"✅ Data request sent successfully")
            print(f"Message ID: {result['message_id']}")
        else:
            print(f"❌ Data request failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Error requesting data: {e}")


@cli.command("ml-job")
@click.argument("target_agent")
@click.option("--model-type", default="random_forest", help="ML model type")
@click.option("--dataset", help="Dataset path or identifier")
@click.pass_context
@async_command
async def submit_ml_job(ctx, target_agent, model_type, dataset):
    """Submit ML training job to A2A ML agent (convenience command)"""
    comm_manager = ctx.obj["comm_manager"]

    try:
        payload = {
            "job_type": "training",
            "model_config": {
                "model_type": model_type,
                "hyperparameters": {"n_estimators": 100, "max_depth": 10},
            },
            "dataset": dataset or "default_crypto_dataset",
            "priority": "normal",
        }

        print(f"🤖 Submitting ML job to {target_agent}...")
        print(f"Model type: {model_type}")
        print(f"Dataset: {payload['dataset']}")

        result = await comm_manager.send_message(target_agent, "ml_training_job_request", payload)

        if result["status"] == "delivered":
            print(f"✅ ML job submitted successfully")
            print(f"Message ID: {result['message_id']}")
        else:
            print(f"❌ ML job submission failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Error submitting ML job: {e}")


if __name__ == "__main__":
    cli()
