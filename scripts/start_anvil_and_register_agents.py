#!/usr/bin/env python3
"""
Start local Anvil blockchain and register all A2A agents.

This script:
1. Starts a local Anvil instance for A2A messaging
2. Deploys A2A smart contracts
3. Registers all agents using the Agent Manager CLI
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.cryptotrading.core.agents.cli.agent_manager_cli import AgentManagerCLI
    from src.cryptotrading.core.blockchain.anvil_client import AnvilA2AClient
except ImportError as e:
    print(f"Warning: Could not import agent modules: {e}")
    print("Agent registration will be skipped")
    AnvilA2AClient = None
    AgentManagerCLI = None

# Agents to register with their capabilities and MCP tools
AGENTS_TO_REGISTER = [
    {
        "agent_id": "mcts_calculation_agent",
        "agent_type": "mcts_calculation",
        "capabilities": [
            "monte_carlo_simulation",
            "strategy_optimization",
            "risk_assessment",
            "portfolio_optimization",
            "backtesting",
            "performance_analysis",
        ],
        "mcp_tools": ["mcts_calculate", "optimize_strategy", "analyze_risk"],
    },
    {
        "agent_id": "technical_analysis_agent",
        "agent_type": "technical_analysis",
        "capabilities": [
            "momentum_analysis",
            "volume_analysis",
            "pattern_recognition",
            "indicator_calculation",
            "trend_analysis",
            "support_resistance",
        ],
        "mcp_tools": ["calculate_indicators", "detect_patterns", "analyze_trends"],
    },
    {
        "agent_id": "ml_agent",
        "agent_type": "ml_agent",
        "capabilities": [
            "model_training",
            "prediction",
            "feature_engineering",
            "model_evaluation",
            "hyperparameter_tuning",
        ],
        "mcp_tools": ["train_model", "predict", "evaluate_model"],
    },
    {
        "agent_id": "trading_algorithm_agent",
        "agent_type": "trading_algorithm",
        "capabilities": [
            "grid_trading",
            "dollar_cost_averaging",
            "arbitrage_detection",
            "momentum_trading",
            "mean_reversion",
            "signal_generation",
        ],
        "mcp_tools": ["grid_create", "dca_execute", "arbitrage_scan", "risk_calculate"],
    },
    {
        "agent_id": "data_analysis_agent",
        "agent_type": "data_analysis",
        "capabilities": [
            "data_processing",
            "statistical_analysis",
            "pattern_recognition",
            "anomaly_detection",
            "correlation_analysis",
        ],
        "mcp_tools": ["analyze_data", "detect_anomalies", "generate_report"],
    },
    {
        "agent_id": "feature_store_agent",
        "agent_type": "feature_store",
        "capabilities": [
            "feature_storage",
            "feature_retrieval",
            "feature_versioning",
            "feature_validation",
            "metadata_management",
        ],
        "mcp_tools": ["store_feature", "retrieve_feature", "validate_feature"],
    },
    {
        "agent_id": "aws_data_exchange_agent",
        "agent_type": "data_exchange",
        "capabilities": [
            "data_import",
            "data_export",
            "data_validation",
            "schema_management",
            "data_transformation",
        ],
        "mcp_tools": ["import_data", "export_data", "validate_schema"],
    },
    {
        "agent_id": "strands_glean_agent",
        "agent_type": "glean_agent",
        "capabilities": [
            "code_analysis",
            "dependency_tracking",
            "impact_analysis",
            "code_search",
            "documentation_generation",
        ],
        "mcp_tools": ["analyze_code", "track_dependencies", "search_codebase"],
    },
]


def start_anvil():
    """Start local Anvil blockchain."""
    print("Starting Anvil blockchain...")

    # Check if anvil is installed
    try:
        subprocess.run(["anvil", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Anvil not found. Please install Foundry:")
        print("   curl -L https://foundry.paradigm.xyz | bash")
        print("   foundryup")
        return None

    # Start Anvil in background
    anvil_process = subprocess.Popen(
        ["anvil", "--port", "8545", "--accounts", "10", "--balance", "10000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for Anvil to start
    time.sleep(3)

    # Check if Anvil started successfully
    if anvil_process.poll() is not None:
        print("❌ Failed to start Anvil")
        return None

    print("✅ Anvil started on http://localhost:8545")
    return anvil_process


async def initialize_blockchain():
    """Initialize blockchain with A2A contracts."""
    print("\nInitializing A2A blockchain...")

    # Create Anvil client
    client = AnvilA2AClient()

    # Initialize (deploys contracts)
    success = await client.initialize()

    if success:
        print("✅ A2A contracts deployed")
        print(f"   Registry: {client.registry_contract_address}")
        print(f"   Messages: {client.message_contract_address}")
    else:
        print("❌ Failed to initialize blockchain")
        return None

    return client


async def register_agent_on_blockchain(
    client: AnvilA2AClient, agent_id: str, capabilities: List[str]
):
    """Register an agent on the blockchain."""
    try:
        # Register agent
        tx_hash = await client.register_agent(
            agent_id=agent_id,
            capabilities=capabilities,
            endpoint=f"http://localhost:8000/agents/{agent_id}",
        )

        if tx_hash:
            print(f"   Blockchain TX: {tx_hash}")
            return True
        else:
            print("   ❌ Blockchain registration failed")
            return False

    except (ConnectionError, ValueError, RuntimeError) as e:
        print(f"   ❌ Blockchain error: {e}")
        return False


async def register_agents():
    """Register all agents using Agent Manager CLI."""
    print("\n" + "=" * 60)
    print("Registering A2A Agents")
    print("=" * 60)

    # Initialize Agent Manager CLI
    manager = AgentManagerCLI()
    await manager.initialize()

    # Initialize blockchain client
    blockchain_client = await initialize_blockchain()

    # Register each agent
    for agent_config in AGENTS_TO_REGISTER:
        print(f"\n📝 Registering {agent_config['agent_id']}...")
        print("-" * 40)

        try:
            # Register with Agent Manager (includes A2A compliance)
            result = await manager.register_agent(
                agent_id=agent_config["agent_id"],
                agent_type=agent_config["agent_type"],
                capabilities=agent_config["capabilities"],
                mcp_tools=agent_config["mcp_tools"],
                blockchain=False,  # We'll do blockchain separately
            )

            if result.get("status") == "success" or result.get("success"):
                print(f"   ✅ A2A Registration successful")
                print(f"   Compliance Score: {result.get('compliance_score', 'N/A')}")

                # Register on blockchain if client available
                if blockchain_client:
                    await register_agent_on_blockchain(
                        blockchain_client, agent_config["agent_id"], agent_config["capabilities"]
                    )
            else:
                print("   ❌ Registration failed: {}".format(result.get("error", "Unknown error")))

        except (ConnectionError, ValueError, RuntimeError) as e:
            print(f"   ❌ Error: {e}")

    # Query and display all registered agents
    print("\n" + "=" * 60)
    print("Registered Agents Summary")
    print("=" * 60)

    agents = await manager.query_agents()
    print(f"\nTotal registered agents: {len(agents)}")

    for agent in agents:
        print(f"  • {agent['agent_id']}")
        print(f"    Status: {agent.get('status', 'unknown')}")
        print(f"    Capabilities: {len(agent.get('capabilities', []))}")

    # Audit compliance
    print("\n" + "=" * 60)
    print("Compliance Audit")
    print("=" * 60)

    audit_result = await manager.audit_compliance()
    print(f"Compliant agents: {len(audit_result.get('compliant_agents', []))}")
    print(f"Non-compliant agents: {len(audit_result.get('non_compliant_agents', []))}")
    print(f"Overall compliance score: {audit_result.get('total_score', 0):.2f}")

    # Check health
    print("\n" + "=" * 60)
    print("Health Check")
    print("=" * 60)

    health = await manager.monitor_health()
    print(f"Healthy agents: {len(health.get('healthy_agents', []))}")
    print(f"Unhealthy agents: {len(health.get('unhealthy_agents', []))}")

    if health.get("unhealthy_agents"):
        print("\nUnhealthy agents:")
        for agent_id in health["unhealthy_agents"]:
            print(f"  ⚠️  {agent_id}")


async def main():
    """Main function."""
    print("=" * 60)
    print("A2A Agent Registration System")
    print("=" * 60)

    # Check if we should skip Anvil start (already running)
    skip_anvil = os.getenv("SKIP_ANVIL_START", "").lower() == "true"
    anvil_process = None

    if not skip_anvil:
        # Start Anvil
        anvil_process = start_anvil()

        if not anvil_process:
            print("\n⚠️  Running without blockchain (Anvil not available)")
            print("    Agents will be registered in A2A system only\n")
    else:
        print("⚠️  Skipping Anvil start (using existing instance)")

    try:
        # Register agents
        await register_agents()

        print("\n" + "=" * 60)
        print("✅ Registration Complete!")
        print("=" * 60)

        if anvil_process:
            print("\n⚠️  Anvil is still running. Press Ctrl+C to stop.")
            # Keep running
            while True:
                await asyncio.sleep(1)
        else:
            # Exit immediately if we didn't start Anvil
            print("✅ Agent registration completed")

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        if anvil_process:
            print("Stopping Anvil...")
            anvil_process.terminate()
            anvil_process.wait()
            print("✅ Anvil stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted by user")
