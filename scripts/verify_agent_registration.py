#!/usr/bin/env python3
"""
Verify A2A Agent Registration Status

Checks that all agents are properly registered with the A2A network.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cryptotrading.core.agents.mcp_agent_lifecycle_manager import MCPAgentLifecycleManager
from src.cryptotrading.core.protocols.a2a.a2a_protocol import A2A_CAPABILITIES, A2AAgentRegistry


async def verify_agents():
    """Verify all agents are properly registered."""
    print("=" * 60)
    print("A2A Agent Registration Verification")
    print("=" * 60)

    # Initialize lifecycle manager
    manager = MCPAgentLifecycleManager()

    # Check configured agents
    print("\n1. Configured Agents in Lifecycle Manager:")
    print("-" * 40)
    for agent_id, config in manager.agent_configs.items():
        print(f"  ✓ {agent_id}")
        print(f"    Type: {config.agent_type.value}")
        print(f"    Auto-start: {config.auto_start}")
        print(f"    Dependencies: {config.dependencies}")

    # Check A2A capabilities
    print("\n2. A2A Protocol Capabilities:")
    print("-" * 40)
    for agent_id, capabilities in A2A_CAPABILITIES.items():
        print(f"  ✓ {agent_id}")
        print(f"    Capabilities: {len(capabilities)} registered")
        print(f"    Sample: {capabilities[:3]}...")

    # Check for mismatches
    print("\n3. Registration Status Check:")
    print("-" * 40)

    lifecycle_agents = set(manager.agent_configs.keys())
    protocol_agents = set(A2A_CAPABILITIES.keys())

    # Agents in lifecycle but not in protocol
    missing_in_protocol = lifecycle_agents - protocol_agents
    if missing_in_protocol:
        print("  ⚠️  Agents in Lifecycle Manager but NOT in A2A Protocol:")
        for agent in missing_in_protocol:
            print(f"    - {agent}")

    # Agents in protocol but not in lifecycle
    missing_in_lifecycle = protocol_agents - lifecycle_agents
    if missing_in_lifecycle:
        print("  ⚠️  Agents in A2A Protocol but NOT in Lifecycle Manager:")
        for agent in missing_in_lifecycle:
            print(f"    - {agent}")

    if not missing_in_protocol and not missing_in_lifecycle:
        print("  ✅ All agents properly registered in both systems!")

    # Test registration
    print("\n4. Testing Dynamic Registration:")
    print("-" * 40)

    # Register trading algorithm agent
    success = A2AAgentRegistry.register_agent(
        agent_id="trading_algorithm_agent",
        capabilities=A2A_CAPABILITIES.get("trading_algorithm_agent", []),
    )
    print(f"  Trading Algorithm Agent registration: {'✅ Success' if success else '❌ Failed'}")

    # Register data analysis agent
    success = A2AAgentRegistry.register_agent(
        agent_id="data_analysis_agent", capabilities=A2A_CAPABILITIES.get("data_analysis_agent", [])
    )
    print(f"  Data Analysis Agent registration: {'✅ Success' if success else '❌ Failed'}")

    # Register feature store agent
    success = A2AAgentRegistry.register_agent(
        agent_id="feature_store_agent", capabilities=A2A_CAPABILITIES.get("feature_store_agent", [])
    )
    print(f"  Feature Store Agent registration: {'✅ Success' if success else '❌ Failed'}")

    # Show all registered agents
    print("\n5. Currently Registered Agents:")
    print("-" * 40)
    all_agents = A2AAgentRegistry.get_all_agents()
    for agent_id, info in all_agents.items():
        print(f"  ✓ {agent_id}")
        print(
            f"    Status: {info['status'].value if hasattr(info['status'], 'value') else info['status']}"
        )
        print(f"    Capabilities: {len(info['capabilities'])}")

    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_agents())
