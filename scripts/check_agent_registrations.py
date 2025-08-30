#!/usr/bin/env python3
"""
Check current A2A agent registrations.
"""

import os
import sys
from pathlib import Path

# Setup environment
os.environ["ENVIRONMENT"] = "development"
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["DATABASE_USERNAME"] = "dev"
os.environ["DATABASE_PASSWORD"] = "dev"
os.environ["JWT_SECRET"] = "test"

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cryptotrading.core.protocols.a2a.a2a_protocol import A2A_CAPABILITIES, A2AAgentRegistry


def main():
    print("=" * 60)
    print("A2A Agent Registration Status")
    print("=" * 60)

    # Check what's defined in A2A_CAPABILITIES
    print("\n1. Agents defined in A2A_CAPABILITIES:")
    print("-" * 40)
    for agent_id in A2A_CAPABILITIES.keys():
        caps = A2A_CAPABILITIES[agent_id]
        print(f"  • {agent_id}")
        print(f"    Capabilities: {len(caps)} defined")
        print(f"    Sample: {', '.join(caps[:3])}...")

    # Check what's registered in A2AAgentRegistry
    print("\n2. Agents in A2AAgentRegistry:")
    print("-" * 40)
    all_agents = A2AAgentRegistry.get_all_agents()

    if all_agents:
        for agent_id, info in all_agents.items():
            status = info.get("status")
            if hasattr(status, "value"):
                status = status.value
            print(f"  • {agent_id}")
            print(f"    Status: {status}")
            print(f"    Registered: {info.get('registered_at', 'Unknown')}")
    else:
        print("  No agents registered in runtime registry")

    # Register agents that are defined but not registered
    print("\n3. Registering missing agents:")
    print("-" * 40)

    registered_ids = set(all_agents.keys())
    defined_ids = set(A2A_CAPABILITIES.keys())
    missing_ids = defined_ids - registered_ids

    if missing_ids:
        print(f"Found {len(missing_ids)} unregistered agents:")
        for agent_id in missing_ids:
            print(f"  Registering {agent_id}...")
            success = A2AAgentRegistry.register_agent(
                agent_id=agent_id, capabilities=A2A_CAPABILITIES[agent_id]
            )
            if success:
                print(f"    ✅ Success")
            else:
                print(f"    ❌ Failed")
    else:
        print("  All defined agents are already registered")

    # Final status
    print("\n4. Final Registration Status:")
    print("-" * 40)
    all_agents = A2AAgentRegistry.get_all_agents()
    print(f"Total registered agents: {len(all_agents)}")

    for agent_id in sorted(all_agents.keys()):
        info = all_agents[agent_id]
        status = info.get("status")
        if hasattr(status, "value"):
            status = status.value
        print(f"  ✅ {agent_id} - {status}")

    # Test capability search
    print("\n5. Capability Search Test:")
    print("-" * 40)

    test_caps = ["grid_trading", "monte_carlo_simulation", "agent_registration", "feature_storage"]

    for cap in test_caps:
        agents = A2AAgentRegistry.find_agents_by_capability(cap)
        if agents:
            print(f"  {cap}: {', '.join(agents)}")
        else:
            print(f"  {cap}: No agents found")

    print("\n" + "=" * 60)
    print("✅ Registration check complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
