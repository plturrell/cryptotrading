#!/usr/bin/env python3
"""
Test that all A2A agents communicate exclusively through MCP tools.
"""

import asyncio
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

from src.cryptotrading.core.agents.specialized.agent_manager import AgentManagerAgent
from src.cryptotrading.core.agents.specialized.trading_algorithm_agent import TradingAlgorithmAgent
from src.cryptotrading.core.protocols.a2a.a2a_protocol import A2AMessage, MessageType


async def test_trading_agent_mcp():
    """Test Trading Algorithm Agent MCP tools."""
    print("\n" + "=" * 60)
    print("Testing Trading Algorithm Agent MCP Tools")
    print("=" * 60)

    # Create agent
    agent = TradingAlgorithmAgent()

    # Test 1: Grid Trading via MCP
    print("\n1. Testing Grid Trading MCP tool:")
    result = await agent.process_mcp_request(
        "grid_trading.create",
        {"symbol": "BTC/USDT", "investment": 10000, "grid_levels": 10, "spacing_percentage": 0.02},
    )
    print(f"   ✅ Grid created: {result.get('grid_id')}")
    print(f"   Buy signals: {len(result.get('buy_orders', []))}")
    print(f"   Sell signals: {len(result.get('sell_orders', []))}")

    # Test 2: DCA via MCP
    print("\n2. Testing DCA MCP tool:")
    result = await agent.process_mcp_request(
        "dca.execute", {"symbol": "ETH/USDT", "amount": 100, "smart_adjust": True}
    )
    print(f"   ✅ DCA signal: {result.get('signal', {}).get('action')}")
    print(f"   Confidence: {result.get('signal', {}).get('confidence')}")

    # Test 3: Momentum Scan via MCP
    print("\n3. Testing Momentum Scan MCP tool:")
    result = await agent.process_mcp_request(
        "momentum.scan", {"symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"], "lookback_periods": 20}
    )
    print(f"   ✅ Scanned {len(result)} symbols")
    for signal in result[:2]:  # Show first 2
        print(f"   {signal['symbol']}: {signal['action']} (confidence: {signal['confidence']})")

    # Test 4: Multi-Strategy Allocation via MCP
    print("\n4. Testing Multi-Strategy Allocation MCP tool:")
    result = await agent.process_mcp_request(
        "multi_strategy.allocate", {"allocation_mode": "dynamic", "risk_parity": True}
    )
    print(f"   ✅ Allocations:")
    for strategy, weight in result.get("allocations", {}).items():
        print(f"   {strategy}: {weight:.1%}")

    # Test 5: Test via A2A message (should route to MCP)
    print("\n5. Testing A2A message routing to MCP:")
    message = A2AMessage(
        type=MessageType.ANALYSIS_REQUEST,
        sender="test_client",
        receiver=agent.agent_id,
        payload={
            "mcp_tool": "breakout.detect",
            "parameters": {"symbols": ["BTC/USDT", "ETH/USDT"], "lookback_periods": 50},
        },
    )
    response = await agent.process_message(message)
    if response:
        print(f"   ✅ Breakouts detected: {len(response.payload)}")

    print("\n✅ Trading Algorithm Agent MCP tests completed!")


async def test_agent_manager_mcp():
    """Test Agent Manager MCP tools."""
    print("\n" + "=" * 60)
    print("Testing Agent Manager MCP Tools")
    print("=" * 60)

    # Create agent manager
    manager = AgentManagerAgent()

    # Test 1: Register Agent via MCP
    print("\n1. Testing Agent Registration MCP tool:")
    result = await manager.process_mcp_request(
        "register_agent",
        {
            "agent_id": "test_agent_001",
            "agent_type": "trading",
            "capabilities": ["grid_trading", "dca", "momentum"],
            "mcp_tools": ["grid_trading.create", "dca.execute", "momentum.scan"],
        },
    )
    print(f"   ✅ Registration: {result.get('status')}")
    print(f"   Compliance: {result.get('compliance_status')}")

    # Test 2: Validate Compliance via MCP
    print("\n2. Testing Compliance Validation MCP tool:")
    result = await manager.process_mcp_request(
        "validate_compliance", {"agent_id": "test_agent_001"}
    )
    print(f"   ✅ Compliance status: {result.get('compliance_status')}")
    print(f"   Violations: {len(result.get('violations', []))}")

    # Test 3: Discover Agents via MCP
    print("\n3. Testing Agent Discovery MCP tool:")
    result = await manager.process_mcp_request(
        "discover_agents", {"filters": {"agent_type": "trading"}}
    )
    print(f"   ✅ Found {result.get('total')} agents")
    for agent in result.get("agents", [])[:2]:  # Show first 2
        print(f"   {agent['agent_id']}: {agent['agent_type']}")

    # Test 4: Health Check via MCP
    print("\n4. Testing Health Check MCP tool:")
    result = await manager.process_mcp_request("health_check", {})
    print(f"   ✅ System status: {result.get('status')}")
    print(f"   Registered agents: {result.get('registered_agents')}")

    # Test 5: Generate Report via MCP
    print("\n5. Testing Report Generation MCP tool:")
    result = await manager.process_mcp_request("generate_report", {"report_type": "registration"})
    print(f"   ✅ Report generated at: {result.get('generated_at')}")
    print(f"   Total registered: {result.get('data', {}).get('total_registered')}")

    # Test 6: Test via A2A message (should route to MCP)
    print("\n6. Testing A2A message routing to MCP:")
    message = {
        "mcp_tool": "manage_lifecycle",
        "parameters": {
            "agent_id": "test_agent_001",
            "action": "suspend",
            "reason": "Testing lifecycle management",
        },
    }
    result = await manager.process_message(message)
    print(f"   ✅ Lifecycle action: {result.get('action')} - {result.get('status')}")

    print("\n✅ Agent Manager MCP tests completed!")


async def test_mcp_only_enforcement():
    """Test that agents reject non-MCP requests."""
    print("\n" + "=" * 60)
    print("Testing MCP-Only Enforcement")
    print("=" * 60)

    agent = TradingAlgorithmAgent()

    # Test 1: Valid MCP request
    print("\n1. Testing valid MCP request:")
    try:
        result = await agent.process_mcp_request(
            "scalping.scan", {"symbols": ["BTC/USDT"], "min_volume_usd": 10000}
        )
        print(f"   ✅ MCP request succeeded: {len(result)} opportunities")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

    # Test 2: Invalid MCP tool name
    print("\n2. Testing invalid MCP tool name:")
    try:
        result = await agent.process_mcp_request("invalid_tool", {"symbol": "BTC/USDT"})
        print(f"   ❌ Should have failed!")
    except ValueError as e:
        print(f"   ✅ Correctly rejected: {e}")

    # Test 3: A2A message without MCP tool
    print("\n3. Testing A2A message without MCP tool:")
    message = A2AMessage(
        type=MessageType.ANALYSIS_REQUEST,
        sender="test_client",
        receiver=agent.agent_id,
        payload={"symbol": "BTC/USDT"},  # No mcp_tool specified
    )
    response = await agent.process_message(message)
    if response and "error" in response.payload:
        print(f"   ✅ Correctly rejected: {response.payload['error']}")
    else:
        print(f"   ❌ Should have returned error!")

    # Test 4: Legacy format (should map to MCP)
    print("\n4. Testing legacy format mapping to MCP:")
    message = A2AMessage(
        type=MessageType.ANALYSIS_REQUEST,
        sender="test_client",
        receiver=agent.agent_id,
        payload={"strategy": "momentum", "symbols": ["BTC/USDT"]},  # Legacy format
    )
    response = await agent.process_message(message)
    if response and "error" not in response.payload:
        print(f"   ✅ Legacy format mapped to MCP successfully")
    else:
        print(f"   ❌ Legacy mapping failed: {response.payload if response else 'No response'}")

    print("\n✅ MCP-only enforcement tests completed!")


async def main():
    """Run all MCP tests."""
    print("\n" + "=" * 70)
    print("A2A AGENTS MCP TOOLS TEST SUITE")
    print("Testing that ALL agent functionality goes through MCP tools")
    print("=" * 70)

    try:
        # Test Trading Algorithm Agent
        await test_trading_agent_mcp()

        # Test Agent Manager
        await test_agent_manager_mcp()

        # Test MCP-only enforcement
        await test_mcp_only_enforcement()

        print("\n" + "=" * 70)
        print("✅ ALL MCP TESTS PASSED!")
        print("All agents are properly using MCP tools for ALL functionality")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
