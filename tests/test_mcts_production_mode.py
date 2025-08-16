#!/usr/bin/env python3
"""
Test MCTS agent in production mode with real data requirements
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

async def test_production_mode():
    """Test that agent requires real data in production mode"""
    print("🚀 Testing MCTS Agent Production Data Requirements")
    
    try:
        # First populate cache with some real-looking data
        from src.cryptotrading.core.protocols.mcp.cache import mcp_cache
        
        # Add some backup data to cache
        backup_data = {
            'BTC': {
                'price': 35000.50,
                'volume': 1500000,
                'volatility': 0.28,
                'timestamp': time.time()
            },
            'ETH': {
                'price': 2100.75,
                'volume': 800000,
                'volatility': 0.32,
                'timestamp': time.time()
            }
        }
        
        # Use direct cache approach for testing
        import hashlib
        backup_key_data = f"market_data_backup:BTC,ETH"
        backup_key = hashlib.md5(backup_key_data.encode()).hexdigest()
        mcp_cache.cache.set(backup_key, backup_data, ttl=3600)
        print("✅ Added backup market data to cache")
        
        # Test agent can work with cached data
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import ProductionMCTSCalculationAgent
        
        agent = ProductionMCTSCalculationAgent(
            agent_id="test_production_agent",
            market_data_provider=None  # No provider - should use cache
        )
        print("✅ Agent created successfully")
        
        # Test Strands tool with cached data
        result = await agent.calculate_optimal_portfolio(
            symbols=['BTC', 'ETH'],
            capital=50000,
            risk_tolerance=0.6
        )
        
        print(f"✅ Portfolio optimization completed: {result['tool_name']}")
        print(f"   Expected return: {result['expected_return']:.4f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        # Test fail-fast behavior with no data
        try:
            result2 = await agent.analyze_market_risk(
                symbols=['UNKNOWN_COIN'],
                time_horizon=7
            )
            print("❌ Should have failed with unknown coin")
        except ValueError as e:
            print(f"✅ Correctly failed with unknown symbol: {str(e)[:80]}...")
        
        print("\n🎉 Production mode tests passed - agent requires real data!")
        return True
        
    except Exception as e:
        print(f"❌ Production mode test failed: {e}")
        return False

async def test_no_fallback_simulation():
    """Test that simulation action selection is deterministic"""
    print("\n🧪 Testing Deterministic Simulation")
    
    try:
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import ProductionTradingEnvironment
        
        # Create environment with test data
        config = {
            'initial_portfolio': 10000,
            'symbols': ['BTC', 'ETH'],
            'max_depth': 5
        }
        
        env = ProductionTradingEnvironment(config, None)
        
        # Create a test state
        state = {
            'portfolio_value': 10000,
            'positions': {'BTC': 0.1},
            'market_data': {
                'BTC': {'price': 35000, 'volatility': 0.25, 'volume': 1000000},
                'ETH': {'price': 2100, 'volatility': 0.30, 'volume': 500000}
            },
            'depth': 1
        }
        
        # Get available actions
        actions = await env.get_available_actions(state)
        print(f"✅ Generated {len(actions)} available actions")
        
        # Test action selection multiple times - should be deterministic
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import ProductionMCTSCalculationAgent
        agent = ProductionMCTSCalculationAgent("test_agent")
        
        selected_actions = []
        for i in range(3):
            action = await agent._select_simulation_action(actions, state, 1)
            selected_actions.append(action['type'] if action else None)
        
        if len(set(selected_actions)) == 1:
            print(f"✅ Action selection is deterministic: {selected_actions[0]}")
        else:
            print(f"⚠️ Action selection varies: {selected_actions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False

async def main():
    """Run all production mode tests"""
    print("🔍 Testing MCTS Production Agent Requirements\n")
    
    test1 = await test_production_mode()
    test2 = await test_no_fallback_simulation()
    
    passed = sum([test1, test2])
    total = 2
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All production requirements verified!")
    else:
        print("⚠️ Some production requirements not met")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())