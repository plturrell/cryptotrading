#!/usr/bin/env python3
"""
Simple test script for MCTS Production Agent
Tests basic functionality without complex dependencies
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_config():
    """Test configuration loading"""
    print("🧪 Testing Configuration...")
    try:
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import MCTSConfig
        
        config = MCTSConfig()
        print(f"✅ Config loaded - iterations: {config.iterations}")
        print(f"✅ Exploration constant: {config.exploration_constant}")
        print(f"✅ Timeout: {config.timeout_seconds}s")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_input_validation():
    """Test input validation"""
    print("\n🧪 Testing Input Validation...")
    try:
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import InputValidator, ValidationError
        
        # Valid input
        valid_params = {
            'initial_portfolio': 10000,
            'symbols': ['BTC', 'ETH'],
            'max_depth': 5
        }
        result = InputValidator.validate_calculation_params(valid_params)
        print("✅ Valid input passed")
        
        # Invalid input
        try:
            InputValidator.validate_calculation_params({'initial_portfolio': -1000})
            print("❌ Should have failed validation")
            return False
        except ValidationError:
            print("✅ Invalid input correctly rejected")
        
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_circuit_breaker():
    """Test circuit breaker"""
    print("\n🧪 Testing Circuit Breaker...")
    try:
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=2, timeout=1)
        
        # Test normal operation
        assert cb.can_execute() == True
        print("✅ Initial state: closed")
        
        # Test failure handling
        cb.record_failure()
        cb.record_failure()
        assert cb.state == 'open'
        assert cb.can_execute() == False
        print("✅ Circuit opened after failures")
        
        # Test recovery
        cb.record_success()
        assert cb.state == 'closed'
        print("✅ Circuit closed after success")
        
        return True
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
        return False

async def test_trading_environment():
    """Test trading environment"""
    print("\n🧪 Testing Trading Environment...")
    try:
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import ProductionTradingEnvironment
        
        config = {
            'initial_portfolio': 10000,
            'symbols': ['BTC', 'ETH'],
            'max_depth': 3
        }
        
        env = ProductionTradingEnvironment(config)
        
        # Test initial state
        state = await env.get_initial_state()
        assert state['portfolio_value'] == 10000
        assert 'market_data' in state
        print("✅ Initial state created")
        
        # Test actions
        actions = await env.get_available_actions(state)
        assert len(actions) > 0
        print(f"✅ {len(actions)} actions available")
        
        # Test action application
        buy_action = next(a for a in actions if a['type'] == 'buy')
        new_state = await env.apply_action(state, buy_action)
        assert new_state['depth'] == 1
        print("✅ Action applied successfully")
        
        # Test evaluation
        value = await env.evaluate_state(new_state)
        print(f"✅ State evaluated: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Trading environment test failed: {e}")
        return False

async def test_mcts_node():
    """Test MCTS node functionality"""
    print("\n🧪 Testing MCTS Node...")
    try:
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import MCTSNodeV2
        
        # Create root node
        root = MCTSNodeV2(
            state={'test': 'state'},
            untried_actions=[{'action': 'test1'}, {'action': 'test2'}]
        )
        
        # Test expansion
        assert not root.is_fully_expanded()
        child = root.add_child({'action': 'test1'}, {'new': 'state'})
        assert len(root.children) == 1
        assert len(root.untried_actions) == 1
        print("✅ Node expansion works")
        
        # Test RAVE
        root.update_rave([{'action': 'test1'}], 0.5)
        action_key = str({'action': 'test1'})
        assert root.rave_visits[action_key] == 1
        print("✅ RAVE updates work")
        
        return True
    except Exception as e:
        print(f"❌ MCTS node test failed: {e}")
        return False

async def test_performance_monitoring():
    """Test performance monitoring"""
    print("\n🧪 Testing Performance Monitoring...")
    try:
        from src.cryptotrading.core.agents.specialized.mcts_monitoring import MCTSMonitor, PerformanceMetrics
        
        monitor = MCTSMonitor("test_agent")
        
        # Create test metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            calculation_id="test_001",
            iterations=100,
            execution_time=1.5,
            tree_size=50,
            memory_usage_mb=100.0,
            best_action_confidence=0.85,
            expected_value=0.12
        )
        
        # Record metrics
        await monitor.record_calculation(metrics)
        assert monitor.calculation_count == 1
        print("✅ Metrics recorded")
        
        # Test health status
        health = await monitor.get_health_status('closed', 100, True)
        assert health.status in ['healthy', 'degraded', 'unhealthy']
        print(f"✅ Health status: {health.status}")
        
        # Test performance summary
        summary = await monitor.get_performance_summary(hours=1)
        assert summary['calculation_count'] == 1
        print("✅ Performance summary generated")
        
        return True
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting MCTS Production Agent Tests\n")
    
    tests = [
        ("Configuration", test_config),
        ("Input Validation", test_input_validation),
        ("Circuit Breaker", test_circuit_breaker),
        ("Trading Environment", test_trading_environment),
        ("MCTS Node", test_mcts_node),
        ("Performance Monitoring", test_performance_monitoring)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {name} test PASSED")
            else:
                print(f"❌ {name} test FAILED")
        except Exception as e:
            print(f"❌ {name} test ERROR: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Agent is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Review errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)