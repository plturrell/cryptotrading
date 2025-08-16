"""
Tests for Production MCTS Calculation Agent
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import (
    ProductionMCTSCalculationAgent, 
    ProductionTradingEnvironment,
    MCTSConfig,
    InputValidator,
    ValidationError,
    CircuitBreaker
)
from src.cryptotrading.core.agents.specialized.mcts_monitoring import (
    MCTSMonitor,
    PerformanceMetrics,
    HealthStatus
)
from src.cryptotrading.core.protocols.mcp.tools import ToolResult


class TestProductionMCTSAgent:
    """Test suite for Production MCTS Agent"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = MCTSConfig()
        config.iterations = 100
        config.timeout_seconds = 10
        config.parallel_simulations = 2
        return config
    
    @pytest.fixture
    def agent(self, config):
        """Create test agent instance"""
        return ProductionMCTSCalculationAgent(
            agent_id="test_prod_mcts_001",
            config=config
        )
    
    @pytest.fixture
    def trading_environment(self):
        """Create test trading environment"""
        config = {
            'initial_portfolio': 10000,
            'symbols': ['BTC', 'ETH'],
            'max_depth': 3
        }
        return ProductionTradingEnvironment(config)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent, config):
        """Test agent initializes correctly"""
        assert agent.agent_id == "test_prod_mcts_001"
        assert agent.agent_type == "mcts_calculation_v2"
        assert agent.config == config
        assert "monte_carlo_search" in agent.capabilities
        assert isinstance(agent.circuit_breaker, CircuitBreaker)
    
    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input validation"""
        # Valid parameters
        valid_params = {
            'initial_portfolio': 10000,
            'symbols': ['BTC', 'ETH'],
            'max_depth': 5
        }
        result = InputValidator.validate_calculation_params(valid_params)
        assert result == valid_params
        
        # Invalid portfolio
        with pytest.raises(ValidationError, match="positive number"):
            InputValidator.validate_calculation_params({'initial_portfolio': -1000})
        
        # Invalid symbols
        with pytest.raises(ValidationError, match="non-empty list"):
            InputValidator.validate_calculation_params({'symbols': []})
        
        # Portfolio too large
        with pytest.raises(ValidationError, match="exceeds maximum"):
            InputValidator.validate_calculation_params({'initial_portfolio': 1e10})
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        cb = CircuitBreaker(failure_threshold=3, timeout=1)
        
        # Initial state
        assert cb.state == 'closed'
        assert cb.can_execute() == True
        
        # Record failures
        for _ in range(3):
            cb.record_failure()
        
        # Should open circuit
        assert cb.state == 'open'
        assert cb.can_execute() == False
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        assert cb.can_execute() == True  # Should be half-open
        
        # Record success
        cb.record_success()
        assert cb.state == 'closed'
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, agent):
        """Test rate limiting"""
        # Simulate multiple rapid requests
        messages = [{'type': 'calculate', 'parameters': {
            'initial_portfolio': 1000,
            'symbols': ['BTC'],
            'max_depth': 1
        }} for _ in range(5)]
        
        # Should handle first few, then rate limit
        results = []
        for msg in messages:
            result = await agent.process_message(msg)
            results.append(result)
        
        # At least one should succeed
        success_count = sum(1 for r in results if 'error' not in r)
        assert success_count >= 1
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, agent):
        """Test calculation timeout"""
        message = {
            'type': 'calculate',
            'problem_type': 'trading',
            'parameters': {
                'initial_portfolio': 10000,
                'symbols': ['BTC'],
                'max_depth': 2
            },
            'iterations': 10000,  # Large number
            'timeout': 1  # Short timeout
        }
        
        result = await agent.process_message(message)
        
        # Should timeout gracefully
        assert 'error' in result or 'timeout' in result.get('error', '')
    
    @pytest.mark.asyncio
    async def test_caching(self, agent):
        """Test result caching"""
        message = {
            'type': 'calculate',
            'problem_type': 'trading',
            'parameters': {
                'initial_portfolio': 5000,
                'symbols': ['BTC'],
                'max_depth': 2
            },
            'iterations': 50
        }
        
        # First call
        start_time = time.time()
        result1 = await agent.process_message(message)
        time1 = time.time() - start_time
        
        # Second call (should be cached)
        start_time = time.time()
        result2 = await agent.process_message(message)
        time2 = time.time() - start_time
        
        # Second call should be faster (cached)
        if not result1.get('error') and not result2.get('error'):
            assert time2 < time1 or result2.get('cached', False)
    
    @pytest.mark.asyncio
    async def test_parallel_mcts(self, agent, trading_environment):
        """Test parallel MCTS execution"""
        agent.environment = trading_environment
        
        result = await agent.run_mcts_parallel(iterations=20)
        
        assert 'best_action' in result
        assert 'stats' in result
        assert result['stats']['iterations'] == 20
        assert 'tree_size' in result['stats']
    
    @pytest.mark.asyncio
    async def test_rave_algorithm(self, trading_environment):
        """Test RAVE (Rapid Action Value Estimation)"""
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent_v2 import MCTSNodeV2
        
        root = MCTSNodeV2(state={'test': 'state'})
        
        # Update RAVE statistics
        actions = [{'type': 'buy'}, {'type': 'sell'}]
        root.update_rave(actions, 0.5)
        
        assert 'buy' in str(root.rave_visits)
        assert root.rave_visits[str({'type': 'buy'})] == 1
        assert root.rave_values[str({'type': 'buy'})] == 0.5
    
    @pytest.mark.asyncio
    async def test_risk_calculation(self, trading_environment):
        """Test risk metrics calculation"""
        state = await trading_environment.get_initial_state()
        risk_metrics = await trading_environment._calculate_risk_metrics(state)
        
        assert 'volatility' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'var_95' in risk_metrics
        assert risk_metrics['volatility'] >= 0
    
    @pytest.mark.asyncio
    async def test_progressive_widening(self, trading_environment):
        """Test progressive widening of actions"""
        # Shallow state should have more actions
        shallow_state = {'depth': 1, 'positions': {}}
        shallow_actions = await trading_environment.get_available_actions(shallow_state)
        
        # Deep state should have fewer actions
        deep_state = {'depth': 8, 'positions': {}}
        deep_actions = await trading_environment.get_available_actions(deep_state)
        
        assert len(deep_actions) <= len(shallow_actions)
    
    @pytest.mark.asyncio
    async def test_mcp_tools(self, agent):
        """Test MCP tool integration"""
        # Test health check tool
        health_result = await agent._health_check()
        assert isinstance(health_result, ToolResult)
        assert not health_result.isError
        
        # Test metrics tool
        metrics_result = await agent._get_metrics()
        assert isinstance(metrics_result, ToolResult)
        assert not metrics_result.isError
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test comprehensive error handling"""
        # Invalid message type
        result = await agent.process_message({'type': 'invalid'})
        assert 'error' in result
        
        # Missing parameters
        result = await agent.process_message({'type': 'calculate'})
        assert 'error' in result
        
        # Invalid parameters
        result = await agent.process_message({
            'type': 'calculate',
            'parameters': {'invalid': 'params'}
        })
        assert 'error' in result


class TestMCTSMonitoring:
    """Test suite for MCTS monitoring"""
    
    @pytest.fixture
    def monitor(self):
        """Create test monitor"""
        return MCTSMonitor("test_agent", retention_hours=1)
    
    @pytest.mark.asyncio
    async def test_performance_recording(self, monitor):
        """Test performance metrics recording"""
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            calculation_id="test_calc_001",
            iterations=1000,
            execution_time=2.5,
            tree_size=500,
            memory_usage_mb=150.0,
            best_action_confidence=0.85,
            expected_value=0.12
        )
        
        await monitor.record_calculation(metrics)
        
        assert monitor.calculation_count == 1
        assert len(monitor.performance_metrics) == 1
        assert monitor.performance_metrics[0] == metrics
    
    @pytest.mark.asyncio
    async def test_error_recording(self, monitor):
        """Test error recording"""
        error = ValueError("Test error")
        context = {"test": "context"}
        
        await monitor.record_error(error, context)
        
        assert monitor.error_count == 1
        assert len(monitor.error_log) == 1
        assert monitor.error_log[0]['error_type'] == 'ValueError'
    
    @pytest.mark.asyncio
    async def test_health_status(self, monitor):
        """Test health status calculation"""
        health = await monitor.get_health_status(
            circuit_breaker_state='closed',
            rate_limit_remaining=90,
            memory_ok=True
        )
        
        assert isinstance(health, HealthStatus)
        assert health.status in ['healthy', 'degraded', 'unhealthy']
        assert health.agent_id == "test_agent"
    
    @pytest.mark.asyncio
    async def test_performance_summary(self, monitor):
        """Test performance summary generation"""
        # Add test metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                calculation_id=f"calc_{i}",
                iterations=100,
                execution_time=1.0,
                tree_size=50,
                memory_usage_mb=100.0,
                best_action_confidence=0.8,
                expected_value=0.1
            )
            await monitor.record_calculation(metrics)
        
        summary = await monitor.get_performance_summary(hours=1)
        
        assert summary['calculation_count'] == 5
        assert 'metrics' in summary
        assert summary['metrics']['average_iterations'] == 100
    
    @pytest.mark.asyncio
    async def test_alerts(self, monitor):
        """Test alert generation"""
        # Add high-error scenario
        monitor.calculation_count = 10
        monitor.error_count = 3  # 30% error rate
        
        alerts = await monitor.get_alerts()
        
        # Should trigger error rate alert
        error_alerts = [a for a in alerts if a['type'] == 'error_rate']
        assert len(error_alerts) > 0
    
    def test_metrics_export(self, monitor):
        """Test metrics export"""
        export_data = monitor.export_metrics(format='json')
        assert isinstance(export_data, str)
        
        import json
        parsed = json.loads(export_data)
        assert 'agent_id' in parsed
        assert 'performance_metrics' in parsed


class TestProductionTradingEnvironment:
    """Test suite for Production Trading Environment"""
    
    @pytest.fixture
    def environment(self):
        """Create test environment"""
        config = {
            'initial_portfolio': 10000,
            'symbols': ['BTC', 'ETH'],
            'max_depth': 5
        }
        return ProductionTradingEnvironment(config)
    
    @pytest.mark.asyncio
    async def test_transaction_costs(self, environment):
        """Test transaction cost calculation"""
        state = await environment.get_initial_state()
        
        buy_action = {
            'type': 'buy',
            'symbol': 'BTC',
            'percentage': 0.5
        }
        
        new_state = await environment.apply_action(state, buy_action)
        
        # Portfolio value should decrease more than investment due to costs
        investment = state['portfolio_value'] * 0.5
        actual_decrease = state['portfolio_value'] - new_state['portfolio_value']
        assert actual_decrease > investment * 0.999  # Account for costs and slippage
    
    @pytest.mark.asyncio
    async def test_risk_limits(self, environment):
        """Test risk-based terminal conditions"""
        # Create state with large loss
        state = {
            'portfolio_value': 5000,  # 50% loss from 10000
            'positions': {},
            'depth': 2
        }
        
        is_terminal = await environment.is_terminal_state(state)
        assert is_terminal  # Should stop due to stop loss
    
    @pytest.mark.asyncio
    async def test_market_data_caching(self, environment):
        """Test market data caching"""
        # First call
        data1 = await environment._fetch_market_data()
        
        # Second call (should be cached)
        data2 = await environment._fetch_market_data()
        
        assert data1 == data2  # Should be identical due to caching
    
    @pytest.mark.asyncio
    async def test_portfolio_value_calculation(self, environment):
        """Test portfolio value calculation"""
        state = {
            'portfolio_value': 5000,
            'positions': {'BTC': 0.1, 'ETH': 2.0},
            'market_data': {
                'BTC': {'price': 30000},
                'ETH': {'price': 2000}
            }
        }
        
        total_value = await environment._calculate_portfolio_value(state)
        expected = 5000 + (0.1 * 30000) + (2.0 * 2000)  # 5000 + 3000 + 4000
        assert total_value == expected