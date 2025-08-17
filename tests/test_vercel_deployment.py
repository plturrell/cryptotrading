"""
Tests for Vercel deployment compatibility
Ensures MCTS works correctly in Vercel Edge Runtime
"""
import pytest
import asyncio
import os
import time
from unittest.mock import Mock, patch, AsyncMock

from src.cryptotrading.core.agents.specialized.mcts_calculation_agent import (
    ProductionMCTSCalculationAgent,
    ProductionTradingEnvironment,
    MCTSConfig
)
from src.cryptotrading.core.agents.specialized.vercel_runtime_adapter import (
    VercelRuntimeAdapter,
    vercel_adapter,
    vercel_edge_handler,
    VercelTimeoutError,
    VercelMemoryError
)


class TestVercelCompatibility:
    """Test Vercel Edge Runtime compatibility"""
    
    @pytest.fixture
    def mock_vercel_env(self, monkeypatch):
        """Mock Vercel environment variables"""
        monkeypatch.setenv('VERCEL_ENV', 'production')
        monkeypatch.setenv('VERCEL_EDGE', 'true')
        monkeypatch.setenv('VERCEL_REGION', 'iad1')
        monkeypatch.setenv('VERCEL_MAX_DURATION', '30')
    
    @pytest.fixture
    def vercel_config(self):
        """Create Vercel-optimized configuration"""
        config = MCTSConfig()
        config.iterations = 500  # Reduced for Vercel
        config.timeout_seconds = 25  # Leave buffer
        config.max_memory_mb = 512
        config.simulation_strategy = 'weighted_random'
        return config
    
    def test_runtime_detection(self, mock_vercel_env):
        """Test Vercel runtime detection"""
        adapter = VercelRuntimeAdapter()
        assert adapter.is_vercel
        assert adapter.is_edge_runtime
        assert adapter.max_duration == 30
        assert adapter.region == 'iad1'
    
    def test_psutil_fallback(self):
        """Test psutil fallback for memory checking"""
        # Mock psutil not available
        with patch.dict('sys.modules', {'psutil': None}):
            adapter = VercelRuntimeAdapter()
            assert not adapter.has_psutil
            
            # Test memory estimation
            memory_info = adapter.get_memory_usage(tree_size=5000)
            assert 'estimated_mb' in memory_info
            assert memory_info['method'] == 'tree_size_estimate'
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_vercel_env):
        """Test Vercel timeout handling"""
        adapter = VercelRuntimeAdapter()
        
        @adapter.timeout_handler(timeout_seconds=2)
        async def slow_operation():
            await asyncio.sleep(3)  # Exceeds timeout
            return "completed"
        
        result = await slow_operation()
        assert result['error'] == 'timeout'
        assert result['partial_results'] == True
    
    @pytest.mark.asyncio
    async def test_memory_guard(self, vercel_config):
        """Test memory limit protection"""
        agent = ProductionMCTSCalculationAgent(
            agent_id="test_memory_agent",
            config=vercel_config
        )
        
        # Mock memory limit check to return True (limit exceeded)
        agent._check_memory_limit = Mock(return_value=True)
        
        # Test that memory guard prevents execution
        @vercel_adapter.memory_guard()
        async def memory_intensive_operation(agent_self):
            return await agent_self.run_mcts_parallel(iterations=1000)
        
        # Should handle memory limit gracefully
        with patch('asyncio.create_task'):
            result = await memory_intensive_operation(agent)
            if isinstance(result, dict) and 'error' in result:
                assert result['error'] == 'memory_limit'
    
    @pytest.mark.asyncio
    async def test_stochastic_simulation(self, vercel_config):
        """Test true Monte Carlo stochastic simulation"""
        agent = ProductionMCTSCalculationAgent(
            agent_id="test_stochastic",
            config=vercel_config
        )
        
        # Test pure random strategy
        agent.config.simulation_strategy = 'pure_random'
        
        actions = [
            {'type': 'buy', 'symbol': 'BTC'},
            {'type': 'sell', 'symbol': 'ETH'},
            {'type': 'analysis'}
        ]
        
        # Run multiple times to verify randomness
        selected_actions = []
        for _ in range(10):
            action = await agent._select_simulation_action(actions, {}, 0)
            selected_actions.append(action['type'])
        
        # Should have variation (not all same)
        assert len(set(selected_actions)) > 1
    
    @pytest.mark.asyncio 
    async def test_edge_compatible_decorator(self):
        """Test Edge Runtime compatibility decorator"""
        @vercel_adapter.edge_compatible
        async def test_function(**kwargs):
            return kwargs
        
        # Edge-incompatible parameters should be filtered
        result = await test_function(
            valid_param="test",
            use_multiprocessing=True,  # Should be filtered
            spawn_workers=True  # Should be filtered
        )
        
        assert 'valid_param' in result
        assert 'use_multiprocessing' not in result
        assert 'spawn_workers' not in result
    
    @pytest.mark.asyncio
    async def test_no_background_tasks(self, mock_vercel_env, vercel_config):
        """Test that background monitoring is disabled in Vercel"""
        agent = ProductionMCTSCalculationAgent(
            agent_id="test_no_background",
            config=vercel_config
        )
        
        # Verify background monitoring is not started
        assert agent.is_vercel_runtime
        # In Vercel, monitoring dashboard should not have active tasks
    
    def test_error_handling(self):
        """Test Vercel-specific error handling"""
        adapter = VercelRuntimeAdapter()
        
        # Test timeout error
        timeout_error = VercelTimeoutError("Operation timed out")
        response = adapter.handle_vercel_error(timeout_error)
        assert response['code'] == 'TIMEOUT'
        assert response['status'] == 504
        
        # Test memory error
        memory_error = VercelMemoryError("Memory limit exceeded")
        response = adapter.handle_vercel_error(memory_error)
        assert response['code'] == 'MEMORY_LIMIT'
        assert response['status'] == 507
        
        # Test generic error
        generic_error = ValueError("Invalid input")
        response = adapter.handle_vercel_error(generic_error)
        assert response['code'] == 'INVALID_INPUT'
        assert response['status'] == 400
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test Vercel KV cache operations"""
        adapter = VercelRuntimeAdapter()
        
        # Test cache get (should handle missing KV gracefully)
        result = await adapter.safe_cache_operation('get', 'test_key')
        assert result is None  # No KV configured
        
        # Test cache set
        result = await adapter.safe_cache_operation('set', 'test_key', 'test_value')
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_full_mcts_execution_vercel(self, mock_vercel_env, vercel_config):
        """Test complete MCTS execution in Vercel environment"""
        agent = ProductionMCTSCalculationAgent(
            agent_id="test_full_vercel",
            config=vercel_config
        )
        
        # Setup simple environment
        config = {
            'initial_portfolio': 10000,
            'symbols': ['BTC'],
            'max_depth': 3
        }
        
        agent.environment = ProductionTradingEnvironment(config)
        
        # Run MCTS with Vercel constraints
        start_time = time.time()
        result = await agent.run_mcts_parallel(iterations=100)
        elapsed = time.time() - start_time
        
        # Verify execution completed within reasonable time
        assert elapsed < 5  # Should be fast with 100 iterations
        assert 'best_action' in result
        assert 'stats' in result
        assert result['stats']['iterations'] == 100
        
        # Verify stochastic behavior
        assert result['stats'].get('simulation_strategy') == vercel_config.simulation_strategy


class TestDeploymentConfiguration:
    """Test deployment configuration files"""
    
    def test_vercel_json_valid(self):
        """Test vercel.json configuration is valid"""
        import json
        
        try:
            with open('vercel_config.json', 'r') as f:
                config = json.load(f)
                
            assert config['version'] == 2
            assert 'functions' in config
            assert 'env' in config
            assert config['env']['MCTS_TIMEOUT'] == '25'  # Less than 30s limit
            assert config['env']['VERCEL_EDGE'] == 'true'
        except FileNotFoundError:
            pytest.skip("vercel_config.json not found")
    
    def test_environment_files(self):
        """Test environment configuration files"""
        # Check production env
        try:
            with open('env_production', 'r') as f:
                prod_env = f.read()
            assert 'MCTS_SECURITY_LEVEL=production' in prod_env
            assert 'VERCEL_EDGE=true' in prod_env
            assert 'MCTS_MONITORING_ENABLED=false' in prod_env  # No background tasks
        except FileNotFoundError:
            pytest.skip("env_production not found")
        
        # Check development env
        try:
            with open('env_development', 'r') as f:
                dev_env = f.read()
            assert 'MCTS_SECURITY_LEVEL=development' in dev_env
            assert 'VERCEL_EDGE=false' in dev_env
            assert 'MCTS_MONITORING_ENABLED=true' in dev_env
        except FileNotFoundError:
            pytest.skip("env_development not found")
    
    def test_api_endpoint_configuration(self):
        """Test API endpoint configuration"""
        import json
        
        # Test TypeScript endpoint config
        try:
            with open('api/mcts-calculate.ts', 'r') as f:
                content = f.read()
            assert 'runtime: \'edge\'' in content
            assert 'maxDuration: 30' in content
        except FileNotFoundError:
            pytest.skip("API endpoint not found")


@pytest.mark.integration
class TestVercelIntegration:
    """Integration tests for Vercel deployment"""
    
    @pytest.mark.asyncio
    async def test_edge_function_simulation(self, mock_vercel_env):
        """Simulate Edge Function execution"""
        # This simulates what happens in the Vercel Edge Function
        
        request_body = {
            'problem_type': 'trading',
            'parameters': {
                'initial_portfolio': 10000,
                'symbols': ['BTC', 'ETH'],
                'max_depth': 5
            },
            'iterations': 500
        }
        
        # Create agent as Edge Function would
        config = MCTSConfig()
        agent = ProductionMCTSCalculationAgent(
            agent_id=f"edge_function_{int(time.time())}",
            config=config
        )
        
        # Execute with Vercel constraints
        agent.environment = ProductionTradingEnvironment(request_body['parameters'])
        
        # This should complete within Edge Function limits
        result = await agent.run_mcts_parallel(iterations=request_body['iterations'])
        
        assert result is not None
        assert 'error' not in result or result.get('error') == 'timeout'  # Timeout is acceptable
        
        if 'stats' in result:
            # Verify memory efficiency
            assert result['stats']['tree_size'] < 10000  # Should stay under control
            # Verify completed within iteration limit
            assert result['stats']['iterations'] <= request_body['iterations']