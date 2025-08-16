"""
Tests for MCTS Calculation Agent
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.cryptotrading.core.agents.specialized import (
    MCTSCalculationAgent, 
    CalculationEnvironment,
    TradingCalculationEnvironment
)
from src.cryptotrading.core.protocols.mcp.tools import ToolResult


class TestMCTSCalculationAgent:
    """Test suite for MCTS Calculation Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create test agent instance"""
        return MCTSCalculationAgent(
            agent_id="test_mcts_001",
            mcts_config={
                'iterations': 100,
                'exploration_constant': 1.4,
                'simulation_depth': 5
            }
        )
    
    @pytest.fixture
    def trading_environment(self):
        """Create test trading environment"""
        config = {
            'initial_portfolio': 10000,
            'symbols': ['BTC', 'ETH'],
            'max_depth': 3
        }
        return TradingCalculationEnvironment(config)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.agent_id == "test_mcts_001"
        assert agent.agent_type == "mcts_calculation"
        assert agent.iterations == 100
        assert agent.exploration_constant == 1.4
        assert agent.simulation_depth == 5
        assert "monte_carlo_search" in agent.capabilities
    
    @pytest.mark.asyncio
    async def test_process_calculate_message(self, agent):
        """Test processing calculation request"""
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
        
        result = await agent.process_message(message)
        
        assert result['type'] == 'calculation_result'
        assert result['problem_type'] == 'trading'
        assert 'best_action' in result
        assert 'expected_value' in result
        assert 'confidence' in result
        assert 'exploration_stats' in result
    
    @pytest.mark.asyncio
    async def test_mcts_algorithm(self, agent, trading_environment):
        """Test MCTS algorithm execution"""
        agent.environment = trading_environment
        
        result = await agent.run_mcts(iterations=10)
        
        assert 'best_action' in result
        assert 'expected_value' in result
        assert 'confidence' in result
        assert result['stats']['iterations'] == 10
        assert result['stats']['elapsed_time'] > 0
        assert 'average_value' in result['stats']
    
    @pytest.mark.asyncio
    async def test_trading_environment_initial_state(self, trading_environment):
        """Test trading environment initialization"""
        state = await trading_environment.get_initial_state()
        
        assert state['portfolio_value'] == 10000
        assert state['positions'] == {}
        assert 'market_data' in state
        assert state['depth'] == 0
        assert len(state['available_actions']) > 0
    
    @pytest.mark.asyncio
    async def test_trading_environment_apply_buy_action(self, trading_environment):
        """Test applying buy action in trading environment"""
        state = await trading_environment.get_initial_state()
        action = {
            'type': 'buy',
            'symbol': 'BTC',
            'percentage': 0.5
        }
        
        new_state = await trading_environment.apply_action(state, action)
        
        assert new_state['portfolio_value'] < state['portfolio_value']
        assert 'BTC' in new_state['positions']
        assert new_state['positions']['BTC'] > 0
        assert new_state['depth'] == 1
    
    @pytest.mark.asyncio
    async def test_trading_environment_terminal_state(self, trading_environment):
        """Test terminal state detection"""
        state = {'depth': 0}
        assert not await trading_environment.is_terminal_state(state)
        
        state = {'depth': 3}
        assert await trading_environment.is_terminal_state(state)
    
    @pytest.mark.asyncio
    async def test_mcp_tool_registration(self, agent):
        """Test MCP tool registration"""
        with patch.object(agent.mcp_bridge, 'mcp_server') as mock_server:
            mock_server.register_tool = Mock()
            agent._register_calculation_tools()
            
            # Should register 3 tools
            assert mock_server.register_tool.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_mcts_calculate_tool(self, agent):
        """Test MCTS calculation MCP tool"""
        result = await agent._mcts_calculate(
            problem_type='trading',
            parameters={'initial_portfolio': 10000, 'symbols': ['BTC']},
            iterations=10
        )
        
        assert isinstance(result, ToolResult)
        assert not result.isError
    
    @pytest.mark.asyncio
    async def test_evaluate_strategy_tool(self, agent):
        """Test strategy evaluation MCP tool"""
        strategy = {
            'type': 'momentum',
            'initial_capital': 50000,
            'risk_per_trade': 0.02
        }
        market_conditions = {
            'volatility': 'high',
            'trend': 'bullish'
        }
        
        result = await agent._evaluate_strategy(strategy, market_conditions)
        
        assert isinstance(result, ToolResult)
        assert not result.isError
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in agent"""
        message = {
            'type': 'unknown_type',
            'data': {}
        }
        
        result = await agent.process_message(message)
        
        assert 'error' in result
        assert 'Unknown message type' in result['error']
    
    @pytest.mark.asyncio
    async def test_node_expansion(self, agent):
        """Test MCTS node expansion"""
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent import MCTSNode
        
        root = MCTSNode(
            state={'value': 0},
            untried_actions=[{'action': 'test1'}, {'action': 'test2'}]
        )
        
        assert not root.is_fully_expanded()
        
        child = root.add_child({'action': 'test1'}, {'value': 1})
        
        assert len(root.children) == 1
        assert len(root.untried_actions) == 1
        assert child.parent == root
    
    @pytest.mark.asyncio
    async def test_ucb_selection(self, agent):
        """Test UCB-based child selection"""
        from src.cryptotrading.core.agents.specialized.mcts_calculation_agent import MCTSNode
        
        root = MCTSNode(state={'value': 0}, visits=10)
        
        # Add children with different values
        child1 = MCTSNode(state={'value': 1}, parent=root, visits=5, value=3.0)
        child2 = MCTSNode(state={'value': 2}, parent=root, visits=3, value=2.0)
        root.children = [child1, child2]
        
        best = root.best_child(c_param=1.4)
        
        # Child2 should have higher UCB score despite lower average value
        assert best == child2
    
    @pytest.mark.asyncio
    async def test_strategy_recommendations(self, agent):
        """Test strategy performance analysis"""
        mcts_result = {
            'expected_value': -0.1,
            'confidence': 0.5,
            'stats': {
                'max_value': 0.8,
                'min_value': -0.3,
                'average_value': 0.2
            }
        }
        
        recommendations = agent._analyze_strategy_performance(mcts_result)
        
        assert len(recommendations) > 0
        assert any('conservative' in r for r in recommendations)
        assert any('iterations' in r for r in recommendations)


class TestCalculationEnvironment:
    """Test abstract calculation environment"""
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined"""
        assert hasattr(CalculationEnvironment, 'get_initial_state')
        assert hasattr(CalculationEnvironment, 'get_available_actions')
        assert hasattr(CalculationEnvironment, 'apply_action')
        assert hasattr(CalculationEnvironment, 'is_terminal_state')
        assert hasattr(CalculationEnvironment, 'evaluate_state')