"""
Comprehensive tests for Trading Algorithm Agent.

Tests all 10 trading strategy ANALYSIS and SIGNAL GENERATION capabilities.
Note: This agent generates signals only - no actual trading or portfolio management.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.cryptotrading.core.agents.specialized.trading_algorithm_agent import (
    TradingAlgorithmAgent,
    TradingStrategy,
    TradingSignal,
    OrderGrid,
    ArbitrageOpportunity
)
from src.cryptotrading.core.agents.base import AgentStatus
from src.cryptotrading.core.protocols.a2a.a2a_protocol import A2AMessage, MessageType


@pytest.fixture
async def trading_agent():
    """Create a trading agent instance for testing."""
    agent = TradingAlgorithmAgent()
    agent.data_provider = AsyncMock()
    agent.grok4_client = AsyncMock()
    return agent


@pytest.fixture
def mock_market_data():
    """Mock market data for testing."""
    return {
        "BTC/USDT": {
            "price": Decimal("50000"),
            "volume": Decimal("1000000"),
            "bid": Decimal("49990"),
            "ask": Decimal("50010"),
            "high_24h": Decimal("51000"),
            "low_24h": Decimal("49000")
        },
        "ETH/USDT": {
            "price": Decimal("3000"),
            "volume": Decimal("500000"),
            "bid": Decimal("2998"),
            "ask": Decimal("3002"),
            "high_24h": Decimal("3100"),
            "low_24h": Decimal("2900")
        }
    }


class TestGridTradingStrategy:
    """Test Grid Trading strategy implementation."""
    
    @pytest.mark.asyncio
    async def test_grid_create(self, trading_agent, mock_market_data):
        """Test grid creation with proper order placement."""
        trading_agent._get_current_price = AsyncMock(
            return_value=mock_market_data["BTC/USDT"]["price"]
        )
        trading_agent._place_grid_orders = AsyncMock()
        
        grid = await trading_agent._mcp_grid_create("BTC/USDT", Decimal("10000"))
        
        assert grid.symbol == "BTC/USDT"
        assert grid.center_price == Decimal("50000")
        assert len(grid.buy_orders) == 10
        assert len(grid.sell_orders) == 10
        assert grid.total_investment == Decimal("10000")
        
        # Verify order spacing
        for i, (price, _) in enumerate(grid.buy_orders):
            if i > 0:
                assert grid.buy_orders[i-1][0] > price  # Descending prices
        
        for i, (price, _) in enumerate(grid.sell_orders):
            if i > 0:
                assert grid.sell_orders[i-1][0] < price  # Ascending prices
    
    @pytest.mark.asyncio
    async def test_grid_rebalance(self, trading_agent):
        """Test grid rebalancing when price moves significantly."""
        trading_agent._get_current_price = AsyncMock(return_value=Decimal("55000"))
        trading_agent.order_book = {
            "BTC/USDT": {
                "grid": {
                    "center_price": Decimal("50000"),
                    "total_investment": Decimal("10000")
                }
            }
        }
        trading_agent._cancel_grid_orders = AsyncMock()
        trading_agent._mcp_grid_create = AsyncMock(
            return_value=OrderGrid(
                symbol="BTC/USDT",
                center_price=Decimal("55000"),
                grid_spacing=Decimal("1100"),
                grid_levels=10,
                buy_orders=[],
                sell_orders=[],
                total_investment=Decimal("10000")
            )
        )
        
        result = await trading_agent._mcp_grid_rebalance("BTC/USDT")
        
        assert result["status"] == "rebalanced"
        assert result["old_center"] == Decimal("50000")
        assert result["new_center"] == Decimal("55000")
        assert result["price_change"] > 0.05
    
    @pytest.mark.asyncio
    async def test_grid_monitor(self, trading_agent):
        """Test grid monitoring and performance tracking."""
        trading_agent._get_current_price = AsyncMock(return_value=Decimal("52000"))
        trading_agent._get_filled_orders = AsyncMock(
            return_value=[
                {"profit": 100},
                {"profit": 150},
                {"profit": -50}
            ]
        )
        trading_agent.order_book = {
            "BTC/USDT": {
                "grid": {
                    "center_price": Decimal("50000"),
                    "grid_spacing": Decimal("1000"),
                    "active_orders": [1, 2, 3, 4, 5]
                }
            }
        }
        
        metrics = await trading_agent._mcp_grid_monitor("BTC/USDT")
        
        assert metrics["symbol"] == "BTC/USDT"
        assert metrics["filled_orders"] == 3
        assert metrics["total_profit"] == 200
        assert metrics["active_orders"] == 5


class TestDCAStrategy:
    """Test Dollar-Cost Averaging strategy."""
    
    @pytest.mark.asyncio
    async def test_dca_execute_basic(self, trading_agent):
        """Test basic DCA execution."""
        trading_agent._get_current_price = AsyncMock(return_value=Decimal("50000"))
        trading_agent._calculate_dca_adjustment = AsyncMock(return_value=1.0)
        trading_agent._generate_trade_signal = AsyncMock(
            return_value={"order_id": "123", "status": "filled"}
        )
        trading_agent._update_position = AsyncMock()
        
        result = await trading_agent._mcp_dca_execute("BTC/USDT", Decimal("1000"))
        
        assert result["symbol"] == "BTC/USDT"
        assert result["amount"] == Decimal("1000")
        assert result["quantity"] == Decimal("0.02")  # 1000 / 50000
        assert result["price"] == Decimal("50000")
        assert result["order_id"] == "123"
    
    @pytest.mark.asyncio
    async def test_dca_smart_adjust(self, trading_agent):
        """Test smart DCA adjustments based on indicators."""
        trading_agent._calculate_rsi = AsyncMock(return_value=25)  # Oversold
        trading_agent._calculate_volume_ratio = AsyncMock(return_value=2.0)  # High volume
        trading_agent._get_ai_sentiment = AsyncMock(
            return_value={"sentiment": "bullish", "confidence": 0.8}
        )
        
        adjustment = await trading_agent._mcp_dca_smart_adjust("BTC/USDT")
        
        assert adjustment["symbol"] == "BTC/USDT"
        assert adjustment["base_multiplier"] > 1.0  # Should increase in oversold
        assert adjustment["rsi"] == 25
        assert adjustment["volume_ratio"] == 2.0
        assert adjustment["sentiment"]["sentiment"] == "bullish"
    
    @pytest.mark.asyncio
    async def test_dca_schedule(self, trading_agent):
        """Test DCA scheduling."""
        with patch('asyncio.create_task') as mock_create_task:
            result = await trading_agent._mcp_dca_schedule("BTC/USDT", 24)
            
            assert result["status"] == "scheduled"
            assert result["symbol"] == "BTC/USDT"
            assert result["frequency_hours"] == 24
            mock_create_task.assert_called_once()


class TestArbitrageStrategy:
    """Test Arbitrage strategy."""
    
    @pytest.mark.asyncio
    async def test_arbitrage_scan(self, trading_agent):
        """Test arbitrage opportunity scanning."""
        trading_agent._get_multi_exchange_prices = AsyncMock(
            return_value={
                "binance": Decimal("50000"),
                "coinbase": Decimal("50100"),
                "kraken": Decimal("49950")
            }
        )
        trading_agent._get_arbitrage_volume = AsyncMock(return_value=Decimal("1"))
        
        opportunities = await trading_agent._mcp_arbitrage_scan(["BTC/USDT"])
        
        assert len(opportunities) > 0
        best = opportunities[0]
        assert best.symbol == "BTC/USDT"
        assert best.spread > 0
        assert best.profit_percentage > 0.001  # At least 0.1%
    
    @pytest.mark.asyncio
    async def test_arbitrage_execute(self, trading_agent):
        """Test arbitrage execution with latency check."""
        opportunity = ArbitrageOpportunity(
            symbol="BTC/USDT",
            buy_exchange="kraken",
            sell_exchange="coinbase",
            buy_price=Decimal("49950"),
            sell_price=Decimal("50100"),
            spread=Decimal("150"),
            profit_percentage=0.003,
            volume=Decimal("1"),
            estimated_profit=Decimal("150")
        )
        
        trading_agent._check_exchange_latency = AsyncMock(
            return_value={"kraken": 50, "coinbase": 60}
        )
        trading_agent._generate_trade_signal = AsyncMock(
            return_value={"filled_price": Decimal("50000"), "status": "filled"}
        )
        
        result = await trading_agent._mcp_arbitrage_execute(opportunity)
        
        assert result["status"] == "success"
        assert "buy_order" in result
        assert "sell_order" in result
        assert "actual_profit" in result


class TestMomentumStrategy:
    """Test Momentum/Trend Following strategy."""
    
    @pytest.mark.asyncio
    async def test_momentum_scan(self, trading_agent):
        """Test momentum opportunity scanning."""
        # Mock price history showing uptrend
        prices = np.array([48000, 48500, 49000, 49500, 50000] * 40)  # 200 periods
        trading_agent._get_price_history = AsyncMock(return_value=prices)
        trading_agent._calculate_rsi = AsyncMock(return_value=70)
        trading_agent._check_volume_surge = AsyncMock(return_value=True)
        trading_agent._calculate_position_size = AsyncMock(return_value=Decimal("0.1"))
        
        signals = await trading_agent._mcp_momentum_scan(["BTC/USDT"])
        
        assert len(signals) > 0
        signal = signals[0]
        assert signal.strategy == TradingStrategy.MOMENTUM
        assert signal.action == "BUY"
        assert signal.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_momentum_trailing_stop(self, trading_agent):
        """Test trailing stop updates."""
        trading_agent.positions = {
            "BTC/USDT": {
                "highest_price": Decimal("50000"),
                "stop_loss": Decimal("47500")
            }
        }
        trading_agent._get_current_price = AsyncMock(return_value=Decimal("52000"))
        trading_agent._update_stop_loss = AsyncMock()
        
        result = await trading_agent._mcp_momentum_trail_stop("BTC/USDT")
        
        assert result["status"] == "updated"
        assert result["new_stop"] == Decimal("49400")  # 52000 * 0.95


class TestMeanReversionStrategy:
    """Test Mean Reversion strategy."""
    
    @pytest.mark.asyncio
    async def test_mean_reversion_identify(self, trading_agent):
        """Test mean reversion opportunity identification."""
        # Create price data with extreme deviation
        prices = np.array([50000] * 40 + [45000])  # Current price far below mean
        trading_agent._get_price_history = AsyncMock(return_value=prices)
        trading_agent._calculate_position_size = AsyncMock(return_value=Decimal("0.1"))
        
        signals = await trading_agent._mcp_mean_reversion_identify(["BTC/USDT"])
        
        assert len(signals) > 0
        signal = signals[0]
        assert signal.strategy == TradingStrategy.MEAN_REVERSION
        assert signal.action == "BUY"  # Should buy when oversold
        assert signal.metadata["z_score"] < -2
    
    @pytest.mark.asyncio
    async def test_pairs_trading(self, trading_agent):
        """Test pairs trading execution."""
        # Mock correlated price data
        prices1 = np.random.randn(100) + 50000
        prices2 = prices1 * 0.06 + np.random.randn(100) * 10  # Correlated with noise
        
        trading_agent._get_price_history = AsyncMock(
            side_effect=[prices1, prices2, prices1, prices2]
        )
        trading_agent._generate_trade_signal = AsyncMock(
            return_value={"status": "filled", "filled_price": Decimal("50000")}
        )
        
        result = await trading_agent._mcp_pairs_trading_execute("BTC/USDT", "ETH/USDT")
        
        if result["status"] == "success":
            assert "pair1" in result
            assert "pair2" in result
            assert result["correlation"] > 0.5


class TestRiskManagement:
    """Test risk management tools."""
    
    @pytest.mark.asyncio
    async def test_risk_calculate(self, trading_agent):
        """Test comprehensive risk calculation."""
        portfolio = {
            "positions": {
                "BTC/USDT": {"value": 5000, "quantity": 0.1},
                "ETH/USDT": {"value": 3000, "quantity": 1}
            }
        }
        
        trading_agent._get_historical_returns = AsyncMock(
            return_value=np.random.randn(30) * 0.05
        )
        trading_agent._calculate_correlation_matrix = AsyncMock(
            return_value=np.array([[1.0, 0.7], [0.7, 1.0]])
        )
        trading_agent._calculate_portfolio_beta = AsyncMock(return_value=1.2)
        trading_agent._get_portfolio_history = AsyncMock(
            return_value=np.array([10000, 9500, 9800, 10200, 10100])
        )
        
        metrics = await trading_agent._mcp_risk_calculate(portfolio)
        
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics
        assert metrics["position_count"] == 2
        assert metrics["total_exposure"] == 8000
    
    @pytest.mark.asyncio
    async def test_position_sizing(self, trading_agent):
        """Test position sizing with Kelly Criterion."""
        trading_agent._get_symbol_trade_history = AsyncMock(
            return_value=[
                {"profit": 100},
                {"profit": 150},
                {"profit": -50},
                {"profit": 200},
                {"profit": -75}
            ]
        )
        trading_agent._get_portfolio_value = AsyncMock(return_value=Decimal("10000"))
        
        sizing = await trading_agent._mcp_position_size(
            "BTC/USDT",
            Decimal("50000"),
            Decimal("47500")
        )
        
        assert sizing["symbol"] == "BTC/USDT"
        assert sizing["recommended_size"] > 0
        assert sizing["kelly_fraction"] > 0
        assert sizing["risk_amount"] == Decimal("200")  # 2% of 10000


class TestMLPredictiveStrategy:
    """Test ML/AI Predictive strategy."""
    
    @pytest.mark.asyncio
    async def test_ml_predict(self, trading_agent):
        """Test ML prediction generation."""
        trading_agent._prepare_ml_features = AsyncMock(
            return_value=np.array([50000, 1000000, 65, 0.5, 0.7])
        )
        trading_agent._get_or_create_ml_model = AsyncMock(
            return_value={
                "random_forest": MagicMock(
                    predict=MagicMock(return_value=[0.05]),
                    predict_proba=MagicMock(return_value=[[0.2, 0.8]]),
                    feature_importances_=np.array([0.3, 0.2, 0.2, 0.15, 0.15])
                ),
                "gradient_boosting": MagicMock(
                    predict=MagicMock(return_value=[0.04])
                )
            }
        )
        trading_agent._get_current_price = AsyncMock(return_value=Decimal("50000"))
        
        prediction = await trading_agent._mcp_ml_predict("BTC/USDT", 24)
        
        assert prediction["symbol"] == "BTC/USDT"
        assert prediction["prediction"] > 0
        assert prediction["confidence"] > 0.5
        assert prediction["action"] in ["BUY", "SELL", "HOLD"]
        assert "feature_importance" in prediction


class TestMultiStrategyHybrid:
    """Test Multi-Strategy Hybrid system."""
    
    @pytest.mark.asyncio
    async def test_strategy_allocate(self, trading_agent):
        """Test dynamic strategy allocation."""
        trading_agent._calculate_strategy_metrics = AsyncMock(
            return_value={
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.1,
                "win_rate": 0.65
            }
        )
        trading_agent._calculate_risk_parity_allocation = AsyncMock(
            return_value={
                TradingStrategy.MOMENTUM: 0.3,
                TradingStrategy.MEAN_REVERSION: 0.2,
                TradingStrategy.GRID_TRADING: 0.5
            }
        )
        
        result = await trading_agent._mcp_strategy_allocate()
        
        assert "allocations" in result
        assert sum(result["allocations"].values()) <= 1.0
        assert result["allocation_mode"] == "dynamic"
    
    @pytest.mark.asyncio
    async def test_strategy_switch(self, trading_agent):
        """Test strategy switching based on market conditions."""
        trading_agent._mcp_strategy_allocate = AsyncMock(
            return_value={"allocations": {}}
        )
        
        result = await trading_agent._mcp_strategy_switch("high_volatility")
        
        assert result["market_condition"] == "high_volatility"
        assert "active_strategies" in result
        
        # Verify appropriate strategies are activated for high volatility
        assert trading_agent.active_strategies[TradingStrategy.SCALPING]
        assert trading_agent.active_strategies[TradingStrategy.GRID_TRADING]
        assert trading_agent.active_strategies[TradingStrategy.ARBITRAGE]


class TestIntegration:
    """Test integration with A2A network."""
    
    @pytest.mark.asyncio
    async def test_process_analysis_request(self, trading_agent):
        """Test processing analysis requests via A2A protocol."""
        message = A2AMessage(
            type=MessageType.ANALYSIS_REQUEST,
            sender="test_agent",
            receiver="trading_algorithm_agent",
            payload={
                "strategy": "momentum",
                "symbols": ["BTC/USDT", "ETH/USDT"]
            }
        )
        
        trading_agent._mcp_momentum_scan = AsyncMock(
            return_value=[
                TradingSignal(
                    strategy=TradingStrategy.MOMENTUM,
                    action="BUY",
                    symbol="BTC/USDT",
                    price=Decimal("50000"),
                    quantity=Decimal("0.1"),
                    confidence=0.8,
                    reason="Strong momentum"
                )
            ]
        )
        
        response = await trading_agent.process_message(message)
        
        assert response is not None
        assert response.type == MessageType.ANALYSIS_RESPONSE
        assert "signals" in response.payload
        assert len(response.payload["signals"]) > 0
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, trading_agent):
        """Test agent lifecycle management."""
        # Test initialization
        assert trading_agent.status == AgentStatus.INITIALIZING
        
        # Test activation
        await trading_agent.activate()
        assert trading_agent.status == AgentStatus.ACTIVE
        
        # Test strategy activation
        trading_agent.active_strategies[TradingStrategy.GRID_TRADING] = True
        assert trading_agent.active_strategies[TradingStrategy.GRID_TRADING]
        
        # Test health check
        health = await trading_agent.health_check()
        assert health["status"] in [AgentStatus.ACTIVE, AgentStatus.DEGRADED]


@pytest.mark.asyncio
async def test_full_trading_workflow():
    """Test complete trading workflow from signal to execution."""
    agent = TradingAlgorithmAgent()
    
    # Mock dependencies
    agent._get_current_price = AsyncMock(return_value=Decimal("50000"))
    agent._get_price_history = AsyncMock(
        return_value=np.array([48000, 49000, 50000, 51000, 52000])
    )
    agent._calculate_rsi = AsyncMock(return_value=72)
    agent._check_volume_surge = AsyncMock(return_value=True)
    agent._calculate_position_size = AsyncMock(return_value=Decimal("0.1"))
    agent._check_risk_limits = AsyncMock(
        return_value={"approved": True, "reason": ""}
    )
    agent._generate_trade_signal = AsyncMock(
        return_value={
            "order_id": "test_order",
            "status": "filled",
            "filled_price": Decimal("50100")
        }
    )
    agent._setup_trailing_stop = AsyncMock()
    
    # 1. Scan for opportunities
    signals = await agent._mcp_momentum_scan(["BTC/USDT"])
    assert len(signals) > 0
    
    # 2. Enter trade
    signal = signals[0]
    trade_result = await agent._mcp_momentum_enter(signal)
    assert trade_result["status"] == "success"
    assert trade_result["order"]["order_id"] == "test_order"
    
    # 3. Monitor and update trailing stop
    agent.positions["BTC/USDT"] = {
        "highest_price": Decimal("50000"),
        "stop_loss": Decimal("47500")
    }
    agent._get_current_price = AsyncMock(return_value=Decimal("52000"))
    agent._update_stop_loss = AsyncMock()
    
    trail_result = await agent._mcp_momentum_trail_stop("BTC/USDT")
    assert trail_result["status"] == "updated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])