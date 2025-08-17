"""
Comprehensive Production Readiness Test Suite
Tests for Enhanced Strands Framework production functionality
"""
import pytest
import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Set test environment
os.environ["ENVIRONMENT"] = "testing"

@pytest.mark.asyncio
class TestProductionStrandsFramework:
    """Test suite for production-ready Strands framework"""
    
    @pytest.fixture
    async def agent(self):
        """Create production agent for testing"""
        from cryptotrading.core.agents.strands_enhanced import EnhancedStrandsAgent
        
        agent = EnhancedStrandsAgent('test-production', 'production-test')
        
        # Initialize production systems (will fall back to simulation)
        try:
            await agent.initialize_production_systems()
        except:
            pass  # Expected in test environment
        
        yield agent
        
        # Cleanup
        if agent.database_manager:
            await agent.database_manager.close()
        if agent.exchange_manager:
            await agent.exchange_manager.close()
    
    async def test_agent_initialization(self, agent):
        """Test basic agent initialization"""
        assert agent.agent_id == 'test-production'
        assert agent.agent_type == 'production-test'
        assert len(agent.tool_registry) > 15  # Should have 15+ tools
        assert len(agent.workflow_registry) >= 2  # Should have workflows
        assert agent.security_manager is not None
        assert agent.production_config is not None
    
    async def test_security_validation(self, agent):
        """Test security and authentication"""
        # Test without authentication (should work for non-sensitive tools)
        result = await agent.execute_tool("get_market_data", {"symbol": "BTC"})
        assert result["success"] is True
        
        # Test sensitive tool without auth (should fail)
        result = await agent.execute_tool("execute_trade", {
            "symbol": "BTC", 
            "side": "buy", 
            "amount": 0.1
        })
        assert result["success"] is False  # No auth provided
        
        # Test with admin authentication
        admin_user = list(agent.security_manager.users.values())[0]  # Admin user
        admin_token = agent.security_manager.create_jwt_token(admin_user)
        
        result = await agent.execute_tool("execute_trade", {
            "symbol": "BTC", 
            "side": "buy", 
            "amount": 0.1
        }, auth_token=admin_token)
        assert result["success"] is True
    
    async def test_input_validation(self, agent):
        """Test comprehensive input validation"""
        # Test invalid symbol
        result = await agent.execute_tool("get_market_data", {"symbol": "invalid_symbol"})
        assert result["success"] is False
        assert "validation failed" in result["error"].lower()
        
        # Test invalid amount
        result = await agent.execute_tool("execute_trade", {
            "symbol": "BTC", 
            "side": "buy", 
            "amount": -1  # Invalid negative amount
        })
        assert result["success"] is False
        
        # Test valid inputs
        result = await agent.execute_tool("get_market_data", {"symbol": "BTC"})
        assert result["success"] is True
    
    async def test_market_data_integration(self, agent):
        """Test market data retrieval and storage"""
        result = await agent.execute_tool("get_market_data", {
            "symbol": "BTC",
            "timeframe": "1h",
            "limit": 100
        })
        
        assert result["success"] is True
        market_data = result["result"]
        
        # Validate market data structure
        assert "symbol" in market_data
        assert "price" in market_data
        assert "volume" in market_data
        assert "timestamp" in market_data
        assert market_data["symbol"] == "BTC"
        assert isinstance(market_data["price"], (int, float))
        assert market_data["price"] > 0
    
    async def test_portfolio_management(self, agent):
        """Test portfolio operations"""
        # Get admin user for authentication
        admin_user = list(agent.security_manager.users.values())[0]
        admin_token = agent.security_manager.create_jwt_token(admin_user)
        
        result = await agent.execute_tool("get_portfolio", {
            "include_history": True
        }, auth_token=admin_token)
        
        assert result["success"] is True
        portfolio = result["result"]
        
        # Validate portfolio structure
        assert "total_value" in portfolio
        assert "positions" in portfolio
        assert "timestamp" in portfolio
        assert isinstance(portfolio["total_value"], (int, float))
        assert portfolio["total_value"] > 0
    
    async def test_trading_execution_with_risk_management(self, agent):
        """Test trading with comprehensive risk management"""
        admin_user = list(agent.security_manager.users.values())[0]
        admin_token = agent.security_manager.create_jwt_token(admin_user)
        
        # Test valid trade
        result = await agent.execute_tool("execute_trade", {
            "symbol": "BTC",
            "side": "buy",
            "amount": 0.01,  # Small amount
            "order_type": "market"
        }, auth_token=admin_token)
        
        assert result["success"] is True
        trade_result = result["result"]
        
        # Validate trade result
        assert "order_id" in trade_result
        assert "executed_price" in trade_result
        assert "fees" in trade_result
        assert trade_result["symbol"] == "BTC"
        assert trade_result["side"] == "buy"
        
        # Test risk limit enforcement - try large trade
        result = await agent.execute_tool("execute_trade", {
            "symbol": "BTC",
            "side": "buy",
            "amount": 100,  # Very large amount
            "order_type": "market"
        }, auth_token=admin_token)
        
        # Should fail due to position size limits
        assert result["success"] is False
        assert "position size limit" in result["error"].lower()
    
    async def test_workflow_execution(self, agent):
        """Test workflow orchestration"""
        result = await agent.process_workflow("market_analysis")
        
        assert result["success"] is True
        assert "workflow_id" in result
        assert "execution_id" in result
        assert "results" in result
        assert result["workflow_id"] == "market_analysis"
    
    async def test_comprehensive_risk_assessment(self, agent):
        """Test advanced risk assessment"""
        admin_user = list(agent.security_manager.users.values())[0]
        admin_token = agent.security_manager.create_jwt_token(admin_user)
        
        result = await agent.execute_tool("risk_assessment_comprehensive", {
            "include_stress_test": True
        }, auth_token=admin_token)
        
        assert result["success"] is True
        risk_data = result["result"]
        
        # Validate risk assessment structure
        assert "basic_metrics" in risk_data
        assert "portfolio_concentration" in risk_data
        assert "overall_risk_score" in risk_data
        assert "risk_level" in risk_data
        assert "stress_test" in risk_data  # Should include stress test
        
        # Validate risk score is reasonable
        assert 0 <= risk_data["overall_risk_score"] <= 100
    
    async def test_system_monitoring_and_health(self, agent):
        """Test system health monitoring"""
        result = await agent.execute_tool("system_health_monitor")
        
        assert result["success"] is True
        health_data = result["result"]
        
        # Validate health monitoring
        assert "overall_health_score" in health_data
        assert "health_status" in health_data
        assert "system_metrics" in health_data
        assert health_data["health_status"] in ["healthy", "degraded", "unhealthy"]
        assert 0 <= health_data["overall_health_score"] <= 100
    
    async def test_resource_management(self, agent):
        """Test resource cleanup and memory management"""
        # Fill up context with many executions
        for i in range(50):
            await agent.execute_tool("get_market_data", {"symbol": "BTC"})
        
        initial_count = len(agent.context.tool_executions)
        
        # Trigger cleanup
        await agent.cleanup_resources()
        
        # Context should be cleaned up if over limit
        assert len(agent.context.tool_executions) <= agent.max_context_history
    
    async def test_audit_logging(self, agent):
        """Test comprehensive audit logging"""
        admin_user = list(agent.security_manager.users.values())[0]
        admin_token = agent.security_manager.create_jwt_token(admin_user)
        
        # Execute a trade to generate audit logs
        await agent.execute_tool("execute_trade", {
            "symbol": "BTC",
            "side": "buy",
            "amount": 0.01
        }, auth_token=admin_token)
        
        # Check audit logs
        audit_events = agent.security_manager.get_audit_events(
            user_id=admin_user.user_id,
            limit=10
        )
        
        assert len(audit_events) > 0
        
        # Find trade execution event
        trade_events = [e for e in audit_events if e.event_type.value == "trade_execute"]
        assert len(trade_events) > 0
        
        trade_event = trade_events[0]
        assert trade_event.user_id == admin_user.user_id
        assert trade_event.resource == "trading"
        assert "BTC" in trade_event.metadata.get("symbol", "")
    
    async def test_configuration_management(self, agent):
        """Test production configuration"""
        config = agent.production_config
        
        # Validate configuration structure
        assert hasattr(config, "database")
        assert hasattr(config, "exchange")
        assert hasattr(config, "risk")
        assert hasattr(config, "security")
        assert hasattr(config, "strands")
        
        # Validate risk configuration
        assert 0 < config.risk.max_portfolio_risk <= 0.1
        assert 0 < config.risk.max_daily_loss <= 0.2
        assert config.risk.position_size_limit > 0
        
        # Validate security configuration
        assert config.security.enable_auth in [True, False]
        assert len(config.security.jwt_secret) >= 32
    
    async def test_performance_metrics(self, agent):
        """Test performance monitoring"""
        # Execute several tools to generate metrics
        for _ in range(5):
            await agent.execute_tool("get_market_data", {"symbol": "BTC"})
        
        metrics = await agent.get_strands_metrics()
        
        # Validate metrics structure
        assert "observer_metrics" in metrics
        assert "tool_registry_size" in metrics
        assert "context_stats" in metrics
        
        observer_metrics = metrics["observer_metrics"]
        assert "tools_executed" in observer_metrics
        assert "average_response_time" in observer_metrics
        assert observer_metrics["tools_executed"] >= 5
    
    async def test_error_handling_and_recovery(self, agent):
        """Test error handling and circuit breaker functionality"""
        # Test with invalid tool
        result = await agent.execute_tool("nonexistent_tool", {})
        assert result["success"] is False
        assert "not found in registry" in result["error"]
        
        # Test parameter validation error
        result = await agent.execute_tool("get_market_data", {"symbol": ""})
        assert result["success"] is False
        
        # Verify agent is still functional after errors
        result = await agent.execute_tool("get_market_data", {"symbol": "BTC"})
        assert result["success"] is True

def run_production_tests():
    """Run all production tests"""
    pytest.main([__file__, "-v", "--tb=short"])

if __name__ == "__main__":
    run_production_tests()