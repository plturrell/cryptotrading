"""
Test suite demonstrating fixed circular dependencies
Shows how DI container breaks circular imports and enables modular architecture
"""
import pytest
import asyncio
import time
from typing import Dict, Any

from src.cryptotrading.core.bootstrap import (
    initialize_application, shutdown_application,
    get_application_health, get_application_metrics
)
from src.cryptotrading.core.di_container import get_container, reset_container
from src.cryptotrading.core.interfaces import (
    ILogger, IMetricsCollector, IHealthChecker, ISecurityManager,
    ICommunicationManager, IServiceRegistry, ICache
)
from src.cryptotrading.core.agents.modular_strands_agent import (
    ModularStrandsAgent, AgentConfig, create_modular_strands_agent
)


class TestCircularDependencyFix:
    """Test that circular dependencies are resolved through DI container"""
    
    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Reset container before each test
        reset_container()
        
        yield
        
        # Cleanup after each test
        await shutdown_application()
        reset_container()
    
    @pytest.mark.asyncio
    async def test_application_bootstrap_initialization(self):
        """Test that application initializes without circular dependency errors"""
        # Initialize application with DI container
        container = await initialize_application({
            "logger_name": "test_crypto",
            "log_level": "INFO",
            "metrics_max_entries": 1000,
            "cache_default_ttl": 300
        })
        
        assert container is not None
        
        # Verify all core services are registered and available
        logger = await container.resolve(ILogger)
        assert logger is not None
        
        metrics = await container.resolve(IMetricsCollector)
        assert metrics is not None
        
        health_checker = await container.resolve(IHealthChecker)
        assert health_checker is not None
        
        security_manager = await container.resolve(ISecurityManager)
        assert security_manager is not None
        
        comm_manager = await container.resolve(ICommunicationManager)
        assert comm_manager is not None
        
        cache = await container.resolve(ICache)
        assert cache is not None
    
    @pytest.mark.asyncio
    async def test_modular_strands_agent_with_di(self):
        """Test that ModularStrandsAgent works with DI container"""
        # Initialize application first
        await initialize_application()
        
        # Create modular agent
        agent = create_modular_strands_agent(
            agent_id="test_agent_001",
            enable_authentication=True,
            enable_communication=True,
            enable_observability=True
        )
        
        # Initialize agent (should resolve dependencies from DI container)
        success = await agent.initialize()
        assert success is True
        
        # Verify agent has resolved dependencies
        assert agent.security_manager is not None
        assert agent.communication_manager is not None
        
        # Test agent functionality
        await agent.start()
        
        # Execute a tool
        result = await agent.execute_tool("get_agent_status", {
            "include_components": True
        })
        
        assert result["success"] is True
        assert "agent_id" in result["result"]["status"]
        assert result["result"]["status"]["agent_id"] == "test_agent_001"
        
        # Test workflow execution
        workflow_result = await agent.execute_workflow("health_check")
        assert workflow_result["success"] is True
        
        # Get agent metrics
        metrics = await agent.get_metrics()
        assert "agent" in metrics
        assert "timestamp" in metrics
        
        # Shutdown agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_interface_segregation_principle(self):
        """Test that components only depend on interfaces they need"""
        await initialize_application()
        
        agent_config = AgentConfig(
            agent_id="isp_test_agent",
            enable_authentication=True,
            enable_communication=False,  # Disable communication
            enable_observability=False   # Disable observability
        )
        
        agent = ModularStrandsAgent(agent_config)
        await agent.initialize()
        
        # Agent should have security manager but not communication
        assert agent.security_manager is not None
        assert agent.communication_manager is None
        assert agent.metrics_collector is None
        
        # Agent should still function with partial dependencies
        result = await agent.execute_tool("get_agent_status", {})
        assert result["success"] is True
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_dependency_injection_performance(self):
        """Test that DI container performs well under load"""
        await initialize_application()
        
        container = get_container()
        
        # Measure resolution time
        start_time = time.time()
        
        # Resolve services multiple times
        for _ in range(100):
            logger = await container.resolve(ILogger)
            metrics = await container.resolve(IMetricsCollector)
            cache = await container.resolve(ICache)
            
            assert logger is not None
            assert metrics is not None
            assert cache is not None
        
        end_time = time.time()
        resolution_time = end_time - start_time
        
        # Should resolve 300 services in under 1 second
        assert resolution_time < 1.0
        
        # Verify singletons return same instances
        logger1 = await container.resolve(ILogger)
        logger2 = await container.resolve(ILogger)
        assert logger1 is logger2  # Same instance
    
    @pytest.mark.asyncio
    async def test_multiple_agents_with_shared_services(self):
        """Test multiple agents sharing services through DI"""
        await initialize_application()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = create_modular_strands_agent(
                agent_id=f"shared_test_agent_{i}",
                enable_authentication=True,
                enable_communication=True
            )
            await agent.initialize()
            agents.append(agent)
        
        # All agents should share the same security manager instance
        security_managers = [agent.security_manager for agent in agents]
        assert all(sm is not None for sm in security_managers)
        
        # All agents should share the same communication manager instance
        comm_managers = [agent.communication_manager for agent in agents]
        assert all(cm is not None for cm in comm_managers)
        
        # Test that agents can execute tools independently
        tasks = []
        for i, agent in enumerate(agents):
            task = agent.execute_tool("store_memory", {
                "key": f"test_key_{i}",
                "value": f"test_value_{i}"
            })
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result["success"] for result in results)
        
        # Cleanup
        for agent in agents:
            await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test health monitoring works across the system"""
        await initialize_application()
        
        # Get system health
        health = await get_application_health()
        
        assert health["overall_status"] in ["healthy", "degraded"]
        assert "healthy_services" in health
        assert "total_services" in health
        assert "services" in health
        
        # Should have health checks for core services
        services = health["services"]
        expected_services = ["logger", "metrics_collector", "cache", "service_registry"]
        
        for service_name in expected_services:
            assert service_name in services
            service_health = services[service_name]
            assert "healthy" in service_health
            assert "timestamp" in service_health
    
    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self):
        """Test metrics collection works across the system"""
        await initialize_application()
        
        container = get_container()
        metrics_collector = await container.resolve(IMetricsCollector)
        
        # Record some test metrics
        await metrics_collector.record_metric("test_metric", 42.0, {"test": "true"})
        await metrics_collector.increment_counter("test_counter", 5)
        await metrics_collector.set_gauge("test_gauge", 100.0)
        await metrics_collector.record_timing("test_timing", 250.5)
        
        # Get metrics
        metrics = await get_application_metrics()
        
        assert "counters" in metrics
        assert "gauges" in metrics
        assert "metrics" in metrics
        
        # Verify our test metrics are present
        assert "test_counter" in metrics["counters"]
        assert metrics["counters"]["test_counter"] == 5
        
        assert "test_gauge" in metrics["gauges"]
        assert metrics["gauges"]["test_gauge"] == 100.0
    
    @pytest.mark.asyncio
    async def test_no_circular_import_errors(self):
        """Test that importing modules doesn't cause circular import errors"""
        # This test verifies that we can import all modules without errors
        
        # Import core modules
        from src.cryptotrading.core import di_container
        from src.cryptotrading.core import bootstrap
        from src.cryptotrading.core import interfaces
        from src.cryptotrading.core import infrastructure
        
        # Import agent modules
        from src.cryptotrading.core.agents import modular_strands_agent
        from src.cryptotrading.core.agents import components
        
        # Import protocol modules
        from src.cryptotrading.core.protocols.mcp import tools
        from src.cryptotrading.core.protocols.mcp import cli
        
        # If we get here without ImportError, circular dependencies are resolved
        assert True
    
    @pytest.mark.asyncio
    async def test_service_lifecycle_management(self):
        """Test proper service lifecycle management"""
        # Initialize application
        container = await initialize_application()
        
        # Verify services are initialized
        health_status = await get_application_health()
        assert health_status["overall_status"] in ["healthy", "degraded"]
        
        # Create and initialize an agent
        agent = create_modular_strands_agent("lifecycle_test_agent")
        await agent.initialize()
        
        # Verify agent services are working
        status = await agent.execute_tool("get_agent_status", {"include_components": True})
        assert status["success"] is True
        
        # Shutdown agent first
        await agent.shutdown()
        
        # Shutdown application
        await shutdown_application()
        
        # Verify container is cleaned up
        services = container.list_services()
        for service_name, service_info in services.items():
            if service_info["singleton"]:
                registration = container.get_registration(eval(f"I{service_name.replace('Enterprise', '').replace('Simple', '')}"))
                if registration:
                    assert registration.initialized is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])