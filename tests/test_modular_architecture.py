"""
Test Suite for Modular Architecture Refactoring
Demonstrates the improvement from god object to focused components
"""
import pytest
import asyncio
import time
from datetime import datetime
from typing import Dict, Any

from src.cryptotrading.core.agents.modular_strands_agent import (
    ModularStrandsAgent,
    AgentConfig,
    create_modular_strands_agent
)
from src.cryptotrading.core.agents.components.tool_manager import (
    ToolManager,
    ToolDefinition,
    ToolPriority
)
from src.cryptotrading.core.agents.components.workflow_engine import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowStep
)
from src.cryptotrading.core.agents.components.context_manager import (
    ContextManager,
    StrandsContext
)
from src.cryptotrading.core.agents.secure_code_sandbox import SecurityLevel


class TestModularArchitecture:
    """Test the modular architecture benefits"""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration"""
        return AgentConfig(
            agent_id="test_modular_agent",
            agent_type="test",
            max_concurrent_tools=5,
            max_concurrent_workflows=2,
            max_contexts=10,
            security_level=SecurityLevel.NORMAL,
            enable_authentication=False,  # Disable for testing
            enable_code_execution=True,
            enable_communication=False,  # Not implemented yet
            enable_database=False,       # Not implemented yet
            enable_observability=False   # Not implemented yet
        )
    
    @pytest.fixture
    async def modular_agent(self, agent_config):
        """Create and initialize modular agent"""
        agent = ModularStrandsAgent(agent_config)
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_component_separation(self, modular_agent):
        """Test that components are properly separated with single responsibilities"""
        agent = modular_agent
        
        # Verify each component exists and has focused responsibility
        assert agent.tool_manager is not None
        assert agent.workflow_engine is not None
        assert agent.context_manager is not None
        
        # Tool manager should only handle tools
        assert hasattr(agent.tool_manager, 'register_tool')
        assert hasattr(agent.tool_manager, 'execute_tool')
        assert not hasattr(agent.tool_manager, 'execute_workflow')  # Not tool manager's job
        
        # Workflow engine should only handle workflows
        assert hasattr(agent.workflow_engine, 'register_workflow')
        assert hasattr(agent.workflow_engine, 'execute_workflow')
        assert not hasattr(agent.workflow_engine, 'store_memory')  # Not workflow engine's job
        
        # Context manager should only handle context/memory
        assert hasattr(agent.context_manager, 'create_context')
        assert hasattr(agent.context_manager, 'store_memory')
        assert not hasattr(agent.context_manager, 'execute_tool')  # Not context manager's job
    
    @pytest.mark.asyncio
    async def test_tool_manager_isolation(self, modular_agent):
        """Test tool manager operates independently"""
        tool_manager = modular_agent.tool_manager
        
        # Register a test tool
        async def test_calculation(x: int, y: int) -> int:
            return x + y
        
        success = await tool_manager.register_tool(
            name="test_calc",
            handler=test_calculation,
            description="Test calculation tool",
            category="math"
        )
        assert success is True
        
        # Execute tool independently
        execution = await tool_manager.execute_tool("test_calc", {"x": 5, "y": 3})
        
        assert execution.status.value == "completed"
        assert execution.result == 8
        assert execution.tool_name == "test_calc"
        
        # Verify tool metrics
        metrics = tool_manager.get_tool_metrics("test_calc")
        assert metrics["total_executions"] == 1
        assert metrics["successful_executions"] == 1
        assert metrics["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_workflow_engine_isolation(self, modular_agent):
        """Test workflow engine operates independently"""
        workflow_engine = modular_agent.workflow_engine
        tool_manager = modular_agent.tool_manager
        
        # Register a test tool first
        async def multiply_tool(value: int, factor: int = 2) -> int:
            return value * factor
        
        await tool_manager.register_tool(
            name="multiply",
            handler=multiply_tool,
            description="Multiply values"
        )
        
        # Create and register workflow
        workflow = WorkflowDefinition(
            id="test_workflow",
            name="Test Calculation Workflow",
            description="Test workflow for calculations",
            steps=[
                WorkflowStep(
                    id="step1",
                    name="Multiply by 2",
                    tool_name="multiply",
                    parameters={"value": 5, "factor": 2}
                ),
                WorkflowStep(
                    id="step2", 
                    name="Multiply result by 3",
                    tool_name="multiply",
                    parameters={"value": "${step_step1_result}", "factor": 3},
                    dependencies=["step1"]
                )
            ]
        )
        
        await workflow_engine.register_workflow(workflow)
        
        # Execute workflow independently
        execution = await workflow_engine.execute_workflow("test_workflow")
        
        assert execution.status.value == "completed"
        assert len(execution.step_executions) == 2
        
        # Verify workflow metrics
        metrics = workflow_engine.get_metrics()
        assert metrics["successful_workflows"] == 1
    
    @pytest.mark.asyncio
    async def test_context_manager_isolation(self, modular_agent):
        """Test context manager operates independently"""
        context_manager = modular_agent.context_manager
        
        # Create context
        context = await context_manager.create_context("test_session_123")
        
        assert context.session_id == "test_session_123"
        assert context.agent_id == modular_agent.config.agent_id
        assert len(context.conversation_history) == 0
        
        # Add conversation entries
        await context_manager.add_conversation_entry(
            "test_session_123", "user", "Hello, how are you?"
        )
        await context_manager.add_conversation_entry(
            "test_session_123", "assistant", "I'm doing well, thank you!"
        )
        
        # Verify context was updated
        updated_context = await context_manager.get_context("test_session_123")
        assert len(updated_context.conversation_history) == 2
        
        # Test memory storage
        await context_manager.store_memory("test_session_123", "user_name", "Alice")
        stored_value = await context_manager.retrieve_memory("test_session_123", "user_name")
        assert stored_value == "Alice"
        
        # Verify context stats
        stats = await context_manager.get_context_stats()
        assert stats["active_contexts"] == 1
        assert stats["active_sessions"] == 1
    
    @pytest.mark.asyncio
    async def test_agent_integration(self, modular_agent):
        """Test that components work together through agent interface"""
        agent = modular_agent
        session_id = "integration_test_session"
        
        # Test complete workflow: tool registration → execution → context tracking
        
        # 1. Register custom tool
        async def data_processor(data: str, operation: str = "upper") -> str:
            if operation == "upper":
                return data.upper()
            elif operation == "lower":
                return data.lower()
            else:
                return data
        
        await agent.tool_manager.register_tool(
            name="process_data",
            handler=data_processor,
            description="Process text data"
        )
        
        # 2. Execute tool through agent interface
        result = await agent.execute_tool(
            tool_name="process_data",
            parameters={"data": "Hello World", "operation": "upper"},
            session_id=session_id
        )
        
        assert result["success"] is True
        assert result["result"] == "HELLO WORLD"
        
        # 3. Verify context was updated
        context_info = await agent.get_context(session_id)
        assert context_info is not None
        assert context_info["tool_executions"] == 1
        
        # 4. Test memory through agent interface
        await agent.store_memory("processed_data", "HELLO WORLD", session_id)
        retrieved = await agent.retrieve_memory("processed_data", session_id)
        assert retrieved == "HELLO WORLD"
    
    @pytest.mark.asyncio
    async def test_secure_code_execution_integration(self, modular_agent):
        """Test secure code execution through modular architecture"""
        agent = modular_agent
        session_id = "code_test_session"
        
        # Test safe code execution
        safe_code = """
x = 10
y = 20
result = x + y
"""
        
        result = await agent.execute_tool(
            tool_name="execute_code",
            parameters={"code": safe_code, "language": "python"},
            session_id=session_id
        )
        
        assert result["success"] is True
        assert len(result["security_violations"]) == 0
        
        # Test that dangerous code is blocked
        dangerous_code = """
import os
os.system('rm -rf /')
"""
        
        result = await agent.execute_tool(
            tool_name="execute_code", 
            parameters={"code": dangerous_code, "language": "python"},
            session_id=session_id
        )
        
        assert result["success"] is False
        assert len(result["security_violations"]) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, modular_agent):
        """Test that components handle concurrent operations properly"""
        agent = modular_agent
        
        # Register a slow tool for concurrency testing
        async def slow_tool(delay: float = 0.1) -> str:
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"
        
        await agent.tool_manager.register_tool(
            name="slow_tool",
            handler=slow_tool,
            description="Tool with artificial delay"
        )
        
        # Execute multiple tools concurrently
        tasks = []
        for i in range(5):
            task = agent.execute_tool(
                tool_name="slow_tool",
                parameters={"delay": 0.1},
                session_id=f"concurrent_session_{i}"
            )
            tasks.append(task)
        
        # Wait for all to complete
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all succeeded
        assert all(result["success"] for result in results)
        
        # Verify they ran concurrently (should be much faster than sequential)
        assert total_time < 0.3  # Should be less than 0.5s if truly concurrent
        
        # Verify tool manager tracked all executions
        metrics = agent.tool_manager.get_tool_metrics("slow_tool")
        assert metrics["total_executions"] == 5
        assert metrics["successful_executions"] == 5
    
    @pytest.mark.asyncio
    async def test_component_failure_isolation(self, modular_agent):
        """Test that component failures don't crash the entire agent"""
        agent = modular_agent
        
        # Test tool execution failure doesn't affect context management
        async def failing_tool() -> str:
            raise Exception("Intentional failure for testing")
        
        await agent.tool_manager.register_tool(
            name="failing_tool",
            handler=failing_tool,
            description="Tool that always fails"
        )
        
        # Execute failing tool
        result = await agent.execute_tool(
            tool_name="failing_tool",
            parameters={},
            session_id="failure_test_session"
        )
        
        assert result["success"] is False
        assert "Intentional failure" in result["error"]
        
        # Verify context manager still works
        context_info = await agent.get_context("failure_test_session")
        assert context_info is not None
        assert context_info["tool_executions"] == 1  # Failed execution still recorded
        
        # Verify other tools still work
        result = await agent.execute_tool(
            tool_name="get_agent_status",
            parameters={},
            session_id="failure_test_session"
        )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_memory_management_separation(self, modular_agent):
        """Test that memory management is properly separated"""
        context_manager = modular_agent.context_manager
        
        # Test different memory types are handled separately
        session_id = "memory_test_session"
        
        # Short-term memory
        await context_manager.store_memory(session_id, "temp_data", "temporary", "short_term")
        
        # Long-term memory  
        await context_manager.store_memory(session_id, "perm_data", "permanent", "long_term")
        
        # Working memory
        await context_manager.store_memory(session_id, "work_data", "working", "working")
        
        # Verify all can be retrieved
        temp_data = await context_manager.retrieve_memory(session_id, "temp_data")
        perm_data = await context_manager.retrieve_memory(session_id, "perm_data")
        work_data = await context_manager.retrieve_memory(session_id, "work_data")
        
        assert temp_data == "temporary"
        assert perm_data == "permanent"
        assert work_data == "working"
        
        # Verify memory stats are tracked separately
        stats = await context_manager.get_memory_stats()
        assert "short_term" in stats
        assert "long_term" in stats
        assert "working_memory" in stats


class TestArchitecturalBenefits:
    """Test architectural benefits of modular design"""
    
    def test_component_reusability(self):
        """Test that components can be reused independently"""
        # Create tool manager independently
        tool_manager = ToolManager(max_concurrent_executions=5)
        assert tool_manager is not None
        
        # Create context manager independently
        context_manager = ContextManager(agent_id="test_agent", max_contexts=10)
        assert context_manager is not None
        
        # Components should be usable without full agent
        assert hasattr(tool_manager, 'register_tool')
        assert hasattr(context_manager, 'create_context')
    
    @pytest.mark.asyncio
    async def test_component_testability(self):
        """Test that components can be tested in isolation"""
        # Tool manager can be tested independently
        tool_manager = ToolManager(max_concurrent_executions=2)
        
        def simple_add(x: int, y: int) -> int:
            return x + y
        
        await tool_manager.register_tool(
            name="add",
            handler=simple_add,
            description="Add two numbers"
        )
        
        execution = await tool_manager.execute_tool("add", {"x": 2, "y": 3})
        assert execution.result == 5
        
        # Context manager can be tested independently
        context_manager = ContextManager(agent_id="test", max_contexts=5)
        context = await context_manager.create_context("test_session")
        assert context.session_id == "test_session"
        
        # Cleanup
        await tool_manager.shutdown()
        await context_manager.shutdown()
    
    def test_single_responsibility_principle(self):
        """Test that each component follows Single Responsibility Principle"""
        config = AgentConfig(agent_id="test_agent")
        agent = ModularStrandsAgent(config)
        
        # Each component should have a single, well-defined responsibility
        
        # Tool manager: Only tool-related operations
        tool_methods = [m for m in dir(agent.tool_manager) if not m.startswith('_')]
        tool_related = ['register_tool', 'execute_tool', 'get_tool_metrics', 'shutdown']
        assert all(any(related in method for related in ['tool', 'execution', 'shutdown']) 
                  for method in tool_methods if not method.startswith('registry'))
        
        # Context manager: Only context/memory operations
        context_methods = [m for m in dir(agent.context_manager) if not m.startswith('_')]
        context_related = ['context', 'memory', 'session', 'shutdown']
        assert all(any(related in method.lower() for related in context_related)
                  for method in context_methods if method not in ['memory_manager'])
    
    def test_configuration_flexibility(self):
        """Test that modular design allows flexible configuration"""
        # Agent with minimal components
        minimal_config = AgentConfig(
            agent_id="minimal_agent",
            enable_code_execution=False,
            enable_communication=False,
            enable_database=False,
            enable_observability=False
        )
        minimal_agent = ModularStrandsAgent(minimal_config)
        
        assert minimal_agent.code_executor is None
        assert minimal_agent.communication_manager is None
        assert minimal_agent.database_manager is None
        
        # Agent with all components
        full_config = AgentConfig(
            agent_id="full_agent",
            enable_code_execution=True,
            enable_communication=True,  # Will fail to import but gracefully handle
            enable_database=True,       # Will fail to import but gracefully handle
            enable_observability=True   # Will fail to import but gracefully handle
        )
        full_agent = ModularStrandsAgent(full_config)
        
        assert full_agent.code_executor is not None
        # Other components will be None due to missing implementations, but no crash


def test_factory_function():
    """Test the factory function for easy agent creation"""
    agent = create_modular_strands_agent(
        agent_id="factory_test_agent",
        max_concurrent_tools=15,
        security_level=SecurityLevel.STRICT
    )
    
    assert agent.config.agent_id == "factory_test_agent"
    assert agent.config.max_concurrent_tools == 15
    assert agent.config.security_level == SecurityLevel.STRICT


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])