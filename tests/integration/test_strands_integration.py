"""
Integration tests for Strands framework using real workflow execution
Tests the complete Strands system with actual tool execution and workflow orchestration
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

from cryptotrading.core.agents.strands import (
    StrandsAgent, WorkflowContext, WorkflowStatus, ToolStatus,
    ToolExecutionResult, WorkflowEngine, ToolRegistry
)
from cryptotrading.core.config.production_config import StrandsConfig
from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase
from cryptotrading.core.config.production_config import DatabaseConfig

@pytest.fixture
async def test_database():
    """Create test database for Strands operations"""
    db_config = DatabaseConfig(
        host="localhost",
        database=":memory:",
        connection_pool_size=5
    )
    
    db = UnifiedDatabase(db_config)
    await db.initialize()
    
    # Add test data
    await db.add_position("BTC-USD", 2.0, 45000.0)
    await db.add_position("ETH-USD", 10.0, 2800.0)
    
    # Add test trades
    await db.log_trade({
        "symbol": "BTC-USD",
        "side": "buy", 
        "amount": 1.0,
        "price": 44000.0,
        "status": "filled"
    })
    
    yield db
    
    await db.close()

@pytest.fixture
async def strands_config():
    """Create test Strands configuration"""
    return StrandsConfig(
        max_concurrent_workflows=5,
        workflow_timeout_seconds=30,
        tool_timeout_seconds=10,
        max_retry_attempts=2,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=30,
        worker_pool_size=4,
        event_bus_capacity=100,
        enable_telemetry=True,
        enable_distributed=False
    )

@pytest.fixture
async def strands_agent(strands_config, test_database):
    """Create Strands agent for testing"""
    agent = StrandsAgent(
        agent_id="test_agent",
        agent_type="integration_test",
        capabilities=["memory", "computation", "analysis"],
        model_provider="test",
        config=strands_config
    )
    
    # Add test database connection
    agent.database_connection = test_database
    
    # Register custom test tools
    agent.register_tool(
        "test_calculation",
        lambda x, y: {"result": x + y, "operation": "addition"},
        {"description": "Test calculation tool", "category": "math"}
    )
    
    agent.register_tool(
        "test_data_processing",
        lambda data: {"processed": [item * 2 for item in data], "count": len(data)},
        {"description": "Test data processing tool", "category": "data"}
    )
    
    agent.register_tool(
        "test_error_tool", 
        lambda: (_ for _ in ()).throw(Exception("Test error")),
        {"description": "Tool that always fails", "category": "test"}
    )
    
    agent.register_tool(
        "test_async_tool",
        lambda: asyncio.sleep(0.1),
        {"description": "Async test tool", "category": "async"}
    )
    
    yield agent
    
    await agent.shutdown()

class TestStrandsWorkflowExecution:
    """Test workflow execution capabilities"""
    
    @pytest.mark.asyncio
    async def test_simple_tool_execution(self, strands_agent):
        """Test basic tool execution"""
        result = await strands_agent.execute_tool(
            "test_calculation", 
            {"x": 10, "y": 20}
        )
        
        assert result.status == ToolStatus.SUCCESS
        assert result.result["result"] == 30
        assert result.result["operation"] == "addition"
        assert result.execution_time_ms > 0
        assert result.metadata["execution_id"]
        
    @pytest.mark.asyncio
    async def test_memory_tools(self, strands_agent):
        """Test memory storage and retrieval tools"""
        # Store in memory
        store_result = await strands_agent.execute_tool(
            "store_memory",
            {"key": "test_data", "value": {"important": "information"}}
        )
        
        assert store_result.status == ToolStatus.SUCCESS
        
        # Retrieve from memory
        get_result = await strands_agent.execute_tool(
            "get_memory",
            {"key": "test_data"}
        )
        
        assert get_result.status == ToolStatus.SUCCESS
        assert get_result.result["important"] == "information"
        
    @pytest.mark.asyncio
    async def test_code_execution_tool(self, strands_agent):
        """Test secure code execution tool"""
        # Safe Python code
        result = await strands_agent.execute_tool(
            "execute_code",
            {
                "code": "print('Hello, World!')\nresult = 2 + 3\nprint(f'Result: {result}')",
                "language": "python"
            }
        )
        
        assert result.status == ToolStatus.SUCCESS
        assert "Hello, World!" in result.result["output"]
        assert "Result: 5" in result.result["output"]
        
        # Unsafe code should be rejected
        unsafe_result = await strands_agent.execute_tool(
            "execute_code",
            {
                "code": "import os; os.system('ls')",
                "language": "python"
            }
        )
        
        assert unsafe_result.status == ToolStatus.FAILURE
        assert "unsafe operation" in unsafe_result.error
        
    @pytest.mark.asyncio
    async def test_tool_timeout(self, strands_agent):
        """Test tool execution timeout"""
        # Register slow tool
        async def slow_tool():
            await asyncio.sleep(15)  # Longer than timeout
            return "done"
            
        strands_agent.register_tool("slow_tool", slow_tool)
        
        result = await strands_agent.execute_tool("slow_tool", {})
        
        assert result.status == ToolStatus.TIMEOUT
        assert "timed out" in result.error
        
    @pytest.mark.asyncio
    async def test_tool_error_handling(self, strands_agent):
        """Test tool error handling"""
        result = await strands_agent.execute_tool("test_error_tool", {})
        
        assert result.status == ToolStatus.FAILURE
        assert "Test error" in result.error
        assert result.execution_time_ms > 0
        
    @pytest.mark.asyncio
    async def test_simple_workflow(self, strands_agent):
        """Test simple workflow execution"""
        workflow_id = str(uuid.uuid4())
        
        inputs = {
            "tools": [
                {
                    "name": "test_calculation",
                    "parameters": {"x": 5, "y": 3}
                },
                {
                    "name": "test_data_processing", 
                    "parameters": {"data": [1, 2, 3, 4]}
                }
            ]
        }
        
        result = await strands_agent.process_workflow(workflow_id, inputs)
        
        assert result["status"] == WorkflowStatus.COMPLETED
        assert result["workflow_id"] == workflow_id
        assert "tool_results" in result["outputs"]
        assert len(result["outputs"]["tool_results"]) == 2
        
        # Check individual tool results
        calc_result = result["outputs"]["tool_results"][0]
        assert calc_result["status"] == ToolStatus.SUCCESS
        assert calc_result["result"]["result"] == 8
        
        data_result = result["outputs"]["tool_results"][1]
        assert data_result["status"] == ToolStatus.SUCCESS
        assert data_result["result"]["processed"] == [2, 4, 6, 8]
        
    @pytest.mark.asyncio
    async def test_defined_workflow_sequential(self, strands_agent):
        """Test defined workflow with sequential execution"""
        workflow_id = str(uuid.uuid4())
        
        inputs = {
            "workflow_definition": {
                "steps": [
                    {
                        "id": "step1",
                        "type": "tool",
                        "tool": "test_calculation",
                        "parameters": {"x": 10, "y": 5}
                    },
                    {
                        "id": "step2",
                        "type": "tool",
                        "tool": "test_data_processing",
                        "parameters": {"data": "$step1.result.result"},
                        "dependencies": ["step1"]
                    }
                ],
                "parallel": False
            }
        }
        
        result = await strands_agent.process_workflow(workflow_id, inputs)
        
        assert result["status"] == WorkflowStatus.COMPLETED
        assert "workflow_results" in result["outputs"]
        
        # Verify step execution order
        workflow_results = result["outputs"]["workflow_results"]
        assert "step1" in workflow_results
        assert "step2" in workflow_results
        
        # step2 should use result from step1
        step1_result = workflow_results["step1"]["result"]
        assert step1_result == 15  # 10 + 5
        
    @pytest.mark.asyncio
    async def test_defined_workflow_parallel(self, strands_agent):
        """Test defined workflow with parallel execution"""
        workflow_id = str(uuid.uuid4())
        
        inputs = {
            "workflow_definition": {
                "steps": [
                    {
                        "id": "parallel1",
                        "type": "tool",
                        "tool": "test_calculation",
                        "parameters": {"x": 1, "y": 2}
                    },
                    {
                        "id": "parallel2", 
                        "type": "tool",
                        "tool": "test_calculation",
                        "parameters": {"x": 3, "y": 4}
                    },
                    {
                        "id": "final",
                        "type": "tool", 
                        "tool": "test_data_processing",
                        "parameters": {"data": [1, 2, 3]},
                        "dependencies": ["parallel1", "parallel2"]
                    }
                ],
                "parallel": True
            }
        }
        
        start_time = datetime.utcnow()
        result = await strands_agent.process_workflow(workflow_id, inputs)
        end_time = datetime.utcnow()
        
        assert result["status"] == WorkflowStatus.COMPLETED
        
        # Parallel execution should be faster than sequential
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 1.0  # Should complete quickly
        
        # Verify all steps completed
        workflow_results = result["outputs"]["workflow_results"]
        assert len(workflow_results) == 3
        assert workflow_results["parallel1"]["result"] == 3
        assert workflow_results["parallel2"]["result"] == 7
        
    @pytest.mark.asyncio
    async def test_workflow_with_conditions(self, strands_agent):
        """Test workflow with conditional execution"""
        workflow_id = str(uuid.uuid4())
        
        inputs = {
            "workflow_definition": {
                "steps": [
                    {
                        "id": "calculation",
                        "type": "tool",
                        "tool": "test_calculation",
                        "parameters": {"x": 10, "y": 5}
                    },
                    {
                        "id": "conditional_step",
                        "type": "tool",
                        "tool": "test_data_processing",
                        "parameters": {"data": [1, 2, 3]},
                        "condition": {
                            "type": "simple",
                            "left": "$calculation.result.result",
                            "operator": ">",
                            "right": 10
                        },
                        "dependencies": ["calculation"]
                    }
                ]
            }
        }
        
        result = await strands_agent.process_workflow(workflow_id, inputs)
        
        assert result["status"] == WorkflowStatus.COMPLETED
        
        # Conditional step should execute since 15 > 10
        workflow_results = result["outputs"]["workflow_results"]
        assert "conditional_step" in workflow_results
        assert workflow_results["conditional_step"]["processed"] == [2, 4, 6]
        
    @pytest.mark.asyncio
    async def test_dag_workflow(self, strands_agent):
        """Test DAG (Directed Acyclic Graph) workflow execution"""
        workflow_id = str(uuid.uuid4())
        
        inputs = {
            "dag": {
                "nodes": {
                    "node1": {
                        "type": "tool",
                        "tool": "test_calculation",
                        "parameters": {"x": 2, "y": 3}
                    },
                    "node2": {
                        "type": "tool", 
                        "tool": "test_calculation",
                        "parameters": {"x": 4, "y": 6}
                    },
                    "node3": {
                        "type": "tool",
                        "tool": "test_data_processing",
                        "parameters": {"data": [1, 2, 3, 4, 5]}
                    }
                },
                "edges": [
                    {"from": "node1", "to": "node3"},
                    {"from": "node2", "to": "node3"}
                ]
            }
        }
        
        result = await strands_agent.process_workflow(workflow_id, inputs)
        
        assert result["status"] == WorkflowStatus.COMPLETED
        assert "dag_results" in result["outputs"]
        
        # All nodes should execute
        dag_results = result["outputs"]["dag_results"]
        assert len(dag_results) == 3
        assert "node1" in dag_results
        assert "node2" in dag_results 
        assert "node3" in dag_results
        
    @pytest.mark.asyncio
    async def test_inferred_trading_workflow(self, strands_agent):
        """Test inferred trading workflow"""
        workflow_id = str(uuid.uuid4())
        
        inputs = {
            "symbol": "BTC-USD",
            "action": "buy"
        }
        
        result = await strands_agent.process_workflow(workflow_id, inputs)
        
        assert result["status"] == WorkflowStatus.COMPLETED
        assert "results" in result["outputs"]
        
        # Should contain trading decision
        decision = result["outputs"]["results"]
        assert decision["symbol"] == "BTC-USD"
        assert decision["action"] == "buy"
        assert "recommendation" in decision
        
    @pytest.mark.asyncio
    async def test_concurrent_workflows(self, strands_agent):
        """Test concurrent workflow execution"""
        workflows = []
        
        # Create multiple workflows
        for i in range(5):
            workflow_id = str(uuid.uuid4())
            inputs = {
                "tools": [
                    {
                        "name": "test_calculation",
                        "parameters": {"x": i, "y": i + 1}
                    }
                ]
            }
            workflows.append((workflow_id, inputs))
            
        # Execute all workflows concurrently
        tasks = [
            strands_agent.process_workflow(wf_id, inputs)
            for wf_id, inputs in workflows
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert all(r["status"] == WorkflowStatus.COMPLETED for r in results)
        
        # Verify results are correct
        for i, result in enumerate(results):
            tool_result = result["outputs"]["tool_results"][0]
            expected = i + (i + 1)  # x + y
            assert tool_result["result"]["result"] == expected
            
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, strands_agent):
        """Test workflow error handling and recovery"""
        workflow_id = str(uuid.uuid4())
        
        inputs = {
            "workflow_definition": {
                "steps": [
                    {
                        "id": "good_step",
                        "type": "tool",
                        "tool": "test_calculation", 
                        "parameters": {"x": 5, "y": 5}
                    },
                    {
                        "id": "error_step",
                        "type": "tool",
                        "tool": "test_error_tool",
                        "parameters": {},
                        "continue_on_error": True
                    },
                    {
                        "id": "final_step",
                        "type": "tool",
                        "tool": "test_data_processing",
                        "parameters": {"data": [1, 2, 3]},
                        "dependencies": ["good_step"]
                    }
                ]
            }
        }
        
        result = await strands_agent.process_workflow(workflow_id, inputs)
        
        assert result["status"] == WorkflowStatus.COMPLETED
        
        # Error step should have error but workflow continues
        workflow_results = result["outputs"]["workflow_results"]
        assert "error" in workflow_results["error_step"]
        assert workflow_results["error_step"]["continued"] is True
        
        # Other steps should complete normally
        assert workflow_results["good_step"]["result"] == 10
        assert workflow_results["final_step"]["processed"] == [2, 4, 6]
        
    @pytest.mark.asyncio
    async def test_workflow_metrics_collection(self, strands_agent):
        """Test workflow execution metrics"""
        initial_metrics = strands_agent.get_metrics()
        
        # Execute several workflows
        for i in range(3):
            workflow_id = str(uuid.uuid4())
            inputs = {
                "tools": [
                    {
                        "name": "test_calculation",
                        "parameters": {"x": i, "y": i}
                    }
                ]
            }
            
            await strands_agent.process_workflow(workflow_id, inputs)
            
        # Check updated metrics
        final_metrics = strands_agent.get_metrics()
        
        assert final_metrics["workflows_executed"] == initial_metrics["workflows_executed"] + 3
        assert final_metrics["workflows_succeeded"] == initial_metrics["workflows_succeeded"] + 3
        assert final_metrics["tools_executed"] > initial_metrics["tools_executed"]
        assert final_metrics["avg_workflow_time_ms"] > 0
        assert final_metrics["avg_tool_time_ms"] > 0
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, strands_agent):
        """Test circuit breaker protection"""
        # Execute error tool multiple times to trigger circuit breaker
        for i in range(5):  # Should exceed circuit breaker threshold
            result = await strands_agent.execute_tool("test_error_tool", {})
            assert result.status == ToolStatus.FAILURE
            
        # Circuit breaker should now be open
        tools_list = strands_agent.list_tools()
        error_tool = [t for t in tools_list if t["name"] == "test_error_tool"][0]
        assert not error_tool["available"]  # Circuit breaker should be open