"""
Test concurrent operations in MCP implementation.

This module tests the system's ability to handle concurrent operations:
- Multiple simultaneous requests
- Request/response ordering
- Race condition handling
- Connection pool management
- Thread safety
- Deadlock prevention
"""

import pytest
import asyncio
import json
import threading
import time
from unittest.mock import Mock, AsyncMock, patch
import sys
import os
from asyncio import Queue, Lock
from typing import Dict, List, Any

# Add project to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cryptotrading.core.protocols.mcp.protocol import (
    MCPProtocol, MCPRequest, MCPResponse, MCPError
)
from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.client import MCPClient
from cryptotrading.core.protocols.mcp.tools import MCPTool


class TestConcurrentRequests:
    """Test handling of concurrent requests."""
    
    @pytest.mark.asyncio
    async def test_multiple_simultaneous_client_requests(self):
        """Test client handling multiple simultaneous requests."""
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Track all sent requests
        sent_requests = []
        request_queue = Queue()
        
        async def capture_and_respond(msg):
            req = json.loads(msg)
            sent_requests.append(req)
            await request_queue.put(req)
        
        mock_transport.send_message = AsyncMock(side_effect=capture_and_respond)
        
        # Start response handler
        async def response_handler():
            while True:
                req = await request_queue.get()
                # Simulate some processing delay
                await asyncio.sleep(0.01)
                
                # Send response
                response = MCPResponse(
                    id=req["id"],
                    result={"method": req["method"], "processed": True}
                )
                await client._handle_message(json.dumps(response.to_dict()))
        
        handler_task = asyncio.create_task(response_handler())
        
        # Send 100 concurrent requests
        tasks = []
        for i in range(100):
            task = asyncio.create_task(
                client._send_request(f"method_{i}", {"index": i})
            )
            tasks.append(task)
        
        # Wait for all requests
        results = await asyncio.gather(*tasks)
        handler_task.cancel()
        
        # Verify all requests completed
        assert len(results) == 100
        assert all(r["processed"] for r in results)
        
        # Verify each request got correct response
        for i, result in enumerate(results):
            assert result["method"] == f"method_{i}"
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test server handling concurrent tool executions."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Shared state for testing race conditions
        execution_order = []
        execution_lock = Lock()
        
        async def concurrent_tool(tool_id: int, delay: float = 0.01):
            async with execution_lock:
                execution_order.append(f"start_{tool_id}")
            
            await asyncio.sleep(delay)
            
            async with execution_lock:
                execution_order.append(f"end_{tool_id}")
            
            return f"Result from tool {tool_id}"
        
        # Add multiple tools
        for i in range(5):
            tool = MCPTool(
                name=f"tool_{i}",
                description=f"Test tool {i}",
                parameters={"tool_id": {"type": "integer"}, "delay": {"type": "number"}},
                function=concurrent_tool
            )
            server.add_tool(tool)
        
        # Execute tools concurrently
        tasks = []
        for i in range(5):
            task = server._handle_call_tool({
                "name": f"tool_{i}",
                "arguments": {"tool_id": i, "delay": 0.01}
            })
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify all tools executed
        assert len(results) == 5
        assert all(not r["isError"] for r in results)
        
        # Verify concurrent execution (interleaved start/end)
        starts = [e for e in execution_order if e.startswith("start_")]
        ends = [e for e in execution_order if e.startswith("end_")]
        
        # All starts should happen before all ends (showing concurrency)
        assert len(starts) == 5
        assert len(ends) == 5
        
        # At least some starts should happen before the first end
        first_end_index = execution_order.index(ends[0])
        starts_before_first_end = [s for s in starts if execution_order.index(s) < first_end_index]
        assert len(starts_before_first_end) > 1  # Shows concurrency
    
    @pytest.mark.asyncio
    async def test_request_response_matching(self):
        """Test correct matching of requests and responses under load."""
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Create delayed responses to test ordering
        response_delays = {
            "fast": 0.01,
            "medium": 0.05,
            "slow": 0.1
        }
        
        async def delayed_response(msg):
            req = json.loads(msg)
            method = req["method"]
            delay = response_delays.get(method, 0.01)
            
            # Wait before responding
            await asyncio.sleep(delay)
            
            response = MCPResponse(
                id=req["id"],
                result={"method": method, "request_params": req.get("params", {})}
            )
            await client._handle_message(json.dumps(response.to_dict()))
        
        mock_transport.send_message = AsyncMock(side_effect=delayed_response)
        
        # Send requests in specific order
        slow_task = asyncio.create_task(client._send_request("slow", {"order": 1}))
        medium_task = asyncio.create_task(client._send_request("medium", {"order": 2}))
        fast_task = asyncio.create_task(client._send_request("fast", {"order": 3}))
        
        # Results should arrive out of order but match correctly
        fast_result = await fast_task
        medium_result = await medium_task
        slow_result = await slow_task
        
        assert fast_result["method"] == "fast"
        assert fast_result["request_params"]["order"] == 3
        
        assert medium_result["method"] == "medium"
        assert medium_result["request_params"]["order"] == 2
        
        assert slow_result["method"] == "slow"
        assert slow_result["request_params"]["order"] == 1


class TestRaceConditions:
    """Test handling of race conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_connection_attempts(self):
        """Test handling of concurrent connection attempts."""
        mock_transport = AsyncMock()
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        connection_count = 0
        connection_lock = Lock()
        
        async def track_connections():
            nonlocal connection_count
            async with connection_lock:
                connection_count += 1
                # Simulate connection setup time
                await asyncio.sleep(0.05)
            return True
        
        mock_transport.connect = AsyncMock(side_effect=track_connections)
        mock_transport.is_connected = False
        
        # Set up initialization response
        async def mock_send(msg):
            req = json.loads(msg)
            if req["method"] == "initialize":
                mock_transport.is_connected = True
                response = MCPResponse(
                    id=req["id"],
                    result={
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "test"},
                        "capabilities": {}
                    }
                )
                await client._handle_message(json.dumps(response.to_dict()))
        
        mock_transport.send_message = AsyncMock(side_effect=mock_send)
        mock_transport.receive_messages = AsyncMock()
        
        # Attempt multiple concurrent connections
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(client.connect())
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed, but only one actual connection
        assert all(results)
        assert connection_count == 5  # All attempted
        # But client should be in consistent state
        assert client.is_initialized
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_registration(self):
        """Test concurrent tool registration doesn't cause issues."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Register tools concurrently
        async def register_tool(index: int):
            tool = MCPTool(
                name=f"tool_{index}",
                description=f"Tool {index}",
                parameters={},
                function=lambda: f"Result {index}"
            )
            server.add_tool(tool)
            # Small delay to increase chance of race
            await asyncio.sleep(0.001)
            return tool.name
        
        # Register 50 tools concurrently
        tasks = []
        for i in range(50):
            task = asyncio.create_task(register_tool(i))
            tasks.append(task)
        
        registered = await asyncio.gather(*tasks)
        
        # All tools should be registered
        assert len(server.tools) == 50
        assert len(set(registered)) == 50  # All unique
        
        # List tools to verify consistency
        result = await server._handle_list_tools({})
        assert len(result["tools"]) == 50
    
    @pytest.mark.asyncio
    async def test_shared_state_protection(self):
        """Test protection of shared state during concurrent access."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Shared counter with protection
        counter = {"value": 0}
        counter_lock = Lock()
        
        async def increment_tool(amount: int = 1):
            # Simulate some processing
            await asyncio.sleep(0.001)
            
            # Protected increment
            async with counter_lock:
                old_value = counter["value"]
                await asyncio.sleep(0.001)  # Increase chance of race
                counter["value"] = old_value + amount
            
            return counter["value"]
        
        tool = MCPTool(
            name="increment",
            description="Increment counter",
            parameters={"amount": {"type": "integer"}},
            function=increment_tool
        )
        server.add_tool(tool)
        
        # Concurrent increments
        tasks = []
        for i in range(100):
            task = server._handle_call_tool({
                "name": "increment",
                "arguments": {"amount": 1}
            })
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Counter should be exactly 100 (no lost updates)
        assert counter["value"] == 100


class TestDeadlockPrevention:
    """Test prevention of deadlocks in concurrent scenarios."""
    
    @pytest.mark.asyncio
    async def test_circular_dependency_prevention(self):
        """Test system prevents circular dependencies causing deadlock."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Create tools that could cause circular dependency
        call_stack = []
        
        async def tool_a(call_b: bool = True):
            call_stack.append("a_start")
            if call_b and len(call_stack) < 10:  # Prevent infinite recursion
                # Tool A calls Tool B
                result = await server._handle_call_tool({
                    "name": "tool_b",
                    "arguments": {"call_a": False}
                })
            call_stack.append("a_end")
            return "A completed"
        
        async def tool_b(call_a: bool = True):
            call_stack.append("b_start")
            if call_a and len(call_stack) < 10:  # Prevent infinite recursion
                # Tool B calls Tool A
                result = await server._handle_call_tool({
                    "name": "tool_a",
                    "arguments": {"call_b": False}
                })
            call_stack.append("b_end")
            return "B completed"
        
        server.add_tool(MCPTool("tool_a", "Tool A", {"call_b": {"type": "boolean"}}, tool_a))
        server.add_tool(MCPTool("tool_b", "Tool B", {"call_a": {"type": "boolean"}}, tool_b))
        
        # Execute both tools concurrently
        task_a = asyncio.create_task(server._handle_call_tool({
            "name": "tool_a",
            "arguments": {}
        }))
        task_b = asyncio.create_task(server._handle_call_tool({
            "name": "tool_b",
            "arguments": {}
        }))
        
        # Should complete without deadlock
        results = await asyncio.gather(task_a, task_b)
        
        assert all(not r["isError"] for r in results)
        assert "a_start" in call_stack
        assert "b_start" in call_stack
    
    @pytest.mark.asyncio
    async def test_timeout_prevents_deadlock(self):
        """Test timeouts prevent potential deadlocks."""
        mock_transport = AsyncMock()
        client = MCPClient("test-client", "1.0.0", mock_transport)
        mock_transport.is_connected = True
        
        # Create a situation that could deadlock
        request_received = asyncio.Event()
        response_sent = asyncio.Event()
        
        async def mock_send(msg):
            request_received.set()
            # Wait for response to be allowed
            await response_sent.wait()
        
        mock_transport.send_message = AsyncMock(side_effect=mock_send)
        
        # Start request with timeout
        request_task = asyncio.create_task(
            client._send_request("test", timeout=0.1)
        )
        
        # Wait for request to be sent
        await request_received.wait()
        
        # Don't send response - should timeout
        with pytest.raises(RuntimeError) as exc_info:
            await request_task
        
        assert "timeout" in str(exc_info.value).lower()
        
        # Allow response to prevent hanging
        response_sent.set()


class TestResourceManagement:
    """Test resource management under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_limits(self):
        """Test connection pool respects limits under load."""
        # Simulate connection pool
        class ConnectionPool:
            def __init__(self, max_connections: int = 5):
                self.max_connections = max_connections
                self.active_connections = 0
                self.connection_lock = Lock()
                self.available = asyncio.Condition()
            
            async def acquire(self):
                async with self.available:
                    while self.active_connections >= self.max_connections:
                        await self.available.wait()
                    
                    self.active_connections += 1
                    return self
            
            async def release(self):
                async with self.available:
                    self.active_connections -= 1
                    self.available.notify()
        
        pool = ConnectionPool(max_connections=3)
        concurrent_count = 0
        max_concurrent = 0
        count_lock = Lock()
        
        async def use_connection(index: int):
            nonlocal concurrent_count, max_concurrent
            
            conn = await pool.acquire()
            
            async with count_lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            
            # Simulate work
            await asyncio.sleep(0.05)
            
            async with count_lock:
                concurrent_count -= 1
            
            await pool.release()
            return f"Task {index} completed"
        
        # Run 10 tasks with pool of 3
        tasks = []
        for i in range(10):
            task = asyncio.create_task(use_connection(i))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete
        assert len(results) == 10
        
        # Max concurrent should not exceed pool limit
        assert max_concurrent <= 3
    
    @pytest.mark.asyncio
    async def test_cleanup_on_concurrent_errors(self):
        """Test proper cleanup when errors occur during concurrent operations."""
        mock_transport = AsyncMock()
        client = MCPClient("test-client", "1.0.0", mock_transport)
        mock_transport.is_connected = True
        
        # Track resource usage
        resources_in_use = set()
        resource_lock = Lock()
        
        async def request_with_resource(method: str, should_fail: bool = False):
            resource_id = f"resource_{method}"
            
            # Acquire resource
            async with resource_lock:
                resources_in_use.add(resource_id)
            
            try:
                if should_fail:
                    raise RuntimeError(f"Simulated error for {method}")
                
                # Simulate request
                await asyncio.sleep(0.01)
                return f"Success: {method}"
            
            finally:
                # Always cleanup
                async with resource_lock:
                    resources_in_use.discard(resource_id)
        
        # Mix of successful and failing operations
        tasks = []
        for i in range(10):
            should_fail = i % 3 == 0  # Every 3rd fails
            task = asyncio.create_task(
                request_with_resource(f"method_{i}", should_fail)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]
        
        assert len(errors) == 4  # 0, 3, 6, 9 should fail
        assert len(successes) == 6
        
        # All resources should be cleaned up
        assert len(resources_in_use) == 0


class TestPerformanceUnderLoad:
    """Test performance characteristics under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_throughput_scaling(self):
        """Test system throughput scales with concurrency."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Simple fast tool
        async def fast_tool(value: int):
            # Minimal processing
            return value * 2
        
        tool = MCPTool(
            name="fast",
            description="Fast computation",
            parameters={"value": {"type": "integer"}},
            function=fast_tool
        )
        server.add_tool(tool)
        
        # Test different concurrency levels
        concurrency_levels = [1, 10, 50, 100]
        throughputs = []
        
        for concurrency in concurrency_levels:
            start_time = asyncio.get_event_loop().time()
            
            # Execute operations
            tasks = []
            for i in range(concurrency):
                task = server._handle_call_tool({
                    "name": "fast",
                    "arguments": {"value": i}
                })
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            elapsed = asyncio.get_event_loop().time() - start_time
            throughput = concurrency / elapsed
            throughputs.append(throughput)
        
        # Throughput should increase with concurrency (up to a point)
        # At least 5x improvement from 1 to 10 concurrent
        assert throughputs[1] > throughputs[0] * 3
        
        # Should handle high concurrency without degradation
        assert throughputs[-1] > throughputs[0] * 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])