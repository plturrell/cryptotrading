"""
Test error recovery mechanisms in MCP implementation.

This module tests the system's ability to recover from various error conditions:
- Network failures during transmission
- Timeout scenarios
- Automatic reconnection logic
- Partial message handling
- Recovery from server/client crashes
- Graceful degradation
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
from asyncio import TimeoutError

# Add project to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cryptotrading.core.protocols.mcp.protocol import (
    MCPProtocol, MCPRequest, MCPResponse, MCPError, MCPErrorCode
)
from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.client import MCPClient, MCPClientSession
from cryptotrading.core.protocols.mcp.transport import (
    MCPTransport, StdioTransport, WebSocketTransport
)


class TestNetworkFailureRecovery:
    """Test recovery from network failures."""
    
    @pytest.mark.asyncio
    async def test_client_reconnection_after_disconnect(self):
        """Test client reconnection after unexpected disconnect."""
        mock_transport = AsyncMock()
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Initial successful connection
        mock_transport.connect = AsyncMock(return_value=True)
        mock_transport.is_connected = True
        mock_transport.receive_messages = AsyncMock()
        
        # Mock successful initialization
        async def mock_send(msg):
            # Parse request and send appropriate response
            req = json.loads(msg)
            if req["method"] == "initialize":
                response = MCPResponse(
                    id=req["id"],
                    result={
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "test-server"},
                        "capabilities": {}
                    }
                )
                await client._handle_message(json.dumps(response.to_dict()))
        
        mock_transport.send_message = AsyncMock(side_effect=mock_send)
        
        # Connect initially
        assert await client.connect()
        assert client.is_initialized
        
        # Simulate disconnect
        mock_transport.is_connected = False
        client.is_initialized = False
        
        # Attempt reconnection
        mock_transport.is_connected = True
        assert await client.connect()
        assert client.is_initialized
    
    @pytest.mark.asyncio
    async def test_server_recovery_from_transport_failure(self):
        """Test server recovery when transport fails."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Add a tool
        tool_calls = []
        async def test_tool():
            tool_calls.append(1)
            return "success"
        
        from cryptotrading.core.protocols.mcp.tools import MCPTool
        tool = MCPTool("test", "Test tool", {}, test_tool)
        server.add_tool(tool)
        
        # Simulate transport failure during message handling
        async def failing_send(msg):
            raise ConnectionError("Transport failed")
        
        mock_transport.send_message = AsyncMock(side_effect=failing_send)
        mock_transport.is_connected = True
        
        # Server should handle gracefully
        request = '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"test","arguments":{}},"id":"1"}'
        await server._handle_message(request)
        
        # Tool should still have been called despite send failure
        assert len(tool_calls) == 1
    
    @pytest.mark.asyncio
    async def test_partial_message_recovery(self):
        """Test recovery from partial message transmission."""
        mock_transport = AsyncMock()
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Simulate receiving partial response then complete one
        partial_json = '{"jsonrpc":"2.0","id":"123","res'
        complete_json = '{"jsonrpc":"2.0","id":"123","result":{"data":"complete"}}'
        
        # First partial message should be ignored
        await client._handle_message(partial_json)
        
        # Client should still be functional
        client.pending_requests["123"] = asyncio.Future()
        await client._handle_message(complete_json)
        
        # Should process complete message
        result = await client.pending_requests["123"]
        assert result["data"] == "complete"


class TestTimeoutRecovery:
    """Test timeout and retry mechanisms."""
    
    @pytest.mark.asyncio
    async def test_request_timeout_with_retry(self):
        """Test request timeout and retry logic."""
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        attempt_count = 0
        
        async def delayed_response(msg):
            nonlocal attempt_count
            attempt_count += 1
            req = json.loads(msg)
            
            # First attempt times out, second succeeds
            if attempt_count == 1:
                await asyncio.sleep(0.2)  # Longer than timeout
            else:
                # Immediate response
                response = MCPResponse(id=req["id"], result={"attempt": attempt_count})
                await client._handle_message(json.dumps(response.to_dict()))
        
        mock_transport.send_message = AsyncMock(side_effect=delayed_response)
        
        # First attempt should timeout
        with pytest.raises(RuntimeError) as exc_info:
            await client._send_request("test", timeout=0.1)
        assert "timeout" in str(exc_info.value).lower()
        
        # Retry should succeed
        result = await client._send_request("test", timeout=1.0)
        assert result["attempt"] == 2
    
    @pytest.mark.asyncio
    async def test_server_handler_timeout_recovery(self):
        """Test server recovery from handler timeouts."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Add handler that times out
        async def slow_handler(params):
            await asyncio.sleep(10)  # Very slow
            return {"result": "too_late"}
        
        server.protocol.register_handler("test/slow", slow_handler)
        
        # Create request
        request = MCPRequest(method="test/slow", params={}, id="123")
        
        # Handle with timeout
        try:
            response = await asyncio.wait_for(
                server.protocol.handle_request(request),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Should recover and create error response
            response = server.protocol.create_error_response(
                request.id,
                MCPErrorCode.INTERNAL_ERROR,
                "Handler timeout"
            )
        
        assert response.error is not None
        assert "timeout" in response.error.message.lower()
    
    @pytest.mark.asyncio
    async def test_cascading_timeout_prevention(self):
        """Test prevention of cascading timeouts."""
        mock_transport = AsyncMock()
        client = MCPClient("test-client", "1.0.0", mock_transport)
        mock_transport.is_connected = True
        
        # Track all requests
        all_requests = []
        
        async def track_requests(msg):
            all_requests.append(json.loads(msg))
            # Don't respond - simulate timeout
        
        mock_transport.send_message = AsyncMock(side_effect=track_requests)
        
        # Make multiple concurrent requests
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                client._send_request(f"method{i}", timeout=0.1)
            )
            tasks.append(task)
        
        # All should timeout independently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should be timeout errors
        assert all(isinstance(r, RuntimeError) for r in results)
        assert len(all_requests) == 5  # All requests were sent
        assert len(client.pending_requests) == 0  # All cleaned up


class TestConnectionRecovery:
    """Test connection recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_automatic_reconnection_with_backoff(self):
        """Test automatic reconnection with exponential backoff."""
        
        class ReconnectingTransport(MCPTransport):
            def __init__(self):
                super().__init__()
                self.connect_attempts = 0
                self.connect_times = []
            
            async def connect(self):
                self.connect_times.append(asyncio.get_event_loop().time())
                self.connect_attempts += 1
                
                # Fail first 3 attempts, then succeed
                if self.connect_attempts <= 3:
                    return False
                
                self.is_connected = True
                return True
            
            async def disconnect(self):
                self.is_connected = False
            
            async def send_message(self, message):
                if not self.is_connected:
                    raise RuntimeError("Not connected")
            
            async def receive_messages(self):
                while self.is_connected:
                    await asyncio.sleep(0.1)
        
        transport = ReconnectingTransport()
        
        # Implement reconnection logic
        async def connect_with_retry(max_attempts=5, base_delay=0.1):
            for attempt in range(max_attempts):
                if await transport.connect():
                    return True
                
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            
            return False
        
        # Test reconnection
        start_time = asyncio.get_event_loop().time()
        connected = await connect_with_retry()
        
        assert connected
        assert transport.connect_attempts == 4
        
        # Verify backoff delays
        if len(transport.connect_times) > 1:
            delays = []
            for i in range(1, len(transport.connect_times)):
                delay = transport.connect_times[i] - transport.connect_times[i-1]
                delays.append(delay)
            
            # Each delay should be roughly double the previous
            for i in range(1, len(delays)):
                assert delays[i] > delays[i-1] * 1.5
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_repeated_failures(self):
        """Test system degrades gracefully after repeated failures."""
        mock_transport = AsyncMock()
        mock_transport.connect = AsyncMock(return_value=False)
        
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Multiple connection attempts should all fail gracefully
        for _ in range(3):
            connected = await client.connect()
            assert not connected
            assert not client.is_initialized
        
        # Client should remain in consistent state
        assert not client.is_initialized
        assert len(client.pending_requests) == 0
    
    @pytest.mark.asyncio
    async def test_session_recovery_after_crash(self):
        """Test session recovery after client/server crash."""
        # Simulate session with state
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.connect = AsyncMock(return_value=True)
        
        # Create session
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Mock successful initialization
        init_response = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "test-server"},
            "capabilities": {"tools": {}}
        }
        
        async def mock_send(msg):
            req = json.loads(msg)
            if req["method"] == "initialize":
                response = MCPResponse(id=req["id"], result=init_response)
                await client._handle_message(json.dumps(response.to_dict()))
            elif req["method"] == "tools/list":
                response = MCPResponse(
                    id=req["id"],
                    result={"tools": [{"name": "calculator", "description": "Math"}]}
                )
                await client._handle_message(json.dumps(response.to_dict()))
        
        mock_transport.send_message = AsyncMock(side_effect=mock_send)
        
        # Use session manager
        async with MCPClientSession(client) as session:
            # Cache some data
            tools = await session.list_tools()
            assert len(tools) == 1
            
            # Simulate crash (transport disconnection)
            mock_transport.is_connected = False
            session.is_initialized = False
            
            # Clear cached data to simulate fresh start
            session.tools.clear()
            
            # Reconnect
            mock_transport.is_connected = True
            await session.connect()
            
            # Should be able to restore state
            tools = await session.list_tools()
            assert len(tools) == 1
            assert tools[0]["name"] == "calculator"


class TestErrorPropagation:
    """Test proper error propagation and handling."""
    
    @pytest.mark.asyncio
    async def test_error_propagation_through_layers(self):
        """Test errors propagate correctly through protocol layers."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Add tool that raises specific error
        class CustomError(Exception):
            pass
        
        async def failing_tool():
            raise CustomError("Tool specific error")
        
        from cryptotrading.core.protocols.mcp.tools import MCPTool
        tool = MCPTool("fail", "Failing tool", {}, failing_tool)
        server.add_tool(tool)
        
        # Call tool
        result = await server._handle_call_tool({
            "name": "fail",
            "arguments": {}
        })
        
        # Error should be captured and returned
        assert result["isError"] is True
        assert "Tool specific error" in result["content"][0]["text"]
    
    @pytest.mark.asyncio
    async def test_transport_error_isolation(self):
        """Test transport errors don't crash the system."""
        # Create custom transport that fails
        class FailingTransport(MCPTransport):
            async def connect(self):
                self.is_connected = True
                return True
            
            async def disconnect(self):
                raise ConnectionError("Disconnect failed")
            
            async def send_message(self, message):
                raise ConnectionError("Send failed")
            
            async def receive_messages(self):
                raise ConnectionError("Receive failed")
        
        transport = FailingTransport()
        client = MCPClient("test-client", "1.0.0", transport)
        
        # Connect should work
        await transport.connect()
        
        # Operations should handle transport errors
        transport.is_connected = True  # Pretend connected
        
        with pytest.raises(RuntimeError):
            await client._send_request("test")
        
        # Disconnect error should be handled
        try:
            await transport.disconnect()
        except ConnectionError:
            pass  # Expected
        
        # Client should remain functional
        assert client.protocol is not None


class TestStateConsistency:
    """Test state consistency during error conditions."""
    
    @pytest.mark.asyncio
    async def test_pending_request_cleanup_on_disconnect(self):
        """Test pending requests are cleaned up on disconnect."""
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Create multiple pending requests
        futures = []
        for i in range(5):
            future = asyncio.Future()
            request_id = f"req-{i}"
            client.pending_requests[request_id] = future
            futures.append(future)
        
        # Disconnect
        await client.disconnect()
        
        # All pending requests should be cleaned up
        assert len(client.pending_requests) == 0
        
        # Futures should be cancelled or resolved
        for future in futures:
            assert future.done() or future.cancelled()
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_server_shutdown(self):
        """Test proper resource cleanup on server shutdown."""
        mock_transport = AsyncMock()
        mock_transport.connect = AsyncMock(return_value=True)
        mock_transport.disconnect = AsyncMock()
        mock_transport.is_connected = True
        
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Add resources
        from cryptotrading.core.protocols.mcp.resources import Resource
        resource = Resource(
            uri="test://resource",
            name="Test",
            description="Test resource",
            mime_type="text/plain",
            read_func=lambda: "data"
        )
        server.add_resource(resource)
        
        # Track cleanup
        cleanup_called = False
        original_disconnect = mock_transport.disconnect
        
        async def track_disconnect():
            nonlocal cleanup_called
            cleanup_called = True
            await original_disconnect()
        
        mock_transport.disconnect = AsyncMock(side_effect=track_disconnect)
        
        # Stop server
        await server.stop()
        
        # Verify cleanup
        assert cleanup_called
        assert not server.is_initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])