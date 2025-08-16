"""
Test edge cases for MCP implementation.

This module tests various edge cases that could occur in production:
- Malformed JSON messages
- Missing required fields
- Unexpected disconnections
- Invalid method parameters
- Large message handling
- Unicode and special characters
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add project to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cryptotrading.core.protocols.mcp.protocol import (
    MCPProtocol, MCPRequest, MCPResponse, MCPError, MCPErrorCode
)
from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.client import MCPClient
from cryptotrading.core.protocols.mcp.transport import StdioTransport
from cryptotrading.core.protocols.mcp.tools import MCPTool


class TestMCPProtocolEdgeCases:
    """Test edge cases in MCP protocol handling."""
    
    def test_malformed_json_parsing(self):
        """Test parsing of malformed JSON messages."""
        protocol = MCPProtocol()
        
        # Test cases of malformed JSON
        test_cases = [
            "{invalid json",  # Missing closing brace
            "{'single': 'quotes'}",  # Single quotes instead of double
            "{\"method\": }",  # Missing value
            "null",  # Just null
            "undefined",  # JavaScript undefined
            "{\"jsonrpc\":\"2.0\",\"method\":\"test\",}",  # Trailing comma
            "",  # Empty string
            "   ",  # Just whitespace
        ]
        
        for malformed in test_cases:
            result = protocol.parse_message(malformed)
            assert isinstance(result, MCPError), f"Failed for: {malformed}"
            assert result.code == MCPErrorCode.PARSE_ERROR.value
    
    def test_missing_required_fields(self):
        """Test messages missing required fields."""
        protocol = MCPProtocol()
        
        # Missing jsonrpc field
        result = protocol.parse_message('{"method": "test"}')
        assert isinstance(result, MCPError)
        assert result.code == MCPErrorCode.INVALID_REQUEST.value
        
        # Wrong jsonrpc version
        result = protocol.parse_message('{"jsonrpc": "1.0", "method": "test"}')
        assert isinstance(result, MCPError)
        assert result.code == MCPErrorCode.INVALID_REQUEST.value
        
        # Neither request nor response
        result = protocol.parse_message('{"jsonrpc": "2.0"}')
        assert isinstance(result, MCPError)
        assert result.code == MCPErrorCode.INVALID_REQUEST.value
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        protocol = MCPProtocol()
        
        # Unicode in method name
        request = protocol.create_request("测试方法", {"emoji": "🚀"})
        serialized = protocol.serialize_message(request)
        parsed = protocol.parse_message(serialized)
        
        assert isinstance(parsed, MCPRequest)
        assert parsed.method == "测试方法"
        assert parsed.params["emoji"] == "🚀"
        
        # Special characters in parameters
        special_params = {
            "newline": "line1\nline2",
            "tab": "col1\tcol2",
            "quote": 'He said "Hello"',
            "backslash": "C:\\Users\\test",
            "unicode": "Hello 世界 🌍",
            "null_char": "test\x00null"
        }
        
        request = protocol.create_request("test", special_params)
        serialized = protocol.serialize_message(request)
        parsed = protocol.parse_message(serialized)
        
        assert isinstance(parsed, MCPRequest)
        for key, value in special_params.items():
            assert parsed.params[key] == value
    
    def test_extremely_large_messages(self):
        """Test handling of very large messages."""
        protocol = MCPProtocol()
        
        # Create a large parameter (1MB of text)
        large_text = "x" * (1024 * 1024)
        request = protocol.create_request("test", {"data": large_text})
        
        # Should serialize and parse successfully
        serialized = protocol.serialize_message(request)
        parsed = protocol.parse_message(serialized)
        
        assert isinstance(parsed, MCPRequest)
        assert len(parsed.params["data"]) == 1024 * 1024
    
    def test_deeply_nested_parameters(self):
        """Test handling of deeply nested parameter structures."""
        protocol = MCPProtocol()
        
        # Create deeply nested structure
        nested = {"level": 1}
        current = nested
        for i in range(2, 101):  # 100 levels deep
            current["nested"] = {"level": i}
            current = current["nested"]
        
        request = protocol.create_request("test", nested)
        serialized = protocol.serialize_message(request)
        parsed = protocol.parse_message(serialized)
        
        assert isinstance(parsed, MCPRequest)
        # Verify deep nesting preserved
        current = parsed.params
        for i in range(1, 101):
            assert current["level"] == i
            if i < 100:
                current = current["nested"]


class TestMCPServerEdgeCases:
    """Test edge cases in MCP server handling."""
    
    @pytest.mark.asyncio
    async def test_server_handler_exceptions(self):
        """Test server handling of exceptions in handlers."""
        # Create mock transport
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.connect = AsyncMock(return_value=True)
        
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Add handler that raises exception
        async def failing_handler(params):
            raise ValueError("Test error")
        
        server.protocol.register_handler("test/fail", failing_handler)
        
        # Create request
        request = MCPRequest(method="test/fail", params={})
        response = await server.protocol.handle_request(request)
        
        # Should return error response
        assert isinstance(response, MCPResponse)
        assert response.error is not None
        assert response.error.code == MCPErrorCode.INTERNAL_ERROR.value
        assert "Test error" in response.error.message
    
    @pytest.mark.asyncio
    async def test_tool_execution_with_invalid_params(self):
        """Test tool execution with invalid parameters."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Add tool that expects specific parameters
        def strict_tool(x: int, y: int) -> int:
            return x + y
        
        tool = MCPTool(
            name="strict_tool",
            description="Requires x and y integers",
            parameters={
                "x": {"type": "integer"},
                "y": {"type": "integer"}
            },
            function=strict_tool
        )
        server.add_tool(tool)
        
        # Test with missing parameters
        result = await server._handle_call_tool({"name": "strict_tool", "arguments": {}})
        assert result["isError"] is True
        
        # Test with wrong parameter types
        result = await server._handle_call_tool({
            "name": "strict_tool",
            "arguments": {"x": "not_a_number", "y": "also_not"}
        })
        assert result["isError"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test handling of concurrent tool calls."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        call_count = 0
        
        # Add slow tool
        async def slow_tool(delay: float = 0.1):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(delay)
            return f"Result {call_count}"
        
        tool = MCPTool(
            name="slow_tool",
            description="Tool with delay",
            parameters={"delay": {"type": "number"}},
            function=slow_tool
        )
        server.add_tool(tool)
        
        # Make concurrent calls
        tasks = []
        for i in range(10):
            task = server._handle_call_tool({
                "name": "slow_tool",
                "arguments": {"delay": 0.01}
            })
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        assert all(not r["isError"] for r in results)
        assert call_count == 10


class TestMCPClientEdgeCases:
    """Test edge cases in MCP client handling."""
    
    @pytest.mark.asyncio
    async def test_response_timeout(self):
        """Test client handling of response timeouts."""
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.send_message = AsyncMock()
        
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Don't set up response - will timeout
        with pytest.raises(RuntimeError) as exc_info:
            await client._send_request("test/timeout", timeout=0.1)
        
        assert "timeout" in str(exc_info.value).lower()
        # Pending request should be cleaned up
        assert len(client.pending_requests) == 0
    
    @pytest.mark.asyncio
    async def test_out_of_order_responses(self):
        """Test handling of out-of-order responses."""
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Capture sent messages
        sent_messages = []
        mock_transport.send_message = AsyncMock(side_effect=lambda m: sent_messages.append(m))
        
        # Start two requests
        task1 = asyncio.create_task(client._send_request("method1"))
        task2 = asyncio.create_task(client._send_request("method2"))
        
        # Let them send
        await asyncio.sleep(0.1)
        
        # Parse sent requests to get IDs
        req1 = json.loads(sent_messages[0])
        req2 = json.loads(sent_messages[1])
        
        # Send responses in reverse order
        resp2 = MCPResponse(id=req2["id"], result={"data": "response2"})
        await client._handle_message(json.dumps(resp2.to_dict()))
        
        resp1 = MCPResponse(id=req1["id"], result={"data": "response1"})
        await client._handle_message(json.dumps(resp1.to_dict()))
        
        # Both tasks should complete with correct results
        result1 = await task1
        result2 = await task2
        
        assert result1["data"] == "response1"
        assert result2["data"] == "response2"
    
    @pytest.mark.asyncio
    async def test_unknown_response_id(self):
        """Test handling of responses with unknown request IDs."""
        mock_transport = AsyncMock()
        client = MCPClient("test-client", "1.0.0", mock_transport)
        
        # Send response with unknown ID - should not crash
        unknown_response = MCPResponse(id="unknown-id", result={"data": "test"})
        
        # Should handle gracefully (log warning)
        await client._handle_message(json.dumps(unknown_response.to_dict()))
        
        # Client should still be functional
        assert len(client.pending_requests) == 0


class TestTransportEdgeCases:
    """Test edge cases in transport layer."""
    
    @pytest.mark.asyncio
    async def test_stdio_transport_binary_data(self):
        """Test stdio transport with binary data."""
        transport = StdioTransport()
        
        # Mock stdin/stdout
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        transport.reader = mock_reader
        transport.writer = mock_writer
        transport.is_connected = True
        
        # Send message with binary-like content
        message = json.dumps({
            "jsonrpc": "2.0",
            "method": "test",
            "params": {
                "data": "\\x00\\x01\\x02\\x03\\xff"
            }
        })
        
        await transport.send_message(message)
        
        # Should encode and send
        mock_writer.write.assert_called_once()
        sent_data = mock_writer.write.call_args[0][0]
        assert message.encode() in sent_data
    
    @pytest.mark.asyncio
    async def test_transport_disconnection_during_send(self):
        """Test transport disconnection during send."""
        transport = StdioTransport()
        transport.is_connected = True
        transport.writer = None  # Simulate disconnection
        
        # Should raise error
        with pytest.raises(RuntimeError) as exc_info:
            await transport.send_message("test")
        
        assert "not connected" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_partial_message_handling(self):
        """Test handling of partial messages in transport."""
        transport = StdioTransport()
        
        # Mock reader that returns partial lines
        mock_reader = AsyncMock()
        transport.reader = mock_reader
        transport.is_connected = True
        
        message_handler = AsyncMock()
        transport.set_message_handler(message_handler)
        
        # Simulate receiving message in chunks
        full_message = '{"jsonrpc":"2.0","method":"test","id":"123"}\n'
        mock_reader.readline = AsyncMock(side_effect=[
            full_message.encode()  # Complete line
        ])
        
        # Create stop event to control loop
        transport._stop_event.set()
        
        # Should handle complete message
        await transport.receive_messages()
        
        # Handler should be called with complete message
        message_handler.assert_called_once()
        received = message_handler.call_args[0][0]
        assert json.loads(received)["method"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])