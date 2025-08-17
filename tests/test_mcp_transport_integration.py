"""
MCP Transport Layer Integration Tests
Tests for all transport implementations: stdio, WebSocket, SSE, HTTP
"""

import pytest
import asyncio
import json
import sys
import os
import subprocess
import websockets
import aiohttp
from aiohttp import web
from typing import Dict, Any, List, Optional
import tempfile
from pathlib import Path

from cryptotrading.core.protocols.mcp.transport import (
    StdioTransport, WebSocketTransport, SSETransport, TransportError
)
from cryptotrading.core.protocols.mcp.enhanced_transport import (
    ProcessTransport, HTTPTransport, EnhancedWebSocketTransport,
    TransportManager, create_process_transport, create_http_transport,
    create_websocket_transport
)
from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.client import MCPClient


class TestStdioTransport:
    """Test stdio transport implementation"""
    
    @pytest.fixture
    def stdio_transport(self):
        """Create stdio transport instance"""
        return StdioTransport()
    
    def test_message_formatting(self, stdio_transport):
        """Test stdio message formatting"""
        message = {
            "jsonrpc": "2.0",
            "id": "test_1",
            "method": "tools/list"
        }
        
        # Format for stdio
        formatted = stdio_transport.format_message(message)
        
        # Should be newline-delimited JSON
        assert formatted.endswith('\n')
        assert not formatted.endswith('\n\n')  # Only one newline
        
        # Should be valid JSON
        parsed = json.loads(formatted.strip())
        assert parsed == message
    
    @pytest.mark.asyncio
    async def test_stdio_server_script(self):
        """Test stdio transport with actual subprocess"""
        # Create a simple MCP server script
        server_script = '''
import sys
import json

def handle_request(request):
    if request["method"] == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"tools": [{"name": "test_tool"}]}
        }
    return {
        "jsonrpc": "2.0",
        "id": request["id"],
        "error": {"code": -32601, "message": "Method not found"}
    }

# Read from stdin
for line in sys.stdin:
    try:
        request = json.loads(line.strip())
        response = handle_request(request)
        print(json.dumps(response))
        sys.stdout.flush()
    except Exception as e:
        error = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": str(e)}
        }
        print(json.dumps(error))
        sys.stdout.flush()
'''
        
        # Write server script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(server_script)
            script_path = f.name
        
        try:
            # Create process transport
            transport = ProcessTransport([sys.executable, script_path])
            
            responses = []
            async def message_handler(message):
                responses.append(json.loads(message))
            
            transport.set_message_handler(message_handler)
            
            # Connect and send request
            await transport.connect()
            
            request = json.dumps({
                "jsonrpc": "2.0",
                "id": "stdio_test",
                "method": "tools/list"
            })
            
            await transport.send_message(request)
            
            # Wait for response
            await asyncio.sleep(0.1)
            
            # Verify response
            assert len(responses) == 1
            assert responses[0]["id"] == "stdio_test"
            assert "result" in responses[0]
            assert len(responses[0]["result"]["tools"]) == 1
            
            await transport.disconnect()
            
        finally:
            # Cleanup
            os.unlink(script_path)


class TestWebSocketTransport:
    """Test WebSocket transport implementation"""
    
    @pytest.fixture
    async def websocket_server(self):
        """Create test WebSocket server"""
        clients = set()
        
        async def handle_websocket(websocket, path):
            clients.add(websocket)
            try:
                # Send initial message
                await websocket.send(json.dumps({
                    "type": "welcome",
                    "version": "1.0.0"
                }))
                
                async for message in websocket:
                    data = json.loads(message)
                    
                    # Echo back with result
                    response = {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "result": {"echo": data}
                    }
                    
                    await websocket.send(json.dumps(response))
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                clients.remove(websocket)
        
        # Start server
        server = await websockets.serve(
            handle_websocket,
            "localhost",
            0  # Random port
        )
        
        port = server.sockets[0].getsockname()[1]
        
        yield f"ws://localhost:{port}"
        
        # Cleanup
        server.close()
        await server.wait_closed()
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, websocket_server):
        """Test WebSocket connection and messaging"""
        transport = EnhancedWebSocketTransport(websocket_server)
        
        messages_received = []
        
        def message_handler(message):
            messages_received.append(json.loads(message))
        
        transport.set_message_handler(message_handler)
        
        # Connect
        await transport.connect()
        assert transport.is_connected
        
        # Should receive welcome message
        await asyncio.sleep(0.1)
        assert len(messages_received) == 1
        assert messages_received[0]["type"] == "welcome"
        
        # Send request
        request = {
            "jsonrpc": "2.0",
            "id": "ws_test",
            "method": "test"
        }
        
        await transport.send_message(json.dumps(request))
        
        # Wait for response
        await asyncio.sleep(0.1)
        
        # Should receive echo response
        assert len(messages_received) == 2
        response = messages_received[1]
        assert response["id"] == "ws_test"
        assert response["result"]["echo"]["method"] == "test"
        
        # Disconnect
        await transport.disconnect()
        assert not transport.is_connected
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection(self, websocket_server):
        """Test WebSocket automatic reconnection"""
        # Create transport with auto-reconnect disabled for controlled testing
        transport = EnhancedWebSocketTransport(websocket_server, auto_reconnect=False)
        
        await transport.connect()
        assert transport.is_connected
        
        # Simulate connection loss
        await transport.websocket.close()
        
        # Should detect disconnection
        await asyncio.sleep(0.1)
        assert not transport.is_connected
        
        # Manual reconnect
        await transport.connect()
        assert transport.is_connected
        
        await transport.disconnect()
    
    @pytest.mark.asyncio
    async def test_websocket_binary_frames(self):
        """Test WebSocket binary frame support"""
        transport = WebSocketTransport()
        
        # Create binary data
        binary_data = b'\x00\x01\x02\x03\x04'
        
        # Transport should support binary frames
        assert transport.supports_binary_frames
        
        # Format binary message
        message = {
            "jsonrpc": "2.0",
            "id": "binary_test",
            "result": {
                "contents": [{
                    "uri": "test://binary",
                    "blob": binary_data.hex()
                }]
            }
        }
        
        # Should handle binary content
        formatted = transport.format_message(message)
        assert isinstance(formatted, (str, bytes))


class TestSSETransport:
    """Test Server-Sent Events transport implementation"""
    
    @pytest.fixture
    async def sse_server(self):
        """Create test SSE server"""
        app = web.Application()
        
        async def sse_handler(request):
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            
            await response.prepare(request)
            
            # Send initial event
            await response.write(b'data: {"type": "connected"}\n\n')
            
            # Keep connection open
            try:
                while True:
                    await asyncio.sleep(1)
                    # Send heartbeat
                    await response.write(b': heartbeat\n\n')
            except Exception:
                pass
            
            return response
        
        async def post_handler(request):
            data = await request.json()
            
            # Process request and return response
            response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "result": {"processed": True}
            }
            
            return web.json_response(response)
        
        app.router.add_get('/sse', sse_handler)
        app.router.add_post('/mcp', post_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', 0)
        await site.start()
        
        port = site._server.sockets[0].getsockname()[1]
        
        yield f"http://localhost:{port}"
        
        await runner.cleanup()
    
    @pytest.mark.asyncio
    async def test_sse_connection(self, sse_server):
        """Test SSE connection and event streaming"""
        transport = SSETransport()
        
        # Connect to SSE endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{sse_server}/sse") as response:
                assert response.status == 200
                assert response.headers['Content-Type'] == 'text/event-stream'
                
                # Read first event
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        assert data["type"] == "connected"
                        break
    
    @pytest.mark.asyncio
    async def test_sse_message_format(self):
        """Test SSE message formatting"""
        transport = SSETransport()
        
        message = {
            "jsonrpc": "2.0",
            "id": "sse_test",
            "result": {"status": "ok"}
        }
        
        # Format as SSE
        formatted = transport.format_sse_message(message)
        
        # Verify SSE format
        lines = formatted.split('\n')
        assert any(line.startswith('data: ') for line in lines)
        assert formatted.endswith('\n\n')  # Double newline at end
        
        # Extract JSON from SSE format
        data_lines = [line[6:] for line in lines if line.startswith('data: ')]
        reconstructed = json.loads(''.join(data_lines))
        assert reconstructed == message
    
    def test_sse_multiline_data(self):
        """Test SSE formatting with multiline data"""
        transport = SSETransport()
        
        # Create message with multiline content
        large_content = "Line 1\nLine 2\nLine 3" * 100
        message = {
            "jsonrpc": "2.0",
            "id": "multiline",
            "result": {"content": large_content}
        }
        
        formatted = transport.format_sse_message(message)
        
        # Each line of JSON should be prefixed with 'data: '
        lines = formatted.split('\n')
        data_lines = [line for line in lines if line.startswith('data: ')]
        
        # Should be able to reconstruct original message
        json_str = ''.join(line[6:] for line in data_lines)
        reconstructed = json.loads(json_str)
        assert reconstructed == message


class TestHTTPTransport:
    """Test HTTP transport implementation"""
    
    @pytest.fixture
    async def http_transport_server(self):
        """Create HTTP transport server"""
        transport = HTTPTransport("localhost", 0)  # Random port
        
        # Mock message handler
        async def mock_handler(message):
            data = json.loads(message)
            
            if data["method"] == "tools/list":
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": data["id"],
                    "result": {
                        "tools": [
                            {"name": "tool1", "description": "Test tool 1"},
                            {"name": "tool2", "description": "Test tool 2"}
                        ]
                    }
                })
            
            return json.dumps({
                "jsonrpc": "2.0",
                "id": data["id"],
                "error": {"code": -32601, "message": "Method not found"}
            })
        
        transport.set_message_handler(mock_handler)
        
        await transport.connect()
        
        yield f"http://localhost:{transport.port}"
        
        await transport.disconnect()
    
    @pytest.mark.asyncio
    async def test_http_request_response(self, http_transport_server):
        """Test HTTP request/response cycle"""
        async with aiohttp.ClientSession() as session:
            # Test MCP request
            request = {
                "jsonrpc": "2.0",
                "id": "http_test",
                "method": "tools/list"
            }
            
            async with session.post(
                f"{http_transport_server}/mcp",
                json=request
            ) as response:
                assert response.status == 200
                data = await response.json()
                
                assert data["id"] == "http_test"
                assert "result" in data
                assert len(data["result"]["tools"]) == 2
    
    @pytest.mark.asyncio
    async def test_http_cors_headers(self, http_transport_server):
        """Test CORS header support"""
        async with aiohttp.ClientSession() as session:
            # Test OPTIONS request
            async with session.options(f"{http_transport_server}/mcp") as response:
                assert response.status == 200
                assert response.headers.get('Access-Control-Allow-Origin') == '*'
                assert 'POST' in response.headers.get('Access-Control-Allow-Methods', '')
    
    @pytest.mark.asyncio
    async def test_http_status_endpoint(self, http_transport_server):
        """Test HTTP status endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{http_transport_server}/mcp/status") as response:
                assert response.status == 200
                data = await response.json()
                
                assert data["status"] == "connected"
                assert data["transport"] == "http"
                assert "host" in data
                assert "port" in data
    
    @pytest.mark.asyncio
    async def test_http_error_handling(self, http_transport_server):
        """Test HTTP error handling"""
        async with aiohttp.ClientSession() as session:
            # Test with invalid JSON
            async with session.post(
                f"{http_transport_server}/mcp",
                data="not json"
            ) as response:
                assert response.status == 500
                data = await response.json()
                assert "error" in data
            
            # Test with unknown method
            request = {
                "jsonrpc": "2.0",
                "id": "error_test",
                "method": "unknown/method"
            }
            
            async with session.post(
                f"{http_transport_server}/mcp",
                json=request
            ) as response:
                assert response.status == 200  # JSON-RPC errors return 200
                data = await response.json()
                assert "error" in data
                assert data["error"]["code"] == -32601


class TestTransportManager:
    """Test transport manager functionality"""
    
    @pytest.fixture
    def transport_manager(self):
        """Create transport manager"""
        return TransportManager()
    
    @pytest.mark.asyncio
    async def test_transport_registration(self, transport_manager):
        """Test registering multiple transports"""
        # Add different transport types
        stdio = StdioTransport()
        websocket = WebSocketTransport()
        http = HTTPTransport()
        
        transport_manager.add_transport("stdio", stdio)
        transport_manager.add_transport("websocket", websocket)
        transport_manager.add_transport("http", http)
        
        # Verify all registered
        assert len(transport_manager.transports) == 3
        assert "stdio" in transport_manager.transports
        assert "websocket" in transport_manager.transports
        assert "http" in transport_manager.transports
    
    @pytest.mark.asyncio
    async def test_transport_switching(self, transport_manager):
        """Test switching between transports"""
        # Add transports
        transport_manager.add_transport("t1", StdioTransport())
        transport_manager.add_transport("t2", WebSocketTransport())
        
        # Initially no active transport
        assert transport_manager.active_transport is None
        
        # Activate first transport
        await transport_manager.connect_transport("t1")
        assert transport_manager.active_transport == "t1"
        
        # Switch to second transport
        await transport_manager.connect_transport("t2")
        assert transport_manager.active_transport == "t2"
    
    @pytest.mark.asyncio
    async def test_transport_manager_cleanup(self, transport_manager):
        """Test transport manager cleanup"""
        # Create HTTP transport (actually starts a server)
        http = HTTPTransport("localhost", 0)
        transport_manager.add_transport("http", http)
        
        # Connect
        await transport_manager.connect_transport("http")
        assert http.is_connected
        
        # Disconnect all
        await transport_manager.disconnect_all()
        assert not http.is_connected
        assert transport_manager.active_transport is None


class TestTransportIntegration:
    """Integration tests across transport types"""
    
    @pytest.mark.asyncio
    async def test_mcp_server_with_multiple_transports(self):
        """Test MCP server supporting multiple transport types"""
        server = MCPServer("multi-transport-server", "1.0.0")
        
        # Add test tool
        server.add_tool({
            "name": "echo",
            "description": "Echo input",
            "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
            "function": lambda message: f"Echo: {message}"
        })
        
        # Test request
        request = {
            "jsonrpc": "2.0",
            "id": "integration_test",
            "method": "tools/list"
        }
        
        # Process request (transport-agnostic)
        response = await server.handle_request(request)
        
        assert response["id"] == "integration_test"
        assert "result" in response
        assert len(response["result"]["tools"]) == 1
        assert response["result"]["tools"][0]["name"] == "echo"
    
    @pytest.mark.asyncio
    async def test_transport_error_handling(self):
        """Test error handling across transports"""
        # Test connection failures
        transports = [
            ProcessTransport(["nonexistent_command"]),
            EnhancedWebSocketTransport("ws://localhost:99999"),  # Invalid port
            HTTPTransport("localhost", 99999)  # Invalid port
        ]
        
        for transport in transports:
            with pytest.raises(Exception):
                await transport.connect()
    
    @pytest.mark.asyncio
    async def test_transport_message_size_limits(self):
        """Test message size handling across transports"""
        # Create large message
        large_data = "x" * (10 * 1024 * 1024)  # 10MB
        message = {
            "jsonrpc": "2.0",
            "id": "large",
            "method": "test",
            "params": {"data": large_data}
        }
        
        # Different transports may have different limits
        transports = {
            "stdio": StdioTransport(),
            "websocket": WebSocketTransport(),
            "sse": SSETransport()
        }
        
        for name, transport in transports.items():
            try:
                formatted = transport.format_message(message)
                # Should at least not crash
                assert formatted is not None
            except Exception as e:
                # Some transports may reject large messages
                print(f"{name} transport rejected large message: {e}")


class TestTransportFactory:
    """Test transport factory functions"""
    
    def test_process_transport_creation(self):
        """Test process transport factory"""
        script_path = "/path/to/server.py"
        project_root = "/path/to/project"
        
        transport = create_process_transport(script_path, project_root)
        
        assert isinstance(transport, ProcessTransport)
        assert sys.executable in transport.command
        assert script_path in transport.command
        assert transport.cwd == project_root
    
    def test_http_transport_creation(self):
        """Test HTTP transport factory"""
        transport = create_http_transport("0.0.0.0", 8080)
        
        assert isinstance(transport, HTTPTransport)
        assert transport.host == "0.0.0.0"
        assert transport.port == 8080
    
    def test_websocket_transport_creation(self):
        """Test WebSocket transport factory"""
        transport = create_websocket_transport("ws://example.com:8080", auto_reconnect=True)
        
        assert isinstance(transport, EnhancedWebSocketTransport)
        assert transport.uri == "ws://example.com:8080"
        assert transport.auto_reconnect is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])