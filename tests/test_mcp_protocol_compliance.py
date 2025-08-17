"""
MCP Protocol Compliance Tests
Tests for full MCP specification compliance including all required and optional features
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

from cryptotrading.core.protocols.mcp.protocol import (
    MCPMessage, MCPRequest, MCPResponse, MCPError,
    ErrorCode, create_error_response
)
from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.client import MCPClient
from cryptotrading.core.protocols.mcp.tools import MCPTool, ToolResult
from cryptotrading.core.protocols.mcp.resources import MCPResource, ResourceContent
from cryptotrading.core.protocols.mcp.transport import (
    StdioTransport, WebSocketTransport, SSETransport
)
from cryptotrading.core.protocols.mcp.capabilities import (
    ServerCapabilities, ClientCapabilities, CapabilityNegotiator
)


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance with specification"""
    
    def test_message_structure(self):
        """Test MCP message structure compliance"""
        # Test request structure
        request = MCPRequest(
            method="tools/list",
            params={},
            id="test_123"
        )
        
        msg_dict = request.to_dict()
        assert msg_dict["jsonrpc"] == "2.0"
        assert msg_dict["method"] == "tools/list"
        assert msg_dict["params"] == {}
        assert msg_dict["id"] == "test_123"
        
        # Test response structure
        response = MCPResponse(
            result={"tools": []},
            id="test_123"
        )
        
        resp_dict = response.to_dict()
        assert resp_dict["jsonrpc"] == "2.0"
        assert resp_dict["result"] == {"tools": []}
        assert resp_dict["id"] == "test_123"
        assert "error" not in resp_dict
        
        # Test error response
        error_response = MCPResponse(
            error=MCPError(
                code=ErrorCode.METHOD_NOT_FOUND,
                message="Unknown method",
                data={"method": "unknown"}
            ),
            id="test_123"
        )
        
        err_dict = error_response.to_dict()
        assert err_dict["jsonrpc"] == "2.0"
        assert "result" not in err_dict
        assert err_dict["error"]["code"] == -32601
        assert err_dict["error"]["message"] == "Unknown method"
    
    def test_error_codes(self):
        """Test standard JSON-RPC error codes"""
        # Test all standard error codes
        assert ErrorCode.PARSE_ERROR == -32700
        assert ErrorCode.INVALID_REQUEST == -32600
        assert ErrorCode.METHOD_NOT_FOUND == -32601
        assert ErrorCode.INVALID_PARAMS == -32602
        assert ErrorCode.INTERNAL_ERROR == -32603
        
        # Test error response creation
        error_resp = create_error_response(
            "test_id",
            ErrorCode.INVALID_PARAMS,
            "Missing required parameter",
            {"param": "symbol"}
        )
        
        assert error_resp["error"]["code"] == -32602
        assert error_resp["error"]["data"]["param"] == "symbol"


class TestMCPInitialization:
    """Test MCP initialization protocol"""
    
    @pytest.fixture
    def mock_server(self):
        """Create mock MCP server"""
        server = MCPServer("test-server", "1.0.0")
        server.capabilities = ServerCapabilities()
        server.capabilities.enable_tools()
        server.capabilities.enable_resources()
        server.capabilities.enable_logging()
        return server
    
    @pytest.fixture
    def mock_client(self):
        """Create mock MCP client"""
        client = MCPClient("test-client", "1.0.0")
        client.capabilities = ClientCapabilities()
        client.capabilities.enable_sampling()
        client.capabilities.enable_roots_changed()
        return client
    
    @pytest.mark.asyncio
    async def test_initialization_handshake(self, mock_server, mock_client):
        """Test initialization handshake sequence"""
        # Client sends initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": "init_1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": mock_client.capabilities.to_dict(),
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # Server processes initialize
        response = await mock_server.handle_request(init_request)
        
        # Verify response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "init_1"
        assert "result" in response
        
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert "capabilities" in result
        assert "serverInfo" in result
        
        # Verify server info
        assert result["serverInfo"]["name"] == "test-server"
        assert result["serverInfo"]["version"] == "1.0.0"
        
        # Verify capabilities negotiation
        assert "tools" in result["capabilities"]
        assert "resources" in result["capabilities"]
        assert "logging" in result["capabilities"]
    
    @pytest.mark.asyncio
    async def test_initialized_notification(self, mock_server):
        """Test initialized notification from client"""
        # Client sends initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        # Process notification (no response expected)
        response = await mock_server.handle_request(notification)
        
        # Notifications should not return a response
        assert response is None or response == {}
        
        # Server should mark as initialized
        assert mock_server.is_initialized


class TestMCPToolsProtocol:
    """Test MCP tools protocol implementation"""
    
    @pytest.fixture
    def server_with_tools(self):
        """Create server with test tools"""
        server = MCPServer("test-server", "1.0.0")
        
        # Add test tools
        echo_tool = MCPTool(
            name="echo",
            description="Echo the input",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"}
                },
                "required": ["message"]
            },
            function=lambda message: f"Echo: {message}"
        )
        
        math_tool = MCPTool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            function=lambda a, b: f"Result: {a + b}"
        )
        
        server.add_tool(echo_tool)
        server.add_tool(math_tool)
        
        return server
    
    @pytest.mark.asyncio
    async def test_tools_list(self, server_with_tools):
        """Test tools/list method"""
        request = {
            "jsonrpc": "2.0",
            "id": "list_1",
            "method": "tools/list"
        }
        
        response = await server_with_tools.handle_request(request)
        
        assert response["id"] == "list_1"
        assert "result" in response
        
        tools = response["result"]["tools"]
        assert len(tools) == 2
        
        # Verify tool structure
        echo_tool = next(t for t in tools if t["name"] == "echo")
        assert echo_tool["description"] == "Echo the input"
        assert "inputSchema" in echo_tool
        assert echo_tool["inputSchema"]["type"] == "object"
        assert "message" in echo_tool["inputSchema"]["properties"]
    
    @pytest.mark.asyncio
    async def test_tools_call(self, server_with_tools):
        """Test tools/call method"""
        request = {
            "jsonrpc": "2.0",
            "id": "call_1",
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {"message": "Hello MCP!"}
            }
        }
        
        response = await server_with_tools.handle_request(request)
        
        assert response["id"] == "call_1"
        assert "result" in response
        
        result = response["result"]
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Echo: Hello MCP!"
        assert not result.get("isError", False)
    
    @pytest.mark.asyncio
    async def test_tools_call_error(self, server_with_tools):
        """Test tools/call with invalid parameters"""
        request = {
            "jsonrpc": "2.0",
            "id": "call_error",
            "method": "tools/call",
            "params": {
                "name": "add",
                "arguments": {"a": "not_a_number", "b": 2}
            }
        }
        
        response = await server_with_tools.handle_request(request)
        
        # Should return tool result with error
        assert "result" in response
        result = response["result"]
        assert result.get("isError", False)
        assert len(result["content"]) > 0
        assert "error" in result["content"][0]["text"].lower()
    
    @pytest.mark.asyncio
    async def test_tool_not_found(self, server_with_tools):
        """Test calling non-existent tool"""
        request = {
            "jsonrpc": "2.0",
            "id": "not_found",
            "method": "tools/call",
            "params": {
                "name": "non_existent",
                "arguments": {}
            }
        }
        
        response = await server_with_tools.handle_request(request)
        
        # Should return error
        assert "error" in response
        assert response["error"]["code"] == ErrorCode.INVALID_PARAMS
        assert "not found" in response["error"]["message"].lower()


class TestMCPResourcesProtocol:
    """Test MCP resources protocol implementation"""
    
    @pytest.fixture
    def server_with_resources(self):
        """Create server with test resources"""
        server = MCPServer("test-server", "1.0.0")
        
        # Add test resources
        config_resource = MCPResource(
            uri="config://app/settings.json",
            name="Application Settings",
            description="Current application configuration",
            mime_type="application/json"
        )
        
        data_resource = MCPResource(
            uri="data://portfolio/current",
            name="Current Portfolio",
            description="Current portfolio holdings",
            mime_type="application/json"
        )
        
        server.add_resource(config_resource)
        server.add_resource(data_resource)
        
        # Add resource content providers
        async def get_config_content():
            return ResourceContent(
                uri="config://app/settings.json",
                text=json.dumps({"theme": "dark", "language": "en"}),
                mime_type="application/json"
            )
        
        async def get_portfolio_content():
            return ResourceContent(
                uri="data://portfolio/current",
                text=json.dumps({"BTC": 1.5, "ETH": 10.0}),
                mime_type="application/json"
            )
        
        server.resource_handlers["config://app/settings.json"] = get_config_content
        server.resource_handlers["data://portfolio/current"] = get_portfolio_content
        
        return server
    
    @pytest.mark.asyncio
    async def test_resources_list(self, server_with_resources):
        """Test resources/list method"""
        request = {
            "jsonrpc": "2.0",
            "id": "res_list",
            "method": "resources/list"
        }
        
        response = await server_with_resources.handle_request(request)
        
        assert response["id"] == "res_list"
        assert "result" in response
        
        resources = response["result"]["resources"]
        assert len(resources) == 2
        
        # Verify resource structure
        config_res = next(r for r in resources if r["uri"] == "config://app/settings.json")
        assert config_res["name"] == "Application Settings"
        assert config_res["description"] == "Current application configuration"
        assert config_res["mimeType"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_resources_read(self, server_with_resources):
        """Test resources/read method"""
        request = {
            "jsonrpc": "2.0",
            "id": "res_read",
            "method": "resources/read",
            "params": {
                "uri": "config://app/settings.json"
            }
        }
        
        response = await server_with_resources.handle_request(request)
        
        assert response["id"] == "res_read"
        assert "result" in response
        
        result = response["result"]
        assert "contents" in result
        assert len(result["contents"]) > 0
        
        content = result["contents"][0]
        assert content["uri"] == "config://app/settings.json"
        assert content["mimeType"] == "application/json"
        assert "text" in content
        
        # Verify content
        data = json.loads(content["text"])
        assert data["theme"] == "dark"
        assert data["language"] == "en"
    
    @pytest.mark.asyncio
    async def test_resource_not_found(self, server_with_resources):
        """Test reading non-existent resource"""
        request = {
            "jsonrpc": "2.0",
            "id": "not_found",
            "method": "resources/read",
            "params": {
                "uri": "unknown://resource"
            }
        }
        
        response = await server_with_resources.handle_request(request)
        
        # Should return error
        assert "error" in response
        assert response["error"]["code"] == ErrorCode.RESOURCE_NOT_FOUND


class TestMCPPromptsProtocol:
    """Test MCP prompts protocol implementation"""
    
    @pytest.fixture
    def server_with_prompts(self):
        """Create server with prompt support"""
        server = MCPServer("test-server", "1.0.0")
        
        # Enable prompts capability
        server.capabilities.enable_prompts()
        
        # Add test prompts
        server.prompts = {
            "analyze_code": {
                "name": "analyze_code",
                "description": "Analyze code for issues",
                "arguments": [
                    {
                        "name": "language",
                        "description": "Programming language",
                        "required": True
                    },
                    {
                        "name": "code",
                        "description": "Code to analyze",
                        "required": True
                    }
                ]
            },
            "generate_docs": {
                "name": "generate_docs",
                "description": "Generate documentation",
                "arguments": [
                    {
                        "name": "format",
                        "description": "Documentation format",
                        "required": False
                    }
                ]
            }
        }
        
        return server
    
    @pytest.mark.asyncio
    async def test_prompts_list(self, server_with_prompts):
        """Test prompts/list method"""
        request = {
            "jsonrpc": "2.0",
            "id": "prompts_list",
            "method": "prompts/list"
        }
        
        response = await server_with_prompts.handle_request(request)
        
        assert response["id"] == "prompts_list"
        assert "result" in response
        
        prompts = response["result"]["prompts"]
        assert len(prompts) == 2
        
        # Verify prompt structure
        analyze_prompt = next(p for p in prompts if p["name"] == "analyze_code")
        assert analyze_prompt["description"] == "Analyze code for issues"
        assert len(analyze_prompt["arguments"]) == 2
        assert analyze_prompt["arguments"][0]["required"] is True
    
    @pytest.mark.asyncio
    async def test_prompts_get(self, server_with_prompts):
        """Test prompts/get method"""
        request = {
            "jsonrpc": "2.0",
            "id": "prompts_get",
            "method": "prompts/get",
            "params": {
                "name": "analyze_code",
                "arguments": {
                    "language": "python",
                    "code": "def hello(): print('world')"
                }
            }
        }
        
        response = await server_with_prompts.handle_request(request)
        
        assert response["id"] == "prompts_get"
        assert "result" in response
        
        # Should return prompt messages
        assert "messages" in response["result"]
        messages = response["result"]["messages"]
        assert len(messages) > 0
        assert messages[0]["role"] in ["user", "assistant", "system"]
        assert "content" in messages[0]


class TestMCPSamplingProtocol:
    """Test MCP sampling/completion protocol"""
    
    @pytest.fixture
    def client_with_sampling(self):
        """Create client with sampling capability"""
        client = MCPClient("test-client", "1.0.0")
        client.capabilities.enable_sampling()
        
        # Mock sampling function
        async def mock_create_message(params):
            return {
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": f"Completed: {params.get('prompt', 'No prompt')}"
                }
            }
        
        client.create_message = mock_create_message
        return client
    
    @pytest.mark.asyncio
    async def test_sampling_create_message(self, client_with_sampling):
        """Test sampling/createMessage request from server to client"""
        # Server would send this request to client
        request = {
            "jsonrpc": "2.0",
            "id": "sample_1",
            "method": "sampling/createMessage",
            "params": {
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "Hello"}}
                ],
                "systemPrompt": "You are a helpful assistant",
                "maxTokens": 100
            }
        }
        
        # Client handles request
        response = await client_with_sampling.handle_request(request)
        
        assert response["id"] == "sample_1"
        assert "result" in response
        
        message = response["result"]
        assert message["role"] == "assistant"
        assert message["content"]["type"] == "text"
        assert "Completed:" in message["content"]["text"]


class TestMCPNotifications:
    """Test MCP notification protocol"""
    
    @pytest.fixture
    def notification_server(self):
        """Create server with notification support"""
        server = MCPServer("test-server", "1.0.0")
        server.notification_handlers = {}
        server.pending_notifications = []
        
        async def handle_notification(method: str, params: Dict[str, Any]):
            server.pending_notifications.append({
                "method": method,
                "params": params,
                "timestamp": datetime.utcnow()
            })
        
        server.handle_notification = handle_notification
        return server
    
    @pytest.mark.asyncio
    async def test_progress_notification(self, notification_server):
        """Test progress notification handling"""
        # Send progress notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {
                "progressToken": "task_123",
                "progress": 50,
                "total": 100
            }
        }
        
        # Process notification
        response = await notification_server.handle_request(notification)
        
        # Notifications don't return responses
        assert response is None or response == {}
        
        # Check notification was recorded
        assert len(notification_server.pending_notifications) == 1
        notif = notification_server.pending_notifications[0]
        assert notif["method"] == "notifications/progress"
        assert notif["params"]["progress"] == 50
    
    @pytest.mark.asyncio
    async def test_tools_list_changed_notification(self, notification_server):
        """Test tools/listChanged notification"""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/tools/listChanged"
        }
        
        response = await notification_server.handle_request(notification)
        assert response is None or response == {}
        
        # Server should refresh tools list
        assert len(notification_server.pending_notifications) == 1
        assert notification_server.pending_notifications[0]["method"] == "notifications/tools/listChanged"
    
    @pytest.mark.asyncio
    async def test_cancelled_notification(self, notification_server):
        """Test request cancellation notification"""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {
                "requestId": "long_running_123",
                "reason": "User cancelled operation"
            }
        }
        
        response = await notification_server.handle_request(notification)
        assert response is None or response == {}
        
        # Check cancellation was recorded
        assert len(notification_server.pending_notifications) == 1
        notif = notification_server.pending_notifications[0]
        assert notif["params"]["requestId"] == "long_running_123"


class TestMCPTransportCompliance:
    """Test MCP transport layer compliance"""
    
    @pytest.mark.asyncio
    async def test_stdio_transport_messages(self):
        """Test stdio transport message format"""
        transport = StdioTransport()
        
        # Test message serialization
        message = {
            "jsonrpc": "2.0",
            "id": "test",
            "method": "tools/list"
        }
        
        serialized = transport.serialize_message(message)
        
        # Should be newline-delimited JSON
        assert serialized.endswith('\n')
        assert json.loads(serialized.strip()) == message
    
    @pytest.mark.asyncio
    async def test_sse_transport_format(self):
        """Test SSE transport message format"""
        transport = SSETransport()
        
        message = {
            "jsonrpc": "2.0",
            "id": "test",
            "result": {"tools": []}
        }
        
        # Format as SSE
        sse_message = transport.format_sse_message(message)
        
        # Should follow SSE format
        lines = sse_message.strip().split('\n')
        assert any(line.startswith('data: ') for line in lines)
        assert lines[-1] == ''  # Empty line at end
        
        # Extract and verify JSON
        data_lines = [line[6:] for line in lines if line.startswith('data: ')]
        reconstructed = json.loads(''.join(data_lines))
        assert reconstructed == message
    
    @pytest.mark.asyncio
    async def test_websocket_transport_binary(self):
        """Test WebSocket transport with binary messages"""
        transport = WebSocketTransport()
        
        # Test binary message support
        message = {
            "jsonrpc": "2.0",
            "id": "binary_test",
            "method": "resources/read",
            "params": {"uri": "file://image.png"}
        }
        
        # Should support both text and binary frames
        assert transport.supports_binary_frames
        
        # Binary content in response
        binary_response = {
            "jsonrpc": "2.0",
            "id": "binary_test",
            "result": {
                "contents": [{
                    "uri": "file://image.png",
                    "mimeType": "image/png",
                    "blob": "base64_encoded_data_here"
                }]
            }
        }
        
        # Should handle binary content
        assert transport.can_handle_message(binary_response)


class TestMCPEdgeCases:
    """Test MCP protocol edge cases and error handling"""
    
    @pytest.fixture
    def robust_server(self):
        """Create server with robust error handling"""
        server = MCPServer("robust-server", "1.0.0")
        server.max_request_size = 1024 * 1024  # 1MB
        server.request_timeout = 30  # 30 seconds
        return server
    
    @pytest.mark.asyncio
    async def test_malformed_json(self, robust_server):
        """Test handling of malformed JSON"""
        # Not actually JSON
        malformed = "not json at all"
        
        # Server should handle gracefully
        response = await robust_server.handle_raw_message(malformed)
        
        assert "error" in response
        assert response["error"]["code"] == ErrorCode.PARSE_ERROR
        assert "parse" in response["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, robust_server):
        """Test handling of missing required fields"""
        # Missing jsonrpc version
        request = {
            "id": "test",
            "method": "tools/list"
        }
        
        response = await robust_server.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] == ErrorCode.INVALID_REQUEST
    
    @pytest.mark.asyncio
    async def test_batch_requests_not_supported(self, robust_server):
        """Test that batch requests are properly rejected"""
        # MCP doesn't support batch requests
        batch = [
            {"jsonrpc": "2.0", "id": "1", "method": "tools/list"},
            {"jsonrpc": "2.0", "id": "2", "method": "resources/list"}
        ]
        
        response = await robust_server.handle_raw_message(json.dumps(batch))
        
        assert "error" in response
        assert response["error"]["code"] == ErrorCode.INVALID_REQUEST
        assert "batch" in response["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_request_size_limit(self, robust_server):
        """Test request size limit enforcement"""
        # Create oversized request
        large_data = "x" * (2 * 1024 * 1024)  # 2MB
        request = {
            "jsonrpc": "2.0",
            "id": "large",
            "method": "tools/call",
            "params": {"data": large_data}
        }
        
        response = await robust_server.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] in [ErrorCode.INVALID_REQUEST, -32000]
        assert "size" in response["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, robust_server):
        """Test handling of concurrent requests"""
        # Add slow tool for testing
        async def slow_tool(**kwargs):
            await asyncio.sleep(0.1)
            return "Done"
        
        robust_server.add_tool(MCPTool(
            name="slow_tool",
            description="Slow tool for testing",
            parameters={},
            function=slow_tool
        ))
        
        # Send multiple concurrent requests
        requests = []
        for i in range(10):
            req = {
                "jsonrpc": "2.0",
                "id": f"concurrent_{i}",
                "method": "tools/call",
                "params": {"name": "slow_tool", "arguments": {}}
            }
            requests.append(robust_server.handle_request(req))
        
        # All should complete
        responses = await asyncio.gather(*requests)
        
        # Verify all succeeded
        for i, response in enumerate(responses):
            assert response["id"] == f"concurrent_{i}"
            assert "result" in response
    
    @pytest.mark.asyncio
    async def test_notification_ordering(self, robust_server):
        """Test notification ordering guarantees"""
        notifications_received = []
        
        async def track_notification(method, params):
            notifications_received.append({
                "method": method,
                "params": params,
                "time": datetime.utcnow()
            })
        
        robust_server.handle_notification = track_notification
        
        # Send multiple notifications
        for i in range(5):
            notif = {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progress": i * 20}
            }
            await robust_server.handle_request(notif)
        
        # Verify order preserved
        assert len(notifications_received) == 5
        for i in range(5):
            assert notifications_received[i]["params"]["progress"] == i * 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])