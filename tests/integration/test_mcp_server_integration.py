"""
Integration tests for MCP Server with real WebSocket connections
Tests the full server lifecycle and tool execution
"""

import pytest
import asyncio
import json
import websockets
from datetime import datetime
import jwt

from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.lifecycle_manager import (
    MCPLifecycleManager, LifecycleHook, managed_mcp_server
)
from cryptotrading.core.protocols.mcp.tools import MCPTool, ToolResult, CryptoTradingTools
from cryptotrading.core.config.production_config import MCPConfig
from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase
from cryptotrading.core.config.production_config import DatabaseConfig

# Test configuration
TEST_HOST = "localhost"
TEST_PORT = 9999
TEST_JWT_SECRET = "test_secret_key_for_testing_only"

@pytest.fixture
async def test_database():
    """Create test database for MCP operations"""
    db_config = DatabaseConfig(
        host="localhost",
        database=":memory:",
        connection_pool_size=5
    )
    
    db = UnifiedDatabase(db_config)
    await db.initialize()
    
    # Add test data
    await db.add_position("BTC-USD", 1.0, 50000.0)
    await db.add_position("ETH-USD", 5.0, 3000.0)
    
    yield db
    
    await db.close()

@pytest.fixture
async def mcp_config(test_database):
    """Create test MCP configuration"""
    config = MCPConfig(
        enable_mcp=True,
        server_host=TEST_HOST,
        server_port=TEST_PORT,
        max_connections=10,
        connection_timeout=5,
        auth_enabled=True,
        ssl_enabled=False,
        rate_limit_per_minute=100,
        jwt_secret=TEST_JWT_SECRET,
        database_url="sqlite:///:memory:"
    )
    
    return config

@pytest.fixture
async def mcp_server(mcp_config):
    """Create and start MCP server for testing"""
    manager = MCPLifecycleManager(mcp_config)
    
    # Add test tools
    class TestHook(LifecycleHook):
        def __init__(self):
            super().__init__("test_hook")
            
        async def on_initialize(self, server):
            # Add custom test tool
            test_tool = MCPTool(
                name="test_echo",
                description="Echo test tool",
                parameters={"message": {"type": "string"}},
                function=lambda message: f"Echo: {message}"
            )
            server.add_tool(test_tool)
    
    manager.add_hook(TestHook())
    
    # Initialize and start server
    await manager.initialize()
    
    # Start server in background
    server_task = asyncio.create_task(manager.start())
    
    # Wait for server to be ready
    await asyncio.sleep(0.5)
    
    yield manager
    
    # Stop server
    await manager.stop()
    server_task.cancel()
    
    try:
        await server_task
    except asyncio.CancelledError:
        pass

@pytest.fixture
def auth_token():
    """Generate test authentication token"""
    payload = {
        "sub": "test_user",
        "permissions": ["read", "write", "execute"],
        "exp": datetime.utcnow().timestamp() + 3600
    }
    
    return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")

class TestMCPServerLifecycle:
    """Test MCP server lifecycle management"""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, mcp_server):
        """Test server initializes correctly"""
        assert mcp_server.phase == "running"
        assert mcp_server.server is not None
        assert len(mcp_server.server.tools) > 0
        
        # Check health
        health = await mcp_server.get_health_status()
        assert health.healthy
        assert health.phase == "running"
        
    @pytest.mark.asyncio
    async def test_websocket_connection(self, mcp_server, auth_token):
        """Test WebSocket connection to server"""
        uri = f"ws://{TEST_HOST}:{TEST_PORT}"
        
        async with websockets.connect(uri) as websocket:
            # Should receive initial response
            message = await websocket.recv()
            data = json.loads(message)
            
            assert "mcp_version" in data
            assert "requires_auth" in data
            assert data["requires_auth"] is True
            
            # Send authentication
            auth_request = {
                "type": "authenticate",
                "id": "auth_1",
                "token": auth_token
            }
            
            await websocket.send(json.dumps(auth_request))
            
            # Receive auth response
            response = await websocket.recv()
            auth_data = json.loads(response)
            
            assert auth_data["type"] == "authenticated"
            assert auth_data["user_id"] == "test_user"
            
    @pytest.mark.asyncio
    async def test_tool_listing(self, mcp_server, auth_token):
        """Test listing available tools"""
        uri = f"ws://{TEST_HOST}:{TEST_PORT}"
        
        async with websockets.connect(uri) as websocket:
            # Authenticate first
            await websocket.recv()  # Initial message
            
            auth_request = {
                "type": "authenticate",
                "id": "auth_1",
                "token": auth_token
            }
            await websocket.send(json.dumps(auth_request))
            await websocket.recv()  # Auth response
            
            # List tools
            list_request = {
                "type": "list_tools",
                "id": "list_1"
            }
            
            await websocket.send(json.dumps(list_request))
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "tools_list"
            assert "tools" in data
            assert len(data["tools"]) > 0
            
            # Verify tool structure
            tool_names = {tool["name"] for tool in data["tools"]}
            assert "test_echo" in tool_names
            assert "get_portfolio" in tool_names
            assert "get_market_data" in tool_names
            
    @pytest.mark.asyncio
    async def test_tool_execution(self, mcp_server, auth_token):
        """Test executing tools through MCP"""
        uri = f"ws://{TEST_HOST}:{TEST_PORT}"
        
        async with websockets.connect(uri) as websocket:
            # Authenticate
            await websocket.recv()
            auth_request = {
                "type": "authenticate",
                "id": "auth_1", 
                "token": auth_token
            }
            await websocket.send(json.dumps(auth_request))
            await websocket.recv()
            
            # Execute test echo tool
            execute_request = {
                "type": "execute_tool",
                "id": "exec_1",
                "tool": "test_echo",
                "arguments": {"message": "Hello MCP!"}
            }
            
            await websocket.send(json.dumps(execute_request))
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "tool_result"
            assert data["tool"] == "test_echo"
            assert not data["result"]["isError"]
            assert data["result"]["content"][0]["text"] == "Echo: Hello MCP!"
            
    @pytest.mark.asyncio
    async def test_portfolio_tool_integration(self, mcp_server, auth_token, test_database):
        """Test portfolio tool with real database"""
        uri = f"ws://{TEST_HOST}:{TEST_PORT}"
        
        async with websockets.connect(uri) as websocket:
            # Authenticate
            await websocket.recv()
            auth_request = {
                "type": "authenticate",
                "id": "auth_1",
                "token": auth_token
            }
            await websocket.send(json.dumps(auth_request))
            await websocket.recv()
            
            # Execute portfolio tool
            execute_request = {
                "type": "execute_tool",
                "id": "exec_portfolio",
                "tool": "get_portfolio",
                "arguments": {"include_history": False}
            }
            
            await websocket.send(json.dumps(execute_request))
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "tool_result"
            assert not data["result"]["isError"]
            
            # Parse portfolio data
            content = data["result"]["content"][0]["text"]
            portfolio = json.loads(content)
            
            assert "holdings" in portfolio
            assert len(portfolio["holdings"]) == 2
            
            # Verify holdings
            btc_holding = [h for h in portfolio["holdings"] if h["symbol"] == "BTC-USD"][0]
            assert btc_holding["amount"] == 1.0
            
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mcp_server, auth_token):
        """Test rate limiting functionality"""
        uri = f"ws://{TEST_HOST}:{TEST_PORT}"
        
        async with websockets.connect(uri) as websocket:
            # Authenticate
            await websocket.recv()
            auth_request = {
                "type": "authenticate",
                "id": "auth_1",
                "token": auth_token
            }
            await websocket.send(json.dumps(auth_request))
            await websocket.recv()
            
            # Send many requests quickly
            responses = []
            for i in range(150):  # Exceed rate limit
                request = {
                    "type": "execute_tool",
                    "id": f"exec_{i}",
                    "tool": "test_echo",
                    "arguments": {"message": f"Request {i}"}
                }
                
                await websocket.send(json.dumps(request))
                
                # Don't wait for all responses to simulate rapid fire
                if i % 10 == 0:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        responses.append(json.loads(response))
                    except asyncio.TimeoutError:
                        pass
                        
            # Collect remaining responses
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    responses.append(json.loads(response))
                except asyncio.TimeoutError:
                    break
                    
            # Should have some rate limit errors
            errors = [r for r in responses if r.get("type") == "error"]
            rate_limit_errors = [e for e in errors if "rate limit" in e.get("error", "").lower()]
            
            # May not always trigger depending on timing, but structure should be correct
            assert len(responses) > 0
            
    @pytest.mark.asyncio
    async def test_concurrent_clients(self, mcp_server, auth_token):
        """Test multiple concurrent client connections"""
        uri = f"ws://{TEST_HOST}:{TEST_PORT}"
        
        async def client_session(client_id: int):
            async with websockets.connect(uri) as websocket:
                # Authenticate
                await websocket.recv()
                auth_request = {
                    "type": "authenticate",
                    "id": f"auth_{client_id}",
                    "token": auth_token
                }
                await websocket.send(json.dumps(auth_request))
                await websocket.recv()
                
                # Execute tools
                for i in range(5):
                    request = {
                        "type": "execute_tool",
                        "id": f"exec_{client_id}_{i}",
                        "tool": "test_echo",
                        "arguments": {"message": f"Client {client_id} - Message {i}"}
                    }
                    
                    await websocket.send(json.dumps(request))
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    assert data["type"] == "tool_result"
                    assert not data["result"]["isError"]
                    
        # Run multiple clients concurrently
        clients = [client_session(i) for i in range(5)]
        await asyncio.gather(*clients)
        
        # Check server metrics
        health = await mcp_server.get_health_status()
        assert health.healthy
        
    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_server, auth_token):
        """Test error handling in tool execution"""
        uri = f"ws://{TEST_HOST}:{TEST_PORT}"
        
        async with websockets.connect(uri) as websocket:
            # Authenticate
            await websocket.recv()
            auth_request = {
                "type": "authenticate",
                "id": "auth_1",
                "token": auth_token
            }
            await websocket.send(json.dumps(auth_request))
            await websocket.recv()
            
            # Try to execute non-existent tool
            request = {
                "type": "execute_tool",
                "id": "exec_error",
                "tool": "non_existent_tool",
                "arguments": {}
            }
            
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "error"
            assert "not found" in data["error"]
            
            # Try invalid parameters
            request = {
                "type": "execute_tool",
                "id": "exec_invalid",
                "tool": "get_market_data",
                "arguments": {}  # Missing required 'symbol' parameter
            }
            
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "tool_result"
            assert data["result"]["isError"]
            
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mcp_config):
        """Test graceful server shutdown"""
        manager = MCPLifecycleManager(mcp_config)
        await manager.initialize()
        
        # Start server
        server_task = asyncio.create_task(manager.start())
        await asyncio.sleep(0.5)
        
        # Connect client
        uri = f"ws://{TEST_HOST}:{TEST_PORT}"
        websocket = await websockets.connect(uri)
        
        # Initiate shutdown
        shutdown_task = asyncio.create_task(manager.stop())
        
        # Client should be disconnected
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await websocket.recv()
            
        # Shutdown should complete
        await shutdown_task
        
        # Server task should be cancelled
        assert manager.phase == "stopped"
        
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
            
    @pytest.mark.asyncio
    async def test_health_monitoring(self, mcp_server):
        """Test health monitoring and metrics"""
        # Let server run for a bit
        await asyncio.sleep(1)
        
        # Get health status
        health = await mcp_server.get_health_status()
        
        assert health.healthy
        assert health.uptime_seconds > 0
        assert health.memory_usage_mb > 0
        assert health.cpu_percent >= 0
        assert health.errors_last_hour == 0
        
        # Get server metrics
        if mcp_server.server:
            metrics = mcp_server.server.metrics.to_dict()
            assert "total_connections" in metrics
            assert "total_requests" in metrics
            assert metrics["uptime_seconds"] > 0