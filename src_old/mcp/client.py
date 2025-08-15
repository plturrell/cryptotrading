"""
MCP Client Implementation
Implements a complete MCP client for connecting to MCP servers
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
import uuid

from .protocol import MCPProtocol, MCPRequest, MCPResponse, MCPError
from .transport import MCPTransport, StdioTransport
from .capabilities import ClientCapabilities

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP client implementation"""
    
    def __init__(self, name: str, version: str, transport: Optional[MCPTransport] = None):
        self.name = name
        self.version = version
        self.protocol = MCPProtocol()
        self.transport = transport or StdioTransport()
        self.capabilities = ClientCapabilities()
        self.is_initialized = False
        self.server_info = {}
        self.server_capabilities = {}
        
        # Client state
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Dict[str, Any]] = {}
        
        # Set up transport
        self.transport.set_message_handler(self._handle_message)
    
    async def _handle_message(self, message: str):
        """Handle incoming message from transport"""
        try:
            parsed = self.protocol.parse_message(message)
            
            if isinstance(parsed, MCPResponse):
                # Handle response to pending request
                if parsed.id in self.pending_requests:
                    future = self.pending_requests.pop(parsed.id)
                    if parsed.error:
                        future.set_exception(Exception(f"MCP Error {parsed.error.code}: {parsed.error.message}"))
                    else:
                        future.set_result(parsed.result)
                else:
                    logger.warning(f"Received response for unknown request ID: {parsed.id}")
            
            elif isinstance(parsed, MCPRequest):
                # Handle server-initiated request (notifications, etc.)
                await self._handle_server_request(parsed)
            
            else:
                logger.error(f"Invalid message received: {message}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_server_request(self, request: MCPRequest):
        """Handle server-initiated requests"""
        # Handle notifications and server requests
        if request.method == "notifications/initialized":
            logger.info("Server sent initialized notification")
        elif request.method == "notifications/cancelled":
            # Handle request cancellation
            pass
        else:
            logger.warning(f"Unhandled server request: {request.method}")
    
    async def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None, 
                           request_id: Optional[str] = None) -> Any:
        """Send request and wait for response"""
        if not self.transport.is_connected:
            raise RuntimeError("Transport not connected")
        
        # Create request
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        request = self.protocol.create_request(method, params, request_id)
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send request
        message = self.protocol.serialize_message(request)
        await self.transport.send_message(message)
        
        try:
            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            # Clean up pending request
            self.pending_requests.pop(request_id, None)
            raise RuntimeError(f"Request timeout for method: {method}")
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        logger.info(f"Connecting MCP client: {self.name} v{self.version}")
        
        # Connect transport
        if not await self.transport.connect():
            raise RuntimeError("Failed to connect transport")
        
        # Start receiving messages
        receive_task = asyncio.create_task(self.transport.receive_messages())
        
        try:
            # Send initialize request
            await self.initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            receive_task.cancel()
            await self.transport.disconnect()
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        await self.transport.disconnect()
        self.is_initialized = False
        logger.info("MCP client disconnected")
    
    async def initialize(self):
        """Initialize connection with server"""
        client_info = {
            "name": self.name,
            "version": self.version
        }
        
        params = {
            "protocolVersion": self.protocol.version,
            "clientInfo": client_info,
            "capabilities": self.capabilities.to_dict()
        }
        
        # Send initialize request
        result = await self._send_request("initialize", params)
        
        # Store server info and capabilities
        self.server_info = result.get("serverInfo", {})
        self.server_capabilities = result.get("capabilities", {})
        
        # Send initialized notification
        initialized_request = MCPRequest(method="initialized", params={})
        message = self.protocol.serialize_message(initialized_request)
        await self.transport.send_message(message)
        
        self.is_initialized = True
        logger.info(f"MCP client initialized with server: {self.server_info.get('name', 'Unknown')}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from server"""
        result = await self._send_request("tools/list")
        tools = result.get("tools", [])
        
        # Update local tools cache
        self.tools.clear()
        for tool in tools:
            self.tools[tool["name"]] = tool
        
        return tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the server"""
        params = {
            "name": name,
            "arguments": arguments
        }
        
        result = await self._send_request("tools/call", params)
        return result
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from server"""
        result = await self._send_request("resources/list")
        resources = result.get("resources", [])
        
        # Update local resources cache
        self.resources.clear()
        for resource in resources:
            self.resources[resource["uri"]] = resource
        
        return resources
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from the server"""
        params = {"uri": uri}
        result = await self._send_request("resources/read", params)
        return result
    
    async def ping(self) -> Dict[str, Any]:
        """Ping the server"""
        result = await self._send_request("ping")
        return result
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool information by name"""
        return self.tools.get(name)
    
    def get_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get resource information by URI"""
        return self.resources.get(uri)
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information"""
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol.version,
            "is_initialized": self.is_initialized,
            "capabilities": self.capabilities.to_dict(),
            "server_info": self.server_info,
            "server_capabilities": self.server_capabilities,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources)
        }


class MCPClientSession:
    """High-level MCP client session manager"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.is_connected = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.is_connected = await self.client.connect()
        if not self.is_connected:
            raise RuntimeError("Failed to connect to MCP server")
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.is_connected:
            await self.client.disconnect()
        self.is_connected = False