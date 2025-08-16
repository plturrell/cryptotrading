"""
MCP Client Implementation

This module implements a complete Model Context Protocol (MCP) client with:
- Asynchronous request/response handling
- Transport abstraction (stdio, WebSocket, SSE)
- Automatic request ID management
- Response timeout handling
- Tool and resource discovery
- Session management with context manager

The client follows the official MCP specification and provides a high-level
API for interacting with MCP servers.

Key Components:
    MCPClient: Main client class for server communication
    MCPClientSession: Context manager for connection lifecycle
    Request Management: Async futures for request/response correlation
    Caching: Local cache of discovered tools and resources

Example:
    >>> # Using with context manager
    >>> async with MCPClientSession(client) as session:
    ...     tools = await session.list_tools()
    ...     result = await session.call_tool("calculator", {"x": 5, "y": 3})
    >>> 
    >>> # Manual connection management
    >>> client = MCPClient("my-client", "1.0.0")
    >>> await client.connect()
    >>> tools = await client.list_tools()
    >>> await client.disconnect()
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
    """MCP client implementation with full protocol support.
    
    Provides a complete MCP client that can:
    - Connect to MCP servers via various transports
    - Perform protocol handshake and initialization
    - Discover and call tools
    - List and read resources
    - Handle async request/response patterns
    - Manage connection lifecycle
    
    The client uses futures for request/response correlation,
    allowing multiple concurrent requests.
    
    Attributes:
        name: Client name for identification
        version: Client version string
        protocol: MCPProtocol instance for message handling
        transport: Transport layer for communication
        capabilities: Client capabilities object
        is_initialized: Whether handshake is complete
        server_info: Connected server information
        server_capabilities: Server's declared capabilities
        pending_requests: Map of request IDs to futures
        tools: Cached tool definitions
        resources: Cached resource metadata
    """
    
    def __init__(self, name: str, version: str, transport: Optional[MCPTransport] = None):
        """Initialize MCP client.
        
        Args:
            name: Client name for identification
            version: Client version string (e.g., "1.0.0")
            transport: Optional transport instance (defaults to StdioTransport)
            
        Example:
            >>> # Default stdio transport
            >>> client = MCPClient("my-client", "1.0.0")
            >>> 
            >>> # Custom WebSocket transport
            >>> ws_transport = WebSocketTransport("ws://localhost:8080")
            >>> client = MCPClient("my-client", "1.0.0", ws_transport)
        """
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
        """Handle incoming message from transport.
        
        Processes both responses to client requests and server-initiated
        requests (notifications).
        
        Args:
            message: Raw JSON string from transport
            
        Message Types:
            - MCPResponse: Response to a pending client request
            - MCPRequest: Server-initiated notification or request
            - MCPError: Parsing error (logged and ignored)
            
        Note:
            Unknown request IDs are logged but don't crash the client.
            This handles cases where responses arrive after timeout.
        """
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
        """Handle server-initiated requests.
        
        Processes notifications and requests from the server.
        Common notifications include:
        - initialized: Server confirmed initialization
        - cancelled: Request was cancelled
        
        Args:
            request: MCPRequest from server
            
        Note:
            Currently only handles standard notifications.
            Custom server requests are logged but ignored.
        """
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
        """Send request and wait for response.
        
        Core request/response mechanism with timeout handling.
        
        Args:
            method: MCP method name (e.g., "tools/list")
            params: Optional method parameters
            request_id: Optional request ID (auto-generated if not provided)
            
        Returns:
            Response result from server
            
        Raises:
            RuntimeError: If transport not connected or request times out
            Exception: If server returns error response
            
        Note:
            Default timeout is 30 seconds. Long-running operations
            may need custom timeout handling.
            
        Example:
            >>> result = await client._send_request(
            ...     "tools/call",
            ...     {"name": "calculator", "arguments": {"x": 5}}
            ... )
        """
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
        """Connect to MCP server.
        
        Establishes transport connection and performs MCP handshake.
        
        Connection Process:
        1. Connect transport layer
        2. Start message receive loop
        3. Send initialize request
        4. Wait for server response
        5. Send initialized notification
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            RuntimeError: If transport fails to connect
            
        Example:
            >>> client = MCPClient("my-client", "1.0.0")
            >>> if await client.connect():
            ...     print("Connected to server")
            ... else:
            ...     print("Connection failed")
        """
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
        """Disconnect from MCP server.
        
        Gracefully closes transport connection and resets client state.
        Safe to call multiple times.
        """
        await self.transport.disconnect()
        self.is_initialized = False
        logger.info("MCP client disconnected")
    
    async def initialize(self):
        """Initialize connection with server.
        
        Performs MCP handshake by sending initialize request and
        processing server capabilities.
        
        Handshake Process:
        1. Send initialize with client info and capabilities
        2. Receive server info and capabilities
        3. Send initialized notification
        4. Mark client as initialized
        
        Raises:
            Exception: If initialization fails
            
        Note:
            Called automatically by connect(). Should not be
            called directly unless implementing custom connection logic.
        """
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
        """List available tools from server.
        
        Discovers all tools the server provides and caches them locally.
        
        Returns:
            List of tool definitions, each containing:
                - name: Tool identifier
                - description: Human-readable description
                - inputSchema: JSON Schema for parameters
                
        Example:
            >>> tools = await client.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool['name']}: {tool['description']}")
            
        Note:
            Results are cached in self.tools for quick lookup.
        """
        result = await self._send_request("tools/list")
        tools = result.get("tools", [])
        
        # Update local tools cache
        self.tools.clear()
        for tool in tools:
            self.tools[tool["name"]] = tool
        
        return tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the server.
        
        Executes a specific tool with provided arguments.
        
        Args:
            name: Tool name to execute
            arguments: Tool-specific arguments
            
        Returns:
            Tool execution result containing:
                - content: Array of content blocks
                - isError: Whether execution failed
                
        Raises:
            Exception: If tool not found or execution fails
            
        Example:
            >>> result = await client.call_tool(
            ...     "calculator",
            ...     {"operation": "add", "x": 5, "y": 3}
            ... )
            >>> print(result["content"][0]["text"])  # "8"
        """
        params = {
            "name": name,
            "arguments": arguments
        }
        
        result = await self._send_request("tools/call", params)
        return result
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from server.
        
        Discovers all resources the server provides and caches them locally.
        
        Returns:
            List of resource metadata, each containing:
                - uri: Resource identifier
                - name: Human-readable name
                - description: Resource description
                - mimeType: Content MIME type
                
        Example:
            >>> resources = await client.list_resources()
            >>> for resource in resources:
            ...     print(f"{resource['uri']}: {resource['name']}")
            
        Note:
            Results are cached in self.resources for quick lookup.
        """
        result = await self._send_request("resources/list")
        resources = result.get("resources", [])
        
        # Update local resources cache
        self.resources.clear()
        for resource in resources:
            self.resources[resource["uri"]] = resource
        
        return resources
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from the server.
        
        Retrieves the content of a specific resource.
        
        Args:
            uri: Resource URI to read
            
        Returns:
            Resource content containing:
                - contents: Array with resource data
                    - uri: Resource URI
                    - mimeType: Content MIME type
                    - text: Resource content as string
                    
        Raises:
            Exception: If resource not found or read fails
            
        Example:
            >>> result = await client.read_resource("config://settings")
            >>> content = result["contents"][0]["text"]
            >>> settings = json.loads(content)
        """
        params = {"uri": uri}
        result = await self._send_request("resources/read", params)
        return result
    
    async def ping(self) -> Dict[str, Any]:
        """Ping the server.
        
        Simple health check to verify server is responsive.
        
        Returns:
            Ping response containing:
                - pong: True
                - timestamp: Server timestamp
                
        Example:
            >>> response = await client.ping()
            >>> print(f"Server responded at {response['timestamp']}")
        """
        result = await self._send_request("ping")
        return result
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool information by name.
        
        Looks up tool in local cache. Call list_tools() first to populate.
        
        Args:
            name: Tool name to look up
            
        Returns:
            Tool definition if found, None otherwise
        """
        return self.tools.get(name)
    
    def get_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get resource information by URI.
        
        Looks up resource in local cache. Call list_resources() first to populate.
        
        Args:
            uri: Resource URI to look up
            
        Returns:
            Resource metadata if found, None otherwise
        """
        return self.resources.get(uri)
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information.
        
        Returns current client state and statistics.
        
        Returns:
            Dictionary containing:
                - name: Client name
                - version: Client version
                - protocol_version: MCP protocol version
                - is_initialized: Connection status
                - capabilities: Client capabilities
                - server_info: Connected server details
                - server_capabilities: Server's capabilities
                - tools_count: Number of cached tools
                - resources_count: Number of cached resources
                
        Example:
            >>> info = client.get_client_info()
            >>> print(f"Connected to: {info['server_info']['name']}")
        """
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
    """High-level MCP client session manager.
    
    Provides a context manager for automatic connection lifecycle.
    Ensures proper connection and disconnection even if errors occur.
    
    Example:
        >>> client = MCPClient("my-client", "1.0.0")
        >>> async with MCPClientSession(client) as session:
        ...     # Connection is established
        ...     tools = await session.list_tools()
        ...     result = await session.call_tool("test", {})
        >>> # Connection is automatically closed
        
    Attributes:
        client: MCPClient instance to manage
        is_connected: Current connection status
    """
    
    def __init__(self, client: MCPClient):
        """Initialize session manager.
        
        Args:
            client: MCPClient instance to manage
        """
        self.client = client
        self.is_connected = False
    
    async def __aenter__(self):
        """Async context manager entry.
        
        Connects to server and returns client for use.
        
        Returns:
            Connected MCPClient instance
            
        Raises:
            RuntimeError: If connection fails
        """
        self.is_connected = await self.client.connect()
        if not self.is_connected:
            raise RuntimeError("Failed to connect to MCP server")
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.
        
        Ensures client is disconnected even if exceptions occurred.
        
        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception value if error occurred
            exc_tb: Exception traceback if error occurred
        """
        if self.is_connected:
            await self.client.disconnect()
        self.is_connected = False