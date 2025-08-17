"""
MCP Server Implementation

This module implements a complete Model Context Protocol (MCP) server with:
- Standard MCP method handlers (initialize, list_tools, call_tool, etc.)
- Tool and resource management
- Transport abstraction (stdio, WebSocket, SSE)
- Capability negotiation
- Error handling and logging

The server follows the official MCP specification and provides a foundation
for exposing tools and resources to language models.

Key Components:
    MCPServer: Main server class that handles connections and requests
    Tool Management: Register and execute tools with proper validation
    Resource Management: Serve static and dynamic resources
    Transport Layer: Abstract transport handling for different protocols

Example:
    >>> from mcp.transport import StdioTransport
    >>> from mcp.tools import MCPTool
    >>> 
    >>> # Create server
    >>> server = MCPServer("my-server", "1.0.0")
    >>> 
    >>> # Add a tool
    >>> tool = MCPTool("hello", "Say hello", {}, lambda: "Hello!")
    >>> server.add_tool(tool)
    >>> 
    >>> # Start server
    >>> await server.start()
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .protocol import MCPProtocol, MCPRequest, MCPResponse, MCPErrorCode
from .transport import MCPTransport, StdioTransport
from .capabilities import ServerCapabilities
from .tools import MCPTool, ToolResult
from .resources import Resource
from .security import (
    SecurityMiddleware, 
    VercelSecurityMiddleware,
    AuthenticationError,
    RateLimitExceeded,
    ValidationError,
    SecurityContext
)

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP server implementation with full protocol support.
    
    Provides a complete MCP server that can:
    - Handle client connections via various transports
    - Manage tools and resources
    - Process standard MCP methods
    - Negotiate capabilities with clients
    - Handle errors gracefully
    
    The server lifecycle:
    1. Create server instance
    2. Add tools and resources
    3. Start server (connects transport)
    4. Handle initialize handshake
    5. Process requests
    6. Stop server on shutdown
    
    Attributes:
        name: Server name for identification
        version: Server version string
        protocol: MCPProtocol instance for message handling
        transport: Transport layer for communication
        capabilities: Server capabilities object
        is_initialized: Whether handshake is complete
        client_info: Connected client information
        tools: Registry of available tools
        resources: Registry of available resources
    """
    
    def __init__(self, name: str, version: str, transport: Optional[MCPTransport] = None,
                 security_middleware: Optional[SecurityMiddleware] = None):
        """Initialize MCP server.
        
        Args:
            name: Server name for identification
            version: Server version string (e.g., "1.0.0")
            transport: Optional transport instance (defaults to StdioTransport)
            security_middleware: Optional security middleware (defaults to VercelSecurityMiddleware)
            
        Example:
            >>> # Default stdio transport with Vercel security
            >>> server = MCPServer("my-server", "1.0.0")
            >>> 
            >>> # Custom WebSocket transport
            >>> ws_transport = WebSocketTransport("ws://localhost:8080")
            >>> server = MCPServer("my-server", "1.0.0", ws_transport)
        """
        self.name = name
        self.version = version
        self.protocol = MCPProtocol()
        self.transport = transport or StdioTransport()
        self.capabilities = ServerCapabilities()
        self.is_initialized = False
        self.client_info = {}
        
        # Initialize security middleware
        self.security_middleware = security_middleware or VercelSecurityMiddleware()
        
        # Server state
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, Resource] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        
        # Register standard MCP methods
        self._register_standard_methods()
        
        # Initialize new MCP features
        self._setup_advanced_features()
        
        # Set up transport
        self.transport.set_message_handler(self._handle_message)
    
    def _register_standard_methods(self):
        """Register standard MCP methods.
        
        Registers handlers for all required MCP protocol methods:
        - initialize: Handshake and capability negotiation
        - initialized: Confirmation of successful initialization
        - tools/list: List available tools
        - tools/call: Execute a specific tool
        - resources/list: List available resources
        - resources/read: Read resource content
        - ping: Health check endpoint
        """
        self.protocol.register_handler("initialize", self._handle_initialize)
        self.protocol.register_handler("initialized", self._handle_initialized)
        self.protocol.register_handler("tools/list", self._handle_list_tools)
        self.protocol.register_handler("tools/call", self._handle_call_tool)
        self.protocol.register_handler("resources/list", self._handle_list_resources)
        self.protocol.register_handler("resources/read", self._handle_read_resource)
        self.protocol.register_handler("ping", self._handle_ping)
        self.protocol.register_handler("security/status", self._handle_security_status)
        
        # Register new MCP specification methods
        self.protocol.register_handler("prompts/list", self._handle_list_prompts)
        self.protocol.register_handler("prompts/get", self._handle_get_prompt)
        self.protocol.register_handler("sampling/createMessage", self._handle_sampling_create_message)
        self.protocol.register_handler("resources/subscribe", self._handle_resource_subscribe)
        self.protocol.register_handler("resources/unsubscribe", self._handle_resource_unsubscribe)
        self.protocol.register_handler("roots/list", self._handle_list_roots)
    
    async def _handle_message(self, message: str, headers: Optional[Dict[str, str]] = None):
        """Handle incoming message from transport.
        
        Main message processing pipeline:
        1. Parse JSON-RPC message
        2. Apply security middleware (auth, rate limiting, validation)
        3. Route to appropriate handler
        4. Send response back via transport
        
        Args:
            message: Raw JSON string from transport
            headers: Optional HTTP headers (for WebSocket/SSE transports)
            
        Note:
            Errors are logged but don't crash the server.
            Security violations are handled according to policy.
        """
        try:
            parsed = self.protocol.parse_message(message)
            
            if isinstance(parsed, MCPRequest):
                # Apply security middleware
                try:
                    processed_params, security_context = await self.security_middleware.process_request(
                        parsed.method, 
                        parsed.params or {}, 
                        headers or {}
                    )
                    
                    # Update request with processed params
                    parsed.params = processed_params
                    
                    # Handle the request
                    response = await self.protocol.handle_request(parsed)
                    
                except AuthenticationError as e:
                    response = self.protocol.create_error_response(
                        parsed.id, MCPErrorCode.INTERNAL_ERROR, f"Authentication failed: {str(e)}"
                    )
                except RateLimitExceeded as e:
                    response = self.protocol.create_error_response(
                        parsed.id, MCPErrorCode.INTERNAL_ERROR, f"Rate limit exceeded: {str(e)}"
                    )
                    # Add Retry-After header if transport supports it
                    if hasattr(response, 'headers'):
                        response.headers = {'Retry-After': str(e.retry_after)}
                except ValidationError as e:
                    response = self.protocol.create_error_response(
                        parsed.id, MCPErrorCode.INVALID_PARAMS, f"Validation failed: {str(e)}"
                    )
                
                # Send response with headers
                if response:
                    # Add security context headers to response if available
                    if 'security_context' in locals() and hasattr(security_context, 'response_headers'):
                        if hasattr(response, 'headers'):
                            response.headers.update(security_context.response_headers)
                        else:
                            # Add headers to response dict if it's a dictionary
                            if isinstance(response, dict):
                                response['headers'] = security_context.response_headers
                    
                    response_str = self.protocol.serialize_message(response)
                    await self.transport.send_message(response_str)
            else:
                logger.error(f"Invalid message received: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            # Send generic error response if we can extract request ID
            try:
                import json
                msg_data = json.loads(message)
                request_id = msg_data.get('id')
                error_response = self.protocol.create_error_response(
                    request_id, MCPErrorCode.INTERNAL_ERROR, "Internal server error"
                )
                response_str = self.protocol.serialize_message(error_response)
                await self.transport.send_message(response_str)
            except Exception as transport_error:
                logger.error(f"Failed to send error response: {transport_error}")
                # Can't send error response, connection may be broken
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request.
        
        First method called during MCP handshake. Validates protocol version,
        stores client information, and negotiates capabilities.
        
        Args:
            params: Initialize parameters containing:
                - protocolVersion: Client's protocol version
                - clientInfo: Client identification
                - capabilities: Client capabilities
                
        Returns:
            Initialize response with:
                - protocolVersion: Server's protocol version
                - serverInfo: Server identification
                - capabilities: Negotiated capabilities
                
        Note:
            Protocol version mismatches are logged but not fatal.
            This allows for backward compatibility.
        """
        logger.info("Handling initialize request")
        
        # Validate protocol version
        protocol_version = params.get("protocolVersion")
        if protocol_version != self.protocol.version:
            logger.warning(f"Protocol version mismatch: client={protocol_version}, server={self.protocol.version}")
        
        # Store client info
        self.client_info = params.get("clientInfo", {})
        client_capabilities = params.get("capabilities", {})
        
        # Update server capabilities based on client
        self._negotiate_capabilities(client_capabilities)
        
        # Create server info
        server_info = {
            "name": self.name,
            "version": self.version
        }
        
        # Return initialize response
        return self.protocol.create_initialize_response(
            server_info, 
            self.capabilities.to_dict()
        )
    
    async def _handle_initialized(self, params: Dict[str, Any]) -> None:
        """Handle initialized notification.
        
        Called by client after successful initialize response.
        Marks the server as ready to handle requests.
        
        Args:
            params: Empty parameters (notification has no data)
            
        Note:
            This is a notification (no response expected).
            After this, the server can process tool/resource requests.
        """
        self.is_initialized = True
        logger.info("MCP server initialized successfully")
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list_tools request.
        
        Returns information about all registered tools including their
        names, descriptions, and parameter schemas.
        
        Args:
            params: Empty parameters (list doesn't need filtering)
            
        Returns:
            Dictionary with 'tools' array containing tool definitions.
            Each tool has:
                - name: Tool identifier
                - description: Human-readable description
                - inputSchema: JSON Schema for parameters
                
        Example Response:
            {
                "tools": [{
                    "name": "calculator",
                    "description": "Perform calculations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        }
                    }
                }]
            }
        """
        tools_list = []
        for tool_name, tool in self.tools.items():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": list(tool.parameters.keys()) if tool.parameters else []
                }
            })
        
        return {"tools": tools_list}
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call_tool request.
        
        Executes a specific tool with provided arguments and returns results.
        
        Args:
            params: Tool execution parameters:
                - name: Tool name to execute
                - arguments: Tool-specific arguments
                
        Returns:
            Tool execution result:
                - content: Array of content blocks (text, images, etc.)
                - isError: Whether execution failed
                
        Raises:
            Exception: If tool not found or execution fails
            
        Example:
            Request params: {
                "name": "calculator",
                "arguments": {"expression": "2 + 2"}
            }
            Response: {
                "content": [{"type": "text", "text": "4"}],
                "isError": false
            }
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            raise Exception(f"Tool '{tool_name}' not found")
        
        tool = self.tools[tool_name]
        
        try:
            # Execute tool
            if tool_name in self.tool_handlers:
                result = await self.tool_handlers[tool_name](arguments)
            else:
                # Use tool's function if available
                if hasattr(tool, 'function') and tool.function:
                    if asyncio.iscoroutinefunction(tool.function):
                        result = await tool.function(**arguments)
                    else:
                        result = tool.function(**arguments)
                else:
                    raise Exception(f"No handler for tool '{tool_name}'")
            
            # Create tool result
            if isinstance(result, ToolResult):
                tool_result = result
            else:
                tool_result = ToolResult(
                    content=[{"type": "text", "text": str(result)}],
                    isError=False
                )
            
            return {
                "content": tool_result.content,
                "isError": tool_result.isError
            }
            
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            }
    
    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list_resources request.
        
        Returns information about all available resources including their
        URIs, names, descriptions, and MIME types.
        
        Args:
            params: Empty parameters (list doesn't need filtering)
            
        Returns:
            Dictionary with 'resources' array containing resource metadata.
            Each resource has:
                - uri: Resource identifier (e.g., "file://config.json")
                - name: Human-readable name
                - description: Resource description
                - mimeType: Content MIME type
                
        Example Response:
            {
                "resources": [{
                    "uri": "config://settings",
                    "name": "Server Settings",
                    "description": "Current server configuration",
                    "mimeType": "application/json"
                }]
            }
        """
        resources_list = []
        for resource_uri, resource in self.resources.items():
            resources_list.append({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mime_type
            })
        
        return {"resources": resources_list}
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle read_resource request.
        
        Reads and returns the content of a specific resource.
        
        Args:
            params: Resource read parameters:
                - uri: Resource URI to read
                
        Returns:
            Resource content:
                - contents: Array with single content object containing:
                    - uri: Resource URI
                    - mimeType: Content MIME type
                    - text: Resource content as string
                    
        Raises:
            Exception: If resource not found or read fails
            
        Example:
            Request params: {"uri": "config://settings"}
            Response: {
                "contents": [{
                    "uri": "config://settings",
                    "mimeType": "application/json",
                    "text": "{\"debug\": true}"
                }]
            }
        """
        uri = params.get("uri")
        
        if uri not in self.resources:
            raise Exception(f"Resource '{uri}' not found")
        
        resource = self.resources[uri]
        
        try:
            # Read resource content
            content = await resource.read()
            
            return {
                "contents": [{
                    "uri": resource.uri,
                    "mimeType": resource.mime_type,
                    "text": content
                }]
            }
            
        except Exception as e:
            logger.error(f"Error reading resource '{uri}': {e}")
            raise Exception(f"Failed to read resource: {str(e)}")
    
    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request.
        
        Simple health check endpoint for monitoring.
        
        Args:
            params: Empty parameters
            
        Returns:
            Pong response with current timestamp
            
        Example Response:
            {
                "pong": true,
                "timestamp": "2024-01-15T10:30:00.000Z"
            }
        """
        return {"pong": True, "timestamp": datetime.now().isoformat()}
    
    async def _handle_security_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle security status request.
        
        Returns current security configuration and metrics.
        Requires admin scope for access.
        
        Args:
            params: Empty parameters
            
        Returns:
            Security status including configuration, metrics, and health
            
        Example Response:
            {
                "config": {
                    "require_auth": true,
                    "rate_limiting_enabled": true,
                    "input_validation_enabled": true
                },
                "metrics": {
                    "total_requests": 1234,
                    "authenticated_requests": 1100,
                    "rate_limited_requests": 23,
                    "validation_failures": 5
                }
            }
        """
        return await self.security_middleware.get_security_status()
    
    def _negotiate_capabilities(self, client_capabilities: Dict[str, Any]):
        """Negotiate capabilities with client.
        
        Adjusts server capabilities based on what the client supports.
        This ensures compatibility and optimal feature usage.
        
        Args:
            client_capabilities: Client's capability object
            
        Note:
            Current implementation uses default server capabilities.
            Future versions could disable features not supported by client.
        """
        # For now, use default server capabilities
        # In a full implementation, this would adjust based on client capabilities
        pass
    
    def add_tool(self, tool: MCPTool, handler: Optional[Callable] = None):
        """Add a tool to the server.
        
        Registers a tool that clients can discover and execute.
        
        Args:
            tool: MCPTool instance with name, description, and parameters
            handler: Optional async function to handle execution
                     (if not provided, tool.function will be used)
                     
        Example:
            >>> # Simple tool with inline function
            >>> tool = MCPTool(
            ...     name="echo",
            ...     description="Echo input",
            ...     parameters={"message": {"type": "string"}},
            ...     function=lambda message: f"Echo: {message}"
            ... )
            >>> server.add_tool(tool)
            >>> 
            >>> # Tool with separate handler
            >>> async def complex_handler(params):
            ...     # Complex async logic
            ...     return result
            >>> server.add_tool(tool, complex_handler)
        """
        self.tools[tool.name] = tool
        if handler:
            self.tool_handlers[tool.name] = handler
        logger.info(f"Added tool: {tool.name}")
    
    def add_resource(self, resource: Resource):
        """Add a resource to the server.
        
        Registers a resource that clients can list and read.
        
        Args:
            resource: Resource instance with URI, name, and read method
            
        Example:
            >>> # Static resource
            >>> resource = Resource(
            ...     uri="config://settings",
            ...     name="Settings",
            ...     description="Server settings",
            ...     mime_type="application/json",
            ...     read_func=lambda: '{"debug": true}'
            ... )
            >>> server.add_resource(resource)
        """
        self.resources[resource.uri] = resource
        logger.info(f"Added resource: {resource.uri}")
    
    def register_tool_handler(self, tool_name: str, handler: Callable):
        """Register a handler for a specific tool.
        
        Updates or adds a handler for an existing tool.
        Useful for dynamic handler updates.
        
        Args:
            tool_name: Name of the tool to update
            handler: Async function to handle tool execution
            
        Note:
            Tool must already be registered with add_tool().
        """
        self.tool_handlers[tool_name] = handler
    
    async def start(self):
        """Start the MCP server.
        
        Connects transport and begins listening for messages.
        Blocks until server is stopped or transport disconnects.
        
        Lifecycle:
        1. Connect transport
        2. Start message receive loop
        3. Wait for initialization
        4. Process requests
        5. Clean shutdown on disconnect/interrupt
        
        Raises:
            RuntimeError: If transport fails to connect
            
        Example:
            >>> server = MCPServer("my-server", "1.0.0")
            >>> # Add tools/resources
            >>> await server.start()  # Blocks until shutdown
        """
        logger.info(f"Starting MCP server: {self.name} v{self.version}")
        
        # Connect transport
        if not await self.transport.connect():
            raise RuntimeError("Failed to connect transport")
        
        # Start receiving messages
        receive_task = asyncio.create_task(self.transport.receive_messages())
        
        try:
            # Wait for initialization or shutdown
            while True:
                await asyncio.sleep(1)
                if not self.transport.is_connected:
                    break
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            receive_task.cancel()
            await self.transport.disconnect()
            logger.info("MCP server stopped")
    
    async def stop(self):
        """Stop the MCP server.
        
        Gracefully disconnects transport and cleans up resources.
        Safe to call multiple times.
        """
        await self.transport.disconnect()
        logger.info("MCP server stopped")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information.
        
        Returns current server state and statistics.
        
        Returns:
            Dictionary containing:
                - name: Server name
                - version: Server version
                - protocol_version: MCP protocol version
                - is_initialized: Initialization status
                - capabilities: Current capabilities
                - tools_count: Number of registered tools
                - resources_count: Number of registered resources
                
        Example:
            >>> info = server.get_server_info()
            >>> print(f"Server: {info['name']} v{info['version']}")
            >>> print(f"Tools: {info['tools_count']}")
        """
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol.version,
            "is_initialized": self.is_initialized,
            "capabilities": self.capabilities.to_dict(),
            "tools_count": len(self.tools),
            "resources_count": len(self.resources)
        }
    
    def _setup_advanced_features(self):
        """Setup advanced MCP features"""
        # Import advanced features
        from .prompts import prompt_registry
        from .sampling import sampling_manager
        from .subscriptions import subscription_manager
        from .progress import progress_manager
        from .roots import root_manager
        
        # Store references
        self.prompt_registry = prompt_registry
        self.sampling_manager = sampling_manager
        self.subscription_manager = subscription_manager
        self.progress_manager = progress_manager
        self.root_manager = root_manager
        
        # Set up notification handlers
        self.subscription_manager.add_notification_handler(self._handle_resource_update)
        self.progress_manager.add_notification_handler(self._handle_progress_update)
        
        # Enable advanced capabilities
        self.capabilities.enable_prompts()
        self.capabilities.enable_resources(subscribe=True, list_changed=True)
        self.capabilities.add_experimental("sampling", {"enabled": True})
        self.capabilities.add_experimental("progress", {"enabled": True})
        self.capabilities.add_experimental("roots", {"enabled": True})
    
    async def _handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list request"""
        prompts = self.prompt_registry.list_prompts()
        return {"prompts": prompts}
    
    async def _handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not name:
            raise Exception("Missing required parameter: name")
        
        try:
            messages = await self.prompt_registry.get_prompt_messages(name, arguments)
            return {"messages": messages}
        except ValueError as e:
            raise Exception(str(e))
    
    async def _handle_sampling_create_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sampling/createMessage request"""
        try:
            from .sampling import parse_sampling_request
            request = parse_sampling_request(params)
            response = await self.sampling_manager.create_message(request)
            return response.to_dict()
        except Exception as e:
            raise Exception(f"Sampling failed: {str(e)}")
    
    async def _handle_resource_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/subscribe request"""
        uri = params.get("uri")
        if not uri:
            raise Exception("Missing required parameter: uri")
        
        # Use client ID as subscriber (would be from auth context in real implementation)
        subscriber_id = self.client_info.get("name", "unknown_client")
        
        try:
            subscription = await self.subscription_manager.subscribe(uri, subscriber_id)
            return subscription.to_dict()
        except Exception as e:
            raise Exception(f"Subscription failed: {str(e)}")
    
    async def _handle_resource_unsubscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/unsubscribe request"""
        uri = params.get("uri")
        if not uri:
            raise Exception("Missing required parameter: uri")
        
        subscriber_id = self.client_info.get("name", "unknown_client")
        
        try:
            await self.subscription_manager.unsubscribe(uri, subscriber_id)
            return {"success": True}
        except Exception as e:
            raise Exception(f"Unsubscribe failed: {str(e)}")
    
    async def _handle_list_roots(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle roots/list request"""
        roots = self.root_manager.list_roots()
        return {"roots": roots}
    
    async def _handle_resource_update(self, update):
        """Handle resource update notifications"""
        # Send notification to client
        notification = update.to_dict()
        try:
            response_str = self.protocol.serialize_message(notification)
            await self.transport.send_message(response_str)
        except Exception as e:
            logger.error(f"Failed to send resource update notification: {e}")
    
    async def _handle_progress_update(self, update):
        """Handle progress update notifications"""
        # Send notification to client
        notification = update.to_dict()
        try:
            response_str = self.protocol.serialize_message(notification)
            await self.transport.send_message(response_str)
        except Exception as e:
            logger.error(f"Failed to send progress notification: {e}")
    
    def add_prompt(self, prompt):
        """Add a prompt to the server"""
        self.prompt_registry.register(prompt)
        logger.info(f"Added prompt: {prompt.name}")
    
    def create_progress_tracker(self, total=None, description=None):
        """Create a progress tracker for long-running operations"""
        return self.progress_manager.create_tracker(total, description)
    
    async def subscribe_to_resource(self, uri: str):
        """Subscribe to resource updates (for internal use)"""
        subscriber_id = f"server_{self.name}"
        return await self.subscription_manager.subscribe(uri, subscriber_id)
    
    def add_root_directory(self, path: str, name: str = None):
        """Add a root directory for file access"""
        import os
        abs_path = os.path.abspath(path)
        uri = f"file://{abs_path}"
        self.root_manager.add_root(uri, name)
        logger.info(f"Added root directory: {uri}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request directly (for testing/integration)"""
        from .protocol import MCPRequest
        mcp_request = MCPRequest(
            method=request.get("method"),
            params=request.get("params"),
            id=request.get("id")
        )
        return await self.protocol.handle_request(mcp_request)