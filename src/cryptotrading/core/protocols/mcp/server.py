"""
MCP Server Implementation
Implements a complete MCP server with standard methods and capabilities
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

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP server implementation with crypto trading capabilities"""
    
    def __init__(self, name: str, version: str, transport: Optional[MCPTransport] = None):
        self.name = name
        self.version = version
        self.protocol = MCPProtocol()
        self.transport = transport or StdioTransport()
        self.capabilities = ServerCapabilities()
        self.is_initialized = False
        self.client_info = {}
        
        # Server state
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, Resource] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        
        # Register standard MCP methods
        self._register_standard_methods()
        
        # Set up transport
        self.transport.set_message_handler(self._handle_message)
    
    def _register_standard_methods(self):
        """Register standard MCP methods"""
        self.protocol.register_handler("initialize", self._handle_initialize)
        self.protocol.register_handler("initialized", self._handle_initialized)
        self.protocol.register_handler("tools/list", self._handle_list_tools)
        self.protocol.register_handler("tools/call", self._handle_call_tool)
        self.protocol.register_handler("resources/list", self._handle_list_resources)
        self.protocol.register_handler("resources/read", self._handle_read_resource)
        self.protocol.register_handler("ping", self._handle_ping)
    
    async def _handle_message(self, message: str):
        """Handle incoming message from transport"""
        try:
            parsed = self.protocol.parse_message(message)
            
            if isinstance(parsed, MCPRequest):
                response = await self.protocol.handle_request(parsed)
                if response:
                    response_str = self.protocol.serialize_message(response)
                    await self.transport.send_message(response_str)
            else:
                logger.error(f"Invalid message received: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request"""
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
        """Handle initialized notification"""
        self.is_initialized = True
        logger.info("MCP server initialized successfully")
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list_tools request"""
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
        """Handle call_tool request"""
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
        """Handle list_resources request"""
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
        """Handle read_resource request"""
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
        """Handle ping request"""
        return {"pong": True, "timestamp": datetime.now().isoformat()}
    
    def _negotiate_capabilities(self, client_capabilities: Dict[str, Any]):
        """Negotiate capabilities with client"""
        # For now, use default server capabilities
        # In a full implementation, this would adjust based on client capabilities
        pass
    
    def add_tool(self, tool: MCPTool, handler: Optional[Callable] = None):
        """Add a tool to the server"""
        self.tools[tool.name] = tool
        if handler:
            self.tool_handlers[tool.name] = handler
        logger.info(f"Added tool: {tool.name}")
    
    def add_resource(self, resource: Resource):
        """Add a resource to the server"""
        self.resources[resource.uri] = resource
        logger.info(f"Added resource: {resource.uri}")
    
    def register_tool_handler(self, tool_name: str, handler: Callable):
        """Register a handler for a specific tool"""
        self.tool_handlers[tool_name] = handler
    
    async def start(self):
        """Start the MCP server"""
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
        """Stop the MCP server"""
        await self.transport.disconnect()
        logger.info("MCP server stopped")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol.version,
            "is_initialized": self.is_initialized,
            "capabilities": self.capabilities.to_dict(),
            "tools_count": len(self.tools),
            "resources_count": len(self.resources)
        }