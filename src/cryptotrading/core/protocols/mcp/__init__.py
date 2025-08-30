"""
Model Context Protocol (MCP) implementation for Strands framework
Provides standard MCP server/client architecture with JSON-RPC protocol
"""

from .capabilities import ClientCapabilities, ServerCapabilities
from .client import MCPClient
from .protocol import MCPError, MCPProtocol, MCPRequest, MCPResponse
from .resources import Resource, ResourceTemplate
from .server import MCPServer
from .tools import MCPTool, ToolResult
from .transport import SSETransport, StdioTransport, WebSocketTransport

__all__ = [
    "MCPServer",
    "MCPClient",
    "StdioTransport",
    "WebSocketTransport",
    "SSETransport",
    "MCPProtocol",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "ServerCapabilities",
    "ClientCapabilities",
    "Resource",
    "ResourceTemplate",
    "MCPTool",
    "ToolResult",
]
