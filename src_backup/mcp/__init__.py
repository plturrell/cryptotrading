"""
Model Context Protocol (MCP) implementation for Strands framework
Provides standard MCP server/client architecture with JSON-RPC protocol
"""

from .server import MCPServer
from .client import MCPClient
from .transport import StdioTransport, WebSocketTransport, SSETransport
from .protocol import MCPProtocol, MCPRequest, MCPResponse, MCPError
from .capabilities import ServerCapabilities, ClientCapabilities
from .resources import Resource, ResourceTemplate
from .tools import MCPTool, ToolResult

__all__ = [
    'MCPServer',
    'MCPClient',
    'StdioTransport',
    'WebSocketTransport', 
    'SSETransport',
    'MCPProtocol',
    'MCPRequest',
    'MCPResponse',
    'MCPError',
    'ServerCapabilities',
    'ClientCapabilities',
    'Resource',
    'ResourceTemplate',
    'MCPTool',
    'ToolResult'
]