"""
MCP JSON-RPC Protocol Implementation
Implements the official Model Context Protocol specification
"""
import json
import uuid
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum


class MCPErrorCode(Enum):
    """Standard MCP error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP-specific errors
    INITIALIZATION_FAILED = -32000
    RESOURCE_NOT_FOUND = -32001
    TOOL_NOT_FOUND = -32002
    CAPABILITY_NOT_SUPPORTED = -32003


@dataclass
class MCPError:
    """MCP error response"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"code": self.code, "message": self.message}
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class MCPRequest:
    """MCP JSON-RPC request"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method", ""),
            params=data.get("params"),
            id=data.get("id")
        )


@dataclass
class MCPResponse:
    """MCP JSON-RPC response"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[MCPError] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            result["id"] = self.id
        if self.result is not None:
            result["result"] = self.result
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        error = None
        if "error" in data:
            error_data = data["error"]
            error = MCPError(
                code=error_data.get("code", 0),
                message=error_data.get("message", ""),
                data=error_data.get("data")
            )
        
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            result=data.get("result"),
            error=error
        )


class MCPProtocol:
    """MCP protocol handler for JSON-RPC messaging"""
    
    def __init__(self):
        self.version = "2024-11-05"
        self.handlers: Dict[str, callable] = {}
    
    def register_handler(self, method: str, handler: callable):
        """Register a method handler"""
        self.handlers[method] = handler
    
    def create_request(self, method: str, params: Optional[Dict[str, Any]] = None, 
                      request_id: Optional[Union[str, int]] = None) -> MCPRequest:
        """Create an MCP request"""
        return MCPRequest(method=method, params=params, id=request_id)
    
    def create_response(self, request_id: Optional[Union[str, int]], 
                       result: Optional[Dict[str, Any]] = None,
                       error: Optional[MCPError] = None) -> MCPResponse:
        """Create an MCP response"""
        return MCPResponse(id=request_id, result=result, error=error)
    
    def create_error_response(self, request_id: Optional[Union[str, int]], 
                             code: MCPErrorCode, message: str,
                             data: Optional[Dict[str, Any]] = None) -> MCPResponse:
        """Create an error response"""
        error = MCPError(code=code.value, message=message, data=data)
        return MCPResponse(id=request_id, error=error)
    
    def parse_message(self, message: str) -> Union[MCPRequest, MCPResponse, MCPError]:
        """Parse a JSON message into MCP request or response"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return MCPError(
                code=MCPErrorCode.PARSE_ERROR.value,
                message=f"Parse error: {str(e)}"
            )
        
        # Validate JSON-RPC structure
        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
            return MCPError(
                code=MCPErrorCode.INVALID_REQUEST.value,
                message="Invalid JSON-RPC request"
            )
        
        # Determine if it's a request or response
        if "method" in data:
            return MCPRequest.from_dict(data)
        elif "result" in data or "error" in data:
            return MCPResponse.from_dict(data)
        else:
            return MCPError(
                code=MCPErrorCode.INVALID_REQUEST.value,
                message="Invalid MCP message format"
            )
    
    def serialize_message(self, message: Union[MCPRequest, MCPResponse]) -> str:
        """Serialize MCP message to JSON string"""
        return json.dumps(message.to_dict(), separators=(',', ':'))
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an incoming MCP request"""
        if request.method not in self.handlers:
            return self.create_error_response(
                request.id,
                MCPErrorCode.METHOD_NOT_FOUND,
                f"Method '{request.method}' not found"
            )
        
        try:
            handler = self.handlers[request.method]
            result = await handler(request.params or {})
            return self.create_response(request.id, result)
        except Exception as e:
            return self.create_error_response(
                request.id,
                MCPErrorCode.INTERNAL_ERROR,
                f"Internal error: {str(e)}"
            )
    
    def validate_capabilities(self, capabilities: Dict[str, Any]) -> bool:
        """Validate capability structure"""
        required_fields = ["experimental", "roots"]
        for field in required_fields:
            if field not in capabilities:
                return False
        return True
    
    def create_initialize_request(self, client_info: Dict[str, Any], 
                                 capabilities: Dict[str, Any]) -> MCPRequest:
        """Create standard initialize request"""
        params = {
            "protocolVersion": self.version,
            "clientInfo": client_info,
            "capabilities": capabilities
        }
        return self.create_request("initialize", params)
    
    def create_initialize_response(self, server_info: Dict[str, Any],
                                  capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Create standard initialize response"""
        return {
            "protocolVersion": self.version,
            "serverInfo": server_info,
            "capabilities": capabilities
        }
    
    def create_list_tools_request(self) -> MCPRequest:
        """Create list_tools request"""
        return self.create_request("tools/list")
    
    def create_call_tool_request(self, name: str, arguments: Dict[str, Any]) -> MCPRequest:
        """Create call_tool request"""
        params = {
            "name": name,
            "arguments": arguments
        }
        return self.create_request("tools/call", params)
    
    def create_list_resources_request(self) -> MCPRequest:
        """Create list_resources request"""
        return self.create_request("resources/list")
    
    def create_read_resource_request(self, uri: str) -> MCPRequest:
        """Create read_resource request"""
        params = {"uri": uri}
        return self.create_request("resources/read", params)