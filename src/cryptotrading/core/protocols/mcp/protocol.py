"""
MCP JSON-RPC Protocol Implementation

This module implements the official Model Context Protocol (MCP) specification
for JSON-RPC 2.0 messaging between language models and external tools/resources.

The MCP protocol enables structured communication for:
- Tool discovery and invocation
- Resource management and access
- Capability negotiation
- Error handling

Key Components:
    MCPProtocol: Main protocol handler for JSON-RPC messaging
    MCPRequest: Request message structure
    MCPResponse: Response message structure
    MCPError: Error response structure
    MCPErrorCode: Standard error codes

Example:
    >>> protocol = MCPProtocol()
    >>> request = protocol.create_request("tools/list")
    >>> response = await protocol.handle_request(request)
"""
import json
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .security.error_sanitizer import get_error_sanitizer, sanitize_mcp_error


class MCPErrorCode(Enum):
    """Standard MCP error codes following JSON-RPC 2.0 specification.

    Error codes are divided into:
    - Standard JSON-RPC errors (-32700 to -32603)
    - MCP-specific errors (-32000 and below)

    Attributes:
        PARSE_ERROR: Invalid JSON was received
        INVALID_REQUEST: The JSON sent is not a valid Request object
        METHOD_NOT_FOUND: The method does not exist / is not available
        INVALID_PARAMS: Invalid method parameter(s)
        INTERNAL_ERROR: Internal JSON-RPC error
        INITIALIZATION_FAILED: MCP initialization failed
        RESOURCE_NOT_FOUND: Requested resource does not exist
        TOOL_NOT_FOUND: Requested tool does not exist
        CAPABILITY_NOT_SUPPORTED: Requested capability is not supported
    """

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
    """MCP error response structure.

    Represents an error that occurred during request processing.
    Follows JSON-RPC 2.0 error object specification.

    Args:
        code: A number indicating the error type
        message: A string providing a short description of the error
        data: Optional additional information about the error

    Example:
        >>> error = MCPError(
        ...     code=-32601,
        ...     message="Method not found",
        ...     data={"method": "unknown_method"}
        ... )
    """

    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for JSON serialization.

        Returns:
            Dict containing error code, message, and optional data
        """
        result = {"code": self.code, "message": self.message}
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class MCPRequest:
    """MCP JSON-RPC request message.

    Represents a request from client to server following JSON-RPC 2.0 spec.
    All requests must have a method name and may include parameters.

    Args:
        jsonrpc: Protocol version (always "2.0")
        method: The method name to invoke
        params: Optional parameters for the method
        id: Request identifier for matching responses

    Note:
        If id is not provided, a UUID will be automatically generated.

    Example:
        >>> request = MCPRequest(
        ...     method="tools/call",
        ...     params={"name": "calculator", "arguments": {"x": 5, "y": 3}}
        ... )
    """

    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

    def __post_init__(self):
        """Post-initialization to ensure request has an ID."""
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary format for JSON serialization.

        Returns:
            Dict containing jsonrpc version, method, params, and id
        """
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPRequest":
        """Create MCPRequest from dictionary data.

        Args:
            data: Dictionary containing request fields

        Returns:
            MCPRequest instance

        Note:
            Missing fields will use defaults (jsonrpc="2.0", method="")
        """
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method", ""),
            params=data.get("params"),
            id=data.get("id"),
        )


@dataclass
class MCPResponse:
    """MCP JSON-RPC response message.

    Represents a response from server to client following JSON-RPC 2.0 spec.
    A response must have either a result or an error, but not both.

    Args:
        jsonrpc: Protocol version (always "2.0")
        id: Request ID this response corresponds to
        result: Success result data (mutually exclusive with error)
        error: Error information (mutually exclusive with result)

    Example:
        >>> # Success response
        >>> response = MCPResponse(
        ...     id="123",
        ...     result={"output": "Calculation complete"}
        ... )
        >>>
        >>> # Error response
        >>> response = MCPResponse(
        ...     id="123",
        ...     error=MCPError(code=-32601, message="Method not found")
        ... )
    """

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
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResponse":
        error = None
        if "error" in data:
            error_data = data["error"]
            error = MCPError(
                code=error_data.get("code", 0),
                message=error_data.get("message", ""),
                data=error_data.get("data"),
            )

        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            result=data.get("result"),
            error=error,
        )


class MCPProtocol:
    """MCP protocol handler for JSON-RPC messaging.

    Main protocol implementation that handles:
    - Message parsing and serialization
    - Request/response creation
    - Method handler registration and dispatch
    - Standard MCP method implementations
    - Error handling and validation

    Attributes:
        version: MCP protocol version string
        handlers: Dictionary mapping method names to handler functions

    Example:
        >>> protocol = MCPProtocol()
        >>>
        >>> # Register a custom handler
        >>> async def my_handler(params):
        ...     return {"result": "success"}
        >>> protocol.register_handler("my_method", my_handler)
        >>>
        >>> # Handle incoming request
        >>> request = protocol.parse_message('{"jsonrpc":"2.0","method":"my_method","id":1}')
        >>> response = await protocol.handle_request(request)
    """

    def __init__(self):
        """Initialize MCP protocol handler.

        Sets protocol version and initializes empty handler registry.
        """
        self.version = "2024-11-05"
        self.handlers: Dict[str, callable] = {}

    def register_handler(self, method: str, handler: callable):
        """Register a method handler.

        Args:
            method: The method name (e.g., "tools/list", "resources/read")
            handler: Async callable that takes params dict and returns result dict

        Example:
            >>> async def list_tools_handler(params):
            ...     return {"tools": [{"name": "calculator", "description": "Basic math"}]}
            >>> protocol.register_handler("tools/list", list_tools_handler)
        """
        self.handlers[method] = handler

    def create_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[Union[str, int]] = None,
    ) -> MCPRequest:
        """Create an MCP request.

        Args:
            method: The method name to invoke
            params: Optional parameters for the method
            request_id: Optional request ID (auto-generated if not provided)

        Returns:
            MCPRequest instance ready for serialization

        Example:
            >>> request = protocol.create_request(
            ...     "tools/call",
            ...     {"name": "calculator", "arguments": {"x": 5, "y": 3}}
            ... )
        """
        return MCPRequest(method=method, params=params, id=request_id)

    def create_response(
        self,
        request_id: Optional[Union[str, int]],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[MCPError] = None,
    ) -> MCPResponse:
        """Create an MCP response.

        Args:
            request_id: The ID from the original request
            result: Success result data (mutually exclusive with error)
            error: Error information (mutually exclusive with result)

        Returns:
            MCPResponse instance ready for serialization

        Raises:
            ValueError: If both result and error are provided

        Example:
            >>> # Success response
            >>> response = protocol.create_response("123", {"output": "Success"})
            >>>
            >>> # Error response
            >>> error = MCPError(code=-32601, message="Not found")
            >>> response = protocol.create_response("123", error=error)
        """
        return MCPResponse(id=request_id, result=result, error=error)

    def create_error_response(
        self,
        request_id: Optional[Union[str, int]],
        code: MCPErrorCode,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> MCPResponse:
        """Create an error response.

        Convenience method for creating error responses with proper error codes.

        Args:
            request_id: The ID from the original request
            code: MCPErrorCode enum value
            message: Human-readable error message
            data: Optional additional error information

        Returns:
            MCPResponse with error field populated

        Example:
            >>> response = protocol.create_error_response(
            ...     "123",
            ...     MCPErrorCode.METHOD_NOT_FOUND,
            ...     "Unknown method: foo/bar",
            ...     {"method": "foo/bar"}
            ... )
        """
        error = MCPError(code=code.value, message=message, data=data)
        return MCPResponse(id=request_id, error=error)

    def parse_message(self, message: str) -> Union[MCPRequest, MCPResponse, MCPError]:
        """Parse a JSON message into MCP request or response.

        Handles JSON parsing, validation, and type detection.

        Args:
            message: JSON string to parse

        Returns:
            MCPRequest if message contains "method" field
            MCPResponse if message contains "result" or "error" field
            MCPError if parsing or validation fails

        Error Conditions:
            - Invalid JSON: Returns PARSE_ERROR
            - Missing jsonrpc="2.0": Returns INVALID_REQUEST
            - Missing required fields: Returns INVALID_REQUEST

        Example:
            >>> request = protocol.parse_message('{"jsonrpc":"2.0","method":"test","id":1}')
            >>> if isinstance(request, MCPRequest):
            ...     print(f"Got request: {request.method}")
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return MCPError(code=MCPErrorCode.PARSE_ERROR.value, message=f"Parse error: {str(e)}")

        # Validate JSON-RPC structure
        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
            return MCPError(
                code=MCPErrorCode.INVALID_REQUEST.value, message="Invalid JSON-RPC request"
            )

        # Determine if it's a request or response
        if "method" in data:
            return MCPRequest.from_dict(data)
        elif "result" in data or "error" in data:
            return MCPResponse.from_dict(data)
        else:
            return MCPError(
                code=MCPErrorCode.INVALID_REQUEST.value, message="Invalid MCP message format"
            )

    def serialize_message(self, message: Union[MCPRequest, MCPResponse]) -> str:
        """Serialize MCP message to JSON string.

        Args:
            message: MCPRequest or MCPResponse to serialize

        Returns:
            Compact JSON string representation

        Note:
            Uses compact JSON format (no extra whitespace) for efficiency.

        Example:
            >>> request = protocol.create_request("test")
            >>> json_str = protocol.serialize_message(request)
            >>> print(json_str)  # {"jsonrpc":"2.0","method":"test","id":"..."}
        """
        return json.dumps(message.to_dict(), separators=(",", ":"))

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an incoming MCP request.

        Dispatches request to registered handler and manages error handling.

        Args:
            request: The MCPRequest to process

        Returns:
            MCPResponse with either result or error

        Error Handling:
            - METHOD_NOT_FOUND: If no handler registered for method
            - INTERNAL_ERROR: If handler raises an exception

        Example:
            >>> # Assuming handler is registered
            >>> request = protocol.create_request("tools/list")
            >>> response = await protocol.handle_request(request)
            >>> if response.error:
            ...     print(f"Error: {response.error.message}")
            ... else:
            ...     print(f"Success: {response.result}")
        """
        if request.method not in self.handlers:
            return self.create_error_response(
                request.id, MCPErrorCode.METHOD_NOT_FOUND, f"Method '{request.method}' not found"
            )

        try:
            handler = self.handlers[request.method]
            result = await handler(request.params or {})
            return self.create_response(request.id, result)
        except Exception as e:
            # Sanitize error message for security
            sanitizer = get_error_sanitizer()
            sanitized_error = sanitize_mcp_error(e, sanitizer)

            return self.create_error_response(
                request.id, MCPErrorCode.INTERNAL_ERROR, sanitized_error.public_message
            )

    def validate_capabilities(self, capabilities: Dict[str, Any]) -> bool:
        """Validate capability structure.

        Ensures capabilities object contains required fields.

        Args:
            capabilities: Capabilities dictionary to validate

        Returns:
            True if valid, False otherwise

        Required Fields:
            - experimental: Object for experimental capabilities
            - roots: Object for root capabilities

        Example:
            >>> caps = {"experimental": {}, "roots": {}}
            >>> protocol.validate_capabilities(caps)  # True
        """
        required_fields = ["experimental", "roots"]
        for field in required_fields:
            if field not in capabilities:
                return False
        return True

    def create_initialize_request(
        self, client_info: Dict[str, Any], capabilities: Dict[str, Any]
    ) -> MCPRequest:
        """Create standard initialize request.

        Used during MCP handshake to establish connection.

        Args:
            client_info: Client identification (name, version)
            capabilities: Client capabilities object

        Returns:
            MCPRequest for initialize method

        Example:
            >>> request = protocol.create_initialize_request(
            ...     {"name": "my-client", "version": "1.0.0"},
            ...     {"experimental": {}, "roots": {}}
            ... )
        """
        params = {
            "protocolVersion": self.version,
            "clientInfo": client_info,
            "capabilities": capabilities,
        }
        return self.create_request("initialize", params)

    def create_initialize_response(
        self, server_info: Dict[str, Any], capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create standard initialize response"""
        return {
            "protocolVersion": self.version,
            "serverInfo": server_info,
            "capabilities": capabilities,
        }

    def create_list_tools_request(self) -> MCPRequest:
        """Create list_tools request.

        Standard MCP request to discover available tools.

        Returns:
            MCPRequest for tools/list method

        Example:
            >>> request = protocol.create_list_tools_request()
            >>> # Send to server and expect response with tools array
        """
        return self.create_request("tools/list")

    def create_call_tool_request(self, name: str, arguments: Dict[str, Any]) -> MCPRequest:
        """Create call_tool request.

        Standard MCP request to invoke a specific tool.

        Args:
            name: Tool name to invoke
            arguments: Tool-specific arguments

        Returns:
            MCPRequest for tools/call method

        Example:
            >>> request = protocol.create_call_tool_request(
            ...     "calculator",
            ...     {"operation": "add", "x": 5, "y": 3}
            ... )
        """
        params = {"name": name, "arguments": arguments}
        return self.create_request("tools/call", params)

    def create_list_resources_request(self) -> MCPRequest:
        """Create list_resources request"""
        return self.create_request("resources/list")

    def create_read_resource_request(self, uri: str) -> MCPRequest:
        """Create read_resource request"""
        params = {"uri": uri}
        return self.create_request("resources/read", params)
