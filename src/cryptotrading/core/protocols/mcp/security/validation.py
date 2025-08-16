"""
Input Validation and Sanitization for MCP

This module provides comprehensive input validation and sanitization for MCP requests
to prevent security vulnerabilities like injection attacks, XSS, and data corruption.

Features:
- JSON schema validation
- SQL injection prevention
- XSS protection
- Path traversal prevention
- File upload validation
- Size and rate limiting
- Custom validation rules
- Integration with MCP protocol
"""

import re
import json
import html
import urllib.parse
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import hashlib
import mimetypes
import os

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Input validation failed"""
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


class SanitizationError(Exception):
    """Input sanitization failed"""
    pass


@dataclass
class ValidationRule:
    """Validation rule configuration"""
    name: str
    required: bool = False
    type_check: Optional[type] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[callable] = None
    sanitizer: Optional[callable] = None


class SecurityValidator:
    """Core security validation and sanitization"""
    
    # Dangerous patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(['\"]\s*;\s*)",
        r"(\|\||&&)",
        r"(/\*|\*/|--)",
        r"(\bxp_|\bsp_)"
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"<iframe[^>]*>",
        r"eval\s*\(",
        r"expression\s*\("
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.\/",
        r"\.\.\\",
        r"\/\.\.",
        r"\\\.\.",
        r"%2e%2e%2f",
        r"%2e%2e\\",
        r"file:\/\/",
        r"\/etc\/",
        r"\/proc\/",
        r"\/sys\/",
        r"c:\\windows\\",
        r"c:\\program files\\"
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]<>]",
        r"\|\s*\w+",
        r"&&\s*\w+",
        r";\s*\w+",
        r"`[^`]*`",
        r"\$\([^)]*\)"
    ]
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize security validator
        
        Args:
            strict_mode: Enable strict validation (blocks suspicious content)
        """
        self.strict_mode = strict_mode
        self.compiled_patterns = {
            'sql': [re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_INJECTION_PATTERNS],
            'xss': [re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS],
            'path': [re.compile(pattern, re.IGNORECASE) for pattern in self.PATH_TRAVERSAL_PATTERNS],
            'command': [re.compile(pattern, re.IGNORECASE) for pattern in self.COMMAND_INJECTION_PATTERNS]
        }
    
    def detect_sql_injection(self, value: str) -> List[str]:
        """Detect SQL injection attempts"""
        threats = []
        for pattern in self.compiled_patterns['sql']:
            if pattern.search(value):
                threats.append(f"SQL injection pattern detected: {pattern.pattern}")
        return threats
    
    def detect_xss(self, value: str) -> List[str]:
        """Detect XSS attempts"""
        threats = []
        for pattern in self.compiled_patterns['xss']:
            if pattern.search(value):
                threats.append(f"XSS pattern detected: {pattern.pattern}")
        return threats
    
    def detect_path_traversal(self, value: str) -> List[str]:
        """Detect path traversal attempts"""
        threats = []
        for pattern in self.compiled_patterns['path']:
            if pattern.search(value):
                threats.append(f"Path traversal pattern detected: {pattern.pattern}")
        return threats
    
    def detect_command_injection(self, value: str) -> List[str]:
        """Detect command injection attempts"""
        threats = []
        for pattern in self.compiled_patterns['command']:
            if pattern.search(value):
                threats.append(f"Command injection pattern detected: {pattern.pattern}")
        return threats
    
    def validate_string_security(self, value: str, context: str = "general") -> Tuple[bool, List[str]]:
        """
        Validate string for security threats
        
        Args:
            value: String to validate
            context: Context for validation (filename, query, etc.)
            
        Returns:
            Tuple of (is_safe, threat_descriptions)
        """
        threats = []
        
        # Check for various injection types
        threats.extend(self.detect_sql_injection(value))
        threats.extend(self.detect_xss(value))
        
        if context in ['filename', 'path']:
            threats.extend(self.detect_path_traversal(value))
        
        if context in ['command', 'shell']:
            threats.extend(self.detect_command_injection(value))
        
        # Check for suspicious encodings
        if '%' in value:
            try:
                decoded = urllib.parse.unquote(value)
                if decoded != value:
                    # Recursively check decoded content
                    _, decoded_threats = self.validate_string_security(decoded, context)
                    threats.extend([f"URL-encoded: {threat}" for threat in decoded_threats])
            except Exception:
                pass
        
        # Check for excessive length (potential DoS)
        if len(value) > 10000:  # 10KB limit
            threats.append(f"Excessive length: {len(value)} characters")
        
        # Check for null bytes
        if '\x00' in value:
            threats.append("Null byte detected")
        
        # Check for control characters (except normal whitespace)
        control_chars = [c for c in value if ord(c) < 32 and c not in '\t\n\r']
        if control_chars:
            threats.append(f"Control characters detected: {control_chars}")
        
        is_safe = len(threats) == 0 or not self.strict_mode
        return is_safe, threats
    
    def sanitize_string(self, value: str, context: str = "general") -> str:
        """
        Sanitize string input
        
        Args:
            value: String to sanitize
            context: Context for sanitization
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove excessive control characters
        value = ''.join(c for c in value if ord(c) >= 32 or c in '\t\n\r')
        
        if context == 'html':
            # HTML escape
            value = html.escape(value, quote=True)
        
        elif context == 'filename':
            # Sanitize filename
            value = re.sub(r'[<>:"/\\|?*]', '_', value)
            value = value.strip('. ')  # Remove leading/trailing dots and spaces
            
        elif context == 'path':
            # Sanitize path
            value = value.replace('..', '')
            value = value.replace('//', '/')
            value = value.replace('\\\\', '\\')
            
        elif context == 'sql':
            # Escape SQL special characters
            value = value.replace("'", "''")
            value = value.replace('"', '""')
            value = value.replace('\\', '\\\\')
            
        # Limit length
        max_length = 1000 if context != 'content' else 10000
        if len(value) > max_length:
            value = value[:max_length]
            logger.warning(f"Truncated input to {max_length} characters")
        
        return value


class SchemaValidator:
    """JSON schema validator with security enhancements"""
    
    def __init__(self, security_validator: SecurityValidator):
        """
        Initialize schema validator
        
        Args:
            security_validator: SecurityValidator instance
        """
        self.security_validator = security_validator
    
    def validate_schema(self, data: Any, schema: Dict[str, Any], 
                       path: str = "") -> Tuple[bool, List[str]]:
        """
        Validate data against JSON schema with security checks
        
        Args:
            data: Data to validate
            schema: JSON schema
            path: Current path in data structure
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Type validation
        expected_type = schema.get("type")
        if expected_type:
            if not self._check_type(data, expected_type):
                errors.append(f"{path}: Expected {expected_type}, got {type(data).__name__}")
                return False, errors
        
        # Type-specific validation
        if expected_type == "string":
            errors.extend(self._validate_string(data, schema, path))
        elif expected_type == "number" or expected_type == "integer":
            errors.extend(self._validate_number(data, schema, path))
        elif expected_type == "array":
            errors.extend(self._validate_array(data, schema, path))
        elif expected_type == "object":
            errors.extend(self._validate_object(data, schema, path))
        
        return len(errors) == 0, errors
    
    def _check_type(self, data: Any, expected_type: str) -> bool:
        """Check if data matches expected type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(data, expected_python_type)
        
        return True
    
    def _validate_string(self, data: str, schema: Dict[str, Any], path: str) -> List[str]:
        """Validate string with security checks"""
        errors = []
        
        # Length validation
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        
        if min_length is not None and len(data) < min_length:
            errors.append(f"{path}: String too short (min: {min_length})")
        
        if max_length is not None and len(data) > max_length:
            errors.append(f"{path}: String too long (max: {max_length})")
        
        # Pattern validation
        pattern = schema.get("pattern")
        if pattern:
            if not re.match(pattern, data):
                errors.append(f"{path}: String doesn't match pattern {pattern}")
        
        # Enum validation
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: Value not in allowed list: {enum_values}")
        
        # Security validation
        context = schema.get("x-security-context", "general")
        is_safe, threats = self.security_validator.validate_string_security(data, context)
        
        if not is_safe:
            errors.extend([f"{path}: Security threat - {threat}" for threat in threats])
        
        return errors
    
    def _validate_number(self, data: Union[int, float], schema: Dict[str, Any], path: str) -> List[str]:
        """Validate number"""
        errors = []
        
        # Range validation
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        
        if minimum is not None and data < minimum:
            errors.append(f"{path}: Number too small (min: {minimum})")
        
        if maximum is not None and data > maximum:
            errors.append(f"{path}: Number too large (max: {maximum})")
        
        # Multiple validation
        multiple_of = schema.get("multipleOf")
        if multiple_of and data % multiple_of != 0:
            errors.append(f"{path}: Number must be multiple of {multiple_of}")
        
        return errors
    
    def _validate_array(self, data: List[Any], schema: Dict[str, Any], path: str) -> List[str]:
        """Validate array"""
        errors = []
        
        # Length validation
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        
        if min_items is not None and len(data) < min_items:
            errors.append(f"{path}: Array too short (min items: {min_items})")
        
        if max_items is not None and len(data) > max_items:
            errors.append(f"{path}: Array too long (max items: {max_items})")
        
        # Items validation
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]"
                is_valid, item_errors = self.validate_schema(item, items_schema, item_path)
                errors.extend(item_errors)
        
        # Unique items
        if schema.get("uniqueItems"):
            if len(data) != len(set(str(item) for item in data)):
                errors.append(f"{path}: Array items must be unique")
        
        return errors
    
    def _validate_object(self, data: Dict[str, Any], schema: Dict[str, Any], path: str) -> List[str]:
        """Validate object"""
        errors = []
        
        # Required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data:
                errors.append(f"{path}: Missing required property '{prop}'")
        
        # Properties validation
        properties = schema.get("properties", {})
        for prop, value in data.items():
            prop_path = f"{path}.{prop}" if path else prop
            
            if prop in properties:
                is_valid, prop_errors = self.validate_schema(value, properties[prop], prop_path)
                errors.extend(prop_errors)
            elif not schema.get("additionalProperties", True):
                errors.append(f"{path}: Additional property '{prop}' not allowed")
        
        # Additional properties schema
        additional_schema = schema.get("additionalProperties")
        if isinstance(additional_schema, dict):
            for prop, value in data.items():
                if prop not in properties:
                    prop_path = f"{path}.{prop}" if path else prop
                    is_valid, prop_errors = self.validate_schema(value, additional_schema, prop_path)
                    errors.extend(prop_errors)
        
        return errors


class MCPInputValidator:
    """MCP-specific input validation"""
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize MCP input validator
        
        Args:
            strict_mode: Enable strict security validation
        """
        self.security_validator = SecurityValidator(strict_mode)
        self.schema_validator = SchemaValidator(self.security_validator)
        
        # Define MCP-specific schemas
        self.mcp_schemas = self._load_mcp_schemas()
    
    def _load_mcp_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load MCP method schemas"""
        return {
            "initialize": {
                "type": "object",
                "required": ["protocolVersion", "clientInfo", "capabilities"],
                "properties": {
                    "protocolVersion": {
                        "type": "string",
                        "pattern": r"^\d{4}-\d{2}-\d{2}$"
                    },
                    "clientInfo": {
                        "type": "object",
                        "required": ["name", "version"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "maxLength": 100,
                                "x-security-context": "general"
                            },
                            "version": {
                                "type": "string",
                                "maxLength": 50,
                                "pattern": r"^\d+\.\d+\.\d+.*$"
                            }
                        }
                    },
                    "capabilities": {
                        "type": "object"
                    }
                }
            },
            
            "tools/call": {
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {
                        "type": "string",
                        "maxLength": 100,
                        "pattern": r"^[a-zA-Z0-9_\-]+$",
                        "x-security-context": "general"
                    },
                    "arguments": {
                        "type": "object",
                        "maxProperties": 50
                    }
                }
            },
            
            "resources/read": {
                "type": "object",
                "required": ["uri"],
                "properties": {
                    "uri": {
                        "type": "string",
                        "maxLength": 500,
                        "x-security-context": "path"
                    }
                }
            }
        }
    
    def validate_mcp_request(self, method: str, params: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate and sanitize MCP request
        
        Args:
            method: MCP method name
            params: Request parameters
            
        Returns:
            Tuple of (is_valid, errors, sanitized_params)
        """
        errors = []
        sanitized_params = {}
        
        # Basic method validation
        if not method or not isinstance(method, str):
            errors.append("Invalid method name")
            return False, errors, sanitized_params
        
        # Check method name security
        is_safe, threats = self.security_validator.validate_string_security(method, "general")
        if not is_safe:
            errors.extend([f"Method name security threat: {threat}" for threat in threats])
            return False, errors, sanitized_params
        
        # Validate against schema if available
        if method in self.mcp_schemas:
            schema = self.mcp_schemas[method]
            is_valid, schema_errors = self.schema_validator.validate_schema(params, schema)
            errors.extend(schema_errors)
            
            if not is_valid:
                return False, errors, sanitized_params
        
        # Sanitize parameters
        try:
            sanitized_params = self._sanitize_params(params)
        except Exception as e:
            errors.append(f"Sanitization failed: {str(e)}")
            return False, errors, sanitized_params
        
        return len(errors) == 0, errors, sanitized_params
    
    def _sanitize_params(self, params: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
        """
        Recursively sanitize parameters
        
        Args:
            params: Parameters to sanitize
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            Sanitized parameters
        """
        if current_depth > max_depth:
            raise SanitizationError("Maximum recursion depth exceeded")
        
        if isinstance(params, dict):
            sanitized = {}
            for key, value in params.items():
                # Sanitize key
                if isinstance(key, str):
                    clean_key = self.security_validator.sanitize_string(key, "general")
                    sanitized[clean_key] = self._sanitize_params(value, max_depth, current_depth + 1)
                else:
                    sanitized[key] = self._sanitize_params(value, max_depth, current_depth + 1)
            return sanitized
        
        elif isinstance(params, list):
            return [self._sanitize_params(item, max_depth, current_depth + 1) for item in params]
        
        elif isinstance(params, str):
            return self.security_validator.sanitize_string(params, "general")
        
        else:
            return params
    
    def validate_file_upload(self, filename: str, content: bytes, 
                           allowed_types: Optional[List[str]] = None,
                           max_size: int = 10 * 1024 * 1024) -> Tuple[bool, List[str]]:
        """
        Validate file upload
        
        Args:
            filename: Original filename
            content: File content
            allowed_types: Allowed MIME types
            max_size: Maximum file size in bytes
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Filename validation
        is_safe, threats = self.security_validator.validate_string_security(filename, "filename")
        if not is_safe:
            errors.extend([f"Filename security threat: {threat}" for threat in threats])
        
        # File size validation
        if len(content) > max_size:
            errors.append(f"File too large: {len(content)} bytes (max: {max_size})")
        
        # MIME type validation
        if allowed_types:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type not in allowed_types:
                errors.append(f"File type not allowed: {mime_type}")
        
        # Content validation (basic)
        if b'\x00' in content:
            errors.append("File contains null bytes")
        
        # Check for embedded scripts in text files
        if filename.lower().endswith(('.txt', '.json', '.xml', '.html', '.css', '.js')):
            try:
                text_content = content.decode('utf-8')
                is_safe, threats = self.security_validator.validate_string_security(text_content, "content")
                if not is_safe:
                    errors.extend([f"File content security threat: {threat}" for threat in threats])
            except UnicodeDecodeError:
                errors.append("File encoding not supported")
        
        return len(errors) == 0, errors


class ValidationMiddleware:
    """Validation middleware for MCP requests"""
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validation middleware
        
        Args:
            strict_mode: Enable strict security validation
        """
        self.validator = MCPInputValidator(strict_mode)
        self.blocked_methods = set()
        self.allowed_methods = set()
        
        # Load configuration from environment
        self._load_config()
    
    def _load_config(self):
        """Load validation configuration from environment"""
        # Blocked methods
        blocked = os.getenv("MCP_BLOCKED_METHODS", "")
        if blocked:
            self.blocked_methods = set(blocked.split(","))
        
        # Allowed methods (if set, only these are allowed)
        allowed = os.getenv("MCP_ALLOWED_METHODS", "")
        if allowed:
            self.allowed_methods = set(allowed.split(","))
    
    async def validate_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize MCP request
        
        Args:
            method: MCP method name
            params: Request parameters
            
        Returns:
            Sanitized parameters
            
        Raises:
            ValidationError: If validation fails
        """
        # Check method allowlist/blocklist
        if self.allowed_methods and method not in self.allowed_methods:
            raise ValidationError(f"Method not allowed: {method}")
        
        if method in self.blocked_methods:
            raise ValidationError(f"Method blocked: {method}")
        
        # Validate and sanitize
        is_valid, errors, sanitized_params = self.validator.validate_mcp_request(method, params)
        
        if not is_valid:
            error_msg = "; ".join(errors)
            raise ValidationError(f"Validation failed: {error_msg}")
        
        return sanitized_params


# Convenience functions
def create_security_validator(strict_mode: bool = True) -> SecurityValidator:
    """Create security validator"""
    return SecurityValidator(strict_mode)


def create_mcp_validator(strict_mode: bool = True) -> MCPInputValidator:
    """Create MCP input validator"""
    return MCPInputValidator(strict_mode)


def create_validation_middleware(strict_mode: bool = True) -> ValidationMiddleware:
    """Create validation middleware"""
    return ValidationMiddleware(strict_mode)