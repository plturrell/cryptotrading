"""
Production-grade input validation and sanitization
Prevents injection attacks, validates data types, and sanitizes user input
"""

import re
import json
import logging
import bleach
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid
import ipaddress
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Input validation failed"""
    pass

class SanitizationError(Exception):
    """Input sanitization failed"""
    pass

class ValidationType(Enum):
    """Types of validation"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    IP_ADDRESS = "ip_address"
    DATE = "date"
    JSON = "json"
    AGENT_ID = "agent_id"
    WORKFLOW_ID = "workflow_id"
    MESSAGE_TYPE = "message_type"

@dataclass
class ValidationRule:
    """Validation rule configuration"""
    field_name: str
    validation_type: ValidationType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None

class SecurityValidator:
    """Production security validator with comprehensive checks"""
    
    # Common dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"('\s*(or|and)\s*')|('\s*(union|select|insert|update|delete|drop|create|alter)\s*)",
        r"(\s*(union|select|insert|update|delete|drop|create|alter)\s+)",
        r"(--|\#|\/\*|\*\/)",
        r"(\s*(script|javascript|vbscript|onload|onerror|onclick)\s*[:=])",
        r"(<\s*script|<\s*iframe|<\s*object|<\s*embed)"
    ]
    
    XSS_PATTERNS = [
        r"<\s*script",
        r"javascript\s*:",
        r"vbscript\s*:",
        r"on\w+\s*=",
        r"<\s*iframe",
        r"<\s*object",
        r"<\s*embed"
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c"
    ]
    
    def __init__(self):
        self.compiled_sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_INJECTION_PATTERNS]
        self.compiled_xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS]
        self.compiled_path_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.PATH_TRAVERSAL_PATTERNS]
    
    def check_sql_injection(self, value: str) -> bool:
        """Check for SQL injection patterns"""
        for pattern in self.compiled_sql_patterns:
            if pattern.search(value):
                return True
        return False
    
    def check_xss(self, value: str) -> bool:
        """Check for XSS patterns"""
        for pattern in self.compiled_xss_patterns:
            if pattern.search(value):
                return True
        return False
    
    def check_path_traversal(self, value: str) -> bool:
        """Check for path traversal patterns"""
        for pattern in self.compiled_path_patterns:
            if pattern.search(value):
                return True
        return False
    
    def validate_security(self, value: str, field_name: str = "unknown"):
        """Comprehensive security validation"""
        if self.check_sql_injection(value):
            raise ValidationError(f"Potential SQL injection detected in {field_name}")
        
        if self.check_xss(value):
            raise ValidationError(f"Potential XSS detected in {field_name}")
        
        if self.check_path_traversal(value):
            raise ValidationError(f"Potential path traversal detected in {field_name}")

class DataValidator:
    """Data type and format validator"""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
    
    def validate_string(self, value: Any, rule: ValidationRule) -> str:
        """Validate string type and constraints"""
        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                raise ValidationError(f"{rule.field_name}: Cannot convert to string")
        
        # Security validation
        self.security_validator.validate_security(value, rule.field_name)
        
        # Length validation
        if rule.min_length is not None and len(value) < rule.min_length:
            raise ValidationError(f"{rule.field_name}: Minimum length {rule.min_length} required")
        
        if rule.max_length is not None and len(value) > rule.max_length:
            raise ValidationError(f"{rule.field_name}: Maximum length {rule.max_length} exceeded")
        
        # Pattern validation
        if rule.pattern:
            if not re.match(rule.pattern, value):
                raise ValidationError(f"{rule.field_name}: Pattern validation failed")
        
        # Allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            raise ValidationError(f"{rule.field_name}: Value not in allowed list")
        
        return value
    
    def validate_integer(self, value: Any, rule: ValidationRule) -> int:
        """Validate integer type and constraints"""
        try:
            if isinstance(value, str):
                value = int(value)
            elif not isinstance(value, int):
                raise ValueError("Not an integer")
        except ValueError:
            raise ValidationError(f"{rule.field_name}: Must be a valid integer")
        
        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            raise ValidationError(f"{rule.field_name}: Minimum value {rule.min_value} required")
        
        if rule.max_value is not None and value > rule.max_value:
            raise ValidationError(f"{rule.field_name}: Maximum value {rule.max_value} exceeded")
        
        return value
    
    def validate_float(self, value: Any, rule: ValidationRule) -> float:
        """Validate float type and constraints"""
        try:
            if isinstance(value, str):
                value = float(value)
            elif not isinstance(value, (int, float)):
                raise ValueError("Not a number")
            value = float(value)
        except ValueError:
            raise ValidationError(f"{rule.field_name}: Must be a valid number")
        
        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            raise ValidationError(f"{rule.field_name}: Minimum value {rule.min_value} required")
        
        if rule.max_value is not None and value > rule.max_value:
            raise ValidationError(f"{rule.field_name}: Maximum value {rule.max_value} exceeded")
        
        return value
    
    def validate_boolean(self, value: Any, rule: ValidationRule) -> bool:
        """Validate boolean type"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ('true', '1', 'yes', 'on'):
                return True
            elif value_lower in ('false', '0', 'no', 'off'):
                return False
        elif isinstance(value, int):
            return bool(value)
        
        raise ValidationError(f"{rule.field_name}: Must be a valid boolean")
    
    def validate_email(self, value: Any, rule: ValidationRule) -> str:
        """Validate email format"""
        value = self.validate_string(value, rule)
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            raise ValidationError(f"{rule.field_name}: Invalid email format")
        
        return value
    
    def validate_url(self, value: Any, rule: ValidationRule) -> str:
        """Validate URL format"""
        value = self.validate_string(value, rule)
        
        try:
            parsed = urlparse(value)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValidationError(f"{rule.field_name}: Invalid URL format")
        except Exception:
            raise ValidationError(f"{rule.field_name}: Invalid URL format")
        
        return value
    
    def validate_uuid(self, value: Any, rule: ValidationRule) -> str:
        """Validate UUID format"""
        value = self.validate_string(value, rule)
        
        try:
            uuid.UUID(value)
        except ValueError:
            raise ValidationError(f"{rule.field_name}: Invalid UUID format")
        
        return value
    
    def validate_ip_address(self, value: Any, rule: ValidationRule) -> str:
        """Validate IP address format"""
        value = self.validate_string(value, rule)
        
        try:
            ipaddress.ip_address(value)
        except ValueError:
            raise ValidationError(f"{rule.field_name}: Invalid IP address format")
        
        return value
    
    def validate_date(self, value: Any, rule: ValidationRule) -> str:
        """Validate ISO date format"""
        value = self.validate_string(value, rule)
        
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            raise ValidationError(f"{rule.field_name}: Invalid ISO date format")
        
        return value
    
    def validate_json(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate JSON format"""
        if isinstance(value, dict):
            return value
        
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValidationError(f"{rule.field_name}: Invalid JSON format")
        
        raise ValidationError(f"{rule.field_name}: Must be valid JSON")
    
    def validate_agent_id(self, value: Any, rule: ValidationRule) -> str:
        """Validate agent ID format"""
        value = self.validate_string(value, rule)
        
        # Agent ID format: type-number (e.g., "transform-001", "illuminate-001")
        agent_pattern = r'^[a-z][a-z0-9-]*-\d{3}$'
        if not re.match(agent_pattern, value):
            raise ValidationError(f"{rule.field_name}: Invalid agent ID format")
        
        return value
    
    def validate_workflow_id(self, value: Any, rule: ValidationRule) -> str:
        """Validate workflow ID format"""
        value = self.validate_string(value, rule)
        
        # Workflow ID format: kebab-case
        workflow_pattern = r'^[a-z][a-z0-9-]*[a-z0-9]$'
        if not re.match(workflow_pattern, value):
            raise ValidationError(f"{rule.field_name}: Invalid workflow ID format")
        
        return value
    
    def validate_message_type(self, value: Any, rule: ValidationRule) -> str:
        """Validate message type format"""
        value = self.validate_string(value, rule)
        
        # Valid message types from protocol
        valid_types = [
            "DATA_LOAD_REQUEST", "DATA_LOAD_RESPONSE",
            "ANALYSIS_REQUEST", "ANALYSIS_RESPONSE",
            "DATA_QUERY", "DATA_QUERY_RESPONSE",
            "TRADE_EXECUTION", "TRADE_RESPONSE",
            "WORKFLOW_REQUEST", "WORKFLOW_RESPONSE",
            "WORKFLOW_STATUS", "HEARTBEAT", "ERROR"
        ]
        
        if value not in valid_types:
            raise ValidationError(f"{rule.field_name}: Invalid message type")
        
        return value
    
    def validate_field(self, value: Any, rule: ValidationRule) -> Any:
        """Validate single field according to rule"""
        # Check if required
        if rule.required and (value is None or value == ""):
            raise ValidationError(f"{rule.field_name}: Field is required")
        
        # Skip validation if not required and empty
        if not rule.required and (value is None or value == ""):
            return value
        
        # Custom validator
        if rule.custom_validator:
            return rule.custom_validator(value, rule)
        
        # Type-specific validation
        validators = {
            ValidationType.STRING: self.validate_string,
            ValidationType.INTEGER: self.validate_integer,
            ValidationType.FLOAT: self.validate_float,
            ValidationType.BOOLEAN: self.validate_boolean,
            ValidationType.EMAIL: self.validate_email,
            ValidationType.URL: self.validate_url,
            ValidationType.UUID: self.validate_uuid,
            ValidationType.IP_ADDRESS: self.validate_ip_address,
            ValidationType.DATE: self.validate_date,
            ValidationType.JSON: self.validate_json,
            ValidationType.AGENT_ID: self.validate_agent_id,
            ValidationType.WORKFLOW_ID: self.validate_workflow_id,
            ValidationType.MESSAGE_TYPE: self.validate_message_type
        }
        
        validator = validators.get(rule.validation_type)
        if not validator:
            raise ValidationError(f"Unknown validation type: {rule.validation_type}")
        
        return validator(value, rule)

class InputSanitizer:
    """Input sanitization for safe processing"""
    
    def __init__(self):
        # Configure bleach for HTML sanitization
        self.allowed_tags = ['b', 'i', 'u', 'em', 'strong']
        self.allowed_attributes = {}
    
    def sanitize_html(self, value: str) -> str:
        """Sanitize HTML content"""
        return bleach.clean(
            value,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
    
    def sanitize_sql_string(self, value: str) -> str:
        """Sanitize string for SQL usage"""
        # Remove or escape dangerous characters
        return value.replace("'", "''").replace(";", "").replace("--", "")
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem usage"""
        # Remove path traversal and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\.\.', '', filename)
        filename = filename.strip('. ')
        
        if not filename:
            filename = "default"
        
        return filename
    
    def sanitize_command_arg(self, arg: str) -> str:
        """Sanitize command line argument"""
        # Remove shell metacharacters
        dangerous_chars = ['&', '|', ';', '$', '`', '>', '<', '(', ')', '{', '}', '[', ']']
        for char in dangerous_chars:
            arg = arg.replace(char, '')
        
        return arg.strip()

class ValidationSchemas:
    """Predefined validation schemas for common objects"""
    
    @staticmethod
    def a2a_message_schema() -> List[ValidationRule]:
        """Validation schema for A2A messages"""
        return [
            ValidationRule("sender_id", ValidationType.AGENT_ID),
            ValidationRule("receiver_id", ValidationType.AGENT_ID),
            ValidationRule("message_type", ValidationType.MESSAGE_TYPE),
            ValidationRule("payload", ValidationType.JSON),
            ValidationRule("message_id", ValidationType.STRING, max_length=100),
            ValidationRule("timestamp", ValidationType.DATE),
            ValidationRule("protocol_version", ValidationType.STRING, max_length=10),
            ValidationRule("correlation_id", ValidationType.STRING, required=False, max_length=100),
            ValidationRule("priority", ValidationType.INTEGER, min_value=0, max_value=3),
        ]
    
    @staticmethod
    def user_registration_schema() -> List[ValidationRule]:
        """Validation schema for user registration"""
        return [
            ValidationRule("username", ValidationType.STRING, min_length=3, max_length=50, 
                         pattern=r'^[a-zA-Z0-9_-]+$'),
            ValidationRule("email", ValidationType.EMAIL, max_length=255),
            ValidationRule("password", ValidationType.STRING, min_length=8, max_length=128),
            ValidationRule("roles", ValidationType.JSON),
        ]
    
    @staticmethod
    def workflow_execution_schema() -> List[ValidationRule]:
        """Validation schema for workflow execution"""
        return [
            ValidationRule("workflow_id", ValidationType.WORKFLOW_ID),
            ValidationRule("input_data", ValidationType.JSON),
            ValidationRule("priority", ValidationType.INTEGER, min_value=1, max_value=10, required=False),
        ]

class RequestValidator:
    """Main request validator"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.sanitizer = InputSanitizer()
    
    def validate_data(self, data: Dict[str, Any], schema: List[ValidationRule]) -> Dict[str, Any]:
        """Validate data against schema"""
        validated_data = {}
        errors = []
        
        # Check for unknown fields
        schema_fields = {rule.field_name for rule in schema}
        unknown_fields = set(data.keys()) - schema_fields
        if unknown_fields:
            errors.append(f"Unknown fields: {', '.join(unknown_fields)}")
        
        # Validate each field
        for rule in schema:
            try:
                value = data.get(rule.field_name)
                validated_value = self.validator.validate_field(value, rule)
                
                # Sanitize string values
                if isinstance(validated_value, str):
                    validated_value = self.sanitizer.sanitize_html(validated_value)
                
                validated_data[rule.field_name] = validated_value
                
            except ValidationError as e:
                errors.append(str(e))
        
        if errors:
            raise ValidationError(f"Validation failed: {'; '.join(errors)}")
        
        return validated_data
    
    def validate_a2a_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate A2A message"""
        return self.validate_data(message_data, ValidationSchemas.a2a_message_schema())
    
    def validate_user_registration(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user registration data"""
        return self.validate_data(user_data, ValidationSchemas.user_registration_schema())
    
    def validate_workflow_execution(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow execution data"""
        return self.validate_data(workflow_data, ValidationSchemas.workflow_execution_schema())

# Global instances
request_validator = RequestValidator()
input_sanitizer = InputSanitizer()

def validate_request(schema: List[ValidationRule]):
    """Decorator for request validation"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request data from kwargs
            request_data = kwargs.get('request_data', {})
            
            try:
                validated_data = request_validator.validate_data(request_data, schema)
                kwargs['validated_data'] = validated_data
            except ValidationError as e:
                logger.warning(f"Validation error in {func.__name__}: {e}")
                raise
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator