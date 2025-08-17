"""
Error Message Sanitization for MCP Security

This module provides secure error message handling to prevent information disclosure
while maintaining useful error information for debugging and user experience.
"""

import re
import logging
import traceback
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSensitivity(str, Enum):
    """Error message sensitivity levels"""
    PUBLIC = "public"           # Safe for end users
    INTERNAL = "internal"       # Internal systems only
    SENSITIVE = "sensitive"     # Contains sensitive data
    DEBUG = "debug"            # Debug information only


class SanitizationLevel(str, Enum):
    """Error sanitization levels"""
    NONE = "none"              # No sanitization
    BASIC = "basic"            # Basic sanitization
    STRICT = "strict"          # Strict sanitization
    MAXIMUM = "maximum"        # Maximum sanitization


@dataclass
class SanitizedError:
    """Sanitized error response"""
    public_message: str
    error_code: str
    error_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        result = {
            "message": self.public_message,
            "code": self.error_code
        }
        
        if self.error_id:
            result["error_id"] = self.error_id
        
        if self.details:
            result["details"] = self.details
        
        if self.suggestions:
            result["suggestions"] = self.suggestions
        
        return result


class ErrorMessageSanitizer:
    """Sanitizes error messages to prevent information disclosure"""
    
    # Patterns that should be sanitized from error messages
    SENSITIVE_PATTERNS = [
        # File paths
        r'/[a-zA-Z0-9_/\.-]+\.py',
        r'[A-Za-z]:\\[a-zA-Z0-9_\\\.-]+',
        
        # Database connection strings and credentials
        r'(?:mysql|postgresql|redis)://[^"\s]+',
        r'password["\s]*[:=]["\s]*[^"\s]+',
        r'token["\s]*[:=]["\s]*[^"\s]+',
        r'key["\s]*[:=]["\s]*[^"\s]+',
        r'secret["\s]*[:=]["\s]*[^"\s]+',
        
        # IP addresses and internal hostnames
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        r'\b[a-zA-Z0-9-]+\.internal\b',
        r'\blocalhost\b',
        r'\b127\.0\.0\.1\b',
        
        # API keys and tokens
        r'\b[A-Za-z0-9]{32,}\b',  # Long alphanumeric strings
        r'Bearer\s+[A-Za-z0-9\-\._~\+\/=]+',
        r'Basic\s+[A-Za-z0-9\+\/=]+',
        
        # Stack trace file references
        r'File "[^"]+", line \d+',
        r'at [a-zA-Z0-9_\.]+ \([^)]+\)',
        
        # Environment variables
        r'\$[A-Z_][A-Z0-9_]*',
        r'%[A-Z_][A-Z0-9_]*%',
        
        # SQL query fragments
        r'SELECT\s+[^;]+\s+FROM\s+[^;]+',
        r'INSERT\s+INTO\s+[^;]+',
        r'UPDATE\s+[^;]+\s+SET\s+[^;]+',
        r'DELETE\s+FROM\s+[^;]+',
        
        # Email addresses
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        # Internal service names
        r'\b[a-z]+-service-[a-z0-9-]+\b',
        r'\bmcp-[a-z]+-[a-z0-9-]+\b',
    ]
    
    # Generic error messages for different error types
    GENERIC_MESSAGES = {
        "authentication": "Authentication failed. Please check your credentials.",
        "authorization": "Access denied. You don't have permission to perform this action.",
        "validation": "Request validation failed. Please check your input parameters.",
        "rate_limit": "Rate limit exceeded. Please try again later.",
        "not_found": "The requested resource was not found.",
        "internal": "An internal error occurred. Please try again or contact support.",
        "network": "Network error occurred. Please check your connection and try again.",
        "timeout": "Request timed out. Please try again.",
        "invalid_input": "Invalid input provided. Please check your request format.",
        "service_unavailable": "Service temporarily unavailable. Please try again later.",
        "method_not_allowed": "Method not allowed for this resource.",
        "conflict": "Request conflicts with current state. Please refresh and try again.",
        "gone": "The requested resource is no longer available.",
        "payload_too_large": "Request payload is too large. Please reduce the data size.",
        "unsupported_media": "Unsupported media type. Please check the content type.",
        "upgrade_required": "Protocol upgrade required. Please update your client.",
        "precondition_failed": "Request precondition failed. Please check the requirements.",
        "range_not_satisfiable": "Requested range cannot be satisfied.",
        "expectation_failed": "Request expectation could not be met.",
        "unprocessable_entity": "Request could not be processed due to semantic errors.",
        "locked": "Resource is currently locked. Please try again later.",
        "failed_dependency": "Request failed due to dependency issues.",
        "too_early": "Request submitted too early. Please wait and retry.",
        "too_many_requests": "Too many requests. Please slow down and try again."
    }
    
    def __init__(self, sanitization_level: SanitizationLevel = SanitizationLevel.STRICT,
                 environment: str = "production"):
        """
        Initialize error message sanitizer
        
        Args:
            sanitization_level: Level of sanitization to apply
            environment: Environment (affects sanitization strictness)
        """
        self.sanitization_level = sanitization_level
        self.environment = environment
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SENSITIVE_PATTERNS]
        
        # Adjust sanitization based on environment
        if environment == "development" and sanitization_level == SanitizationLevel.STRICT:
            self.sanitization_level = SanitizationLevel.BASIC
            logger.info("Reduced sanitization level for development environment")
    
    def sanitize_error_message(self, error: Exception, error_type: str = "internal",
                             include_details: bool = False) -> SanitizedError:
        """
        Sanitize an error message for safe public display
        
        Args:
            error: The original exception
            error_type: Type of error (used for generic message lookup)
            include_details: Whether to include additional details
            
        Returns:
            SanitizedError with safe public message
        """
        original_message = str(error)
        error_class = type(error).__name__
        
        # Generate error ID for correlation
        import uuid
        import hashlib
        error_id = hashlib.md5(f"{error_class}:{original_message}".encode()).hexdigest()[:8]
        
        # Determine sanitization approach
        if self.sanitization_level == SanitizationLevel.NONE:
            public_message = original_message
        elif self.sanitization_level == SanitizationLevel.BASIC:
            public_message = self._basic_sanitization(original_message, error_type)
        elif self.sanitization_level == SanitizationLevel.STRICT:
            public_message = self._strict_sanitization(original_message, error_type)
        else:  # MAXIMUM
            public_message = self._maximum_sanitization(error_type)
        
        # Create sanitized error
        sanitized = SanitizedError(
            public_message=public_message,
            error_code=self._generate_error_code(error_class, error_type),
            error_id=error_id
        )
        
        # Add details if requested and safe
        if include_details and self.sanitization_level != SanitizationLevel.MAXIMUM:
            sanitized.details = self._get_safe_details(error, error_type)
        
        # Add helpful suggestions
        sanitized.suggestions = self._get_error_suggestions(error_type, error_class)
        
        # Log original error for debugging
        self._log_original_error(error, error_id, sanitized.public_message)
        
        return sanitized
    
    def _basic_sanitization(self, message: str, error_type: str) -> str:
        """Apply basic sanitization to error message"""
        sanitized = message
        
        # Remove sensitive patterns
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub('[REDACTED]', sanitized)
        
        # Limit message length
        if len(sanitized) > 200:
            sanitized = sanitized[:197] + "..."
        
        return sanitized
    
    def _strict_sanitization(self, message: str, error_type: str) -> str:
        """Apply strict sanitization to error message"""
        # For most error types, use generic messages
        if error_type in self.GENERIC_MESSAGES:
            return self.GENERIC_MESSAGES[error_type]
        
        # For unknown types, use basic sanitization
        return self._basic_sanitization(message, error_type)
    
    def _maximum_sanitization(self, error_type: str) -> str:
        """Apply maximum sanitization (generic messages only)"""
        return self.GENERIC_MESSAGES.get(error_type, self.GENERIC_MESSAGES["internal"])
    
    def _generate_error_code(self, error_class: str, error_type: str) -> str:
        """Generate standardized error code"""
        # Map common exception types to error codes
        error_code_map = {
            "ValidationError": "VALIDATION_ERROR",
            "AuthenticationError": "AUTHENTICATION_ERROR",
            "AuthorizationError": "AUTHORIZATION_ERROR",
            "RateLimitExceeded": "RATE_LIMIT_EXCEEDED",
            "ValueError": "INVALID_INPUT",
            "TypeError": "INVALID_INPUT",
            "KeyError": "MISSING_PARAMETER",
            "AttributeError": "INTERNAL_ERROR",
            "ConnectionError": "NETWORK_ERROR",
            "TimeoutError": "TIMEOUT_ERROR",
            "FileNotFoundError": "NOT_FOUND",
            "PermissionError": "AUTHORIZATION_ERROR",
            "NotImplementedError": "NOT_IMPLEMENTED",
            "RuntimeError": "INTERNAL_ERROR",
        }
        
        # Try to map by exception class first
        if error_class in error_code_map:
            return error_code_map[error_class]
        
        # Fallback to error type
        type_code_map = {
            "authentication": "AUTHENTICATION_ERROR",
            "authorization": "AUTHORIZATION_ERROR",
            "validation": "VALIDATION_ERROR",
            "rate_limit": "RATE_LIMIT_EXCEEDED",
            "not_found": "NOT_FOUND",
            "network": "NETWORK_ERROR",
            "timeout": "TIMEOUT_ERROR",
            "invalid_input": "INVALID_INPUT",
        }
        
        return type_code_map.get(error_type, "INTERNAL_ERROR")
    
    def _get_safe_details(self, error: Exception, error_type: str) -> Optional[Dict[str, Any]]:
        """Get safe error details that can be exposed"""
        details = {}
        
        # Add error type information
        details["error_type"] = error_type
        details["error_class"] = type(error).__name__
        
        # Add safe attributes based on error type
        if error_type == "validation":
            # For validation errors, we can expose field-level information
            if hasattr(error, 'field'):
                details["field"] = getattr(error, 'field')
            if hasattr(error, 'code'):
                details["validation_code"] = getattr(error, 'code')
        
        elif error_type == "rate_limit":
            # For rate limiting, we can expose retry information
            if hasattr(error, 'retry_after'):
                details["retry_after"] = getattr(error, 'retry_after')
        
        elif error_type == "authentication":
            # For auth errors, minimal safe information
            details["auth_method_supported"] = ["jwt", "api_key"]
        
        return details if details else None
    
    def _get_error_suggestions(self, error_type: str, error_class: str) -> List[str]:
        """Get helpful suggestions for error resolution"""
        suggestions = []
        
        suggestion_map = {
            "authentication": [
                "Check that your authentication credentials are correct",
                "Ensure your token has not expired",
                "Verify you're using the correct authentication method"
            ],
            "authorization": [
                "Verify you have the required permissions for this operation",
                "Contact your administrator if you believe you should have access"
            ],
            "validation": [
                "Check the request format and parameter types",
                "Ensure all required parameters are provided",
                "Verify parameter values are within acceptable ranges"
            ],
            "rate_limit": [
                "Reduce the frequency of your requests",
                "Implement exponential backoff in your retry logic",
                "Consider upgrading your plan for higher rate limits"
            ],
            "not_found": [
                "Verify the resource identifier is correct",
                "Check that the resource exists and is accessible"
            ],
            "network": [
                "Check your internet connection",
                "Verify the service endpoint is correct",
                "Try again in a few moments"
            ]
        }
        
        return suggestion_map.get(error_type, ["Contact support if the problem persists"])
    
    def _log_original_error(self, error: Exception, error_id: str, public_message: str):
        """Log original error for debugging purposes"""
        logger.error(
            f"Error sanitized and returned to client",
            extra={
                "error_id": error_id,
                "error_class": type(error).__name__,
                "original_message": str(error),
                "public_message": public_message,
                "sanitization_level": self.sanitization_level.value
            },
            exc_info=True
        )
    
    def sanitize_exception_details(self, exc_info: tuple) -> Dict[str, Any]:
        """Sanitize exception details from exc_info tuple"""
        exc_type, exc_value, exc_traceback = exc_info
        
        if self.sanitization_level == SanitizationLevel.MAXIMUM:
            return {
                "error": "An error occurred",
                "details": "Contact support for assistance"
            }
        
        result = {
            "error_type": exc_type.__name__ if exc_type else "Unknown",
            "message": self._strict_sanitization(str(exc_value) if exc_value else "", "internal")
        }
        
        # Only include traceback in development or with basic sanitization
        if (self.environment == "development" or 
            self.sanitization_level == SanitizationLevel.BASIC):
            
            # Sanitize the traceback
            if exc_traceback:
                tb_lines = traceback.format_tb(exc_traceback)
                sanitized_tb = []
                
                for line in tb_lines:
                    sanitized_line = line
                    for pattern in self.compiled_patterns:
                        sanitized_line = pattern.sub('[REDACTED]', sanitized_line)
                    sanitized_tb.append(sanitized_line)
                
                result["traceback"] = sanitized_tb[-3:]  # Only last 3 frames
        
        return result


# Utility functions for common error types
def sanitize_mcp_error(error: Exception, sanitizer: Optional[ErrorMessageSanitizer] = None) -> SanitizedError:
    """Sanitize MCP-specific errors"""
    if sanitizer is None:
        sanitizer = ErrorMessageSanitizer()
    
    # Determine error type based on exception
    error_type_map = {
        "AuthenticationError": "authentication",
        "AuthorizationError": "authorization", 
        "ValidationError": "validation",
        "RateLimitExceeded": "rate_limit",
        "ValueError": "invalid_input",
        "KeyError": "not_found",
        "ConnectionError": "network",
        "TimeoutError": "timeout"
    }
    
    error_class = type(error).__name__
    error_type = error_type_map.get(error_class, "internal")
    
    return sanitizer.sanitize_error_message(error, error_type)


def create_safe_error_response(error: Exception, sanitization_level: SanitizationLevel = SanitizationLevel.STRICT) -> Dict[str, Any]:
    """Create a safe error response dictionary"""
    sanitizer = ErrorMessageSanitizer(sanitization_level)
    sanitized = sanitize_mcp_error(error, sanitizer)
    return sanitized.to_dict()


# Global sanitizer instance
_global_sanitizer: Optional[ErrorMessageSanitizer] = None


def get_error_sanitizer() -> ErrorMessageSanitizer:
    """Get global error sanitizer instance"""
    global _global_sanitizer
    if _global_sanitizer is None:
        from .secure_defaults import get_security_defaults
        defaults = get_security_defaults()
        
        level = SanitizationLevel.STRICT
        if not defaults.sanitize_error_messages:
            level = SanitizationLevel.BASIC
        
        environment = "development" if defaults.debug_mode_enabled else "production"
        _global_sanitizer = ErrorMessageSanitizer(level, environment)
    
    return _global_sanitizer