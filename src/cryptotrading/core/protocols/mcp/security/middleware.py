"""
Integrated Security Middleware for MCP

This module provides a unified security middleware that combines authentication,
rate limiting, and input validation for MCP servers deployed on Vercel.

Features:
- Unified security pipeline
- Configurable security policies
- Request/response logging
- Security metrics and monitoring
- Integration with Vercel environment
- Performance optimized for Edge Functions
"""

import asyncio
import time
import json
import os
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .auth import AuthenticationMiddleware, AuthToken, AuthenticationError, AuthorizationError, create_auth_middleware
from .rate_limiter import RateLimitMiddleware, RateLimitExceeded, create_rate_limit_middleware
from .validation import ValidationMiddleware, ValidationError, create_validation_middleware
from .authentication import (
    SecureAuthenticator, 
    AuthenticationContext, 
    Permission, 
    AuthenticationMethod,
    require_permissions,
    SecureValidator
)
from .permissions import (
    get_permission_validator,
    validate_method_permission,
    MethodSecurityLevel
)
from .audit_logger import (
    get_audit_logger,
    create_security_context,
    extract_security_context_from_headers,
    SecurityEventType
)
from .secure_defaults import (
    get_secure_config_manager,
    get_security_defaults,
    validate_security_config
)
from .rate_limit_headers import (
    get_rate_limit_header_manager,
    get_multi_tier_manager,
    RateLimitInfo
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration"""
    # Authentication
    require_auth: bool = True
    allow_anonymous_read: bool = False
    jwt_enabled: bool = True
    api_key_enabled: bool = True
    
    # Rate limiting
    rate_limiting_enabled: bool = True
    global_rate_limit: int = 1000  # requests per minute
    user_rate_limit: int = 100
    
    # Validation
    input_validation_enabled: bool = True
    strict_validation: bool = True
    max_request_size: int = 1024 * 1024  # 1MB
    
    # Logging and monitoring
    log_requests: bool = True
    log_security_events: bool = True
    metrics_enabled: bool = True
    
    # Method-specific settings
    allowed_methods: Optional[List[str]] = None
    blocked_methods: Optional[List[str]] = None
    read_only_methods: List[str] = None
    
    def __post_init__(self):
        if self.read_only_methods is None:
            self.read_only_methods = [
                "initialize",
                "tools/list",
                "resources/list",
                "resources/read",
                "ping"
            ]


class SecurityMetrics:
    """Security metrics collector"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "authenticated_requests": 0,
            "rate_limited_requests": 0,
            "validation_failures": 0,
            "security_threats": 0,
            "method_calls": {},
            "auth_methods": {},
            "start_time": time.time()
        }
    
    def record_request(self, method: str, auth_method: Optional[str] = None):
        """Record a request"""
        self.metrics["total_requests"] += 1
        self.metrics["method_calls"][method] = self.metrics["method_calls"].get(method, 0) + 1
        
        if auth_method:
            self.metrics["authenticated_requests"] += 1
            self.metrics["auth_methods"][auth_method] = self.metrics["auth_methods"].get(auth_method, 0) + 1
    
    def record_rate_limit(self):
        """Record rate limit hit"""
        self.metrics["rate_limited_requests"] += 1
    
    def record_validation_failure(self):
        """Record validation failure"""
        self.metrics["validation_failures"] += 1
    
    def record_security_threat(self):
        """Record security threat detected"""
        self.metrics["security_threats"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = time.time() - self.metrics["start_time"]
        metrics_copy = self.metrics.copy()
        metrics_copy.update({
            "uptime_seconds": uptime,
            "requests_per_second": self.metrics["total_requests"] / max(uptime, 1)
        })
        return metrics_copy
    
    def reset(self):
        """Reset metrics"""
        self.__init__()


class SecurityContext:
    """Security context for a request"""
    
    def __init__(self):
        self.start_time = time.time()
        self.auth_token: Optional[AuthToken] = None
        self.user_id: Optional[str] = None
        self.api_key_id: Optional[str] = None
        self.method: Optional[str] = None
        self.request_size: int = 0
        self.security_threats: List[str] = []
        self.rate_limit_status: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
        self.rate_limit_info: Optional[RateLimitInfo] = None
        self.response_headers: Dict[str, str] = {}
    
    def add_security_threat(self, threat: str):
        """Add security threat"""
        self.security_threats.append(threat)
        logger.warning(f"Security threat detected: {threat}")
    
    def set_rate_limit_info(self, rate_limit_info: RateLimitInfo):
        """Set rate limit information and add headers"""
        self.rate_limit_info = rate_limit_info
        self.response_headers.update(rate_limit_info.to_headers())
    
    def add_response_header(self, key: str, value: str):
        """Add a response header"""
        self.response_headers[key] = value
    
    def get_duration(self) -> float:
        """Get request duration"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "duration": self.get_duration(),
            "user_id": self.user_id,
            "api_key_id": self.api_key_id,
            "method": self.method,
            "request_size": self.request_size,
            "security_threats": self.security_threats,
            "validation_errors": self.validation_errors,
            "authenticated": self.auth_token is not None
        }


class SecurityMiddleware:
    """Integrated security middleware for MCP"""
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize security middleware
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.metrics = SecurityMetrics()
        
        # Initialize sub-middlewares
        self.auth_middleware = create_auth_middleware() if config.require_auth else None
        self.rate_limit_middleware = create_rate_limit_middleware() if config.rate_limiting_enabled else None
        self.validation_middleware = create_validation_middleware(config.strict_validation) if config.input_validation_enabled else None
        
        logger.info("Security middleware initialized", extra={
            "auth_enabled": config.require_auth,
            "rate_limiting_enabled": config.rate_limiting_enabled,
            "validation_enabled": config.input_validation_enabled
        })
    
    async def process_request(self, method: str, params: Dict[str, Any], 
                            headers: Dict[str, str]) -> Tuple[Dict[str, Any], SecurityContext]:
        """
        Process incoming request through security pipeline
        
        Args:
            method: MCP method name
            params: Request parameters
            headers: Request headers
            
        Returns:
            Tuple of (processed_params, security_context)
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitExceeded: If rate limit exceeded
            ValidationError: If validation fails
        """
        context = SecurityContext()
        context.method = method
        context.request_size = len(json.dumps(params)) if params else 0
        
        try:
            # Record request
            self.metrics.record_request(method)
            
            # Size check
            if context.request_size > self.config.max_request_size:
                raise ValidationError(f"Request too large: {context.request_size} bytes")
            
            # Method filtering
            if self.config.allowed_methods and method not in self.config.allowed_methods:
                raise ValidationError(f"Method not allowed: {method}")
            
            if self.config.blocked_methods and method in self.config.blocked_methods:
                raise ValidationError(f"Method blocked: {method}")
            
            # Authentication
            auth_method = None
            if self.config.require_auth:
                if self.config.allow_anonymous_read and method in self.config.read_only_methods:
                    # Allow anonymous read operations
                    pass
                else:
                    context.auth_token = await self.auth_middleware.authenticate_request(headers)
                    context.user_id = context.auth_token.user_id
                    context.api_key_id = context.auth_token.api_key_id
                    auth_method = "jwt" if not context.api_key_id else "api_key"
                    
                    # Check authorization for method
                    required_scope = self._get_required_scope(method)
                    self.auth_middleware.check_authorization(context.auth_token, required_scope)
            
            # Update metrics with auth info
            if auth_method:
                self.metrics.record_request(method, auth_method)
            
            # Rate limiting with headers
            if self.config.rate_limiting_enabled and self.rate_limit_middleware:
                identifier = context.user_id or context.api_key_id or "anonymous"
                
                # Check rate limits and get status
                try:
                    await self.rate_limit_middleware.check_request_limits(
                        identifier, method, context.api_key_id
                    )
                    
                    # Get rate limit status for headers
                    if hasattr(self.rate_limit_middleware, 'get_rate_limit_status'):
                        rate_status = await self.rate_limit_middleware.get_rate_limit_status(
                            identifier, method
                        )
                        
                        # Create rate limit info and add headers
                        header_manager = get_rate_limit_header_manager()
                        rate_limit_info = header_manager.create_from_rate_limiter_status(rate_status)
                        context.set_rate_limit_info(rate_limit_info)
                        
                except Exception as e:
                    # Still add default rate limit headers even if status fetch fails
                    header_manager = get_rate_limit_header_manager()
                    default_info = RateLimitInfo(
                        limit=self.config.user_rate_limit,
                        remaining=max(0, self.config.user_rate_limit - 1),
                        reset=int(time.time() + 60),
                        window=60,
                        policy="unknown"
                    )
                    context.set_rate_limit_info(default_info)
                    raise
            
            # Input validation
            processed_params = params
            if self.config.input_validation_enabled and self.validation_middleware:
                processed_params = await self.validation_middleware.validate_request(method, params)
            
            # Log successful request with structured logging
            if self.config.log_requests:
                self._log_request_structured(context, "success", headers)
            
            return processed_params, context
            
        except RateLimitExceeded as e:
            self.metrics.record_rate_limit()
            context.add_security_threat(f"Rate limit exceeded: {str(e)}")
            if self.config.log_security_events:
                self._log_request_structured(context, "rate_limited", headers, str(e))
            raise
            
        except ValidationError as e:
            self.metrics.record_validation_failure()
            context.validation_errors.append(str(e))
            if "security threat" in str(e).lower():
                self.metrics.record_security_threat()
                context.add_security_threat(str(e))
            if self.config.log_security_events:
                self._log_request_structured(context, "validation_failed", headers, str(e))
            raise
            
        except Exception as e:
            context.add_security_threat(f"Unexpected error: {str(e)}")
            if self.config.log_security_events:
                self._log_request_structured(context, "error", headers, str(e))
            raise
    
    def _validate_method_permission(self, method: str, auth_context: Optional[AuthenticationContext]) -> Dict[str, Any]:
        """Validate method permissions using secure permission system"""
        permission_validator = get_permission_validator()
        return permission_validator.validate_method_access(method, auth_context)
    
    def _log_request_structured(self, context: SecurityContext, status: str, 
                               headers: Dict[str, str], error: Optional[str] = None):
        """Log request using structured security audit logger"""
        audit_logger = get_audit_logger()
        
        # Create comprehensive security context
        security_context = extract_security_context_from_headers(
            headers, 
            method=context.method,
            user_id=context.user_id
        )
        
        # Add additional context
        security_context.request_size = context.request_size
        security_context.processing_time_ms = context.get_duration() * 1000
        security_context.auth_method = "jwt" if context.auth_token else "anonymous"
        security_context.threat_indicators = context.security_threats
        security_context.metadata.update({
            "api_key_id": context.api_key_id,
            "validation_errors": context.validation_errors,
            "authenticated": context.auth_token is not None
        })
        
        # Log based on status
        if status == "success":
            audit_logger.log_method_execution(
                security_context, 
                success=True, 
                processing_time_ms=security_context.processing_time_ms
            )
        elif status == "rate_limited":
            audit_logger.log_rate_limit_exceeded(
                security_context, 
                "method_rate_limit"
            )
        elif status == "validation_failed":
            audit_logger.log_input_validation_failure(
                security_context, 
                context.validation_errors
            )
        elif status == "security_violation":
            audit_logger.log_security_threat(
                security_context, 
                "security_policy_violation", 
                error or "Unknown security violation"
            )
        elif context.security_threats:
            audit_logger.log_security_threat(
                security_context, 
                "threat_detected", 
                "; ".join(context.security_threats)
            )
        else:
            audit_logger.log_method_execution(
                security_context, 
                success=False, 
                processing_time_ms=security_context.processing_time_ms,
                error=error
            )
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        status = {
            "config": asdict(self.config),
            "metrics": self.metrics.get_metrics(),
            "middleware_status": {
                "auth_enabled": self.auth_middleware is not None,
                "rate_limiting_enabled": self.rate_limit_middleware is not None,
                "validation_enabled": self.validation_middleware is not None
            }
        }
        
        # Add rate limiter status if available
        if self.rate_limit_middleware:
            try:
                status["rate_limits"] = await self.rate_limit_middleware.rate_limiter.get_rate_limit_status(
                    "global", "global"
                )
            except Exception as e:
                status["rate_limits"] = {"error": str(e)}
        
        return status
    
    def reset_metrics(self):
        """Reset security metrics"""
        self.metrics.reset()
        logger.info("Security metrics reset")


class VercelSecurityMiddleware(SecurityMiddleware):
    """Vercel-optimized security middleware"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize Vercel security middleware with environment configuration
        
        Args:
            config: Optional security config (defaults loaded from env vars)
        """
        if config is None:
            config = self._load_vercel_config()
        
        super().__init__(config)
    
    def _load_vercel_config(self) -> SecurityConfig:
        """Load security config from secure defaults with Vercel environment overrides"""
        # Start with secure defaults based on environment
        defaults = get_security_defaults()
        config = SecurityConfig()
        
        # Apply secure defaults
        config.require_auth = defaults.require_authentication
        config.allow_anonymous_read = defaults.allow_anonymous_access
        config.jwt_enabled = defaults.jwt_enabled
        config.api_key_enabled = defaults.api_key_enabled
        config.rate_limiting_enabled = defaults.rate_limiting_enabled
        config.global_rate_limit = defaults.global_rate_limit_per_minute
        config.user_rate_limit = defaults.user_rate_limit_per_minute
        config.input_validation_enabled = defaults.input_validation_enabled
        config.strict_validation = defaults.strict_validation
        config.max_request_size = defaults.max_request_size_bytes
        config.log_requests = defaults.security_logging_enabled
        config.log_security_events = defaults.audit_logging_enabled
        config.metrics_enabled = defaults.security_logging_enabled
        
        # Environment overrides (already handled by secure_defaults module)
        # These will only override if explicitly set
        
        # Method filtering from environment (if specified)
        allowed_methods = os.getenv("MCP_ALLOWED_METHODS")
        if allowed_methods:
            config.allowed_methods = [m.strip() for m in allowed_methods.split(",")]
        
        blocked_methods = os.getenv("MCP_BLOCKED_METHODS")
        if blocked_methods:
            config.blocked_methods = [m.strip() for m in blocked_methods.split(",")]
        
        logger.info("Loaded Vercel security configuration", extra={
            "require_auth": config.require_auth,
            "rate_limiting": config.rate_limiting_enabled,
            "validation": config.input_validation_enabled
        })
        
        return config


# Convenience functions for different environments
def create_security_middleware(config: Optional[SecurityConfig] = None) -> SecurityMiddleware:
    """Create security middleware with optional config"""
    if config is None:
        config = SecurityConfig()
    return SecurityMiddleware(config)


def create_vercel_security_middleware() -> VercelSecurityMiddleware:
    """Create Vercel-optimized security middleware"""
    return VercelSecurityMiddleware()


def create_development_security_middleware() -> SecurityMiddleware:
    """Create development-friendly security middleware (less strict)"""
    config = SecurityConfig(
        require_auth=False,
        allow_anonymous_read=True,
        rate_limiting_enabled=False,
        strict_validation=False,
        log_requests=True,
        log_security_events=True
    )
    return SecurityMiddleware(config)


def create_production_security_middleware() -> SecurityMiddleware:
    """Create production security middleware (strict settings)"""
    config = SecurityConfig(
        require_auth=True,
        allow_anonymous_read=False,
        rate_limiting_enabled=True,
        strict_validation=True,
        log_requests=True,
        log_security_events=True,
        max_request_size=512 * 1024  # 512KB for production
    )
    return SecurityMiddleware(config)


class SecureMiddleware(SecurityMiddleware):
    """
    Enhanced Security Middleware with No Authentication Bypass
    
    This middleware addresses critical security vulnerabilities by:
    - Using SecureAuthenticator with no anonymous access in strict mode
    - Implementing granular permission checking
    - Adding comprehensive input validation
    - Providing secure rate limiting per authenticated user
    - Blocking all unauthenticated requests except health checks
    """
    
    def __init__(self, config: SecurityConfig, jwt_secret: str, enable_strict_mode: bool = True):
        """
        Initialize secure middleware with mandatory authentication
        
        Args:
            config: Security configuration 
            jwt_secret: JWT secret key (must be at least 32 characters)
            enable_strict_mode: If True, blocks all unauthenticated requests
        """
        super().__init__(config)
        
        # Initialize secure authenticator (no bypass vulnerabilities)
        self.secure_authenticator = SecureAuthenticator(
            secret_key=jwt_secret, 
            enable_strict_mode=enable_strict_mode
        )
        
        # Secure validator for input sanitization
        self.secure_validator = SecureValidator()
        
        # Method-permission mapping
        self.method_permissions = {
            # Read permissions
            "initialize": [Permission.HEALTH_CHECK],
            "ping": [Permission.HEALTH_CHECK],
            "tools/list": [Permission.READ_TOOLS],
            "resources/list": [Permission.READ_RESOURCES],
            "resources/read": [Permission.READ_RESOURCES],
            
            # Execute permissions  
            "tools/call": [Permission.EXECUTE_TOOLS],
            
            # Write permissions
            "resources/write": [Permission.WRITE_RESOURCES],
            
            # Admin permissions
            "security/status": [Permission.ADMIN_SERVER],
            "metrics": [Permission.METRICS_READ],
        }
        
        logger.info("Secure middleware initialized with no authentication bypass", extra={
            "strict_mode": enable_strict_mode,
            "authenticated_only": True
        })
    
    async def process_request(self, method: str, params: Dict[str, Any], 
                            headers: Dict[str, str]) -> Tuple[Dict[str, Any], SecurityContext]:
        """
        Process request through secure pipeline with mandatory authentication
        
        Security Pipeline:
        1. Extract and validate authentication
        2. Check method permissions
        3. Apply rate limiting per authenticated user
        4. Validate and sanitize all inputs
        5. Log security events
        
        Args:
            method: MCP method name
            params: Request parameters  
            headers: Request headers
            
        Returns:
            Tuple of (processed_params, security_context)
            
        Raises:
            AuthenticationError: If authentication fails or is missing
            AuthorizationError: If user lacks required permissions
            RateLimitExceeded: If rate limit exceeded for user
            ValidationError: If input validation fails
        """
        context = SecurityContext()
        context.method = method
        context.request_size = len(json.dumps(params, default=str))
        
        try:
            # Step 1: Mandatory Authentication - NO BYPASS
            auth_context = await self.secure_authenticator.authenticate_request(
                headers, 
                ip_address=headers.get("x-forwarded-for") or headers.get("x-real-ip")
            )
            
            if not auth_context:
                # In strict mode, this raises exception automatically
                # But we double-check to prevent any bypass
                context.add_security_threat("authentication_required_but_missing")
                self.metrics.record_security_threat()
                raise AuthenticationError("Authentication required - no anonymous access allowed")
            
            # Verify authentication context is valid
            if auth_context.is_expired():
                context.add_security_threat("expired_authentication_used")
                self.metrics.record_security_threat()
                raise AuthenticationError("Authentication expired")
            
            # Set authenticated user context
            context.user_id = auth_context.user_id
            context.auth_token = AuthToken(
                user_id=auth_context.user_id,
                token_type=auth_context.method.value,
                expires_at=auth_context.expires_at
            )
            
            # Step 2: Permission Checking using secure permission system
            permission_result = self._validate_method_permission(method, auth_context)
            if not permission_result["allowed"]:
                context.add_security_threat(f"permission_denied_for_{method}_{permission_result['reason']}")
                self.metrics.record_security_threat()
                raise AuthorizationError(f"Permission denied for {method}: {permission_result['reason']}")
            
            # Step 3: Rate Limiting (per authenticated user)
            if self.rate_limit_middleware:
                rate_limit_key = f"user:{auth_context.user_id}"
                rate_limited = await self.rate_limit_middleware.check_rate_limit(
                    rate_limit_key, method
                )
                if rate_limited:
                    context.add_security_threat("rate_limit_exceeded")
                    self.metrics.record_rate_limit()
                    raise RateLimitExceeded(f"Rate limit exceeded for user {auth_context.user_id}")
            
            # Step 4: Input Validation and Sanitization
            processed_params = await self._secure_validate_params(method, params, context)
            
            # Step 5: Record successful authentication
            self.metrics.record_request(method, auth_context.method.value)
            
            # Log successful secure request
            if self.config.log_requests:
                self._log_request(context, "authenticated_success")
            
            return processed_params, context
            
        except (AuthenticationError, AuthorizationError, RateLimitExceeded, ValidationError) as e:
            # Security violations - log and re-raise
            context.add_security_threat(f"security_violation_{type(e).__name__}")
            self.metrics.record_security_threat()
            
            if self.config.log_security_events:
                self._log_request(context, "security_violation", str(e))
            
            raise
        except Exception as e:
            # Unexpected errors - log and convert to security error
            context.add_security_threat("unexpected_error_during_security_processing")
            self.metrics.record_security_threat()
            logger.error(f"Unexpected security processing error: {e}", exc_info=True)
            raise AuthenticationError("Security processing failed")
    
    async def _secure_validate_params(self, method: str, params: Dict[str, Any], 
                                    context: SecurityContext) -> Dict[str, Any]:
        """
        Secure parameter validation with comprehensive sanitization
        
        Args:
            method: MCP method name
            params: Request parameters
            context: Security context for logging
            
        Returns:
            Sanitized parameters
            
        Raises:
            ValidationError: If validation fails
        """
        if not params:
            return {}
        
        # Check size limits to prevent DoS
        if not self.secure_validator.validate_size_limits(params):
            context.add_security_threat("oversized_request_params")
            raise ValidationError("Request parameters exceed size limits")
        
        sanitized_params = {}
        
        for key, value in params.items():
            # Sanitize parameter names
            clean_key = self.secure_validator.sanitize_string(key, max_length=100)
            if not clean_key or clean_key != key:
                context.add_security_threat(f"suspicious_parameter_name_{key}")
                raise ValidationError(f"Invalid parameter name: {key}")
            
            # Sanitize parameter values based on type
            if isinstance(value, str):
                # Check for path traversal in string values
                if not self.secure_validator.validate_path(value) and "/" in value:
                    context.add_security_threat(f"path_traversal_attempt_{value}")
                    raise ValidationError(f"Path traversal detected in parameter: {key}")
                
                # Sanitize string value
                sanitized_value = self.secure_validator.sanitize_string(value)
                sanitized_params[clean_key] = sanitized_value
                
            elif isinstance(value, (int, float)):
                # Validate numeric ranges
                if abs(value) > 1e15:  # Prevent numeric overflow
                    context.add_security_threat(f"numeric_overflow_attempt_{value}")
                    raise ValidationError(f"Numeric value too large: {key}")
                sanitized_params[clean_key] = value
                
            elif isinstance(value, (list, dict)):
                # Validate nested structures
                if not self.secure_validator.validate_size_limits(value, max_depth=5):
                    context.add_security_threat(f"nested_structure_too_deep_{key}")
                    raise ValidationError(f"Nested structure too deep: {key}")
                sanitized_params[clean_key] = value
                
            else:
                # Allow None and bool, reject other types
                if value is not None and not isinstance(value, bool):
                    context.add_security_threat(f"unsupported_parameter_type_{type(value)}")
                    raise ValidationError(f"Unsupported parameter type for {key}: {type(value)}")
                sanitized_params[clean_key] = value
        
        return sanitized_params
    
    def create_admin_token(self, admin_user_id: str = "admin") -> str:
        """
        Create admin token for system operations
        
        Args:
            admin_user_id: Admin user identifier
            
        Returns:
            JWT token with admin permissions
        """
        admin_permissions = [
            Permission.READ_TOOLS,
            Permission.EXECUTE_TOOLS, 
            Permission.READ_RESOURCES,
            Permission.WRITE_RESOURCES,
            Permission.ADMIN_SERVER,
            Permission.METRICS_READ,
            Permission.HEALTH_CHECK
        ]
        
        return self.secure_authenticator.create_user_token(admin_user_id, admin_permissions)
    
    def create_read_only_token(self, user_id: str) -> str:
        """
        Create read-only token for limited operations
        
        Args:
            user_id: User identifier
            
        Returns:
            JWT token with read-only permissions
        """
        read_permissions = [
            Permission.READ_TOOLS,
            Permission.READ_RESOURCES,
            Permission.HEALTH_CHECK
        ]
        
        return self.secure_authenticator.create_user_token(user_id, read_permissions)


def create_secure_middleware(jwt_secret: str, enable_strict_mode: bool = True) -> SecureMiddleware:
    """
    Create secure middleware with no authentication bypass vulnerabilities
    
    Args:
        jwt_secret: JWT secret key (must be at least 32 characters)
        enable_strict_mode: If True, blocks all unauthenticated requests
        
    Returns:
        SecureMiddleware instance
        
    Raises:
        ValueError: If JWT secret is too short
    """
    if len(jwt_secret) < 32:
        raise ValueError("JWT secret must be at least 32 characters for security")
    
    config = SecurityConfig(
        require_auth=True,
        allow_anonymous_read=False,  # No anonymous access
        rate_limiting_enabled=True,
        input_validation_enabled=True,
        strict_validation=True,
        log_security_events=True,
        max_request_size=256 * 1024  # 256KB limit
    )
    
    return SecureMiddleware(config, jwt_secret, enable_strict_mode)