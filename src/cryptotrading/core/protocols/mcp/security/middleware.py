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

from .auth import AuthenticationMiddleware, AuthToken, create_auth_middleware
from .rate_limiter import RateLimitMiddleware, RateLimitExceeded, create_rate_limit_middleware
from .validation import ValidationMiddleware, ValidationError, create_validation_middleware

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
    
    def add_security_threat(self, threat: str):
        """Add security threat"""
        self.security_threats.append(threat)
        logger.warning(f"Security threat detected: {threat}")
    
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
            
            # Rate limiting
            if self.config.rate_limiting_enabled and self.rate_limit_middleware:
                identifier = context.user_id or context.api_key_id or "anonymous"
                await self.rate_limit_middleware.check_request_limits(
                    identifier, method, context.api_key_id
                )
            
            # Input validation
            processed_params = params
            if self.config.input_validation_enabled and self.validation_middleware:
                processed_params = await self.validation_middleware.validate_request(method, params)
            
            # Log successful request
            if self.config.log_requests:
                self._log_request(context, "success")
            
            return processed_params, context
            
        except RateLimitExceeded as e:
            self.metrics.record_rate_limit()
            context.add_security_threat(f"Rate limit exceeded: {str(e)}")
            if self.config.log_security_events:
                self._log_request(context, "rate_limited", str(e))
            raise
            
        except ValidationError as e:
            self.metrics.record_validation_failure()
            context.validation_errors.append(str(e))
            if "security threat" in str(e).lower():
                self.metrics.record_security_threat()
                context.add_security_threat(str(e))
            if self.config.log_security_events:
                self._log_request(context, "validation_failed", str(e))
            raise
            
        except Exception as e:
            context.add_security_threat(f"Unexpected error: {str(e)}")
            if self.config.log_security_events:
                self._log_request(context, "error", str(e))
            raise
    
    def _get_required_scope(self, method: str) -> str:
        """Get required scope for method"""
        if method.startswith("tools/call"):
            return "tools:call"
        elif method.startswith("tools/"):
            return "tools:read"
        elif method.startswith("resources/read"):
            return "resources:read"
        elif method.startswith("resources/"):
            return "resources:read"
        elif method in ["initialize", "ping"]:
            return "read"
        else:
            return "admin"
    
    def _log_request(self, context: SecurityContext, status: str, error: Optional[str] = None):
        """Log request with context"""
        log_data = {
            "status": status,
            "method": context.method,
            "duration": context.get_duration(),
            "user_id": context.user_id,
            "api_key_id": context.api_key_id,
            "request_size": context.request_size,
            "authenticated": context.auth_token is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error:
            log_data["error"] = error
        
        if context.security_threats:
            log_data["security_threats"] = context.security_threats
        
        if context.validation_errors:
            log_data["validation_errors"] = context.validation_errors
        
        if status == "success":
            logger.info("MCP request processed", extra=log_data)
        elif status in ["rate_limited", "validation_failed", "error"]:
            logger.warning(f"MCP request {status}", extra=log_data)
        else:
            logger.error("MCP request failed", extra=log_data)
    
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
        """Load security config from Vercel environment variables"""
        config = SecurityConfig()
        
        # Authentication settings
        config.require_auth = os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true"
        config.allow_anonymous_read = os.getenv("MCP_ALLOW_ANONYMOUS_READ", "false").lower() == "true"
        config.jwt_enabled = os.getenv("MCP_JWT_ENABLED", "true").lower() == "true"
        config.api_key_enabled = os.getenv("MCP_API_KEY_ENABLED", "true").lower() == "true"
        
        # Rate limiting settings
        config.rate_limiting_enabled = os.getenv("MCP_RATE_LIMITING", "true").lower() == "true"
        config.global_rate_limit = int(os.getenv("MCP_GLOBAL_RATE_LIMIT", "1000"))
        config.user_rate_limit = int(os.getenv("MCP_USER_RATE_LIMIT", "100"))
        
        # Validation settings
        config.input_validation_enabled = os.getenv("MCP_INPUT_VALIDATION", "true").lower() == "true"
        config.strict_validation = os.getenv("MCP_STRICT_VALIDATION", "true").lower() == "true"
        config.max_request_size = int(os.getenv("MCP_MAX_REQUEST_SIZE", str(1024 * 1024)))
        
        # Logging settings
        config.log_requests = os.getenv("MCP_LOG_REQUESTS", "true").lower() == "true"
        config.log_security_events = os.getenv("MCP_LOG_SECURITY", "true").lower() == "true"
        config.metrics_enabled = os.getenv("MCP_METRICS", "true").lower() == "true"
        
        # Method filtering
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