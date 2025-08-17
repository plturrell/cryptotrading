"""
MCP Security Package

This package provides comprehensive security features for MCP implementations:
- JWT and API key authentication
- Rate limiting with Vercel Edge Config
- Input validation and sanitization
- Security middleware integration
"""

from .auth import (
    AuthenticationError,
    AuthorizationError,
    AuthToken,
    APIKey,
    JWTManager,
    APIKeyManager,
    AuthenticationMiddleware,
    TokenScope,
    create_jwt_manager,
    create_api_key_manager,
    create_auth_middleware
)

from .rate_limiter import (
    RateLimitExceeded,
    RateLimitConfig,
    RateLimitAlgorithm,
    RateLimiter,
    RateLimitMiddleware,
    VercelEdgeConfigClient,
    create_edge_config_client,
    create_rate_limiter,
    create_rate_limit_middleware
)

from .validation import (
    ValidationError,
    SanitizationError,
    ValidationRule,
    SecurityValidator,
    SchemaValidator,
    MCPInputValidator,
    ValidationMiddleware,
    create_security_validator,
    create_mcp_validator,
    create_validation_middleware
)

from .middleware import (
    SecurityMiddleware,
    VercelSecurityMiddleware,
    SecureMiddleware,
    SecurityContext,
    SecurityConfig,
    create_security_middleware,
    create_secure_middleware
)

__all__ = [
    # Auth
    'AuthenticationError',
    'AuthorizationError', 
    'AuthToken',
    'APIKey',
    'JWTManager',
    'APIKeyManager',
    'AuthenticationMiddleware',
    'TokenScope',
    'create_jwt_manager',
    'create_api_key_manager',
    'create_auth_middleware',
    
    # Rate Limiting
    'RateLimitExceeded',
    'RateLimitConfig',
    'RateLimitAlgorithm',
    'RateLimiter',
    'RateLimitMiddleware',
    'VercelEdgeConfigClient',
    'create_edge_config_client',
    'create_rate_limiter',
    'create_rate_limit_middleware',
    
    # Validation
    'ValidationError',
    'SanitizationError',
    'ValidationRule',
    'SecurityValidator',
    'SchemaValidator',
    'MCPInputValidator',
    'ValidationMiddleware',
    'create_security_validator',
    'create_mcp_validator',
    'create_validation_middleware',
    
    # Middleware
    'SecurityMiddleware',
    'VercelSecurityMiddleware',
    'SecureMiddleware',
    'SecurityContext',
    'SecurityConfig',
    'create_security_middleware',
    'create_secure_middleware'
]