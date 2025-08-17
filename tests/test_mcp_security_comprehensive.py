"""
Comprehensive Security Tests for MCP Implementation
Tests all security features including authentication, authorization, rate limiting,
input validation, audit logging, and secure defaults.
"""

import pytest
import asyncio
import json
import time
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import redis.asyncio as redis
from unittest.mock import Mock, AsyncMock, patch

from cryptotrading.core.protocols.mcp.security.middleware import (
    SecurityConfig, SecurityMiddleware, SecureMiddleware,
    create_secure_middleware, VercelSecurityMiddleware
)
from cryptotrading.core.protocols.mcp.security.authentication import (
    SecureAuthenticator, AuthenticationContext, Permission,
    AuthenticationMethod, AuthenticationError, AuthorizationError
)
from cryptotrading.core.protocols.mcp.security.token_storage import (
    RedisTokenStorage, DatabaseTokenStorage, InMemoryTokenStorage,
    get_token_storage_manager
)
from cryptotrading.core.protocols.mcp.security.permissions import (
    get_permission_validator, MethodSecurityLevel
)
from cryptotrading.core.protocols.mcp.security.audit_logger import (
    get_audit_logger, SecurityEventType
)
from cryptotrading.core.protocols.mcp.security.secure_defaults import (
    get_secure_config_manager, get_security_defaults
)
from cryptotrading.core.protocols.mcp.security.error_sanitizer import (
    get_error_sanitizer
)
from cryptotrading.core.protocols.mcp.security.rate_limit_headers import (
    get_rate_limit_header_manager, RateLimitInfo
)


class TestSecureAuthenticator:
    """Test SecureAuthenticator with no authentication bypass"""
    
    @pytest.fixture
    def authenticator(self):
        """Create secure authenticator instance"""
        return SecureAuthenticator(
            secret_key="test_secret_key_32_chars_minimum!!",
            enable_strict_mode=True
        )
    
    @pytest.mark.asyncio
    async def test_no_anonymous_access_in_strict_mode(self, authenticator):
        """Test that strict mode blocks all anonymous access"""
        headers = {}  # No auth headers
        
        with pytest.raises(AuthenticationError) as exc_info:
            await authenticator.authenticate_request(headers)
        
        assert "Authentication required" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_jwt_authentication_success(self, authenticator):
        """Test successful JWT authentication"""
        # Create valid token
        token = authenticator.create_user_token(
            "test_user",
            [Permission.READ_TOOLS, Permission.EXECUTE_TOOLS]
        )
        
        headers = {"Authorization": f"Bearer {token}"}
        
        auth_context = await authenticator.authenticate_request(headers)
        
        assert auth_context is not None
        assert auth_context.user_id == "test_user"
        assert auth_context.method == AuthenticationMethod.JWT
        assert Permission.READ_TOOLS in auth_context.permissions
        assert Permission.EXECUTE_TOOLS in auth_context.permissions
    
    @pytest.mark.asyncio
    async def test_jwt_token_expiration(self, authenticator):
        """Test JWT token expiration handling"""
        # Create token that expires immediately
        payload = {
            "sub": "test_user",
            "permissions": ["read_tools"],
            "exp": (datetime.utcnow() - timedelta(seconds=1)).timestamp()
        }
        
        expired_token = jwt.encode(
            payload,
            authenticator.secret_key,
            algorithm="HS256"
        )
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        
        with pytest.raises(AuthenticationError) as exc_info:
            await authenticator.authenticate_request(headers)
        
        assert "expired" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_api_key_authentication(self, authenticator):
        """Test API key authentication"""
        # Add API key
        authenticator.add_api_key(
            "test_api_key_123",
            user_id="api_user",
            permissions=[Permission.READ_RESOURCES]
        )
        
        headers = {"X-API-Key": "test_api_key_123"}
        
        auth_context = await authenticator.authenticate_request(headers)
        
        assert auth_context is not None
        assert auth_context.user_id == "api_user"
        assert auth_context.method == AuthenticationMethod.API_KEY
        assert Permission.READ_RESOURCES in auth_context.permissions
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self, authenticator):
        """Test invalid API key rejection"""
        headers = {"X-API-Key": "invalid_key"}
        
        with pytest.raises(AuthenticationError) as exc_info:
            await authenticator.authenticate_request(headers)
        
        assert "Invalid API key" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_permission_checking(self, authenticator):
        """Test permission checking functionality"""
        # Create token with limited permissions
        token = authenticator.create_user_token(
            "limited_user",
            [Permission.READ_TOOLS]  # Only read permission
        )
        
        headers = {"Authorization": f"Bearer {token}"}
        auth_context = await authenticator.authenticate_request(headers)
        
        # Should have read permission
        assert authenticator.check_permission(auth_context, Permission.READ_TOOLS)
        
        # Should not have write permission
        assert not authenticator.check_permission(auth_context, Permission.WRITE_RESOURCES)
    
    @pytest.mark.asyncio
    async def test_token_validation_with_missing_fields(self, authenticator):
        """Test token validation with missing required fields"""
        # Token without permissions
        payload = {
            "sub": "test_user",
            "exp": (datetime.utcnow() + timedelta(hours=1)).timestamp()
        }
        
        invalid_token = jwt.encode(
            payload,
            authenticator.secret_key,
            algorithm="HS256"
        )
        
        headers = {"Authorization": f"Bearer {invalid_token}"}
        
        with pytest.raises(AuthenticationError) as exc_info:
            await authenticator.authenticate_request(headers)
        
        assert "Invalid token" in str(exc_info.value)


class TestTokenStorage:
    """Test token revocation storage implementations"""
    
    @pytest.mark.asyncio
    async def test_in_memory_token_storage(self):
        """Test in-memory token storage"""
        storage = InMemoryTokenStorage()
        
        # Revoke token
        token_hash = "test_token_hash_123"
        assert await storage.revoke_token(token_hash)
        
        # Check if revoked
        assert await storage.is_revoked(token_hash)
        
        # Non-revoked token
        assert not await storage.is_revoked("other_token_hash")
        
        # Clear storage
        await storage.clear()
        assert not await storage.is_revoked(token_hash)
    
    @pytest.mark.asyncio
    async def test_redis_token_storage(self):
        """Test Redis token storage with mocked Redis"""
        mock_redis = AsyncMock()
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            storage = RedisTokenStorage("redis://localhost:6379")
            
            # Test revoke
            token_hash = "test_token_hash"
            mock_redis.setex.return_value = True
            
            assert await storage.revoke_token(token_hash)
            mock_redis.setex.assert_called_once()
            
            # Test is_revoked
            mock_redis.get.return_value = b"1"
            assert await storage.is_revoked(token_hash)
            
            mock_redis.get.return_value = None
            assert not await storage.is_revoked("other_token")
    
    @pytest.mark.asyncio
    async def test_database_token_storage(self):
        """Test database token storage with mocked database"""
        mock_db = AsyncMock()
        storage = DatabaseTokenStorage(mock_db)
        
        # Test revoke
        token_hash = "test_token_hash"
        mock_db.execute.return_value = None
        
        assert await storage.revoke_token(token_hash)
        mock_db.execute.assert_called_once()
        
        # Test is_revoked
        mock_db.fetch_one.return_value = {"revoked_at": datetime.utcnow()}
        assert await storage.is_revoked(token_hash)
        
        mock_db.fetch_one.return_value = None
        assert not await storage.is_revoked("other_token")
    
    @pytest.mark.asyncio
    async def test_token_storage_manager(self):
        """Test token storage manager with fallback"""
        manager = get_token_storage_manager()
        
        # Should default to in-memory storage
        assert manager.get_storage() is not None
        
        # Test operations
        token_hash = "test_token"
        assert await manager.revoke_token(token_hash)
        assert await manager.is_revoked(token_hash)


class TestSecurePermissions:
    """Test secure permission system"""
    
    def test_permission_validator_initialization(self):
        """Test permission validator with secure defaults"""
        validator = get_permission_validator()
        
        # Check that no methods allow anonymous access
        for method, config in validator.registry.method_configs.items():
            assert not config.allow_anonymous, f"Method {method} allows anonymous access!"
    
    def test_method_permission_validation(self):
        """Test method-level permission validation"""
        validator = get_permission_validator()
        
        # Test with valid authentication context
        auth_context = AuthenticationContext(
            user_id="test_user",
            method=AuthenticationMethod.JWT,
            permissions=[Permission.READ_TOOLS],
            authenticated_at=datetime.utcnow()
        )
        
        # Should allow read operations
        result = validator.validate_method_access("tools/list", auth_context)
        assert result["allowed"]
        
        # Should deny write operations
        result = validator.validate_method_access("resources/write", auth_context)
        assert not result["allowed"]
        assert "insufficient_permissions" in result["reason"]
    
    def test_admin_method_protection(self):
        """Test admin methods require admin permissions"""
        validator = get_permission_validator()
        
        # Non-admin user
        auth_context = AuthenticationContext(
            user_id="regular_user",
            method=AuthenticationMethod.JWT,
            permissions=[Permission.READ_TOOLS, Permission.EXECUTE_TOOLS],
            authenticated_at=datetime.utcnow()
        )
        
        # Should deny admin methods
        admin_methods = ["security/status", "server/shutdown", "admin/config"]
        for method in admin_methods:
            result = validator.validate_method_access(method, auth_context)
            assert not result["allowed"], f"Non-admin accessed {method}!"


class TestSecurityAuditLogging:
    """Test structured security audit logging"""
    
    def test_audit_logger_initialization(self):
        """Test audit logger setup"""
        logger = get_audit_logger()
        assert logger is not None
        assert logger.logger is not None
    
    def test_security_event_logging(self):
        """Test logging of security events"""
        logger = get_audit_logger()
        
        # Create security context
        from cryptotrading.core.protocols.mcp.security.audit_logger import SecurityContext
        
        context = SecurityContext(
            request_id="test_req_123",
            user_id="test_user",
            ip_address="192.168.1.1",
            method="tools/call"
        )
        
        # Test different event types
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_authentication_success(
                context,
                AuthenticationMethod.JWT
            )
            mock_info.assert_called_once()
            
        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.log_authentication_failure(
                context,
                "Invalid credentials"
            )
            mock_warning.assert_called_once()
            
        with patch.object(logger.logger, 'error') as mock_error:
            logger.log_security_threat(
                context,
                "sql_injection_attempt",
                "Detected SQL injection in parameter"
            )
            mock_error.assert_called_once()


class TestSecureDefaults:
    """Test secure default configurations"""
    
    def test_production_defaults(self):
        """Test production environment defaults"""
        import os
        os.environ['ENVIRONMENT'] = 'production'
        
        defaults = get_security_defaults()
        
        # Production should have strict settings
        assert defaults.require_authentication
        assert not defaults.allow_anonymous_access
        assert defaults.jwt_enabled
        assert defaults.rate_limiting_enabled
        assert defaults.strict_validation
        assert defaults.audit_logging_enabled
    
    def test_development_defaults(self):
        """Test development environment defaults"""
        import os
        os.environ['ENVIRONMENT'] = 'development'
        
        defaults = get_security_defaults()
        
        # Development can be less strict
        assert defaults.security_logging_enabled
        assert defaults.max_request_size_bytes > 0
    
    def test_environment_overrides(self):
        """Test environment variable overrides"""
        import os
        os.environ['MCP_REQUIRE_AUTH'] = 'false'
        os.environ['MCP_RATE_LIMIT_GLOBAL'] = '5000'
        
        config_manager = get_secure_config_manager()
        config = config_manager.get_config()
        
        # Should respect environment overrides
        assert not config.require_authentication
        assert config.global_rate_limit_per_minute == 5000
        
        # Cleanup
        del os.environ['MCP_REQUIRE_AUTH']
        del os.environ['MCP_RATE_LIMIT_GLOBAL']


class TestErrorSanitization:
    """Test error message sanitization"""
    
    def test_error_sanitizer_patterns(self):
        """Test sanitization of sensitive patterns"""
        sanitizer = get_error_sanitizer()
        
        # Test file path sanitization
        error = "Error in /home/user/project/secret.py:123"
        sanitized = sanitizer.sanitize_error_message(error)
        assert "/home/user/project/secret.py" not in sanitized
        assert "[REDACTED_PATH]" in sanitized
        
        # Test password sanitization
        error = 'Authentication failed: password="secret123"'
        sanitized = sanitizer.sanitize_error_message(error)
        assert "secret123" not in sanitized
        assert "[REDACTED]" in sanitized
        
        # Test API key sanitization
        error = "Invalid API key: sk-1234567890abcdef"
        sanitized = sanitizer.sanitize_error_message(error)
        assert "sk-1234567890abcdef" not in sanitized
        assert "[REDACTED_KEY]" in sanitized
    
    def test_safe_error_messages(self):
        """Test that safe messages are not modified"""
        sanitizer = get_error_sanitizer()
        
        safe_messages = [
            "Invalid request format",
            "Method not found",
            "Rate limit exceeded",
            "Authentication required"
        ]
        
        for message in safe_messages:
            assert sanitizer.sanitize_error_message(message) == message


class TestRateLimitHeaders:
    """Test rate limit response headers"""
    
    def test_rate_limit_header_creation(self):
        """Test creation of RFC-compliant headers"""
        manager = get_rate_limit_header_manager()
        
        info = RateLimitInfo(
            limit=100,
            remaining=75,
            reset=int(time.time() + 3600),
            window=60,
            policy="user_tier_1"
        )
        
        headers = manager.create_headers(info)
        
        # Check RFC 6585 compliant headers
        assert headers["X-RateLimit-Limit"] == "100"
        assert headers["X-RateLimit-Remaining"] == "75"
        assert headers["X-RateLimit-Reset"] == str(info.reset)
        assert headers["X-RateLimit-Policy"] == "user_tier_1"
        
        # Check draft-ietf headers
        assert headers["RateLimit-Limit"] == "100"
        assert headers["RateLimit-Remaining"] == "75"
        assert headers["RateLimit-Reset"] == str(info.reset)
    
    def test_retry_after_header(self):
        """Test Retry-After header when rate limited"""
        manager = get_rate_limit_header_manager()
        
        info = RateLimitInfo(
            limit=100,
            remaining=0,  # No requests remaining
            reset=int(time.time() + 300),  # Reset in 5 minutes
            window=60
        )
        
        headers = manager.create_headers(info)
        
        # Should include Retry-After
        assert "Retry-After" in headers
        retry_after = int(headers["Retry-After"])
        assert 295 <= retry_after <= 305  # Allow some timing variance


class TestSecureMiddleware:
    """Test integrated secure middleware"""
    
    @pytest.fixture
    def secure_middleware(self):
        """Create secure middleware instance"""
        return create_secure_middleware(
            jwt_secret="test_secret_key_32_chars_minimum!!",
            enable_strict_mode=True
        )
    
    @pytest.mark.asyncio
    async def test_complete_security_pipeline(self, secure_middleware):
        """Test full security pipeline with all features"""
        # Create valid auth token
        token = secure_middleware.create_admin_token("admin_user")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Request-ID": "test_req_123",
            "X-Forwarded-For": "192.168.1.100"
        }
        
        params = {
            "tool": "get_portfolio",
            "arguments": {"include_history": True}
        }
        
        # Process request through secure pipeline
        processed_params, context = await secure_middleware.process_request(
            "tools/call",
            params,
            headers
        )
        
        # Verify authentication
        assert context.user_id == "admin_user"
        assert context.auth_token is not None
        
        # Verify sanitization
        assert processed_params == params  # Valid params should pass through
        
        # Verify response headers include rate limit info
        assert context.response_headers  # Should have rate limit headers
    
    @pytest.mark.asyncio
    async def test_authentication_bypass_prevention(self, secure_middleware):
        """Test that authentication bypass is prevented"""
        # Try various bypass attempts
        bypass_attempts = [
            {},  # No headers
            {"Authorization": "Bearer invalid_token"},  # Invalid token
            {"Authorization": ""},  # Empty auth
            {"X-API-Key": ""},  # Empty API key
        ]
        
        for headers in bypass_attempts:
            with pytest.raises(AuthenticationError):
                await secure_middleware.process_request(
                    "tools/list",
                    {},
                    headers
                )
    
    @pytest.mark.asyncio
    async def test_permission_enforcement(self, secure_middleware):
        """Test permission enforcement for different operations"""
        # Create read-only token
        token = secure_middleware.create_read_only_token("readonly_user")
        headers = {"Authorization": f"Bearer {token}"}
        
        # Should allow read operations
        params, context = await secure_middleware.process_request(
            "tools/list",
            {},
            headers
        )
        assert context.user_id == "readonly_user"
        
        # Should deny write operations
        with pytest.raises(AuthorizationError) as exc_info:
            await secure_middleware.process_request(
                "resources/write",
                {"path": "/test", "content": "data"},
                headers
            )
        assert "Permission denied" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_input_validation_security(self, secure_middleware):
        """Test input validation prevents malicious inputs"""
        token = secure_middleware.create_admin_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test path traversal prevention
        with pytest.raises(Exception) as exc_info:
            await secure_middleware.process_request(
                "resources/read",
                {"path": "../../etc/passwd"},
                headers
            )
        assert "Path traversal" in str(exc_info.value)
        
        # Test oversized request prevention
        large_data = "x" * (257 * 1024)  # Over 256KB limit
        with pytest.raises(Exception) as exc_info:
            await secure_middleware.process_request(
                "tools/call",
                {"data": large_data},
                headers
            )
        assert "too large" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_security_event_tracking(self, secure_middleware):
        """Test security events are properly tracked"""
        # Trigger various security events
        headers = {"Authorization": "Bearer invalid_token"}
        
        # Authentication failure
        with pytest.raises(AuthenticationError):
            await secure_middleware.process_request("tools/list", {}, headers)
        
        # Check metrics
        metrics = secure_middleware.metrics.get_metrics()
        assert metrics["security_threats"] > 0
        
        # Valid request
        token = secure_middleware.create_admin_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        await secure_middleware.process_request("tools/list", {}, headers)
        
        updated_metrics = secure_middleware.metrics.get_metrics()
        assert updated_metrics["authenticated_requests"] > metrics.get("authenticated_requests", 0)


class TestVercelSecurityMiddleware:
    """Test Vercel-specific security middleware"""
    
    def test_vercel_environment_configuration(self):
        """Test Vercel environment configuration loading"""
        import os
        
        # Set Vercel environment variables
        os.environ['VERCEL_ENV'] = 'production'
        os.environ['MCP_JWT_SECRET'] = 'vercel_production_secret_key_32_chars!!'
        
        middleware = VercelSecurityMiddleware()
        
        # Should load production defaults
        assert middleware.config.require_auth
        assert middleware.config.rate_limiting_enabled
        assert middleware.config.strict_validation
        
        # Cleanup
        del os.environ['VERCEL_ENV']
        del os.environ['MCP_JWT_SECRET']
    
    @pytest.mark.asyncio
    async def test_vercel_edge_function_compatibility(self):
        """Test middleware works with Vercel Edge Function constraints"""
        middleware = VercelSecurityMiddleware()
        
        # Should handle requests efficiently
        start_time = time.time()
        
        # Create minimal request
        with pytest.raises(AuthenticationError):
            await middleware.process_request(
                "tools/list",
                {},
                {}
            )
        
        # Should be fast (under 50ms for auth check)
        duration = time.time() - start_time
        assert duration < 0.05, f"Auth check too slow: {duration}s"


class TestEndToEndSecurity:
    """End-to-end security integration tests"""
    
    @pytest.fixture
    def mcp_server_with_security(self):
        """Create MCP server with full security enabled"""
        from cryptotrading.core.protocols.mcp.server import MCPServer
        
        server = MCPServer("secure-test-server", "1.0.0")
        
        # Add security middleware
        security_config = SecurityConfig(
            require_auth=True,
            rate_limiting_enabled=True,
            input_validation_enabled=True,
            log_security_events=True
        )
        
        server.security_middleware = SecurityMiddleware(security_config)
        
        return server
    
    @pytest.mark.asyncio
    async def test_secure_mcp_request_flow(self, mcp_server_with_security):
        """Test complete secure MCP request flow"""
        server = mcp_server_with_security
        
        # Create request without auth
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/list",
            "params": {}
        }
        
        # Should fail without authentication
        response = await server.handle_request(request)
        assert "error" in response
        assert response["error"]["code"] == -32001  # Unauthorized
        
        # Create authenticated request
        # (In real scenario, would include proper JWT token)
        auth_request = {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/list",
            "params": {},
            "auth": {
                "token": "valid_jwt_token_here"
            }
        }
        
        # Would succeed with valid auth in real implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])