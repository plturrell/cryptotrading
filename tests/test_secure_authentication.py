"""
Test Suite for Secure Authentication System
Tests the fixed authentication system with no bypass vulnerabilities
"""
import pytest
import asyncio
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any

from src.cryptotrading.core.protocols.mcp.security.authentication import (
    SecureAuthenticator,
    AuthenticationContext, 
    Permission,
    AuthenticationMethod,
    SecureValidator,
    RateLimiter
)
from src.cryptotrading.core.protocols.mcp.security.middleware import (
    SecureMiddleware,
    SecurityConfig,
    create_secure_middleware
)
from src.cryptotrading.core.protocols.mcp.security.auth import (
    AuthenticationError,
    AuthorizationError
)
from src.cryptotrading.core.protocols.mcp.security.validation import ValidationError
from src.cryptotrading.core.protocols.mcp.security.rate_limiter import RateLimitExceeded


class TestSecureAuthenticator:
    """Test the SecureAuthenticator class"""
    
    @pytest.fixture
    def secure_secret(self):
        """Generate secure JWT secret"""
        return secrets.token_urlsafe(32)
    
    @pytest.fixture
    def authenticator(self, secure_secret):
        """Create SecureAuthenticator instance"""
        return SecureAuthenticator(secure_secret, enable_strict_mode=True)
    
    def test_initialization_requires_secure_secret(self):
        """Test that initialization requires a secure secret"""
        # Should fail with short secret
        with pytest.raises(ValueError, match="Secret key must be at least 32 characters"):
            SecureAuthenticator("short_secret")
        
        # Should succeed with secure secret
        secure_secret = secrets.token_urlsafe(32)
        authenticator = SecureAuthenticator(secure_secret)
        assert authenticator is not None
    
    def test_token_creation_and_validation(self, authenticator):
        """Test token creation and validation"""
        permissions = [Permission.READ_TOOLS, Permission.EXECUTE_TOOLS]
        user_id = "test_user"
        
        # Create token
        token = authenticator.create_user_token(user_id, permissions)
        assert token is not None
        assert isinstance(token, str)
        
        # Validate token
        context = authenticator.token_manager.validate_token(token)
        assert context is not None
        assert context.user_id == user_id
        assert context.method == AuthenticationMethod.JWT_BEARER
        assert Permission.READ_TOOLS in context.permissions
        assert Permission.EXECUTE_TOOLS in context.permissions
    
    def test_api_key_creation_and_validation(self, authenticator):
        """Test API key creation and validation"""
        permissions = [Permission.READ_RESOURCES]
        user_id = "api_user"
        
        # Create API key
        key_id, secret = authenticator.create_user_api_key(user_id, permissions, "test_key")
        assert key_id is not None
        assert secret is not None
        assert len(secret) >= 32  # Secure length
        
        # Validate API key
        context = authenticator.api_key_manager.validate_api_key(key_id, secret)
        assert context is not None
        assert context.user_id == user_id
        assert context.method == AuthenticationMethod.API_KEY
        assert Permission.READ_RESOURCES in context.permissions
    
    @pytest.mark.asyncio
    async def test_strict_mode_blocks_unauthenticated_requests(self, authenticator):
        """Test that strict mode blocks all unauthenticated requests"""
        headers = {}  # No authentication headers
        
        # Should return None in strict mode
        context = await authenticator.authenticate_request(headers)
        assert context is None
    
    @pytest.mark.asyncio 
    async def test_anonymous_context_limited_permissions(self, secure_secret):
        """Test that anonymous context has very limited permissions"""
        # Create authenticator with strict mode disabled
        authenticator = SecureAuthenticator(secure_secret, enable_strict_mode=False)
        
        headers = {}  # No authentication
        context = await authenticator.authenticate_request(headers)
        
        assert context is not None
        assert context.user_id == "anonymous"
        assert context.permissions == {Permission.HEALTH_CHECK}  # Only health checks
    
    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, authenticator):
        """Test that expired tokens are rejected"""
        permissions = [Permission.READ_TOOLS]
        user_id = "test_user"
        
        # Create token with very short expiration (0 hours = immediate expiry)
        token = authenticator.token_manager.create_token(user_id, permissions, expires_hours=0)
        
        # Wait a moment to ensure expiration
        await asyncio.sleep(0.1)
        
        # Should be rejected as expired
        context = authenticator.token_manager.validate_token(token)
        assert context is None
    
    @pytest.mark.asyncio
    async def test_rate_limiting_per_user(self, authenticator):
        """Test that rate limiting works per user"""
        rate_limiter = RateLimiter(window_seconds=1, max_requests=2)
        
        user_key = "user:test_user"
        
        # First two requests should succeed
        assert await rate_limiter.check_rate_limit(user_key) is True
        assert await rate_limiter.check_rate_limit(user_key) is True
        
        # Third request should be rate limited
        assert await rate_limiter.check_rate_limit(user_key) is False
        
        # Check remaining quota
        remaining = await rate_limiter.get_remaining_quota(user_key)
        assert remaining == 0


class TestSecureValidator:
    """Test the SecureValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create SecureValidator instance"""
        return SecureValidator()
    
    def test_path_traversal_detection(self, validator):
        """Test path traversal attack detection"""
        # Safe paths
        assert validator.validate_path("documents/file.txt") is True
        assert validator.validate_path("data.json") is True
        
        # Dangerous paths
        assert validator.validate_path("../../../etc/passwd") is False
        assert validator.validate_path("..\\windows\\system32") is False
        assert validator.validate_path("/etc/passwd") is False
        assert validator.validate_path("file://etc/passwd") is False
        assert validator.validate_path("http://evil.com/file") is False
    
    def test_size_limits_validation(self, validator):
        """Test size and depth limits"""
        # Small valid object
        small_obj = {"key": "value", "number": 42}
        assert validator.validate_size_limits(small_obj) is True
        
        # Deeply nested object (should fail)
        deep_obj = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": {"level7": {"level8": {"level9": {"level10": {"level11": "too_deep"}}}}}}}}}}
        assert validator.validate_size_limits(deep_obj, max_depth=10) is False
        
        # Very large object (should fail)
        large_obj = {"data": "x" * (2 * 1024 * 1024)}  # 2MB string
        assert validator.validate_size_limits(large_obj, max_size=1024*1024) is False
    
    def test_string_sanitization(self, validator):
        """Test string sanitization"""
        # Clean string
        clean = validator.sanitize_string("Hello World 123")
        assert clean == "Hello World 123"
        
        # String with dangerous characters
        dangerous = validator.sanitize_string("<script>alert('xss')</script>")
        assert "<" not in dangerous
        assert ">" not in dangerous
        assert "script" in dangerous  # Content preserved, tags removed
        
        # String with null bytes
        with_nulls = validator.sanitize_string("hello\x00world\x01test")
        assert "\x00" not in with_nulls
        assert "\x01" not in with_nulls
        assert "helloworld" in with_nulls


class TestSecureMiddleware:
    """Test the SecureMiddleware class"""
    
    @pytest.fixture
    def jwt_secret(self):
        """Generate secure JWT secret"""
        return secrets.token_urlsafe(32)
    
    @pytest.fixture
    def middleware(self, jwt_secret):
        """Create SecureMiddleware instance"""
        return create_secure_middleware(jwt_secret, enable_strict_mode=True)
    
    @pytest.fixture
    def admin_token(self, middleware):
        """Create admin token for testing"""
        return middleware.create_admin_token("test_admin")
    
    @pytest.fixture
    def read_only_token(self, middleware):
        """Create read-only token for testing"""
        return middleware.create_read_only_token("read_user")
    
    @pytest.mark.asyncio
    async def test_unauthenticated_request_blocked(self, middleware):
        """Test that unauthenticated requests are blocked"""
        headers = {}  # No authentication
        params = {"test": "value"}
        
        with pytest.raises(AuthenticationError, match="Authentication required"):
            await middleware.process_request("tools/call", params, headers)
    
    @pytest.mark.asyncio
    async def test_authenticated_request_allowed(self, middleware, admin_token):
        """Test that properly authenticated requests are allowed"""
        headers = {"authorization": f"Bearer {admin_token}"}
        params = {"tool_name": "test_tool"}
        
        # Should not raise exception
        processed_params, context = await middleware.process_request("tools/call", params, headers)
        
        assert processed_params is not None
        assert context.user_id == "test_admin"
        assert context.auth_token is not None
    
    @pytest.mark.asyncio
    async def test_permission_enforcement(self, middleware, read_only_token):
        """Test that permissions are properly enforced"""
        headers = {"authorization": f"Bearer {read_only_token}"}
        params = {"tool_name": "test_tool"}
        
        # Read-only user should not be able to execute tools
        with pytest.raises(AuthorizationError, match="Insufficient permissions"):
            await middleware.process_request("tools/call", params, headers)
        
        # But should be able to list tools
        processed_params, context = await middleware.process_request("tools/list", {}, headers)
        assert processed_params is not None
        assert context.user_id == "read_user"
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self, middleware, admin_token):
        """Test input sanitization and validation"""
        headers = {"authorization": f"Bearer {admin_token}"}
        
        # Test path traversal attempt
        dangerous_params = {"file_path": "../../../etc/passwd"}
        
        with pytest.raises(ValidationError, match="Path traversal detected"):
            await middleware.process_request("tools/call", dangerous_params, headers)
        
        # Test oversized request
        large_params = {"data": "x" * (2 * 1024 * 1024)}  # 2MB
        
        with pytest.raises(ValidationError, match="size limits"):
            await middleware.process_request("tools/call", large_params, headers)
    
    @pytest.mark.asyncio
    async def test_expired_token_rejection(self, middleware):
        """Test that expired tokens are rejected"""
        # Create a token that expires immediately
        expired_token = middleware.secure_authenticator.token_manager.create_token(
            "test_user", [Permission.READ_TOOLS], expires_hours=0
        )
        
        # Wait for expiration
        await asyncio.sleep(0.1)
        
        headers = {"authorization": f"Bearer {expired_token}"}
        
        with pytest.raises(AuthenticationError, match="Authentication expired"):
            await middleware.process_request("tools/list", {}, headers)
    
    @pytest.mark.asyncio
    async def test_invalid_token_rejection(self, middleware):
        """Test that invalid tokens are rejected"""
        # Invalid token format
        headers = {"authorization": "Bearer invalid_token_format"}
        
        with pytest.raises(AuthenticationError, match="Authentication required"):
            await middleware.process_request("tools/list", {}, headers)
        
        # Token with wrong signature
        fake_token = jwt.encode({"sub": "hacker"}, "wrong_secret", algorithm="HS256")
        headers = {"authorization": f"Bearer {fake_token}"}
        
        with pytest.raises(AuthenticationError, match="Authentication required"):
            await middleware.process_request("tools/list", {}, headers)
    
    def test_secure_secret_requirement(self):
        """Test that secure middleware requires strong JWT secret"""
        # Should fail with weak secret
        with pytest.raises(ValueError, match="JWT secret must be at least 32 characters"):
            create_secure_middleware("weak_secret")
        
        # Should succeed with strong secret
        strong_secret = secrets.token_urlsafe(32)
        middleware = create_secure_middleware(strong_secret)
        assert middleware is not None


class TestSecurityIntegration:
    """Integration tests for complete security system"""
    
    @pytest.fixture
    def full_system(self):
        """Create complete security system"""
        jwt_secret = secrets.token_urlsafe(32)
        middleware = create_secure_middleware(jwt_secret, enable_strict_mode=True)
        return middleware
    
    @pytest.mark.asyncio
    async def test_complete_request_pipeline(self, full_system):
        """Test complete request processing pipeline"""
        # Create admin token
        admin_token = full_system.create_admin_token("admin_user")
        
        # Process valid request
        headers = {"authorization": f"Bearer {admin_token}"}
        params = {"symbol": "BTC", "action": "get_price"}
        
        processed_params, context = await full_system.process_request(
            "tools/call", params, headers
        )
        
        # Verify processing
        assert processed_params["symbol"] == "BTC"
        assert processed_params["action"] == "get_price"
        assert context.user_id == "admin_user"
        assert len(context.security_threats) == 0
    
    @pytest.mark.asyncio
    async def test_security_threat_detection(self, full_system):
        """Test that security threats are properly detected and logged"""
        admin_token = full_system.create_admin_token("admin_user")
        headers = {"authorization": f"Bearer {admin_token}"}
        
        # Test multiple threat types
        threat_params = [
            {"malicious_path": "../../../etc/passwd"},  # Path traversal
            {"script": "<script>alert('xss')</script>"},  # XSS attempt  
            {"overflow": 10**20},  # Numeric overflow
        ]
        
        for params in threat_params:
            with pytest.raises(ValidationError):
                await full_system.process_request("tools/call", params, headers)
    
    @pytest.mark.asyncio
    async def test_no_authentication_bypass(self, full_system):
        """Test that there are no ways to bypass authentication"""
        # Try various bypass attempts
        bypass_attempts = [
            {},  # Empty headers
            {"authorization": ""},  # Empty auth
            {"authorization": "Basic fake"},  # Wrong auth type
            {"authorization": "Bearer "},  # Empty bearer
            {"x-admin": "true"},  # Fake admin header
            {"x-bypass": "please"},  # Fake bypass header
            {"authorization": "Bearer null"},  # Null token
            {"authorization": "Bearer undefined"},  # Undefined token
        ]
        
        for headers in bypass_attempts:
            with pytest.raises(AuthenticationError):
                await full_system.process_request("tools/call", {"test": "value"}, headers)
    
    def test_metrics_and_logging(self, full_system):
        """Test security metrics collection"""
        metrics = full_system.metrics.get_metrics()
        
        # Verify metrics structure
        assert "total_requests" in metrics
        assert "authenticated_requests" in metrics
        assert "security_threats" in metrics
        assert "uptime_seconds" in metrics
        
        # Verify initial state
        assert metrics["total_requests"] >= 0
        assert metrics["security_threats"] >= 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])