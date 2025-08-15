"""
MCP Authentication Framework
Provides API key and JWT-based authentication for MCP servers and clients
"""
import jwt
import hashlib
import secrets
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuthCredentials:
    """Authentication credentials"""
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    username: Optional[str] = None
    tenant_id: Optional[str] = None


@dataclass
class AuthContext:
    """Authentication context for requests"""
    user_id: str
    tenant_id: str
    permissions: List[str]
    expires_at: datetime
    metadata: Dict[str, Any]


class APIKeyManager:
    """Manages API key generation and validation"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.key_prefix = "mcp_"
    
    def generate_api_key(self, user_id: str, tenant_id: str, 
                        permissions: List[str] = None) -> str:
        """Generate a new API key"""
        permissions = permissions or ["read", "write"]
        
        # Generate secure random key
        key_data = secrets.token_urlsafe(32)
        api_key = f"{self.key_prefix}{key_data}"
        
        # Store key metadata
        self.api_keys[api_key] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "permissions": permissions,
            "created_at": datetime.now(),
            "last_used": None,
            "active": True
        }
        
        logger.info(f"Generated API key for user {user_id}, tenant {tenant_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[AuthContext]:
        """Validate API key and return auth context"""
        if not api_key or api_key not in self.api_keys:
            return None
        
        key_data = self.api_keys[api_key]
        if not key_data.get("active", False):
            return None
        
        # Update last used timestamp
        key_data["last_used"] = datetime.now()
        
        return AuthContext(
            user_id=key_data["user_id"],
            tenant_id=key_data["tenant_id"],
            permissions=key_data["permissions"],
            expires_at=datetime.now() + timedelta(hours=24),  # 24 hour session
            metadata={"auth_method": "api_key"}
        )
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info(f"Revoked API key: {api_key[:10]}...")
            return True
        return False


class JWTManager:
    """Manages JWT token generation and validation"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(64)
        self.algorithm = "HS256"
        self.default_expiry = timedelta(hours=1)
    
    def generate_token(self, user_id: str, tenant_id: str, 
                      permissions: List[str] = None,
                      expires_in: timedelta = None) -> str:
        """Generate a JWT token"""
        permissions = permissions or ["read", "write"]
        expires_in = expires_in or self.default_expiry
        
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "permissions": permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + expires_in,
            "iss": "mcp-server",
            "aud": "mcp-client"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Generated JWT token for user {user_id}, tenant {tenant_id}")
        return token
    
    def validate_token(self, token: str) -> Optional[AuthContext]:
        """Validate JWT token and return auth context"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                audience="mcp-client",
                issuer="mcp-server"
            )
            
            return AuthContext(
                user_id=payload["user_id"],
                tenant_id=payload["tenant_id"],
                permissions=payload["permissions"],
                expires_at=datetime.fromtimestamp(payload["exp"]),
                metadata={"auth_method": "jwt", "token_payload": payload}
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None


class MCPAuthenticator:
    """Main authentication manager for MCP"""
    
    def __init__(self, secret_key: str = None):
        self.api_key_manager = APIKeyManager()
        self.jwt_manager = JWTManager(secret_key)
        self.auth_cache: Dict[str, AuthContext] = {}
        self.cache_ttl = 300  # 5 minutes
    
    def authenticate(self, credentials: AuthCredentials) -> Optional[AuthContext]:
        """Authenticate using provided credentials"""
        # Try API key authentication
        if credentials.api_key:
            auth_context = self.api_key_manager.validate_api_key(credentials.api_key)
            if auth_context:
                self._cache_auth_context(credentials.api_key, auth_context)
                return auth_context
        
        # Try JWT authentication
        if credentials.jwt_token:
            auth_context = self.jwt_manager.validate_token(credentials.jwt_token)
            if auth_context:
                self._cache_auth_context(credentials.jwt_token, auth_context)
                return auth_context
        
        logger.warning("Authentication failed for provided credentials")
        return None
    
    def check_permission(self, auth_context: AuthContext, 
                        required_permission: str) -> bool:
        """Check if auth context has required permission"""
        if not auth_context or not auth_context.permissions:
            return False
        
        # Check for specific permission or admin access
        return (required_permission in auth_context.permissions or 
                "admin" in auth_context.permissions)
    
    def check_tenant_access(self, auth_context: AuthContext, 
                           tenant_id: str) -> bool:
        """Check if auth context has access to tenant"""
        if not auth_context:
            return False
        
        # Admin users can access all tenants
        if "admin" in auth_context.permissions:
            return True
        
        return auth_context.tenant_id == tenant_id
    
    def generate_api_key(self, user_id: str, tenant_id: str, 
                        permissions: List[str] = None) -> str:
        """Generate new API key"""
        return self.api_key_manager.generate_api_key(user_id, tenant_id, permissions)
    
    def generate_jwt_token(self, user_id: str, tenant_id: str, 
                          permissions: List[str] = None) -> str:
        """Generate new JWT token"""
        return self.jwt_manager.generate_token(user_id, tenant_id, permissions)
    
    def revoke_credentials(self, credentials: AuthCredentials) -> bool:
        """Revoke credentials"""
        success = False
        
        if credentials.api_key:
            success = self.api_key_manager.revoke_api_key(credentials.api_key)
            self._remove_from_cache(credentials.api_key)
        
        return success
    
    def _cache_auth_context(self, key: str, auth_context: AuthContext):
        """Cache authentication context"""
        cache_key = hashlib.sha256(key.encode()).hexdigest()
        self.auth_cache[cache_key] = auth_context
    
    def _remove_from_cache(self, key: str):
        """Remove authentication context from cache"""
        cache_key = hashlib.sha256(key.encode()).hexdigest()
        self.auth_cache.pop(cache_key, None)
    
    def cleanup_expired_cache(self):
        """Clean up expired authentication cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, context in self.auth_cache.items()
            if context.expires_at < now
        ]
        
        for key in expired_keys:
            del self.auth_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired auth cache entries")


class AuthMiddleware:
    """Authentication middleware for MCP requests"""
    
    def __init__(self, authenticator: MCPAuthenticator):
        self.authenticator = authenticator
        self.public_methods = ["initialize", "initialized", "ping"]
    
    def authenticate_request(self, method: str, headers: Dict[str, str] = None) -> Optional[AuthContext]:
        """Authenticate MCP request"""
        # Allow public methods without authentication
        if method in self.public_methods:
            return AuthContext(
                user_id="anonymous",
                tenant_id="public",
                permissions=["read"],
                expires_at=datetime.now() + timedelta(hours=1),
                metadata={"auth_method": "public"}
            )
        
        if not headers:
            return None
        
        # Extract credentials from headers
        credentials = AuthCredentials()
        
        # Check for API key in Authorization header
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token.startswith("mcp_"):
                credentials.api_key = token
            else:
                credentials.jwt_token = token
        
        # Check for API key in X-API-Key header
        if "X-API-Key" in headers:
            credentials.api_key = headers["X-API-Key"]
        
        return self.authenticator.authenticate(credentials)
    
    def require_permission(self, auth_context: AuthContext, 
                          permission: str) -> bool:
        """Check if request has required permission"""
        return self.authenticator.check_permission(auth_context, permission)
    
    def require_tenant_access(self, auth_context: AuthContext, 
                             tenant_id: str) -> bool:
        """Check if request has access to tenant"""
        return self.authenticator.check_tenant_access(auth_context, tenant_id)
