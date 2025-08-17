"""
Secure Authentication System for MCP Server
Fixes critical authentication bypass vulnerabilities
"""

import asyncio
import logging
import jwt
import hashlib
import secrets
import time
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import bcrypt
from cryptography.fernet import Fernet

from .token_storage import get_token_storage, hash_token

logger = logging.getLogger(__name__)

class AuthenticationMethod(str, Enum):
    """Supported authentication methods"""
    JWT_BEARER = "jwt_bearer"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    MUTUAL_TLS = "mutual_tls"

class Permission(str, Enum):
    """Granular permissions"""
    READ_TOOLS = "read:tools"
    EXECUTE_TOOLS = "execute:tools"
    READ_RESOURCES = "read:resources"
    WRITE_RESOURCES = "write:resources"
    ADMIN_SERVER = "admin:server"
    METRICS_READ = "metrics:read"
    HEALTH_CHECK = "health:check"

@dataclass
class AuthenticationContext:
    """Authentication context with strict validation"""
    user_id: str
    session_id: str
    method: AuthenticationMethod
    permissions: Set[Permission]
    metadata: Dict[str, Any] = field(default_factory=dict)
    authenticated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    rate_limit_key: str = ""
    
    def __post_init__(self):
        if not self.rate_limit_key:
            self.rate_limit_key = f"{self.user_id}:{self.ip_address or 'unknown'}"
            
    def is_expired(self) -> bool:
        """Check if authentication has expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
        
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions
        
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        return any(perm in self.permissions for perm in permissions)

class SecureValidator:
    """Secure input validation to prevent attacks"""
    
    @staticmethod
    def validate_path(path: str) -> bool:
        """Validate file paths against traversal attacks"""
        if not path:
            return False
            
        # Normalize path
        import os
        normalized = os.path.normpath(path)
        
        # Check for dangerous patterns
        dangerous_patterns = [
            "..", "~", "/etc/", "/proc/", "/sys/", "/dev/",
            "\\\\", "%2e%2e", "%2f", "%5c", "\\x2e\\x2e",
            "file://", "http://", "https://", "ftp://"
        ]
        
        path_lower = path.lower()
        for pattern in dangerous_patterns:
            if pattern in path_lower:
                return False
                
        # Must be relative and not start with /
        if normalized.startswith('/') or normalized.startswith('\\'):
            return False
            
        return True
        
    @staticmethod
    def validate_size_limits(data: Any, max_depth: int = 10, max_size: int = 1024*1024) -> bool:
        """Validate object size and depth limits"""
        try:
            import json
            serialized = json.dumps(data)
            
            # Check size limit
            if len(serialized) > max_size:
                return False
                
            # Check nesting depth
            def check_depth(obj, current_depth=0):
                if current_depth > max_depth:
                    return False
                    
                if isinstance(obj, dict):
                    return all(check_depth(v, current_depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, current_depth + 1) for item in obj)
                return True
                
            return check_depth(data)
            
        except Exception:
            return False
            
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return ""
            
        # Truncate
        if len(value) > max_length:
            value = value[:max_length]
            
        # Remove null bytes and control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
        
        # Basic XSS prevention
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
        for char in dangerous_chars:
            value = value.replace(char, '')
            
        return value.strip()

class RateLimiter:
    """Per-user rate limiting with sliding window"""
    
    def __init__(self, window_seconds: int = 60, max_requests: int = 100):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
        
    async def check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit"""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Initialize or clean old requests
            if key not in self._requests:
                self._requests[key] = []
            else:
                self._requests[key] = [
                    req_time for req_time in self._requests[key] 
                    if req_time > window_start
                ]
                
            # Check limit
            if len(self._requests[key]) >= self.max_requests:
                logger.warning(f"Rate limit exceeded for {key}")
                return False
                
            # Add current request
            self._requests[key].append(now)
            return True
            
    async def get_remaining_quota(self, key: str) -> int:
        """Get remaining requests in current window"""
        async with self._lock:
            if key not in self._requests:
                return self.max_requests
                
            now = time.time()
            window_start = now - self.window_seconds
            current_requests = [
                req_time for req_time in self._requests[key] 
                if req_time > window_start
            ]
            
            return max(0, self.max_requests - len(current_requests))

class SecureTokenManager:
    """Secure JWT token management with persistent revocation storage"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        if len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self._encryption = Fernet(Fernet.generate_key()) if len(secret_key) >= 32 else None
        # Remove in-memory revocation storage - using persistent storage now
        
    def create_token(self, user_id: str, permissions: List[Permission], 
                    expires_hours: int = 24, metadata: Dict[str, Any] = None) -> str:
        """Create secure JWT token"""
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "iat": now.timestamp(),
            "exp": (now + timedelta(hours=expires_hours)).timestamp(),
            "jti": secrets.token_urlsafe(16),  # Unique token ID
            "permissions": [perm.value for perm in permissions],
            "metadata": metadata or {}
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
        
    async def validate_token(self, token: str) -> Optional[AuthenticationContext]:
        """Validate JWT token securely with persistent revocation check"""
        try:
            # Check if token is revoked using persistent storage
            token_storage = await get_token_storage()
            if await token_storage.is_token_revoked(hash_token(token)):
                logger.warning("Attempt to use revoked token")
                return None
                
            # Decode and validate
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"require": ["sub", "iat", "exp", "jti"]}
            )
            
            # Extract permissions
            permissions = set()
            for perm_str in payload.get("permissions", []):
                try:
                    permissions.add(Permission(perm_str))
                except ValueError:
                    logger.warning(f"Invalid permission in token: {perm_str}")
                    
            # Create authentication context
            expires_at = datetime.fromtimestamp(payload["exp"])
            
            context = AuthenticationContext(
                user_id=payload["sub"],
                session_id=payload["jti"],
                method=AuthenticationMethod.JWT_BEARER,
                permissions=permissions,
                metadata=payload.get("metadata", {}),
                expires_at=expires_at
            )
            
            return context
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token used")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
            
    async def revoke_token(self, token: str):
        """Revoke a token using persistent storage"""
        try:
            # Decode token to get expiration
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Don't verify expiration for revocation
            )
            expires_at = datetime.fromtimestamp(payload.get("exp", 0)) if payload.get("exp") else None
            
            # Store revocation persistently
            token_storage = await get_token_storage()
            await token_storage.revoke_token(hash_token(token), expires_at)
            
            logger.info(f"Token revoked successfully")
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            # Still try to revoke with hash only
            token_storage = await get_token_storage()
            await token_storage.revoke_token(hash_token(token))

class APIKeyManager:
    """Secure API key management"""
    
    def __init__(self):
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        
    def create_api_key(self, user_id: str, permissions: List[Permission], 
                      name: str = "", expires_days: int = 365) -> tuple[str, str]:
        """Create new API key"""
        # Generate secure key
        key_id = secrets.token_urlsafe(16)
        secret = secrets.token_urlsafe(32)
        
        # Hash secret for storage
        hashed_secret = bcrypt.hashpw(secret.encode(), bcrypt.gensalt()).decode()
        
        # Store key info
        expires_at = datetime.utcnow() + timedelta(days=expires_days)
        self._api_keys[key_id] = {
            "user_id": user_id,
            "hashed_secret": hashed_secret,
            "permissions": [perm.value for perm in permissions],
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "last_used": None,
            "usage_count": 0
        }
        
        return key_id, secret
        
    def validate_api_key(self, key_id: str, secret: str) -> Optional[AuthenticationContext]:
        """Validate API key"""
        if key_id not in self._api_keys:
            return None
            
        key_info = self._api_keys[key_id]
        
        # Check expiration
        expires_at = datetime.fromisoformat(key_info["expires_at"])
        if datetime.utcnow() > expires_at:
            logger.warning(f"Expired API key used: {key_id}")
            return None
            
        # Verify secret
        if not bcrypt.checkpw(secret.encode(), key_info["hashed_secret"].encode()):
            logger.warning(f"Invalid secret for API key: {key_id}")
            return None
            
        # Update usage
        key_info["last_used"] = datetime.utcnow().isoformat()
        key_info["usage_count"] += 1
        
        # Create context
        permissions = set()
        for perm_str in key_info["permissions"]:
            try:
                permissions.add(Permission(perm_str))
            except ValueError:
                pass
                
        return AuthenticationContext(
            user_id=key_info["user_id"],
            session_id=key_id,
            method=AuthenticationMethod.API_KEY,
            permissions=permissions,
            metadata={"api_key_name": key_info["name"]}
        )
        
    def revoke_api_key(self, key_id: str):
        """Revoke API key"""
        if key_id in self._api_keys:
            del self._api_keys[key_id]
            logger.info(f"API key revoked: {key_id}")

class SecureAuthenticator:
    """Main authentication coordinator"""
    
    def __init__(self, secret_key: str, enable_strict_mode: bool = True):
        self.strict_mode = enable_strict_mode
        self.validator = SecureValidator()
        self.rate_limiter = RateLimiter()
        self.token_manager = SecureTokenManager(secret_key)
        self.api_key_manager = APIKeyManager()
        
        # Method routing
        self._methods = {
            "bearer": self._authenticate_bearer,
            "api_key": self._authenticate_api_key,
        }
        
    async def authenticate_request(self, headers: Dict[str, str], 
                                 ip_address: Optional[str] = None) -> Optional[AuthenticationContext]:
        """Authenticate incoming request"""
        try:
            # Extract authentication info
            auth_header = headers.get("authorization", "")
            api_key_header = headers.get("x-api-key", "")
            
            context = None
            
            # Try Bearer token
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                context = await self._authenticate_bearer(token)
                
            # Try API key
            elif api_key_header:
                context = await self._authenticate_api_key(api_key_header)
                
            if context:
                # Set additional context
                context.ip_address = ip_address
                context.user_agent = headers.get("user-agent", "")
                
                # Check rate limiting
                if not await self.rate_limiter.check_rate_limit(context.rate_limit_key):
                    logger.warning(f"Rate limit exceeded for user {context.user_id}")
                    return None
                    
                return context
                
            # In strict mode, reject all unauthenticated requests
            if self.strict_mode:
                logger.warning(f"Unauthenticated request from {ip_address}")
                return None
                
            # Fallback to anonymous (very limited permissions)
            return self._create_anonymous_context(ip_address)
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
            
    async def _authenticate_bearer(self, token: str) -> Optional[AuthenticationContext]:
        """Authenticate bearer token"""
        if not token or len(token) < 10:
            return None
        return self.token_manager.validate_token(token)
        
    async def _authenticate_api_key(self, api_key: str) -> Optional[AuthenticationContext]:
        """Authenticate API key"""
        if not api_key or ":" not in api_key:
            return None
            
        try:
            key_id, secret = api_key.split(":", 1)
            return self.api_key_manager.validate_api_key(key_id, secret)
        except ValueError:
            return None
            
    def _create_anonymous_context(self, ip_address: Optional[str]) -> AuthenticationContext:
        """Create severely limited anonymous context"""
        return AuthenticationContext(
            user_id="anonymous",
            session_id=f"anon_{secrets.token_urlsafe(8)}",
            method=AuthenticationMethod.API_KEY,
            permissions={Permission.HEALTH_CHECK},  # Only health checks
            ip_address=ip_address,
            rate_limit_key=f"anonymous:{ip_address or 'unknown'}"
        )
        
    def create_user_token(self, user_id: str, permissions: List[Permission]) -> str:
        """Create token for user"""
        return self.token_manager.create_token(user_id, permissions)
        
    def create_user_api_key(self, user_id: str, permissions: List[Permission], 
                           name: str = "") -> tuple[str, str]:
        """Create API key for user"""
        return self.api_key_manager.create_api_key(user_id, permissions, name)
        
    def revoke_token(self, token: str):
        """Revoke token"""
        self.token_manager.revoke_token(token)
        
    def revoke_api_key(self, key_id: str):
        """Revoke API key"""
        self.api_key_manager.revoke_api_key(key_id)

# Authorization decorator
def require_permissions(*required_permissions: Permission):
    """Decorator to require specific permissions"""
    def decorator(func):
        async def wrapper(self, context: AuthenticationContext, *args, **kwargs):
            if not context:
                raise PermissionError("Authentication required")
                
            if context.is_expired():
                raise PermissionError("Authentication expired")
                
            if not any(context.has_permission(perm) for perm in required_permissions):
                logger.warning(f"Permission denied for user {context.user_id}: requires {required_permissions}")
                raise PermissionError(f"Insufficient permissions: requires one of {required_permissions}")
                
            return await func(self, context, *args, **kwargs)
        return wrapper
    return decorator