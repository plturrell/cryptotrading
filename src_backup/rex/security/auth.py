"""
Production-ready authentication and authorization system
Implements JWT tokens, API keys, and RBAC for the A2A system
"""

import jwt
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)

class Role(Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    AGENT = "agent"
    VIEWER = "viewer"
    SERVICE = "service"

class Permission(Enum):
    """System permissions"""
    REGISTER_AGENT = "register_agent"
    EXECUTE_WORKFLOW = "execute_workflow"
    VIEW_METRICS = "view_metrics"
    MANAGE_USERS = "manage_users"
    SEND_MESSAGE = "send_message"
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [p for p in Permission],
    Role.AGENT: [
        Permission.SEND_MESSAGE, 
        Permission.READ_DATA, 
        Permission.WRITE_DATA,
        Permission.EXECUTE_WORKFLOW
    ],
    Role.SERVICE: [
        Permission.REGISTER_AGENT,
        Permission.SEND_MESSAGE,
        Permission.READ_DATA,
        Permission.EXECUTE_WORKFLOW
    ],
    Role.VIEWER: [Permission.VIEW_METRICS, Permission.READ_DATA]
}

@dataclass
class User:
    """User account"""
    user_id: str
    username: str
    email: str
    roles: List[Role]
    api_key_hash: Optional[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    is_active: bool = True

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow()

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, []):
                return True
        return False

    def get_permissions(self) -> Set[Permission]:
        """Get all permissions for this user"""
        permissions = set()
        for role in self.roles:
            permissions.update(ROLE_PERMISSIONS.get(role, []))
        return permissions

class SecurityConfig:
    """Security configuration"""
    JWT_SECRET_KEY = secrets.token_urlsafe(32)
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    API_KEY_LENGTH = 32
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15

class AuthenticationError(Exception):
    """Authentication failed"""
    pass

class AuthorizationError(Exception):
    """Authorization failed"""
    pass

class RateLimitError(Exception):
    """Rate limit exceeded"""
    pass

class AuthManager:
    """Production authentication and authorization manager"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self.login_attempts: Dict[str, List[datetime]] = {}
        self.blacklisted_tokens: Set[str] = set()
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user for initial setup"""
        admin_user = User(
            user_id="admin-001",
            username="admin",
            email="admin@example.com",
            roles=[Role.ADMIN]
        )
        self.users[admin_user.user_id] = admin_user
        
        # Generate admin API key
        api_key = self.generate_api_key(admin_user.user_id)
        logger.info(f"Default admin created with API key: {api_key}")
    
    def create_user(
        self, 
        username: str, 
        email: str, 
        roles: List[Role],
        created_by: str
    ) -> tuple[User, str]:
        """Create new user with API key"""
        # Verify creator has permission
        creator = self.get_user(created_by)
        if not creator or not creator.has_permission(Permission.MANAGE_USERS):
            raise AuthorizationError("Insufficient permissions to create users")
        
        user_id = f"user-{secrets.token_urlsafe(8)}"
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles
        )
        
        self.users[user_id] = user
        api_key = self.generate_api_key(user_id)
        
        logger.info(f"User {username} created with ID {user_id}")
        return user, api_key
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate secure API key for user"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        api_key = secrets.token_urlsafe(self.config.API_KEY_LENGTH)
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store hash in user record
        self.users[user_id].api_key_hash = api_key_hash
        self.api_keys[api_key] = user_id
        
        logger.info(f"API key generated for user {user_id}")
        return api_key
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        user_id = self.api_keys.get(api_key)
        if not user_id:
            logger.warning(f"Invalid API key attempted")
            return None
        
        user = self.users.get(user_id)
        if not user or not user.is_active:
            logger.warning(f"Authentication failed for inactive user {user_id}")
            return None
        
        # Verify hash matches
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if user.api_key_hash != api_key_hash:
            logger.warning(f"API key hash mismatch for user {user_id}")
            return None
        
        user.last_login = datetime.utcnow()
        logger.info(f"User {user.username} authenticated via API key")
        return user
    
    def create_jwt_token(self, user: User) -> tuple[str, str]:
        """Create JWT access and refresh tokens"""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [p.value for p in user.get_permissions()],
            "iat": now,
            "exp": now + timedelta(minutes=self.config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": user.user_id,
            "iat": now,
            "exp": now + timedelta(days=self.config.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
            "type": "refresh"
        }
        
        access_token = jwt.encode(
            access_payload, 
            self.config.JWT_SECRET_KEY, 
            algorithm=self.config.JWT_ALGORITHM
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.config.JWT_SECRET_KEY,
            algorithm=self.config.JWT_ALGORITHM
        )
        
        logger.info(f"JWT tokens created for user {user.username}")
        return access_token, refresh_token
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        if token in self.blacklisted_tokens:
            raise AuthenticationError("Token has been revoked")
        
        try:
            payload = jwt.decode(
                token,
                self.config.JWT_SECRET_KEY,
                algorithms=[self.config.JWT_ALGORITHM]
            )
            
            # Verify token type and expiration
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    def refresh_jwt_token(self, refresh_token: str) -> tuple[str, str]:
        """Refresh JWT tokens using refresh token"""
        try:
            payload = jwt.decode(
                refresh_token,
                self.config.JWT_SECRET_KEY,
                algorithms=[self.config.JWT_ALGORITHM]
            )
            
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")
            
            user = self.get_user(payload["user_id"])
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Create new tokens
            return self.create_jwt_token(user)
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Refresh token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid refresh token")
    
    def revoke_token(self, token: str):
        """Revoke a JWT token"""
        self.blacklisted_tokens.add(token)
        logger.info("Token revoked")
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_minutes: int = 1) -> bool:
        """Check if identifier exceeds rate limit"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)
        
        # Clean old attempts
        if identifier in self.login_attempts:
            self.login_attempts[identifier] = [
                attempt for attempt in self.login_attempts[identifier]
                if attempt > window_start
            ]
        
        # Check current count
        current_attempts = len(self.login_attempts.get(identifier, []))
        if current_attempts >= max_requests:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Record this attempt
        if identifier not in self.login_attempts:
            self.login_attempts[identifier] = []
        self.login_attempts[identifier].append(now)
        
        return True

# Global auth manager instance
auth_manager = AuthManager()

def require_auth(permission: Permission = None):
    """Decorator to require authentication and optional permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract token from request headers (implementation depends on framework)
            token = kwargs.get('auth_token') or kwargs.get('api_key')
            
            if not token:
                raise AuthenticationError("Authentication required")
            
            user = None
            
            # Try API key first
            if token.startswith('sk-'):
                user = auth_manager.authenticate_api_key(token)
            else:
                # Try JWT
                try:
                    payload = auth_manager.verify_jwt_token(token)
                    user = auth_manager.get_user(payload['user_id'])
                except AuthenticationError:
                    user = None
            
            if not user:
                raise AuthenticationError("Invalid credentials")
            
            # Check permission if required
            if permission and not user.has_permission(permission):
                raise AuthorizationError(f"Missing permission: {permission.value}")
            
            # Add user to kwargs
            kwargs['current_user'] = user
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(role: Role):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')
            if not user or role not in user.roles:
                raise AuthorizationError(f"Role {role.value} required")
            return await func(*args, **kwargs)
        return wrapper
    return decorator