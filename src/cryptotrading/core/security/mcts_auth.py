"""
Enhanced Security for MCTS Agent
Implements authentication, authorization, and secure communication
"""
import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import jwt

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permission levels for MCTS operations"""

    READ = "read"
    CALCULATE = "calculate"
    OPTIMIZE = "optimize"
    ADMIN = "admin"


class SecurityLevel(Enum):
    """Security levels for different environments"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class UserSession:
    """User session information"""

    user_id: str
    username: str
    permissions: List[Permission]
    expires_at: datetime
    session_token: str
    rate_limit: int = 100  # requests per hour
    current_usage: int = 0
    last_request: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions or Permission.ADMIN in self.permissions

    def check_rate_limit(self) -> bool:
        """Check if user is within rate limits"""
        now = datetime.utcnow()

        # Reset counter if hour has passed
        if now - self.last_request > timedelta(hours=1):
            self.current_usage = 0

        self.last_request = now
        self.current_usage += 1

        return self.current_usage <= self.rate_limit


@dataclass
class APIKey:
    """API key for service-to-service authentication"""

    key_id: str
    key_hash: str
    permissions: List[Permission]
    expires_at: Optional[datetime] = None
    rate_limit: int = 1000  # higher limit for API keys
    current_usage: int = 0
    last_used: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    created_by: str = ""

    def is_expired(self) -> bool:
        return self.expires_at is not None and datetime.utcnow() > self.expires_at

    def verify_key(self, provided_key: str) -> bool:
        """Verify the provided key against stored hash"""
        provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        return hmac.compare_digest(self.key_hash, provided_hash)


class SecurityManager:
    """Manages authentication and authorization for MCTS agent"""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.DEVELOPMENT):
        self.security_level = security_level
        self.jwt_secret = os.getenv("MCTS_JWT_SECRET") or self._generate_secret()
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24

        # Session and API key storage
        self.active_sessions: Dict[str, UserSession] = {}
        self.api_keys: Dict[str, APIKey] = {}

        # Security monitoring
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.blocked_ips: Dict[str, datetime] = {}

        # Load default API keys for development
        if security_level == SecurityLevel.DEVELOPMENT:
            self._setup_development_keys()

        logger.info(f"Security manager initialized with {security_level.value} level")

    def _generate_secret(self) -> str:
        """Generate a secure random secret"""
        return secrets.token_hex(32)

    def _setup_development_keys(self):
        """Setup development API keys"""
        dev_key = "dev_mcts_key_" + secrets.token_hex(16)
        dev_key_hash = hashlib.sha256(dev_key.encode()).hexdigest()

        self.api_keys["dev_key_001"] = APIKey(
            key_id="dev_key_001",
            key_hash=dev_key_hash,
            permissions=[Permission.READ, Permission.CALCULATE, Permission.OPTIMIZE],
            description="Development API key",
            created_by="system",
        )

        logger.info(f"Development API key created: {dev_key}")

    async def authenticate_jwt(self, token: str) -> Optional[UserSession]:
        """Authenticate using JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            user_id = payload.get("user_id")
            session_token = payload.get("session_token")

            if user_id and session_token in self.active_sessions:
                session = self.active_sessions[session_token]

                if not session.is_expired():
                    return session
                else:
                    # Clean up expired session
                    del self.active_sessions[session_token]

            return None

        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None

    async def authenticate_api_key(self, key_id: str, provided_key: str) -> Optional[APIKey]:
        """Authenticate using API key"""
        if key_id not in self.api_keys:
            return None

        api_key = self.api_keys[key_id]

        if api_key.is_expired():
            logger.warning(f"API key {key_id} has expired")
            return None

        if not api_key.verify_key(provided_key):
            logger.warning(f"Invalid API key provided for {key_id}")
            return None

        # Update usage
        api_key.last_used = datetime.utcnow()
        api_key.current_usage += 1

        return api_key

    async def create_user_session(
        self,
        username: str,
        password: str,
        permissions: List[Permission],
        ip_address: str = None,
        user_agent: str = None,
    ) -> Optional[Tuple[str, UserSession]]:
        """Create a new user session"""
        # In production, verify password against secure storage
        if not self._verify_user_credentials(username, password):
            await self._record_failed_attempt(ip_address)
            return None

        # Check if IP is blocked
        if await self._is_ip_blocked(ip_address):
            logger.warning(f"Blocked IP {ip_address} attempted login")
            return None

        # Create session
        user_id = f"user_{hashlib.md5(username.encode()).hexdigest()[:8]}"
        session_token = secrets.token_hex(32)
        expires_at = datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours)

        session = UserSession(
            user_id=user_id,
            username=username,
            permissions=permissions,
            expires_at=expires_at,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.active_sessions[session_token] = session

        # Create JWT token
        jwt_payload = {
            "user_id": user_id,
            "username": username,
            "session_token": session_token,
            "exp": expires_at.timestamp(),
            "iat": time.time(),
        }

        jwt_token = jwt.encode(jwt_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        logger.info(f"Created session for user {username}")
        return jwt_token, session

    def _verify_user_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials - secure implementation"""
        if self.security_level == SecurityLevel.DEVELOPMENT:
            # Development mode - use environment variables for security
            dev_username = os.getenv("MCTS_DEV_USERNAME", "admin")
            dev_password = os.getenv("MCTS_DEV_PASSWORD", "change_me_in_production")

            # Hash password for comparison even in dev mode
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            expected_hash = hashlib.sha256(dev_password.encode()).hexdigest()

            return username == dev_username and password_hash == expected_hash
        else:
            # Production mode - implement secure credential verification
            # This should integrate with your user management system
            logger.error("Production credential verification not implemented")
            return False

    async def _record_failed_attempt(self, ip_address: str):
        """Record failed authentication attempt"""
        if not ip_address:
            return

        now = datetime.utcnow()

        if ip_address not in self.failed_attempts:
            self.failed_attempts[ip_address] = []

        self.failed_attempts[ip_address].append(now)

        # Remove old attempts (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        self.failed_attempts[ip_address] = [
            attempt for attempt in self.failed_attempts[ip_address] if attempt > cutoff
        ]

        # Block IP if too many failed attempts
        if len(self.failed_attempts[ip_address]) >= 5:
            self.blocked_ips[ip_address] = now + timedelta(hours=1)
            logger.warning(f"Blocked IP {ip_address} due to repeated failed attempts")

    async def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        if not ip_address or ip_address not in self.blocked_ips:
            return False

        block_until = self.blocked_ips[ip_address]
        if datetime.utcnow() > block_until:
            # Block expired
            del self.blocked_ips[ip_address]
            return False

        return True

    async def authorize_operation(
        self,
        session_or_key: Union[UserSession, APIKey],
        operation: Permission,
        ip_address: str = None,
    ) -> bool:
        """Authorize operation based on permissions and rate limits"""

        # Check if IP is blocked
        if await self._is_ip_blocked(ip_address):
            return False

        # Check permissions
        if isinstance(session_or_key, UserSession):
            if not session_or_key.has_permission(operation):
                logger.warning(f"User {session_or_key.username} lacks {operation.value} permission")
                return False

            if not session_or_key.check_rate_limit():
                logger.warning(f"User {session_or_key.username} exceeded rate limit")
                return False

        elif isinstance(session_or_key, APIKey):
            if (
                operation not in session_or_key.permissions
                and Permission.ADMIN not in session_or_key.permissions
            ):
                logger.warning(
                    f"API key {session_or_key.key_id} lacks {operation.value} permission"
                )
                return False

            # Check API key rate limit
            now = datetime.utcnow()
            if now - session_or_key.last_used > timedelta(hours=1):
                session_or_key.current_usage = 0

            session_or_key.current_usage += 1
            if session_or_key.current_usage > session_or_key.rate_limit:
                logger.warning(f"API key {session_or_key.key_id} exceeded rate limit")
                return False

        return True

    async def create_api_key(
        self,
        description: str,
        permissions: List[Permission],
        created_by: str,
        expires_in_days: int = None,
    ) -> Tuple[str, str]:
        """Create a new API key"""
        key_id = f"key_{secrets.token_hex(8)}"
        raw_key = f"mcts_{secrets.token_hex(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            permissions=permissions,
            expires_at=expires_at,
            description=description,
            created_by=created_by,
        )

        self.api_keys[key_id] = api_key

        logger.info(f"Created API key {key_id} for {created_by}")
        return key_id, raw_key

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id in self.api_keys:
            del self.api_keys[key_id]
            logger.info(f"Revoked API key {key_id}")
            return True
        return False

    async def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        active_sessions_count = len(
            [s for s in self.active_sessions.values() if not s.is_expired()]
        )
        active_api_keys_count = len([k for k in self.api_keys.values() if not k.is_expired()])

        return {
            "security_level": self.security_level.value,
            "active_sessions": active_sessions_count,
            "active_api_keys": active_api_keys_count,
            "blocked_ips": len(self.blocked_ips),
            "failed_attempts_last_hour": sum(
                len(attempts) for attempts in self.failed_attempts.values()
            ),
            "total_api_keys": len(self.api_keys),
        }


def require_permission(permission: Permission):
    """Decorator to require specific permission for MCTS operations"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get authentication context from self
            if hasattr(self, "_current_auth_context"):
                auth_context = self._current_auth_context
                security_manager = auth_context.get("security_manager")
                session_or_key = auth_context.get("session_or_key")
                ip_address = auth_context.get("ip_address")

                if security_manager and session_or_key:
                    authorized = await security_manager.authorize_operation(
                        session_or_key, permission, ip_address
                    )

                    if not authorized:
                        return {
                            "error": "Insufficient permissions or rate limit exceeded",
                            "required_permission": permission.value,
                            "code": "AUTH_ERROR",
                        }

            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


class SecureMCTSAgent:
    """Mixin class to add security to MCTS agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.security_manager = SecurityManager(
            security_level=SecurityLevel(os.getenv("MCTS_SECURITY_LEVEL", "development"))
        )
        self._current_auth_context = {}

    async def authenticate_request(
        self, auth_header: str, ip_address: str = None
    ) -> Dict[str, Any]:
        """Authenticate incoming request"""
        if not auth_header:
            return {"error": "Missing authentication header", "code": "AUTH_REQUIRED"}

        # Try JWT authentication first
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            session = await self.security_manager.authenticate_jwt(token)

            if session:
                self._current_auth_context = {
                    "security_manager": self.security_manager,
                    "session_or_key": session,
                    "ip_address": ip_address,
                    "auth_type": "jwt",
                }
                return {"success": True, "auth_type": "jwt", "user": session.username}

        # Try API key authentication
        elif auth_header.startswith("ApiKey "):
            parts = auth_header[7:].split(":")
            if len(parts) == 2:
                key_id, provided_key = parts
                api_key = await self.security_manager.authenticate_api_key(key_id, provided_key)

                if api_key:
                    self._current_auth_context = {
                        "security_manager": self.security_manager,
                        "session_or_key": api_key,
                        "ip_address": ip_address,
                        "auth_type": "api_key",
                    }
                    return {"success": True, "auth_type": "api_key", "key_id": key_id}

        return {"error": "Invalid authentication credentials", "code": "AUTH_INVALID"}

    @require_permission(Permission.READ)
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status - requires READ permission"""
        return (
            await super().get_agent_status()
            if hasattr(super(), "get_agent_status")
            else {"status": "active"}
        )

    @require_permission(Permission.CALCULATE)
    async def secure_calculate(self, *args, **kwargs) -> Dict[str, Any]:
        """Secure wrapper for calculate operations"""
        return await self.run_mcts_parallel(*args, **kwargs)
