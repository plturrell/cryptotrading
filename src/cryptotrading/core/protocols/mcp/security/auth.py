"""
JWT Authentication and API Key Management for MCP

This module provides authentication and authorization middleware for MCP servers
running on Vercel architecture. Supports JWT tokens and API key authentication.

Features:
- JWT token validation with configurable expiration
- API key management with scopes and rate limits
- Request signing and validation
- Role-based access control
- Integration with Vercel environment variables
"""

import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    jwt = None
    JWT_AVAILABLE = False

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication failed"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AuthorizationError(Exception):
    """Authorization failed"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TokenScope(Enum):
    """Token scopes for different operations"""

    READ_TOOLS = "tools:read"
    CALL_TOOLS = "tools:call"
    READ_RESOURCES = "resources:read"
    ADMIN = "admin"
    ALL = "*"


@dataclass
class AuthToken:
    """Authentication token data"""

    user_id: str
    scopes: List[str]
    issued_at: datetime
    expires_at: Optional[datetime] = None
    api_key_id: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if token is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def has_scope(self, required_scope: str) -> bool:
        """Check if token has required scope"""
        if TokenScope.ALL.value in self.scopes:
            return True
        return required_scope in self.scopes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT payload"""
        return {
            "user_id": self.user_id,
            "scopes": self.scopes,
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self.expires_at.timestamp()) if self.expires_at else None,
            "api_key_id": self.api_key_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthToken":
        """Create from dictionary (JWT payload)"""
        return cls(
            user_id=data["user_id"],
            scopes=data["scopes"],
            issued_at=datetime.fromtimestamp(data["iat"]),
            expires_at=datetime.fromtimestamp(data["exp"]) if data.get("exp") else None,
            api_key_id=data.get("api_key_id"),
        )


@dataclass
class APIKey:
    """API key configuration"""

    key_id: str
    key_hash: str  # SHA-256 hash of the actual key
    name: str
    scopes: List[str]
    rate_limit: int  # requests per minute
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0

    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def has_scope(self, required_scope: str) -> bool:
        """Check if API key has required scope"""
        if TokenScope.ALL.value in self.scopes:
            return True
        return required_scope in self.scopes

    def verify_key(self, provided_key: str) -> bool:
        """Verify provided key against stored hash"""
        provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        return hmac.compare_digest(self.key_hash, provided_hash)


class JWTManager:
    """JWT token management"""

    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        """
        Initialize JWT manager

        Args:
            secret_key: Secret key for signing (from VERCEL_JWT_SECRET env var if not provided)
            algorithm: JWT signing algorithm
        """
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT library not installed. Install with: pip install PyJWT")

        self.secret_key = secret_key or os.getenv("VERCEL_JWT_SECRET")
        if not self.secret_key:
            raise ValueError(
                "JWT secret key not provided. Set VERCEL_JWT_SECRET environment variable."
            )

        self.algorithm = algorithm
        self.default_expiry = timedelta(hours=24)

    def create_token(
        self,
        user_id: str,
        scopes: List[str],
        expires_in: Optional[timedelta] = None,
        api_key_id: Optional[str] = None,
    ) -> str:
        """
        Create a new JWT token

        Args:
            user_id: User identifier
            scopes: List of token scopes
            expires_in: Token expiration time (default 24h)
            api_key_id: Associated API key ID

        Returns:
            Encoded JWT token string
        """
        now = datetime.utcnow()
        expires_at = now + (expires_in or self.default_expiry)

        token = AuthToken(
            user_id=user_id,
            scopes=scopes,
            issued_at=now,
            expires_at=expires_at,
            api_key_id=api_key_id,
        )

        payload = token.to_dict()
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> AuthToken:
        """
        Verify and decode JWT token

        Args:
            token: JWT token string

        Returns:
            Decoded AuthToken

        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            auth_token = AuthToken.from_dict(payload)

            if auth_token.is_expired():
                raise AuthenticationError("Token has expired")

            return auth_token

        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("Token has expired") from exc
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}") from e

    def refresh_token(self, token: str, extends_by: Optional[timedelta] = None) -> str:
        """
        Refresh an existing token

        Args:
            token: Current JWT token
            extends_by: How much to extend expiry (default 24h)

        Returns:
            New JWT token with extended expiry
        """
        auth_token = self.verify_token(token)

        # Create new token with same data but extended expiry
        return self.create_token(
            user_id=auth_token.user_id,
            scopes=auth_token.scopes,
            expires_in=extends_by or self.default_expiry,
            api_key_id=auth_token.api_key_id,
        )


class APIKeyManager:
    """API key management system"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize API key manager

        Args:
            storage_path: Path to store API keys (defaults to VERCEL_API_KEYS env var)
        """
        self.storage_path = storage_path or os.getenv("VERCEL_API_KEYS", "/tmp/api_keys.json")
        self.keys: Dict[str, APIKey] = {}
        self.usage_tracking: Dict[str, List[datetime]] = {}
        self._load_keys()

    def _load_keys(self):
        """Load API keys from storage"""
        try:
            import json
            from pathlib import Path

            if Path(self.storage_path).exists():
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for key_data in data.get("keys", []):
                    key = APIKey(
                        key_id=key_data["key_id"],
                        key_hash=key_data["key_hash"],
                        name=key_data["name"],
                        scopes=key_data["scopes"],
                        rate_limit=key_data["rate_limit"],
                        created_at=datetime.fromisoformat(key_data["created_at"]),
                        expires_at=datetime.fromisoformat(key_data["expires_at"])
                        if key_data.get("expires_at")
                        else None,
                        last_used=datetime.fromisoformat(key_data["last_used"])
                        if key_data.get("last_used")
                        else None,
                        usage_count=key_data.get("usage_count", 0),
                    )
                    self.keys[key.key_id] = key

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load API keys: %s", str(e))

    def _save_keys(self):
        """Save API keys to storage"""
        try:
            import json
            from pathlib import Path

            # Ensure directory exists
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

            data = {
                "keys": [
                    {
                        "key_id": key.key_id,
                        "key_hash": key.key_hash,
                        "name": key.name,
                        "scopes": key.scopes,
                        "rate_limit": key.rate_limit,
                        "created_at": key.created_at.isoformat(),
                        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                        "last_used": key.last_used.isoformat() if key.last_used else None,
                        "usage_count": key.usage_count,
                    }
                    for key in self.keys.values()
                ]
            }

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except (OSError, json.JSONEncodeError) as e:
            logger.error("Failed to save API keys: %s", str(e))

    def create_api_key(
        self,
        name: str,
        scopes: List[str],
        rate_limit: int = 100,
        expires_in: Optional[timedelta] = None,
    ) -> Tuple[str, str]:
        """
        Create a new API key

        Args:
            name: Human-readable name for the key
            scopes: List of scopes this key can access
            rate_limit: Requests per minute limit
            expires_in: Key expiration time

        Returns:
            Tuple of (key_id, actual_key)
        """
        import secrets

        # Generate key ID and actual key
        key_id = f"mcp_{secrets.token_hex(8)}"
        actual_key = f"mcp_key_{secrets.token_urlsafe(32)}"

        # Hash the key for storage
        key_hash = hashlib.sha256(actual_key.encode()).hexdigest()

        # Create API key object
        now = datetime.utcnow()
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes,
            rate_limit=rate_limit,
            created_at=now,
            expires_at=now + expires_in if expires_in else None,
        )

        # Store key
        self.keys[key_id] = api_key
        self.usage_tracking[key_id] = []
        self._save_keys()

        logger.info("Created API key '%s' with ID %s", name, key_id)
        return key_id, actual_key

    def verify_api_key(self, provided_key: str) -> Optional[APIKey]:
        """
        Verify an API key

        Args:
            provided_key: The API key to verify

        Returns:
            APIKey object if valid, None otherwise
        """
        # Check all keys (could optimize with index)
        for api_key in self.keys.values():
            if api_key.verify_key(provided_key):
                if api_key.is_expired():
                    logger.warning("API key %s is expired", api_key.key_id)
                    return None

                # Update usage
                api_key.last_used = datetime.utcnow()
                api_key.usage_count += 1
                self._save_keys()

                return api_key

        return None

    def check_rate_limit(self, key_id: str) -> bool:
        """
        Check if API key is within rate limit

        Args:
            key_id: API key ID

        Returns:
            True if within limit, False if exceeded
        """
        if key_id not in self.keys:
            return False

        api_key = self.keys[key_id]
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Initialize usage tracking if needed
        if key_id not in self.usage_tracking:
            self.usage_tracking[key_id] = []

        # Remove old usage records
        self.usage_tracking[key_id] = [
            usage_time for usage_time in self.usage_tracking[key_id] if usage_time > minute_ago
        ]

        # Check if under limit
        if len(self.usage_tracking[key_id]) >= api_key.rate_limit:
            return False

        # Record this usage
        self.usage_tracking[key_id].append(now)
        return True

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key

        Args:
            key_id: API key ID to revoke

        Returns:
            True if revoked, False if not found
        """
        if key_id in self.keys:
            del self.keys[key_id]
            if key_id in self.usage_tracking:
                del self.usage_tracking[key_id]
            self._save_keys()
            logger.info("Revoked API key %s", key_id)
            return True
        return False

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys (without sensitive data)

        Returns:
            List of API key metadata
        """
        return [
            {
                "key_id": key.key_id,
                "name": key.name,
                "scopes": key.scopes,
                "rate_limit": key.rate_limit,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "usage_count": key.usage_count,
                "is_expired": key.is_expired(),
            }
            for key in self.keys.values()
        ]


class AuthenticationMiddleware:
    """Authentication middleware for MCP servers"""

    def __init__(self, jwt_manager: JWTManager, api_key_manager: APIKeyManager):
        """
        Initialize authentication middleware

        Args:
            jwt_manager: JWT token manager
            api_key_manager: API key manager
        """
        self.jwt_manager = jwt_manager
        self.api_key_manager = api_key_manager

    async def authenticate_request(self, headers: Dict[str, str]) -> AuthToken:
        """
        Authenticate a request using headers

        Args:
            headers: Request headers

        Returns:
            AuthToken if authenticated

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check Authorization header (case insensitive)
        auth_header = headers.get("authorization", "") or headers.get("Authorization", "")
        auth_header = auth_header.strip()

        if auth_header.startswith("Bearer "):
            # JWT token authentication
            token = auth_header[7:]  # Remove "Bearer "
            try:
                return self.jwt_manager.verify_token(token)
            except AuthenticationError as exc:
                raise AuthenticationError("Token verification failed") from exc

        elif auth_header.startswith("ApiKey "):
            # API key authentication
            api_key = auth_header[7:]  # Remove "ApiKey "
            verified_key = self.api_key_manager.verify_api_key(api_key)

            if not verified_key:
                raise AuthenticationError("Invalid API key")

            # Check rate limit
            if not self.api_key_manager.check_rate_limit(verified_key.key_id):
                raise AuthenticationError("Rate limit exceeded")

            # Convert API key to AuthToken
            return AuthToken(
                user_id=verified_key.key_id,
                scopes=verified_key.scopes,
                issued_at=verified_key.created_at,
                expires_at=verified_key.expires_at,
                api_key_id=verified_key.key_id,
            )

        # Check for X-API-Key header (case insensitive)
        api_key = headers.get("x-api-key") or headers.get("X-API-Key")
        if api_key:
            verified_key = self.api_key_manager.verify_api_key(api_key)

            if not verified_key:
                raise AuthenticationError("Invalid API key")

            if not self.api_key_manager.check_rate_limit(verified_key.key_id):
                raise AuthenticationError("Rate limit exceeded")

            return AuthToken(
                user_id=verified_key.key_id,
                scopes=verified_key.scopes,
                issued_at=verified_key.created_at,
                expires_at=verified_key.expires_at,
                api_key_id=verified_key.key_id,
            )

        raise AuthenticationError("No valid authentication credentials provided")

    def check_authorization(self, auth_token: AuthToken, required_scope: str) -> bool:
        """
        Check if token has required authorization

        Args:
            auth_token: Authenticated token
            required_scope: Required scope for operation

        Returns:
            True if authorized

        Raises:
            AuthorizationError: If not authorized
        """
        if not auth_token.has_scope(required_scope):
            raise AuthorizationError(f"Insufficient permissions. Required scope: {required_scope}")

        return True


# Convenience functions for Vercel environment
def create_jwt_manager() -> JWTManager:
    """Create JWT manager with Vercel environment variables"""
    return JWTManager()


def create_api_key_manager() -> APIKeyManager:
    """Create API key manager with Vercel environment variables"""
    return APIKeyManager()


def create_auth_middleware() -> AuthenticationMiddleware:
    """Create authentication middleware with default managers"""
    jwt_manager = create_jwt_manager()
    api_key_manager = create_api_key_manager()
    return AuthenticationMiddleware(jwt_manager, api_key_manager)
