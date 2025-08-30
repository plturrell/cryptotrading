"""
Enterprise Security Management System
Comprehensive security layer with authentication, authorization, input validation, and audit logging.
"""
import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union

import jwt
from cryptography.fernet import Fernet


class SecurityLevel(Enum):
    """Security access levels"""

    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    TRADER = "trader"
    ADMIN = "admin"
    SYSTEM = "system"


class AuditEventType(Enum):
    """Audit event types"""

    LOGIN = "login"
    LOGOUT = "logout"
    TRADE_EXECUTE = "trade_execute"
    RISK_BREACH = "risk_breach"
    CONFIG_CHANGE = "config_change"
    DATA_ACCESS = "data_access"
    SYSTEM_ERROR = "system_error"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class User:
    """User entity"""

    user_id: str
    username: str
    email: str
    security_level: SecurityLevel
    api_key: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    is_active: bool = True
    permissions: Set[str] = field(default_factory=set)


@dataclass
class AuditEvent:
    """Audit event record"""

    event_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    resource: str
    action: str
    result: str
    ip_address: str
    user_agent: str
    metadata: Dict[str, Any]
    timestamp: datetime


class InputValidator:
    """Comprehensive input validation"""

    # Validation patterns
    PATTERNS = {
        "symbol": re.compile(r"^[A-Z]{2,10}$"),
        "amount": re.compile(r"^\d+(\.\d{1,8})?$"),
        "percentage": re.compile(r"^(0(\.\d+)?|1(\.0+)?)$"),
        "timeframe": re.compile(r"^(1m|5m|15m|30m|1h|4h|12h|1d|1w|1M)$"),
        "order_type": re.compile(r"^(market|limit|stop|stop_limit)$"),
        "side": re.compile(r"^(buy|sell)$"),
        "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        "api_key": re.compile(r"^[a-zA-Z0-9]{32,64}$"),
    }

    # Value ranges
    RANGES = {
        "amount": (0.00000001, 1000000.0),
        "percentage": (0.0, 1.0),
        "timeout": (1, 300),
        "limit": (1, 1000),
        "risk_percentage": (0.001, 0.1),  # 0.1% to 10%
    }

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol"""
        if not isinstance(symbol, str):
            return False
        return bool(InputValidator.PATTERNS["symbol"].match(symbol.upper()))

    @staticmethod
    def validate_amount(amount: Union[str, float, int]) -> bool:
        """Validate trading amount"""
        try:
            amount_float = float(amount)
            min_val, max_val = InputValidator.RANGES["amount"]
            return min_val <= amount_float <= max_val
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_percentage(percentage: Union[str, float, int]) -> bool:
        """Validate percentage value"""
        try:
            pct_float = float(percentage)
            min_val, max_val = InputValidator.RANGES["percentage"]
            return min_val <= pct_float <= max_val
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate timeframe string"""
        if not isinstance(timeframe, str):
            return False
        return bool(InputValidator.PATTERNS["timeframe"].match(timeframe))

    @staticmethod
    def validate_order_type(order_type: str) -> bool:
        """Validate order type"""
        if not isinstance(order_type, str):
            return False
        return bool(InputValidator.PATTERNS["order_type"].match(order_type.lower()))

    @staticmethod
    def validate_side(side: str) -> bool:
        """Validate order side"""
        if not isinstance(side, str):
            return False
        return bool(InputValidator.PATTERNS["side"].match(side.lower()))

    @staticmethod
    def sanitize_string(value: str, max_length: int = 255) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return ""

        # Remove potential injection characters
        sanitized = re.sub(r'[<>"\'\x00-\x1f\x7f-\x9f]', "", value)

        # Limit length
        return sanitized[:max_length].strip()

    @staticmethod
    def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters based on tool requirements"""
        validated = {}
        errors = []

        # Tool-specific validation rules
        validation_rules = {
            "get_market_data": {
                "symbol": (InputValidator.validate_symbol, True),
                "timeframe": (InputValidator.validate_timeframe, False),
                "limit": (lambda x: isinstance(x, int) and 1 <= x <= 1000, False),
            },
            "execute_trade": {
                "symbol": (InputValidator.validate_symbol, True),
                "side": (InputValidator.validate_side, True),
                "amount": (InputValidator.validate_amount, True),
                "order_type": (InputValidator.validate_order_type, False),
            },
            "get_risk_metrics": {
                "scope": (lambda x: x in ["portfolio", "symbol", "strategy"], False),
                "symbols": (
                    lambda x: isinstance(x, list)
                    and all(InputValidator.validate_symbol(s) for s in x),
                    False,
                ),
            },
        }

        rules = validation_rules.get(tool_name, {})

        for param_name, value in parameters.items():
            if param_name in rules:
                validator, required = rules[param_name]
                if not validator(value):
                    errors.append(f"Invalid value for parameter '{param_name}': {value}")
                else:
                    validated[param_name] = value
            else:
                # Generic sanitization for unknown parameters
                if isinstance(value, str):
                    validated[param_name] = InputValidator.sanitize_string(value)
                elif isinstance(value, (int, float, bool, list, dict)):
                    validated[param_name] = value

        # Check required parameters
        for param_name, (validator, required) in rules.items():
            if required and param_name not in parameters:
                errors.append(f"Required parameter '{param_name}' is missing")

        if errors:
            raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")

        return validated


class RateLimiter:
    """Advanced rate limiting with multiple strategies"""

    def __init__(self):
        self.request_counts: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}

    def is_allowed(self, identifier: str, limit: int = 100, window: int = 60) -> bool:
        """Check if request is allowed under rate limit"""
        current_time = time.time()

        # Check if IP is blocked
        if identifier in self.blocked_ips:
            if current_time < self.blocked_ips[identifier]:
                return False
            else:
                del self.blocked_ips[identifier]

        # Initialize request history for new identifiers
        if identifier not in self.request_counts:
            self.request_counts[identifier] = []

        # Clean old requests outside the window
        self.request_counts[identifier] = [
            req_time
            for req_time in self.request_counts[identifier]
            if current_time - req_time < window
        ]

        # Check rate limit
        if len(self.request_counts[identifier]) >= limit:
            # Block IP for progressive time
            block_duration = min(300, 60 * (len(self.request_counts[identifier]) - limit + 1))
            self.blocked_ips[identifier] = current_time + block_duration
            return False

        # Record current request
        self.request_counts[identifier].append(current_time)
        return True


class SecurityManager:
    """
    Enterprise security management system

    Features:
    - JWT-based authentication
    - Role-based access control
    - Input validation and sanitization
    - Rate limiting and IP blocking
    - Comprehensive audit logging
    - Security event monitoring
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("SecurityManager")

        # Initialize components
        self.rate_limiter = RateLimiter()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, User] = {}
        self.audit_events: List[AuditEvent] = []

        # Security settings
        self.jwt_secret = config.security.jwt_secret
        self.jwt_algorithm = "HS256"
        self.session_timeout = timedelta(hours=config.security.jwt_expiry_hours)
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes

        # Initialize admin user if not exists
        self._initialize_admin_user()

        self.logger.info("Security manager initialized")

    def _initialize_admin_user(self):
        """Initialize default admin user"""
        admin_id = "admin-system"
        if admin_id not in self.users:
            admin_user = User(
                user_id=admin_id,
                username="admin",
                email="admin@cryptotrading.local",
                security_level=SecurityLevel.ADMIN,
                api_key=secrets.token_urlsafe(32),
                created_at=datetime.utcnow(),
                permissions={"*"},  # All permissions
            )
            self.users[admin_id] = admin_user
            self.logger.info(f"Created admin user with API key: {admin_user.api_key}")

    def authenticate_user(self, api_key: str, ip_address: str) -> Optional[User]:
        """Authenticate user by API key"""
        # Rate limiting
        if not self.rate_limiter.is_allowed(ip_address, 10, 60):  # 10 requests per minute
            self.audit_log(
                AuditEventType.SECURITY_VIOLATION,
                None,
                "authentication",
                "rate_limit_exceeded",
                "blocked",
                ip_address,
                "",
                {"reason": "rate_limit_exceeded"},
            )
            return None

        # Find user by API key
        user = None
        for u in self.users.values():
            if u.api_key == api_key and u.is_active:
                user = u
                break

        if user:
            user.last_login = datetime.utcnow()
            user.failed_login_attempts = 0

            self.audit_log(
                AuditEventType.LOGIN,
                user.user_id,
                "authentication",
                "login",
                "success",
                ip_address,
                "",
                {"username": user.username},
            )

            return user
        else:
            self.audit_log(
                AuditEventType.SECURITY_VIOLATION,
                None,
                "authentication",
                "login",
                "failed",
                ip_address,
                "",
                {"reason": "invalid_api_key"},
            )
            return None

    def create_jwt_token(self, user: User) -> str:
        """Create JWT token for authenticated user"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "security_level": user.security_level.value,
            "permissions": list(user.permissions),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.session_timeout,
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        # Store active session
        self.active_sessions[token] = {
            "user_id": user.user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
        }

        return token

    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Check if session is still active
            if token in self.active_sessions:
                session = self.active_sessions[token]
                session["last_activity"] = datetime.utcnow()
                return payload
            else:
                return None

        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            if token in self.active_sessions:
                del self.active_sessions[token]
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None

    def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource/action"""
        # Admin has all permissions
        if user.security_level == SecurityLevel.ADMIN:
            return True

        # System level for internal operations
        if user.security_level == SecurityLevel.SYSTEM:
            return True

        # Check specific permissions
        permission_key = f"{resource}:{action}"
        if permission_key in user.permissions or "*" in user.permissions:
            return True

        # Resource-level permissions
        resource_permission = f"{resource}:*"
        if resource_permission in user.permissions:
            return True

        # Default trading permissions for trader level
        if user.security_level == SecurityLevel.TRADER:
            trading_actions = ["read", "execute", "analyze"]
            trading_resources = ["market_data", "portfolio", "trades", "risk_metrics"]

            if resource in trading_resources and action in trading_actions:
                return True

        return False

    def validate_and_authorize(
        self, token: str, resource: str, action: str, parameters: Dict[str, Any] = None
    ) -> tuple[bool, Optional[User], Dict[str, Any]]:
        """Complete validation and authorization check"""
        # Validate token
        payload = self.validate_jwt_token(token)
        if not payload:
            return False, None, {}

        # Get user
        user = self.users.get(payload["user_id"])
        if not user or not user.is_active:
            return False, None, {}

        # Check permission
        if not self.check_permission(user, resource, action):
            self.audit_log(
                AuditEventType.SECURITY_VIOLATION,
                user.user_id,
                resource,
                action,
                "unauthorized",
                "",
                "",
                {"reason": "insufficient_permissions"},
            )
            return False, user, {}

        # Validate parameters if provided
        validated_params = {}
        if parameters:
            try:
                validated_params = InputValidator.validate_tool_parameters(resource, parameters)
            except ValueError as e:
                self.audit_log(
                    AuditEventType.SECURITY_VIOLATION,
                    user.user_id,
                    resource,
                    action,
                    "validation_failed",
                    "",
                    "",
                    {"error": str(e), "parameters": parameters},
                )
                return False, user, {}

        return True, user, validated_params

    def audit_log(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        resource: str,
        action: str,
        result: str,
        ip_address: str,
        user_agent: str,
        metadata: Dict[str, Any],
    ):
        """Log audit event"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata,
            timestamp=datetime.utcnow(),
        )

        self.audit_events.append(event)

        # Log to system logger
        self.logger.info(
            f"AUDIT: {event_type.value} | {user_id or 'anonymous'} | "
            f"{resource}:{action} | {result} | {ip_address}"
        )

        # Trim audit log if too large (keep last 10000 events)
        if len(self.audit_events) > 10000:
            self.audit_events = self.audit_events[-10000:]

    def get_audit_events(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get audit events with filtering"""
        events = self.audit_events

        # Apply filters
        if user_id:
            events = [e for e in events if e.user_id == user_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.utcnow()
        expired_tokens = []

        for token, session in self.active_sessions.items():
            if current_time - session["last_activity"] > self.session_timeout:
                expired_tokens.append(token)

        for token in expired_tokens:
            del self.active_sessions[token]

        if expired_tokens:
            self.logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")


def require_auth(security_level: SecurityLevel = SecurityLevel.AUTHENTICATED):
    """Decorator for requiring authentication and authorization"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get security manager from self
            if not hasattr(self, "security_manager"):
                raise Exception("Security manager not available")

            security_manager: SecurityManager = self.security_manager

            # Extract token from kwargs or context
            token = kwargs.pop("auth_token", None)
            if not token and hasattr(self, "context"):
                token = getattr(self.context, "auth_token", None)

            if not token:
                raise Exception("Authentication token required")

            # Validate and authorize
            resource = func.__name__
            action = "execute"
            parameters = kwargs

            authorized, user, validated_params = security_manager.validate_and_authorize(
                token, resource, action, parameters
            )

            if not authorized:
                raise Exception("Authorization failed")

            # Check security level
            if user.security_level.value < security_level.value:
                raise Exception(f"Insufficient security level: required {security_level.value}")

            # Update kwargs with validated parameters
            kwargs.update(validated_params)
            kwargs["authenticated_user"] = user

            # Execute function
            result = await func(self, *args, **kwargs)

            # Log successful execution
            security_manager.audit_log(
                AuditEventType.DATA_ACCESS,
                user.user_id,
                resource,
                action,
                "success",
                "",
                "",
                {"parameters": list(kwargs.keys())},
            )

            return result

        return wrapper

    return decorator
