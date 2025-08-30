"""
Enhanced Permission System for MCP Security

This module provides granular method-level permissions and secure access control
for MCP protocol methods, addressing authentication bypass vulnerabilities.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Union

from .authentication import AuthenticationContext, Permission

logger = logging.getLogger(__name__)


class MethodSecurityLevel(str, Enum):
    """Security levels for MCP methods"""

    PUBLIC = "public"  # No authentication required
    AUTHENTICATED = "authenticated"  # Basic authentication required
    AUTHORIZED = "authorized"  # Specific permissions required
    ADMIN = "admin"  # Admin-level permissions required
    SYSTEM = "system"  # System/internal use only


@dataclass
class MethodPermissionConfig:
    """Permission configuration for a specific method"""

    method: str
    security_level: MethodSecurityLevel
    required_permissions: List[Permission]
    allow_anonymous: bool = False
    rate_limit_tier: str = "standard"
    audit_required: bool = False
    description: str = ""


class SecureMethodRegistry:
    """Registry of secure method permissions with no bypass vulnerabilities"""

    def __init__(self):
        """Initialize with secure default permissions"""
        self.method_configs: Dict[str, MethodPermissionConfig] = {}
        self._load_secure_defaults()

    def _load_secure_defaults(self):
        """Load secure default method permissions - NO AUTHENTICATION BYPASS"""

        # Core protocol methods - require authentication
        self.register_method(
            MethodPermissionConfig(
                method="initialize",
                security_level=MethodSecurityLevel.AUTHENTICATED,
                required_permissions=[Permission.HEALTH_CHECK],
                allow_anonymous=False,  # SECURE: No anonymous access
                audit_required=True,
                description="MCP initialization handshake",
            )
        )

        self.register_method(
            MethodPermissionConfig(
                method="initialized",
                security_level=MethodSecurityLevel.AUTHENTICATED,
                required_permissions=[Permission.HEALTH_CHECK],
                allow_anonymous=False,
                description="MCP initialization confirmation",
            )
        )

        # Health check - limited anonymous access with restrictions
        self.register_method(
            MethodPermissionConfig(
                method="ping",
                security_level=MethodSecurityLevel.PUBLIC,
                required_permissions=[],
                allow_anonymous=True,
                rate_limit_tier="restrictive",  # Heavy rate limiting
                description="Basic health check (limited info)",
            )
        )

        # Tool operations - require specific permissions
        self.register_method(
            MethodPermissionConfig(
                method="tools/list",
                security_level=MethodSecurityLevel.AUTHORIZED,
                required_permissions=[Permission.READ_TOOLS],
                allow_anonymous=False,
                description="List available tools",
            )
        )

        self.register_method(
            MethodPermissionConfig(
                method="tools/call",
                security_level=MethodSecurityLevel.AUTHORIZED,
                required_permissions=[Permission.EXECUTE_TOOLS],
                allow_anonymous=False,
                rate_limit_tier="strict",
                audit_required=True,
                description="Execute tool functions",
            )
        )

        # Resource operations - require specific permissions
        self.register_method(
            MethodPermissionConfig(
                method="resources/list",
                security_level=MethodSecurityLevel.AUTHORIZED,
                required_permissions=[Permission.READ_RESOURCES],
                allow_anonymous=False,
                description="List available resources",
            )
        )

        self.register_method(
            MethodPermissionConfig(
                method="resources/read",
                security_level=MethodSecurityLevel.AUTHORIZED,
                required_permissions=[Permission.READ_RESOURCES],
                allow_anonymous=False,
                audit_required=True,
                description="Read resource content",
            )
        )

        self.register_method(
            MethodPermissionConfig(
                method="resources/write",
                security_level=MethodSecurityLevel.AUTHORIZED,
                required_permissions=[Permission.WRITE_RESOURCES],
                allow_anonymous=False,
                rate_limit_tier="strict",
                audit_required=True,
                description="Write resource content",
            )
        )

        # Administrative operations - require admin permissions
        self.register_method(
            MethodPermissionConfig(
                method="security/status",
                security_level=MethodSecurityLevel.ADMIN,
                required_permissions=[Permission.ADMIN_SERVER],
                allow_anonymous=False,
                audit_required=True,
                description="Get security system status",
            )
        )

        self.register_method(
            MethodPermissionConfig(
                method="admin/metrics",
                security_level=MethodSecurityLevel.ADMIN,
                required_permissions=[Permission.METRICS_READ, Permission.ADMIN_SERVER],
                allow_anonymous=False,
                audit_required=True,
                description="Get system metrics",
            )
        )

        self.register_method(
            MethodPermissionConfig(
                method="admin/users",
                security_level=MethodSecurityLevel.ADMIN,
                required_permissions=[Permission.ADMIN_SERVER],
                allow_anonymous=False,
                audit_required=True,
                description="User management operations",
            )
        )

        # Crypto trading specific methods (if applicable)
        self.register_method(
            MethodPermissionConfig(
                method="trading/portfolio",
                security_level=MethodSecurityLevel.AUTHORIZED,
                required_permissions=[Permission.READ_RESOURCES],
                allow_anonymous=False,
                description="Get portfolio information",
            )
        )

        self.register_method(
            MethodPermissionConfig(
                method="trading/execute",
                security_level=MethodSecurityLevel.AUTHORIZED,
                required_permissions=[Permission.EXECUTE_TOOLS],
                allow_anonymous=False,
                rate_limit_tier="strict",
                audit_required=True,
                description="Execute trading operations",
            )
        )

        logger.info(f"Loaded {len(self.method_configs)} secure method configurations")

    def register_method(self, config: MethodPermissionConfig):
        """Register a method permission configuration"""
        self.method_configs[config.method] = config
        logger.debug(f"Registered method: {config.method} (level: {config.security_level})")

    def get_method_config(self, method: str) -> Optional[MethodPermissionConfig]:
        """Get permission configuration for a method"""
        return self.method_configs.get(method)

    def is_method_allowed(self, method: str) -> bool:
        """Check if method is registered and allowed"""
        return method in self.method_configs

    def get_required_permissions(self, method: str) -> List[Permission]:
        """Get required permissions for a method"""
        config = self.get_method_config(method)
        return config.required_permissions if config else [Permission.ADMIN_SERVER]

    def requires_authentication(self, method: str) -> bool:
        """Check if method requires authentication"""
        config = self.get_method_config(method)
        if not config:
            return True  # Secure default - require auth for unknown methods

        return config.security_level != MethodSecurityLevel.PUBLIC

    def allows_anonymous(self, method: str) -> bool:
        """Check if method allows anonymous access"""
        config = self.get_method_config(method)
        if not config:
            return False  # Secure default - no anonymous for unknown methods

        return config.allow_anonymous and config.security_level == MethodSecurityLevel.PUBLIC

    def requires_audit(self, method: str) -> bool:
        """Check if method requires audit logging"""
        config = self.get_method_config(method)
        return config.audit_required if config else True  # Secure default

    def get_rate_limit_tier(self, method: str) -> str:
        """Get rate limit tier for method"""
        config = self.get_method_config(method)
        return config.rate_limit_tier if config else "strict"  # Secure default

    def list_methods_by_permission(self, permission: Permission) -> List[str]:
        """List methods that require a specific permission"""
        methods = []
        for method, config in self.method_configs.items():
            if permission in config.required_permissions:
                methods.append(method)
        return methods

    def get_security_summary(self) -> Dict[str, int]:
        """Get summary of security configuration"""
        summary = {
            "total_methods": len(self.method_configs),
            "public_methods": 0,
            "authenticated_methods": 0,
            "authorized_methods": 0,
            "admin_methods": 0,
            "audit_required": 0,
            "anonymous_allowed": 0,
        }

        for config in self.method_configs.values():
            if config.security_level == MethodSecurityLevel.PUBLIC:
                summary["public_methods"] += 1
            elif config.security_level == MethodSecurityLevel.AUTHENTICATED:
                summary["authenticated_methods"] += 1
            elif config.security_level == MethodSecurityLevel.AUTHORIZED:
                summary["authorized_methods"] += 1
            elif config.security_level == MethodSecurityLevel.ADMIN:
                summary["admin_methods"] += 1

            if config.audit_required:
                summary["audit_required"] += 1

            if config.allow_anonymous:
                summary["anonymous_allowed"] += 1

        return summary


class PermissionValidator:
    """Validates permissions for method access with no bypass vulnerabilities"""

    def __init__(self, method_registry: SecureMethodRegistry):
        """Initialize with method registry"""
        self.method_registry = method_registry

    def validate_method_access(
        self, method: str, auth_context: Optional[AuthenticationContext]
    ) -> Dict[str, any]:
        """
        Validate if user has permission to access method

        Args:
            method: MCP method name
            auth_context: Authentication context (None for unauthenticated)

        Returns:
            Dict with validation result and details
        """
        result = {
            "allowed": False,
            "reason": "",
            "required_permissions": [],
            "missing_permissions": [],
            "security_level": "",
            "audit_required": False,
        }

        # Check if method is registered
        config = self.method_registry.get_method_config(method)
        if not config:
            result["reason"] = f"Unknown method: {method}"
            result["required_permissions"] = [Permission.ADMIN_SERVER.value]
            logger.warning(f"Access denied - unknown method: {method}")
            return result

        result["security_level"] = config.security_level.value
        result["required_permissions"] = [p.value for p in config.required_permissions]
        result["audit_required"] = config.audit_required

        # Check public methods
        if config.security_level == MethodSecurityLevel.PUBLIC:
            if config.allow_anonymous:
                result["allowed"] = True
                result["reason"] = "Public method with anonymous access"
                return result

        # All other methods require authentication
        if not auth_context:
            result["reason"] = "Authentication required"
            logger.warning(f"Access denied - no authentication for method: {method}")
            return result

        # Check if user is expired
        if auth_context.is_expired():
            result["reason"] = "Authentication expired"
            logger.warning(f"Access denied - expired authentication for method: {method}")
            return result

        # Check specific permission requirements
        if config.required_permissions:
            missing_permissions = []
            for required_perm in config.required_permissions:
                if not auth_context.has_permission(required_perm):
                    missing_permissions.append(required_perm.value)

            if missing_permissions:
                result["missing_permissions"] = missing_permissions
                result["reason"] = f"Missing required permissions: {missing_permissions}"
                logger.warning(
                    f"Access denied - insufficient permissions for {method}: {missing_permissions}"
                )
                return result

        # Access granted
        result["allowed"] = True
        result["reason"] = "Access granted"
        logger.debug(f"Access granted for method: {method} (user: {auth_context.user_id})")
        return result

    def check_admin_access(self, auth_context: Optional[AuthenticationContext]) -> bool:
        """Check if user has admin access"""
        if not auth_context or auth_context.is_expired():
            return False

        return auth_context.has_permission(Permission.ADMIN_SERVER)

    def get_user_accessible_methods(
        self, auth_context: Optional[AuthenticationContext]
    ) -> List[str]:
        """Get list of methods user can access"""
        accessible_methods = []

        for method in self.method_registry.method_configs.keys():
            validation = self.validate_method_access(method, auth_context)
            if validation["allowed"]:
                accessible_methods.append(method)

        return accessible_methods


# Global instances
_secure_method_registry = SecureMethodRegistry()
_permission_validator = PermissionValidator(_secure_method_registry)


def get_method_registry() -> SecureMethodRegistry:
    """Get the global secure method registry"""
    return _secure_method_registry


def get_permission_validator() -> PermissionValidator:
    """Get the global permission validator"""
    return _permission_validator


def validate_method_permission(
    method: str, auth_context: Optional[AuthenticationContext]
) -> Dict[str, any]:
    """Convenience function to validate method permissions"""
    return _permission_validator.validate_method_access(method, auth_context)


def register_custom_method(
    method: str,
    security_level: MethodSecurityLevel,
    required_permissions: List[Permission],
    **kwargs,
):
    """Register a custom method with permissions"""
    config = MethodPermissionConfig(
        method=method,
        security_level=security_level,
        required_permissions=required_permissions,
        **kwargs,
    )
    _secure_method_registry.register_method(config)


# Decorators for method permission enforcement
def require_permissions(*permissions: Permission):
    """Decorator to require specific permissions for a method"""

    def decorator(func):
        async def wrapper(self, auth_context: Optional[AuthenticationContext], *args, **kwargs):
            if not auth_context:
                raise PermissionError("Authentication required")

            if auth_context.is_expired():
                raise PermissionError("Authentication expired")

            missing_perms = [p for p in permissions if not auth_context.has_permission(p)]
            if missing_perms:
                raise PermissionError(
                    f"Missing required permissions: {[p.value for p in missing_perms]}"
                )

            return await func(self, auth_context, *args, **kwargs)

        return wrapper

    return decorator


def require_admin(func):
    """Decorator to require admin permissions"""
    return require_permissions(Permission.ADMIN_SERVER)(func)
