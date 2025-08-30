"""
Secure Production Defaults for MCP Security

This module provides secure-by-default configuration for production deployments,
eliminating common misconfigurations that lead to security vulnerabilities.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeploymentEnvironment(str, Enum):
    """Deployment environment types"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SecurityProfile(str, Enum):
    """Security profile levels"""

    PERMISSIVE = "permissive"  # Development/testing
    BALANCED = "balanced"  # Staging
    STRICT = "strict"  # Production
    MAXIMUM = "maximum"  # High-security production


@dataclass
class SecureDefaults:
    """Secure default configuration values"""

    # Authentication defaults
    require_authentication: bool = True
    allow_anonymous_access: bool = False
    jwt_enabled: bool = True
    api_key_enabled: bool = True
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5
    token_rotation_enabled: bool = True

    # Authorization defaults
    enforce_permissions: bool = True
    default_deny_unknown_methods: bool = True
    admin_approval_required: bool = True
    audit_admin_actions: bool = True

    # Rate limiting defaults
    rate_limiting_enabled: bool = True
    global_rate_limit_per_minute: int = 1000
    user_rate_limit_per_minute: int = 100
    anonymous_rate_limit_per_minute: int = 10
    burst_limit_multiplier: float = 1.5
    rate_limit_headers_enabled: bool = True

    # Input validation defaults
    input_validation_enabled: bool = True
    strict_validation: bool = True
    max_request_size_bytes: int = 512 * 1024  # 512KB
    max_nested_depth: int = 10
    sanitize_output: bool = True

    # Security monitoring defaults
    security_logging_enabled: bool = True
    audit_logging_enabled: bool = True
    threat_detection_enabled: bool = True
    compliance_logging_enabled: bool = True
    log_sensitive_data: bool = False

    # Network security defaults
    cors_enabled: bool = False
    allowed_origins: List[str] = field(default_factory=list)
    tls_required: bool = True
    secure_headers_enabled: bool = True

    # Error handling defaults
    sanitize_error_messages: bool = True
    expose_stack_traces: bool = False
    detailed_errors_in_dev: bool = False

    # Operational security defaults
    health_check_auth_required: bool = True
    metrics_auth_required: bool = True
    admin_endpoints_enabled: bool = False
    debug_mode_enabled: bool = False

    # Token storage defaults
    token_storage_backend: str = "redis"
    token_revocation_enabled: bool = True
    distributed_session_management: bool = True

    # Additional security features
    ip_whitelisting_enabled: bool = False
    geo_blocking_enabled: bool = False
    anomaly_detection_enabled: bool = True
    brute_force_protection_enabled: bool = True


class SecureConfigurationManager:
    """Manages secure configuration based on deployment environment"""

    def __init__(
        self,
        environment: Optional[DeploymentEnvironment] = None,
        security_profile: Optional[SecurityProfile] = None,
    ):
        """
        Initialize secure configuration manager

        Args:
            environment: Deployment environment (detected from env vars if None)
            security_profile: Security profile (derived from environment if None)
        """
        self.environment = environment or self._detect_environment()
        self.security_profile = security_profile or self._derive_security_profile()

        logger.info(
            f"Initializing secure configuration: {self.environment} ({self.security_profile})"
        )

        # Load defaults based on profile
        self.defaults = self._load_profile_defaults()

        # Apply environment-specific overrides
        self._apply_environment_overrides()

    def _detect_environment(self) -> DeploymentEnvironment:
        """Detect deployment environment from environment variables"""
        env_name = os.getenv(
            "DEPLOYMENT_ENVIRONMENT", os.getenv("NODE_ENV", os.getenv("ENVIRONMENT", "production"))
        ).lower()

        env_mapping = {
            "dev": DeploymentEnvironment.DEVELOPMENT,
            "development": DeploymentEnvironment.DEVELOPMENT,
            "test": DeploymentEnvironment.TESTING,
            "testing": DeploymentEnvironment.TESTING,
            "stage": DeploymentEnvironment.STAGING,
            "staging": DeploymentEnvironment.STAGING,
            "prod": DeploymentEnvironment.PRODUCTION,
            "production": DeploymentEnvironment.PRODUCTION,
        }

        detected = env_mapping.get(env_name, DeploymentEnvironment.PRODUCTION)
        logger.info(f"Detected environment: {detected}")
        return detected

    def _derive_security_profile(self) -> SecurityProfile:
        """Derive security profile from environment"""
        profile_mapping = {
            DeploymentEnvironment.DEVELOPMENT: SecurityProfile.PERMISSIVE,
            DeploymentEnvironment.TESTING: SecurityProfile.PERMISSIVE,
            DeploymentEnvironment.STAGING: SecurityProfile.BALANCED,
            DeploymentEnvironment.PRODUCTION: SecurityProfile.STRICT,
        }

        # Allow override via environment variable
        profile_override = os.getenv("SECURITY_PROFILE")
        if profile_override:
            try:
                return SecurityProfile(profile_override.lower())
            except ValueError:
                logger.warning(f"Invalid security profile override: {profile_override}")

        return profile_mapping[self.environment]

    def _load_profile_defaults(self) -> SecureDefaults:
        """Load defaults based on security profile"""
        if self.security_profile == SecurityProfile.PERMISSIVE:
            return self._get_permissive_defaults()
        elif self.security_profile == SecurityProfile.BALANCED:
            return self._get_balanced_defaults()
        elif self.security_profile == SecurityProfile.STRICT:
            return self._get_strict_defaults()
        elif self.security_profile == SecurityProfile.MAXIMUM:
            return self._get_maximum_defaults()
        else:
            # Default to strict for unknown profiles
            return self._get_strict_defaults()

    def _get_permissive_defaults(self) -> SecureDefaults:
        """Permissive defaults for development/testing"""
        return SecureDefaults(
            # More relaxed for development
            allow_anonymous_access=True,
            session_timeout_minutes=480,  # 8 hours
            max_concurrent_sessions=10,
            user_rate_limit_per_minute=1000,
            max_request_size_bytes=2 * 1024 * 1024,  # 2MB
            health_check_auth_required=False,
            detailed_errors_in_dev=True,
            debug_mode_enabled=True,
            admin_endpoints_enabled=True,
            tls_required=False,
            cors_enabled=True,
            allowed_origins=["*"],  # Only for development!
        )

    def _get_balanced_defaults(self) -> SecureDefaults:
        """Balanced defaults for staging"""
        return SecureDefaults(
            # Staging configuration
            allow_anonymous_access=False,
            session_timeout_minutes=120,  # 2 hours
            max_concurrent_sessions=8,
            user_rate_limit_per_minute=200,
            max_request_size_bytes=1024 * 1024,  # 1MB
            health_check_auth_required=True,
            detailed_errors_in_dev=False,
            debug_mode_enabled=False,
            admin_endpoints_enabled=True,
            tls_required=True,
            cors_enabled=True,
            allowed_origins=[],  # Must be configured explicitly
        )

    def _get_strict_defaults(self) -> SecureDefaults:
        """Strict defaults for production"""
        return SecureDefaults(
            # Production security defaults - all secure options enabled
            require_authentication=True,
            allow_anonymous_access=False,
            session_timeout_minutes=60,
            max_concurrent_sessions=5,
            token_rotation_enabled=True,
            enforce_permissions=True,
            default_deny_unknown_methods=True,
            admin_approval_required=True,
            audit_admin_actions=True,
            rate_limiting_enabled=True,
            user_rate_limit_per_minute=100,
            anonymous_rate_limit_per_minute=10,
            input_validation_enabled=True,
            strict_validation=True,
            max_request_size_bytes=512 * 1024,  # 512KB
            security_logging_enabled=True,
            audit_logging_enabled=True,
            threat_detection_enabled=True,
            compliance_logging_enabled=True,
            log_sensitive_data=False,
            sanitize_error_messages=True,
            expose_stack_traces=False,
            health_check_auth_required=True,
            metrics_auth_required=True,
            admin_endpoints_enabled=False,
            debug_mode_enabled=False,
            tls_required=True,
            secure_headers_enabled=True,
            cors_enabled=False,
            token_revocation_enabled=True,
            distributed_session_management=True,
            anomaly_detection_enabled=True,
            brute_force_protection_enabled=True,
        )

    def _get_maximum_defaults(self) -> SecureDefaults:
        """Maximum security defaults for high-security environments"""
        strict_defaults = self._get_strict_defaults()

        # Enhanced security settings
        strict_defaults.session_timeout_minutes = 30
        strict_defaults.max_concurrent_sessions = 3
        strict_defaults.user_rate_limit_per_minute = 50
        strict_defaults.anonymous_rate_limit_per_minute = 5
        strict_defaults.max_request_size_bytes = 256 * 1024  # 256KB
        strict_defaults.ip_whitelisting_enabled = True
        strict_defaults.geo_blocking_enabled = True

        return strict_defaults

    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        overrides = {
            # Authentication overrides
            "MCP_REQUIRE_AUTH": ("require_authentication", bool),
            "MCP_ALLOW_ANONYMOUS": ("allow_anonymous_access", bool),
            "MCP_SESSION_TIMEOUT": ("session_timeout_minutes", int),
            # Rate limiting overrides
            "MCP_RATE_LIMITING": ("rate_limiting_enabled", bool),
            "MCP_USER_RATE_LIMIT": ("user_rate_limit_per_minute", int),
            "MCP_GLOBAL_RATE_LIMIT": ("global_rate_limit_per_minute", int),
            # Validation overrides
            "MCP_INPUT_VALIDATION": ("input_validation_enabled", bool),
            "MCP_STRICT_VALIDATION": ("strict_validation", bool),
            "MCP_MAX_REQUEST_SIZE": ("max_request_size_bytes", int),
            # Security overrides
            "MCP_SECURITY_LOGGING": ("security_logging_enabled", bool),
            "MCP_AUDIT_LOGGING": ("audit_logging_enabled", bool),
            "MCP_TLS_REQUIRED": ("tls_required", bool),
            "MCP_SANITIZE_ERRORS": ("sanitize_error_messages", bool),
            # Operational overrides
            "MCP_DEBUG_MODE": ("debug_mode_enabled", bool),
            "MCP_ADMIN_ENDPOINTS": ("admin_endpoints_enabled", bool),
            "MCP_TOKEN_STORAGE": ("token_storage_backend", str),
        }

        for env_var, (attr_name, value_type) in overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if value_type == bool:
                        parsed_value = env_value.lower() in ("true", "1", "yes", "on")
                    elif value_type == int:
                        parsed_value = int(env_value)
                    else:
                        parsed_value = env_value

                    setattr(self.defaults, attr_name, parsed_value)
                    logger.info(f"Applied environment override: {attr_name} = {parsed_value}")

                except (ValueError, TypeError) as e:
                    logger.error(f"Failed to parse environment variable {env_var}: {e}")

    def get_security_config(self) -> Dict[str, Any]:
        """Get complete security configuration as dictionary"""
        config = {
            "environment": self.environment.value,
            "security_profile": self.security_profile.value,
            "defaults": {
                # Convert defaults to dictionary
                field.name: getattr(self.defaults, field.name)
                for field in self.defaults.__dataclass_fields__.values()
            },
        }

        return config

    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate security configuration and return warnings/errors"""
        warnings = []
        errors = []

        # Check for insecure configurations
        if (
            self.defaults.allow_anonymous_access
            and self.environment == DeploymentEnvironment.PRODUCTION
        ):
            errors.append("Anonymous access enabled in production environment")

        if not self.defaults.tls_required and self.environment == DeploymentEnvironment.PRODUCTION:
            errors.append("TLS not required in production environment")

        if (
            self.defaults.debug_mode_enabled
            and self.environment == DeploymentEnvironment.PRODUCTION
        ):
            errors.append("Debug mode enabled in production environment")

        if (
            not self.defaults.sanitize_error_messages
            and self.environment != DeploymentEnvironment.DEVELOPMENT
        ):
            warnings.append("Error message sanitization disabled")

        if self.defaults.session_timeout_minutes > 480:  # 8 hours
            warnings.append("Session timeout is very long (>8 hours)")

        if not self.defaults.rate_limiting_enabled:
            warnings.append("Rate limiting is disabled")

        if self.defaults.max_request_size_bytes > 5 * 1024 * 1024:  # 5MB
            warnings.append("Maximum request size is very large (>5MB)")

        # Check for missing critical settings
        if (
            not self.defaults.audit_logging_enabled
            and self.environment == DeploymentEnvironment.PRODUCTION
        ):
            warnings.append("Audit logging disabled in production")

        if self.defaults.cors_enabled and "*" in self.defaults.allowed_origins:
            errors.append("CORS wildcard origins enabled - security risk")

        return {"warnings": warnings, "errors": errors}

    def get_configuration_summary(self) -> str:
        """Get human-readable configuration summary"""
        validation = self.validate_configuration()

        summary = f"""
MCP Security Configuration Summary
=================================
Environment: {self.environment.value}
Security Profile: {self.security_profile.value}

Key Security Settings:
- Authentication Required: {self.defaults.require_authentication}
- Anonymous Access: {self.defaults.allow_anonymous_access}
- Rate Limiting: {self.defaults.rate_limiting_enabled}
- Input Validation: {self.defaults.input_validation_enabled}
- TLS Required: {self.defaults.tls_required}
- Security Logging: {self.defaults.security_logging_enabled}
- Debug Mode: {self.defaults.debug_mode_enabled}

Session Management:
- Timeout: {self.defaults.session_timeout_minutes} minutes
- Max Concurrent: {self.defaults.max_concurrent_sessions}
- Token Rotation: {self.defaults.token_rotation_enabled}

Rate Limits:
- User: {self.defaults.user_rate_limit_per_minute}/min
- Global: {self.defaults.global_rate_limit_per_minute}/min
- Anonymous: {self.defaults.anonymous_rate_limit_per_minute}/min

Request Limits:
- Max Size: {self.defaults.max_request_size_bytes} bytes
- Max Depth: {self.defaults.max_nested_depth} levels
"""

        if validation["errors"]:
            summary += f"\nSECURITY ERRORS ({len(validation['errors'])}):\n"
            for error in validation["errors"]:
                summary += f"  ❌ {error}\n"

        if validation["warnings"]:
            summary += f"\nSECURITY WARNINGS ({len(validation['warnings'])}):\n"
            for warning in validation["warnings"]:
                summary += f"  ⚠️  {warning}\n"

        if not validation["errors"] and not validation["warnings"]:
            summary += "\n✅ Configuration validated successfully - no issues detected"

        return summary


# Global configuration manager instance
_config_manager: Optional[SecureConfigurationManager] = None


def get_secure_config_manager() -> SecureConfigurationManager:
    """Get global secure configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SecureConfigurationManager()
    return _config_manager


def initialize_secure_config(
    environment: Optional[DeploymentEnvironment] = None,
    security_profile: Optional[SecurityProfile] = None,
) -> SecureConfigurationManager:
    """Initialize global secure configuration"""
    global _config_manager
    _config_manager = SecureConfigurationManager(environment, security_profile)
    return _config_manager


def get_security_defaults() -> SecureDefaults:
    """Get secure defaults for current configuration"""
    return get_secure_config_manager().defaults


def validate_security_config() -> Dict[str, List[str]]:
    """Validate current security configuration"""
    return get_secure_config_manager().validate_configuration()


def print_security_summary():
    """Print security configuration summary"""
    manager = get_secure_config_manager()
    print(manager.get_configuration_summary())
