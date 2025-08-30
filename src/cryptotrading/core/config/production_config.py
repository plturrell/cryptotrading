"""
Production Configuration Management System
Enterprise-grade configuration with environment-based settings, validation, and security.
"""
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet


class Environment(Enum):
    """Environment types"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration"""

    host: str = "localhost"
    port: int = 5432
    database: str = "cryptotrading"
    username: str = ""
    password: str = ""
    ssl_mode: str = "require"
    connection_pool_size: int = 10
    connection_timeout: float = 30.0
    query_timeout: float = 60.0
    retry_attempts: int = 3


# Exchange configuration removed - only Yahoo Finance and FRED data sources


@dataclass
class RiskConfig:
    """Risk management configuration"""

    max_portfolio_risk: float = 0.02  # 2% max risk per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.15  # 15% max drawdown
    # position_size_limit removed - analysis only, no trading
    stop_loss_percentage: float = 0.03  # 3% stop loss
    take_profit_ratio: float = 2.0  # 2:1 risk/reward
    var_confidence: float = 0.95  # 95% VaR confidence
    stress_test_scenarios: List[str] = field(
        default_factory=lambda: ["market_crash", "flash_crash", "vol_spike", "liquidity_crisis"]
    )


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""

    enable_metrics: bool = True
    metrics_port: int = 8080
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_tracing: bool = True
    jaeger_endpoint: str = "http://localhost:14268"
    prometheus_endpoint: str = "http://localhost:9090"
    alert_webhook_url: str = ""
    health_check_interval: int = 30
    performance_threshold_ms: int = 1000


@dataclass
class SecurityConfig:
    """Security configuration"""

    enable_auth: bool = True
    jwt_secret: str = ""
    jwt_expiry_hours: int = 24
    api_key_required: bool = True
    rate_limit_enabled: bool = True
    encryption_key: str = ""
    allowed_ips: List[str] = field(default_factory=list)
    audit_log_enabled: bool = True
    audit_log_retention_days: int = 90


@dataclass
class StrandsConfig:
    """Strands framework configuration"""

    max_concurrent_workflows: int = 10
    workflow_timeout_seconds: int = 300
    tool_timeout_seconds: int = 30
    max_retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    context_cleanup_interval: int = 3600  # 1 hour
    max_context_history: int = 1000
    enable_a2a_communication: bool = True
    a2a_port: int = 9000
    worker_pool_size: int = 4
    event_bus_capacity: int = 10000
    enable_telemetry: bool = True
    enable_distributed: bool = False
    redis_url: Optional[str] = None


@dataclass
class CodeManagementConfig:
    """Code management configuration"""

    project_path: Optional[Path] = None
    scan_interval: int = 3600  # 1 hour
    quality_check_interval: int = 900  # 15 minutes
    proactive_scan_interval: int = 3600  # 1 hour
    dashboard_port: int = 5001
    auto_fix_enabled: bool = True
    max_auto_fixes_per_cycle: int = 10
    enable_seal_integration: bool = True
    max_concurrent_scans: int = 3
    issue_retention_days: int = 30


@dataclass
class MCPConfig:
    """Model Context Protocol configuration"""

    enable_mcp: bool = True
    server_host: str = "0.0.0.0"
    server_port: int = 8765
    max_connections: int = 100
    connection_timeout: int = 30
    auth_enabled: bool = True
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:*"])
    rate_limit_per_minute: int = 60
    max_request_size_mb: int = 10


class ProductionConfig:
    """
    Enterprise production configuration management system

    Features:
    - Environment-based configuration
    - Encrypted credential storage
    - Configuration validation
    - Hot reloading
    - Audit logging
    """

    def __init__(self, environment: Environment = Environment.PRODUCTION):
        self.environment = environment
        self.config_dir = Path(os.getenv("CONFIG_DIR", "config"))
        self.logger = logging.getLogger("ProductionConfig")

        # Initialize encryption
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)

        # Load configurations
        self.database = DatabaseConfig()
        # Exchange removed - only Yahoo Finance and FRED data
        self.risk = RiskConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        self.strands = StrandsConfig()
        self.code_management = CodeManagementConfig()
        self.mcp = MCPConfig()

        # Load from files
        self._load_configurations()

        # Validate configuration
        self._validate_configuration()

        self.logger.info(f"Configuration loaded for {environment.value} environment")

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for credential storage"""
        key_file = self.config_dir / ".encryption_key"

        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Create new key
            key = Fernet.generate_key()
            self.config_dir.mkdir(exist_ok=True)
            with open(key_file, "wb") as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Secure permissions
            return key

    def _load_configurations(self):
        """Load configurations from environment and files"""
        # Database configuration
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USERNAME", self.database.username)
        self.database.password = self._decrypt_credential(
            os.getenv("DB_PASSWORD_ENCRYPTED", ""), os.getenv("DB_PASSWORD", self.database.password)
        )

        # Exchange configuration removed - only Yahoo Finance and FRED data sources

        # Risk configuration
        self.risk.max_portfolio_risk = float(
            os.getenv("MAX_PORTFOLIO_RISK", self.risk.max_portfolio_risk)
        )
        self.risk.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", self.risk.max_daily_loss))
        self.risk.max_drawdown = float(os.getenv("MAX_DRAWDOWN", self.risk.max_drawdown))

        # Security configuration
        self.security.enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
        self.security.jwt_secret = self._get_or_generate_jwt_secret()
        self.security.encryption_key = self._encryption_key.decode()

        # Strands configuration
        self.strands.max_concurrent_workflows = int(
            os.getenv("MAX_CONCURRENT_WORKFLOWS", self.strands.max_concurrent_workflows)
        )
        self.strands.workflow_timeout_seconds = int(
            os.getenv("WORKFLOW_TIMEOUT", self.strands.workflow_timeout_seconds)
        )
        self.strands.worker_pool_size = int(
            os.getenv("STRANDS_WORKER_POOL_SIZE", self.strands.worker_pool_size)
        )
        self.strands.enable_distributed = (
            os.getenv("STRANDS_ENABLE_DISTRIBUTED", "false").lower() == "true"
        )
        self.strands.redis_url = os.getenv("STRANDS_REDIS_URL", self.strands.redis_url)

        # Code management configuration
        project_path = os.getenv("PROJECT_PATH", os.getcwd())
        self.code_management.project_path = Path(project_path) if project_path else None
        self.code_management.scan_interval = int(
            os.getenv("CODE_SCAN_INTERVAL", self.code_management.scan_interval)
        )
        self.code_management.auto_fix_enabled = os.getenv("CODE_AUTO_FIX", "true").lower() == "true"
        self.code_management.dashboard_port = int(
            os.getenv("CODE_DASHBOARD_PORT", self.code_management.dashboard_port)
        )

        # MCP configuration
        self.mcp.enable_mcp = os.getenv("MCP_ENABLED", "true").lower() == "true"
        self.mcp.server_host = os.getenv("MCP_HOST", self.mcp.server_host)
        self.mcp.server_port = int(os.getenv("MCP_PORT", self.mcp.server_port))
        self.mcp.auth_enabled = os.getenv("MCP_AUTH_ENABLED", "true").lower() == "true"
        self.mcp.ssl_enabled = os.getenv("MCP_SSL_ENABLED", "false").lower() == "true"

        # Load from configuration files if they exist
        self._load_from_files()

    def _load_from_files(self):
        """Load configuration from JSON files"""
        config_files = {
            "database.json": self.database,
            # Exchange configuration removed
            "risk.json": self.risk,
            "monitoring.json": self.monitoring,
            "security.json": self.security,
            "strands.json": self.strands,
        }

        for filename, config_obj in config_files.items():
            config_path = self.config_dir / self.environment.value / filename
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        data = json.load(f)

                    # Update configuration object with file data
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)

                    self.logger.debug(f"Loaded configuration from {config_path}")
                except Exception as e:
                    self.logger.error(f"Error loading {config_path}: {e}")

    def _decrypt_credential(self, encrypted_value: str, fallback_value: str) -> str:
        """Decrypt credential or return fallback"""
        if encrypted_value:
            try:
                return self._cipher.decrypt(encrypted_value.encode()).decode()
            except Exception:
                self.logger.warning("Failed to decrypt credential, using fallback")
                return fallback_value
        return fallback_value

    def _get_or_generate_jwt_secret(self) -> str:
        """Get or generate JWT secret"""
        jwt_secret = os.getenv("JWT_SECRET")
        if not jwt_secret:
            jwt_secret = secrets.token_urlsafe(32)
            self.logger.warning("Generated new JWT secret - should be persisted for production")
        return jwt_secret

    def _validate_configuration(self):
        """Validate configuration for production readiness"""
        errors = []

        # Database validation (skip for testing environment)
        if self.environment != Environment.TESTING:
            if not self.database.host:
                errors.append("Database host is required")
            if not self.database.username:
                errors.append("Database username is required")
            if not self.database.password and self.environment == Environment.PRODUCTION:
                errors.append("Database password is required for production")

        # Exchange removed - no validation needed

        # Risk validation
        if self.risk.max_portfolio_risk <= 0 or self.risk.max_portfolio_risk > 0.1:
            errors.append("Max portfolio risk should be between 0 and 10%")
        if self.risk.max_daily_loss <= 0 or self.risk.max_daily_loss > 0.2:
            errors.append("Max daily loss should be between 0 and 20%")

        # Security validation (only for production)
        if self.environment == Environment.PRODUCTION:
            if not self.security.enable_auth:
                errors.append("Authentication should be enabled in production")
            if len(self.security.jwt_secret) < 32:
                errors.append("JWT secret should be at least 32 characters")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def encrypt_credential(self, credential: str) -> str:
        """Encrypt a credential for secure storage"""
        return self._cipher.encrypt(credential.encode()).decode()

    def get_database_url(self) -> str:
        """Get database connection URL"""
        return (
            f"postgresql://{self.database.username}:{self.database.password}@"
            f"{self.database.host}:{self.database.port}/{self.database.database}"
            f"?sslmode={self.database.ssl_mode}"
        )

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration as dictionary"""
        return {
            "log_level": self.monitoring.log_level,
            "metrics_enabled": self.monitoring.enable_metrics,
            "metrics_port": self.monitoring.metrics_port,
            "tracing_enabled": self.monitoring.enable_tracing,
            "jaeger_endpoint": self.monitoring.jaeger_endpoint,
        }

    def reload_configuration(self):
        """Hot reload configuration from files and environment"""
        self.logger.info("Reloading configuration...")
        self._load_configurations()
        self._validate_configuration()
        self.logger.info("Configuration reloaded successfully")

    def save_configuration_template(self):
        """Save configuration templates for each environment"""
        for env in Environment:
            env_dir = self.config_dir / env.value
            env_dir.mkdir(parents=True, exist_ok=True)

            # Save template files
            templates = {
                "database.json": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "cryptotrading",
                    "connection_pool_size": 10,
                    "connection_timeout": 30.0,
                },
                "exchange.json": {
                    "name": "binance",
                    "sandbox": env != Environment.PRODUCTION,
                    "rate_limit_requests": 100,
                    "rate_limit_window": 60,
                    "timeout": 30.0,
                },
                "risk.json": {
                    "max_portfolio_risk": 0.02,
                    "max_daily_loss": 0.05,
                    "max_drawdown": 0.15,
                    "position_size_limit": 0.2,
                },
            }

            for filename, template in templates.items():
                template_path = env_dir / filename
                if not template_path.exists():
                    with open(template_path, "w") as f:
                        json.dump(template, f, indent=2)


# Global configuration instance
_config_instance: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        env_name = os.getenv("ENVIRONMENT", "production").lower()
        environment = Environment(env_name)
        _config_instance = ProductionConfig(environment)
    return _config_instance


def reload_config():
    """Reload global configuration"""
    global _config_instance
    if _config_instance:
        _config_instance.reload_configuration()
