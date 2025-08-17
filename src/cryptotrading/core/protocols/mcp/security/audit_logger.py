"""
Structured Security Logging for MCP Security

This module provides comprehensive security audit logging with structured context
for forensic analysis, threat detection, and compliance requirements.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Types of security events"""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_GRANTED = "authz_granted"
    AUTHORIZATION_DENIED = "authz_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INPUT_VALIDATION_FAILED = "input_validation_failed"
    SECURITY_THREAT_DETECTED = "security_threat_detected"
    TOKEN_REVOKED = "token_revoked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ADMIN_ACTION = "admin_action"
    SYSTEM_ERROR = "system_error"
    METHOD_EXECUTED = "method_executed"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGED = "config_changed"


class SecuritySeverity(str, Enum):
    """Security event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Comprehensive security context for audit logging"""
    # Request identification
    request_id: str
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # User context
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    auth_method: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    
    # Network context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    forwarded_for: Optional[str] = None
    
    # Request context
    method: Optional[str] = None
    endpoint: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    
    # Temporal context
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_time_ms: Optional[float] = None
    
    # Security context
    security_level: Optional[str] = None
    threat_indicators: List[str] = field(default_factory=list)
    risk_score: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Structured security event for audit logging"""
    # Core event data
    event_id: str
    event_type: SecurityEventType
    severity: SecuritySeverity
    message: str
    
    # Context
    security_context: SecurityContext
    
    # Event details
    success: bool = True
    error_code: Optional[str] = None
    error_details: Optional[str] = None
    
    # Threat analysis
    threat_detected: bool = False
    threat_types: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    
    # Compliance
    compliance_tags: List[str] = field(default_factory=list)
    retention_period_days: int = 365
    
    # Performance
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str, separators=(',', ':'))


class SecurityAuditLogger:
    """Centralized security audit logger with structured context"""
    
    def __init__(self, service_name: str = "mcp-security", environment: str = "production"):
        """
        Initialize security audit logger
        
        Args:
            service_name: Name of the service for log identification
            environment: Environment (production, staging, development)
        """
        self.service_name = service_name
        self.environment = environment
        
        # Configure structured logger
        self.audit_logger = logging.getLogger(f"{service_name}.security_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Add structured formatter if not already configured
        if not self.audit_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
        
        # Event counters for metrics
        self.event_counters: Dict[str, int] = {}
        
        logger.info(f"Initialized security audit logger: {service_name} ({environment})")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return f"mcp-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    
    def _increment_counter(self, event_type: SecurityEventType):
        """Increment event counter for metrics"""
        key = f"{self.environment}.{event_type.value}"
        self.event_counters[key] = self.event_counters.get(key, 0) + 1
    
    def _determine_severity(self, event_type: SecurityEventType, threat_detected: bool = False) -> SecuritySeverity:
        """Determine event severity based on type and context"""
        if threat_detected:
            return SecuritySeverity.HIGH
        
        severity_map = {
            SecurityEventType.AUTHENTICATION_FAILURE: SecuritySeverity.MEDIUM,
            SecurityEventType.AUTHORIZATION_DENIED: SecuritySeverity.MEDIUM,
            SecurityEventType.RATE_LIMIT_EXCEEDED: SecuritySeverity.MEDIUM,
            SecurityEventType.INPUT_VALIDATION_FAILED: SecuritySeverity.MEDIUM,
            SecurityEventType.SECURITY_THREAT_DETECTED: SecuritySeverity.HIGH,
            SecurityEventType.TOKEN_REVOKED: SecuritySeverity.MEDIUM,
            SecurityEventType.SUSPICIOUS_ACTIVITY: SecuritySeverity.HIGH,
            SecurityEventType.ADMIN_ACTION: SecuritySeverity.MEDIUM,
            SecurityEventType.SYSTEM_ERROR: SecuritySeverity.LOW,
            SecurityEventType.AUTHENTICATION_SUCCESS: SecuritySeverity.LOW,
            SecurityEventType.AUTHORIZATION_GRANTED: SecuritySeverity.LOW,
            SecurityEventType.METHOD_EXECUTED: SecuritySeverity.LOW,
            SecurityEventType.DATA_ACCESS: SecuritySeverity.LOW,
            SecurityEventType.CONFIG_CHANGED: SecuritySeverity.HIGH,
        }
        
        return severity_map.get(event_type, SecuritySeverity.MEDIUM)
    
    def log_security_event(self, event_type: SecurityEventType, message: str, 
                          security_context: SecurityContext, **kwargs) -> SecurityEvent:
        """
        Log a security event with full structured context
        
        Args:
            event_type: Type of security event
            message: Human-readable event message
            security_context: Security context information
            **kwargs: Additional event parameters
            
        Returns:
            SecurityEvent object that was logged
        """
        # Create security event
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            severity=self._determine_severity(event_type, kwargs.get('threat_detected', False)),
            message=message,
            security_context=security_context,
            **kwargs
        )
        
        # Add service context
        event.security_context.metadata.update({
            "service_name": self.service_name,
            "environment": self.environment,
            "log_version": "1.0"
        })
        
        # Increment counters
        self._increment_counter(event_type)
        
        # Determine log level based on severity
        log_level = {
            SecuritySeverity.LOW: logging.INFO,
            SecuritySeverity.MEDIUM: logging.WARNING,
            SecuritySeverity.HIGH: logging.ERROR,
            SecuritySeverity.CRITICAL: logging.CRITICAL
        }.get(event.severity, logging.INFO)
        
        # Log structured event
        self.audit_logger.log(
            log_level,
            f"SECURITY_EVENT: {event.event_type.value}",
            extra={
                "event_data": event.to_dict(),
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "user_id": security_context.user_id,
                "ip_address": security_context.ip_address,
                "method": security_context.method,
                "threat_detected": event.threat_detected
            }
        )
        
        # Log to file for compliance (if configured)
        if event.severity in [SecuritySeverity.HIGH, SecuritySeverity.CRITICAL]:
            self._log_to_compliance_file(event)
        
        return event
    
    def _log_to_compliance_file(self, event: SecurityEvent):
        """Log high-severity events to compliance file"""
        try:
            compliance_logger = logging.getLogger(f"{self.service_name}.compliance")
            compliance_logger.info(event.to_json())
        except Exception as e:
            logger.error(f"Failed to log compliance event: {e}")
    
    # Convenience methods for common security events
    
    def log_authentication_success(self, security_context: SecurityContext):
        """Log successful authentication"""
        return self.log_security_event(
            SecurityEventType.AUTHENTICATION_SUCCESS,
            f"User {security_context.user_id} authenticated successfully",
            security_context,
            success=True
        )
    
    def log_authentication_failure(self, security_context: SecurityContext, reason: str):
        """Log failed authentication"""
        return self.log_security_event(
            SecurityEventType.AUTHENTICATION_FAILURE,
            f"Authentication failed: {reason}",
            security_context,
            success=False,
            error_details=reason
        )
    
    def log_authorization_denied(self, security_context: SecurityContext, required_permissions: List[str]):
        """Log authorization denial"""
        return self.log_security_event(
            SecurityEventType.AUTHORIZATION_DENIED,
            f"Authorization denied for method {security_context.method}",
            security_context,
            success=False,
            error_details=f"Required permissions: {required_permissions}"
        )
    
    def log_rate_limit_exceeded(self, security_context: SecurityContext, limit_type: str):
        """Log rate limit exceeded"""
        return self.log_security_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            f"Rate limit exceeded: {limit_type}",
            security_context,
            success=False,
            threat_detected=True,
            threat_types=["rate_limit_abuse"]
        )
    
    def log_security_threat(self, security_context: SecurityContext, threat_type: str, details: str):
        """Log security threat detection"""
        return self.log_security_event(
            SecurityEventType.SECURITY_THREAT_DETECTED,
            f"Security threat detected: {threat_type}",
            security_context,
            success=False,
            threat_detected=True,
            threat_types=[threat_type],
            error_details=details
        )
    
    def log_input_validation_failure(self, security_context: SecurityContext, validation_errors: List[str]):
        """Log input validation failure"""
        return self.log_security_event(
            SecurityEventType.INPUT_VALIDATION_FAILED,
            f"Input validation failed for method {security_context.method}",
            security_context,
            success=False,
            error_details="; ".join(validation_errors),
            threat_detected=any("security" in error.lower() for error in validation_errors)
        )
    
    def log_method_execution(self, security_context: SecurityContext, success: bool, 
                            processing_time_ms: float, error: Optional[str] = None):
        """Log method execution"""
        security_context.processing_time_ms = processing_time_ms
        
        return self.log_security_event(
            SecurityEventType.METHOD_EXECUTED,
            f"Method {security_context.method} executed",
            security_context,
            success=success,
            error_details=error
        )
    
    def log_admin_action(self, security_context: SecurityContext, action: str, target: str):
        """Log administrative action"""
        return self.log_security_event(
            SecurityEventType.ADMIN_ACTION,
            f"Admin action: {action} on {target}",
            security_context,
            success=True,
            compliance_tags=["admin_audit", "privileged_access"]
        )
    
    def log_token_revocation(self, security_context: SecurityContext, token_id: str, reason: str):
        """Log token revocation"""
        return self.log_security_event(
            SecurityEventType.TOKEN_REVOKED,
            f"Token revoked: {token_id[:8]}...",
            security_context,
            success=True,
            error_details=reason,
            compliance_tags=["token_management"]
        )
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get security event statistics"""
        return {
            "service_name": self.service_name,
            "environment": self.environment,
            "event_counters": self.event_counters.copy(),
            "total_events": sum(self.event_counters.values()),
            "logger_status": "active"
        }


# Utility functions for creating security context
def create_security_context(request_id: str = None, **kwargs) -> SecurityContext:
    """Create security context with automatic request ID generation"""
    if not request_id:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
    
    return SecurityContext(request_id=request_id, **kwargs)


def extract_security_context_from_headers(headers: Dict[str, str], 
                                        method: str = None, 
                                        user_id: str = None) -> SecurityContext:
    """Extract security context from HTTP headers"""
    return SecurityContext(
        request_id=headers.get("x-request-id", f"req-{uuid.uuid4().hex[:12]}"),
        ip_address=headers.get("x-forwarded-for") or headers.get("x-real-ip"),
        user_agent=headers.get("user-agent"),
        forwarded_for=headers.get("x-forwarded-for"),
        method=method,
        user_id=user_id,
        metadata={
            "headers_count": len(headers),
            "has_auth": "authorization" in headers.keys(),
            "has_api_key": "x-api-key" in headers.keys()
        }
    )


# Global audit logger instance
_audit_logger: Optional[SecurityAuditLogger] = None


def get_audit_logger() -> SecurityAuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger()
    return _audit_logger


def initialize_audit_logger(service_name: str = "mcp-security", environment: str = "production"):
    """Initialize global audit logger"""
    global _audit_logger
    _audit_logger = SecurityAuditLogger(service_name, environment)
    return _audit_logger