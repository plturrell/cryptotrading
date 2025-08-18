"""
Error Tracking and Aggregation System
Captures, categorizes, and aggregates errors across the system
"""

import os
import json
import hashlib
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import asyncio

from .logger import get_logger

logger = get_logger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories"""
    API_ERROR = "api_error"
    DATABASE_ERROR = "database_error"
    VALIDATION_ERROR = "validation_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    INTEGRATION_ERROR = "integration_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AI_ERROR = "ai_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorContext:
    """Context information for an error"""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[int] = None
    agent_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: Optional[str] = None
    endpoint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TrackedError:
    """Tracked error with full context"""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    error_traceback: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    fingerprint: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'error_traceback': self.error_traceback,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': asdict(self.context),
            'fingerprint': self.fingerprint
        }

class ErrorTracker:
    """Central error tracking and aggregation system"""
    
    def __init__(self, max_errors: int = 10000, alert_threshold: int = 10):
        self.max_errors = max_errors
        self.alert_threshold = alert_threshold
        
        # Error storage
        self.errors: List[TrackedError] = []
        self.error_counts: Counter = Counter()
        self.error_fingerprints: Dict[str, List[TrackedError]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Start background tasks
        self._start_background_tasks()
    
    def track_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                   category: Optional[ErrorCategory] = None, 
                   context: Optional[ErrorContext] = None) -> str:
        """Track an error with context"""
        # Auto-categorize if not provided
        if category is None:
            category = self._categorize_error(error)
        
        # Create error context if not provided
        if context is None:
            context = self._get_current_context()
        
        # Generate error fingerprint
        fingerprint = self._generate_fingerprint(error)
        
        # Create tracked error
        tracked_error = TrackedError(
            error_id=self._generate_error_id(),
            timestamp=datetime.utcnow(),
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            severity=severity,
            category=category,
            context=context,
            fingerprint=fingerprint
        )
        
        # Store error
        with self.lock:
            # Add to main list
            self.errors.append(tracked_error)
            
            # Maintain max size
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)
            
            # Update counts
            self.error_counts[fingerprint] += 1
            self.error_fingerprints[fingerprint].append(tracked_error)
            
            # Check for alerts
            if self.error_counts[fingerprint] == self.alert_threshold:
                self._trigger_alert(tracked_error, self.error_counts[fingerprint])
        
        # Log the error
        logger.error(f"Tracked error: {tracked_error.error_type}", error=error, extra={
            'error_id': tracked_error.error_id,
            'error_fingerprint': fingerprint,
            'error_severity': severity.value,
            'error_category': category.value
        })
        
        return tracked_error.error_id
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Auto-categorize error based on type and message"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # API errors
        if any(keyword in error_msg for keyword in ['api', 'endpoint', 'http', 'request']):
            return ErrorCategory.API_ERROR
        
        # Database errors
        elif any(keyword in error_msg for keyword in ['database', 'sql', 'query', 'transaction']):
            return ErrorCategory.DATABASE_ERROR
        
        # Network errors
        elif any(keyword in error_msg for keyword in ['connection', 'timeout', 'network', 'socket']):
            return ErrorCategory.NETWORK_ERROR
        
        # Authentication errors
        elif any(keyword in error_msg for keyword in ['auth', 'permission', 'unauthorized', 'forbidden']):
            return ErrorCategory.AUTHENTICATION_ERROR
        
        # Rate limit errors
        elif any(keyword in error_msg for keyword in ['rate limit', 'throttle', 'quota']):
            return ErrorCategory.RATE_LIMIT_ERROR
        
        # Validation errors
        elif any(keyword in error_type for keyword in ['Validation', 'ValueError', 'TypeError']):
            return ErrorCategory.VALIDATION_ERROR
        
        # Default
        else:
            return ErrorCategory.UNKNOWN_ERROR
    
    def _get_current_context(self) -> ErrorContext:
        """Get current execution context"""
        from .logger import trace_context as ctx_var
        
        # Get trace context
        trace_ctx = ctx_var.get()
        
        return ErrorContext(
            trace_id=trace_ctx.get('trace_id'),
            span_id=trace_ctx.get('span_id'),
            service_name=trace_ctx.get('service'),
            metadata={}
        )
    
    def _generate_fingerprint(self, error: Exception) -> str:
        """Generate error fingerprint for deduplication"""
        # Use error type, key parts of message, and top stack frames
        tb = traceback.extract_tb(error.__traceback__)
        
        # Get top 3 stack frames
        stack_sig = []
        for frame in tb[-3:]:
            stack_sig.append(f"{frame.filename}:{frame.lineno}:{frame.name}")
        
        # Create fingerprint
        fingerprint_data = {
            'type': type(error).__name__,
            'message_pattern': self._extract_message_pattern(str(error)),
            'stack': '|'.join(stack_sig)
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def _extract_message_pattern(self, message: str) -> str:
        """Extract pattern from error message, removing variable parts"""
        import re
        
        # Remove numbers
        pattern = re.sub(r'\d+', 'N', message)
        
        # Remove quoted strings
        pattern = re.sub(r'"[^"]*"', '"..."', pattern)
        pattern = re.sub(r"'[^']*'", "'...'", pattern)
        
        # Remove hex values
        pattern = re.sub(r'0x[0-9a-fA-F]+', '0x...', pattern)
        
        return pattern[:100]  # Limit length
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        import uuid
        return f"err_{uuid.uuid4().hex[:12]}"
    
    def _trigger_alert(self, error: TrackedError, count: int):
        """Trigger alert for repeated errors"""
        alert_data = {
            'error': error.to_dict(),
            'occurrence_count': count,
            'first_seen': self.error_fingerprints[error.fingerprint][0].timestamp.isoformat(),
            'alert_time': datetime.utcnow().isoformat()
        }
        
        # Log alert
        logger.critical(f"Error alert triggered: {error.error_type}", extra=alert_data)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback):
        """Add callback for error alerts"""
        self.alert_callbacks.append(callback)
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            recent_errors = [e for e in self.errors if e.timestamp >= cutoff]
        
        # Group by category
        by_category = defaultdict(list)
        for error in recent_errors:
            by_category[error.category.value].append(error)
        
        # Group by severity
        by_severity = defaultdict(list)
        for error in recent_errors:
            by_severity[error.severity.value].append(error)
        
        # Top error types
        error_types = Counter(e.error_type for e in recent_errors)
        
        return {
            'time_range': {
                'start': cutoff.isoformat(),
                'end': datetime.utcnow().isoformat(),
                'hours': hours
            },
            'total_errors': len(recent_errors),
            'by_category': {cat: len(errors) for cat, errors in by_category.items()},
            'by_severity': {sev: len(errors) for sev, errors in by_severity.items()},
            'top_error_types': dict(error_types.most_common(10)),
            'top_fingerprints': [
                {
                    'fingerprint': fp,
                    'count': count,
                    'sample': self.error_fingerprints[fp][-1].to_dict()
                }
                for fp, count in self.error_counts.most_common(5)
            ]
        }
    
    def get_error_details(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific error"""
        with self.lock:
            for error in self.errors:
                if error.error_id == error_id:
                    return error.to_dict()
        return None
    
    def get_errors_by_fingerprint(self, fingerprint: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get errors with the same fingerprint"""
        with self.lock:
            errors = self.error_fingerprints.get(fingerprint, [])
            return [e.to_dict() for e in errors[-limit:]]
    
    def purge_expired_errors(self, days: int = 7):
        """Purge errors older than N days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self.lock:
            # Filter errors
            self.errors = [e for e in self.errors if e.timestamp >= cutoff]
            
            # Update fingerprints
            for fingerprint in list(self.error_fingerprints.keys()):
                self.error_fingerprints[fingerprint] = [
                    e for e in self.error_fingerprints[fingerprint]
                    if e.timestamp >= cutoff
                ]
                
                # Remove empty fingerprints
                if not self.error_fingerprints[fingerprint]:
                    del self.error_fingerprints[fingerprint]
                    del self.error_counts[fingerprint]
    
    def export_errors(self, filepath: str, format: str = "json"):
        """Export errors to file"""
        with self.lock:
            errors_data = [e.to_dict() for e in self.errors]
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(errors_data, f, indent=2)
        elif format == "csv":
            import csv
            if errors_data:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=errors_data[0].keys())
                    writer.writeheader()
                    writer.writerows(errors_data)
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        import threading
        
        def cleanup_task():
            while True:
                try:
                    # Purge expired errors every hour
                    self.purge_expired_errors(days=7)
                    time.sleep(3600)
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

# Global error tracker instance
_error_tracker = None

def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance"""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker

def track_error(error: Exception, **kwargs) -> str:
    """Track an error with the global tracker"""
    return get_error_tracker().track_error(error, **kwargs)

# Decorator for automatic error tracking
def track_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                category: Optional[ErrorCategory] = None):
    """Decorator to automatically track function errors"""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                track_error(e, severity=severity, category=category)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                track_error(e, severity=severity, category=category)
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

import time