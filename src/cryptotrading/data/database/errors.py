"""
Standardized Database Error Handling and Messaging
Provides consistent error messages and handling across the database layer
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """Standardized error codes for database operations"""
    
    # Connection errors
    CONNECTION_FAILED = "DB_CONN_001"
    CONNECTION_TIMEOUT = "DB_CONN_002"
    CONNECTION_LOST = "DB_CONN_003"
    
    # Validation errors
    VALIDATION_FAILED = "DB_VAL_001"
    CONSTRAINT_VIOLATION = "DB_VAL_002"
    FOREIGN_KEY_VIOLATION = "DB_VAL_003"
    UNIQUE_CONSTRAINT = "DB_VAL_004"
    INVALID_DATA_TYPE = "DB_VAL_005"
    REQUIRED_FIELD_MISSING = "DB_VAL_006"
    
    # Query errors
    QUERY_SYNTAX_ERROR = "DB_QRY_001"
    QUERY_EXECUTION_ERROR = "DB_QRY_002"
    QUERY_TIMEOUT = "DB_QRY_003"
    TABLE_NOT_FOUND = "DB_QRY_004"
    COLUMN_NOT_FOUND = "DB_QRY_005"
    
    # Transaction errors
    TRANSACTION_FAILED = "DB_TXN_001"
    DEADLOCK_DETECTED = "DB_TXN_002"
    TRANSACTION_TIMEOUT = "DB_TXN_003"
    
    # Migration errors
    MIGRATION_FAILED = "DB_MIG_001"
    MIGRATION_CONFLICT = "DB_MIG_002"
    SCHEMA_MISMATCH = "DB_MIG_003"
    
    # Performance errors
    SLOW_QUERY_DETECTED = "DB_PERF_001"
    CONNECTION_POOL_EXHAUSTED = "DB_PERF_002"
    MEMORY_LIMIT_EXCEEDED = "DB_PERF_003"
    
    # Security errors
    SQL_INJECTION_ATTEMPT = "DB_SEC_001"
    UNAUTHORIZED_ACCESS = "DB_SEC_002"
    INVALID_CREDENTIALS = "DB_SEC_003"
    
    # General errors
    UNKNOWN_ERROR = "DB_GEN_001"
    CONFIGURATION_ERROR = "DB_GEN_002"
    INTERNAL_ERROR = "DB_GEN_003"

class DatabaseError(Exception):
    """Base class for all database errors"""
    
    def __init__(self, 
                 message: str,
                 error_code: ErrorCode,
                 details: Optional[Dict[str, Any]] = None,
                 original_error: Optional[Exception] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_error = original_error
        
        # Build comprehensive error message
        full_message = f"[{error_code.value}] {message}"
        if details:
            detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            full_message += f" ({detail_str})"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization"""
        return {
            'error_code': self.error_code.value,
            'message': self.message,
            'details': self.details,
            'original_error': str(self.original_error) if self.original_error else None
        }

class DatabaseConnectionError(DatabaseError):
    """Database connection related errors"""
    pass

class DatabaseValidationError(DatabaseError):
    """Data validation related errors"""
    pass

class DatabaseQueryError(DatabaseError):
    """Query execution related errors"""
    pass

class DatabaseTransactionError(DatabaseError):
    """Transaction related errors"""
    pass

class DatabaseMigrationError(DatabaseError):
    """Migration related errors"""
    pass

class DatabasePerformanceError(DatabaseError):
    """Performance related errors"""
    pass

class DatabaseSecurityError(DatabaseError):
    """Security related errors"""
    pass

class ErrorMessageBuilder:
    """Builds standardized error messages"""
    
    # Error message templates
    MESSAGES = {
        # Connection messages
        ErrorCode.CONNECTION_FAILED: "Failed to connect to database",
        ErrorCode.CONNECTION_TIMEOUT: "Database connection timed out",
        ErrorCode.CONNECTION_LOST: "Database connection was lost",
        
        # Validation messages
        ErrorCode.VALIDATION_FAILED: "Data validation failed",
        ErrorCode.CONSTRAINT_VIOLATION: "Database constraint violated",
        ErrorCode.FOREIGN_KEY_VIOLATION: "Foreign key constraint violated",
        ErrorCode.UNIQUE_CONSTRAINT: "Unique constraint violated",
        ErrorCode.INVALID_DATA_TYPE: "Invalid data type provided",
        ErrorCode.REQUIRED_FIELD_MISSING: "Required field is missing",
        
        # Query messages
        ErrorCode.QUERY_SYNTAX_ERROR: "SQL query syntax error",
        ErrorCode.QUERY_EXECUTION_ERROR: "Query execution failed",
        ErrorCode.QUERY_TIMEOUT: "Query execution timed out",
        ErrorCode.TABLE_NOT_FOUND: "Table does not exist",
        ErrorCode.COLUMN_NOT_FOUND: "Column does not exist",
        
        # Transaction messages
        ErrorCode.TRANSACTION_FAILED: "Transaction failed",
        ErrorCode.DEADLOCK_DETECTED: "Database deadlock detected",
        ErrorCode.TRANSACTION_TIMEOUT: "Transaction timed out",
        
        # Migration messages
        ErrorCode.MIGRATION_FAILED: "Database migration failed",
        ErrorCode.MIGRATION_CONFLICT: "Migration conflict detected",
        ErrorCode.SCHEMA_MISMATCH: "Schema version mismatch",
        
        # Performance messages
        ErrorCode.SLOW_QUERY_DETECTED: "Slow query detected",
        ErrorCode.CONNECTION_POOL_EXHAUSTED: "Connection pool exhausted",
        ErrorCode.MEMORY_LIMIT_EXCEEDED: "Memory limit exceeded",
        
        # Security messages
        ErrorCode.SQL_INJECTION_ATTEMPT: "Potential SQL injection attempt detected",
        ErrorCode.UNAUTHORIZED_ACCESS: "Unauthorized database access attempt",
        ErrorCode.INVALID_CREDENTIALS: "Invalid database credentials",
        
        # General messages
        ErrorCode.UNKNOWN_ERROR: "An unknown database error occurred",
        ErrorCode.CONFIGURATION_ERROR: "Database configuration error",
        ErrorCode.INTERNAL_ERROR: "Internal database error"
    }
    
    @classmethod
    def build_message(cls, error_code: ErrorCode, 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Build error message with context"""
        base_message = cls.MESSAGES.get(error_code, "Unknown error")
        
        if context:
            # Add context-specific information
            if error_code == ErrorCode.VALIDATION_FAILED and 'field' in context:
                base_message += f" for field '{context['field']}'"
            elif error_code == ErrorCode.UNIQUE_CONSTRAINT and 'field' in context:
                base_message += f" on field '{context['field']}'"
            elif error_code == ErrorCode.TABLE_NOT_FOUND and 'table' in context:
                base_message += f": '{context['table']}'"
            elif error_code == ErrorCode.FOREIGN_KEY_VIOLATION and 'table' in context:
                base_message += f" in table '{context['table']}'"
        
        return base_message

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.error_stats = {}
    
    def handle_error(self, error: Exception, 
                    operation: str = "unknown",
                    context: Optional[Dict[str, Any]] = None) -> DatabaseError:
        """Handle and standardize errors"""
        
        # Determine error type and code
        error_code, db_error_class = self._classify_error(error)
        
        # Build standardized message
        message = ErrorMessageBuilder.build_message(error_code, context)
        
        # Create standardized error
        db_error = db_error_class(
            message=message,
            error_code=error_code,
            details=context or {},
            original_error=error
        )
        
        # Log error
        self._log_error(db_error, operation)
        
        # Update statistics
        self._update_error_stats(error_code, operation)
        
        return db_error
    
    def _classify_error(self, error: Exception) -> tuple:
        """Classify error and determine appropriate code and class"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Connection errors
        if any(keyword in error_str for keyword in ['connection', 'connect', 'timeout']):
            if 'timeout' in error_str:
                return ErrorCode.CONNECTION_TIMEOUT, DatabaseConnectionError
            elif 'lost' in error_str or 'closed' in error_str:
                return ErrorCode.CONNECTION_LOST, DatabaseConnectionError
            else:
                return ErrorCode.CONNECTION_FAILED, DatabaseConnectionError
        
        # Validation errors
        if any(keyword in error_str for keyword in ['validation', 'constraint', 'unique', 'foreign key']):
            if 'unique' in error_str:
                return ErrorCode.UNIQUE_CONSTRAINT, DatabaseValidationError
            elif 'foreign key' in error_str:
                return ErrorCode.FOREIGN_KEY_VIOLATION, DatabaseValidationError
            elif 'constraint' in error_str:
                return ErrorCode.CONSTRAINT_VIOLATION, DatabaseValidationError
            else:
                return ErrorCode.VALIDATION_FAILED, DatabaseValidationError
        
        # Query errors
        if any(keyword in error_str for keyword in ['syntax', 'table', 'column', 'sql']):
            if 'syntax' in error_str:
                return ErrorCode.QUERY_SYNTAX_ERROR, DatabaseQueryError
            elif 'table' in error_str and 'not' in error_str:
                return ErrorCode.TABLE_NOT_FOUND, DatabaseQueryError
            elif 'column' in error_str and 'not' in error_str:
                return ErrorCode.COLUMN_NOT_FOUND, DatabaseQueryError
            else:
                return ErrorCode.QUERY_EXECUTION_ERROR, DatabaseQueryError
        
        # Transaction errors
        if any(keyword in error_str for keyword in ['deadlock', 'transaction', 'rollback']):
            if 'deadlock' in error_str:
                return ErrorCode.DEADLOCK_DETECTED, DatabaseTransactionError
            else:
                return ErrorCode.TRANSACTION_FAILED, DatabaseTransactionError
        
        # Performance errors
        if any(keyword in error_str for keyword in ['pool', 'memory', 'slow']):
            if 'pool' in error_str:
                return ErrorCode.CONNECTION_POOL_EXHAUSTED, DatabasePerformanceError
            elif 'memory' in error_str:
                return ErrorCode.MEMORY_LIMIT_EXCEEDED, DatabasePerformanceError
            else:
                return ErrorCode.SLOW_QUERY_DETECTED, DatabasePerformanceError
        
        # Security errors
        if any(keyword in error_str for keyword in ['injection', 'unauthorized', 'credentials']):
            if 'injection' in error_str:
                return ErrorCode.SQL_INJECTION_ATTEMPT, DatabaseSecurityError
            elif 'unauthorized' in error_str:
                return ErrorCode.UNAUTHORIZED_ACCESS, DatabaseSecurityError
            else:
                return ErrorCode.INVALID_CREDENTIALS, DatabaseSecurityError
        
        # Default to unknown error
        return ErrorCode.UNKNOWN_ERROR, DatabaseError
    
    def _log_error(self, error: DatabaseError, operation: str):
        """Log error with appropriate level"""
        log_message = f"Database error in operation '{operation}': {error.message}"
        
        # Determine log level based on error type
        if error.error_code in [ErrorCode.CONNECTION_FAILED, ErrorCode.MIGRATION_FAILED]:
            logger.error(log_message, extra={'error_details': error.details})
        elif error.error_code in [ErrorCode.SLOW_QUERY_DETECTED, ErrorCode.CONNECTION_TIMEOUT]:
            logger.warning(log_message, extra={'error_details': error.details})
        elif error.error_code in [ErrorCode.SQL_INJECTION_ATTEMPT, ErrorCode.UNAUTHORIZED_ACCESS]:
            logger.critical(log_message, extra={'error_details': error.details})
        else:
            logger.info(log_message, extra={'error_details': error.details})
    
    def _update_error_stats(self, error_code: ErrorCode, operation: str):
        """Update error statistics"""
        key = f"{operation}:{error_code.value}"
        if key not in self.error_stats:
            self.error_stats[key] = {
                'count': 0,
                'first_seen': logger.time(),
                'last_seen': logger.time()
            }
        
        self.error_stats[key]['count'] += 1
        self.error_stats[key]['last_seen'] = logger.time()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'total_errors': sum(stat['count'] for stat in self.error_stats.values()),
            'unique_error_types': len(self.error_stats),
            'error_breakdown': self.error_stats.copy()
        }

# Global error handler instance
error_handler = ErrorHandler()

def handle_database_error(operation: str = "unknown"):
    """Decorator to handle database errors consistently"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DatabaseError:
                # Already a standardized error, re-raise
                raise
            except Exception as e:
                # Convert to standardized error
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                standardized_error = error_handler.handle_error(e, operation, context)
                raise standardized_error
        return wrapper
    return decorator

def validate_sql_operation(operation_type: str, table_name: str = None):
    """Validate SQL operation is allowed"""
    allowed_operations = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
    
    if operation_type.upper() not in allowed_operations:
        raise DatabaseSecurityError(
            message=f"SQL operation '{operation_type}' is not allowed",
            error_code=ErrorCode.SQL_INJECTION_ATTEMPT,
            details={'operation': operation_type, 'table': table_name}
        )

def format_user_friendly_error(error: DatabaseError) -> str:
    """Format error message for end users (removes technical details)"""
    user_friendly_messages = {
        ErrorCode.CONNECTION_FAILED: "Unable to connect to the database. Please try again later.",
        ErrorCode.CONNECTION_TIMEOUT: "The request took too long. Please try again.",
        ErrorCode.VALIDATION_FAILED: "The information provided is not valid. Please check and try again.",
        ErrorCode.UNIQUE_CONSTRAINT: "This information already exists. Please use different values.",
        ErrorCode.FOREIGN_KEY_VIOLATION: "This operation would violate data relationships.",
        ErrorCode.QUERY_TIMEOUT: "The operation took too long. Please try again.",
        ErrorCode.MIGRATION_FAILED: "System maintenance is in progress. Please try again later.",
        ErrorCode.UNAUTHORIZED_ACCESS: "You don't have permission to perform this operation.",
    }
    
    return user_friendly_messages.get(
        error.error_code, 
        "An unexpected error occurred. Please contact support if this continues."
    )

# Exception types for backwards compatibility
ValidationError = DatabaseValidationError