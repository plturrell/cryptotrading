"""
Comprehensive diagnostic logger for rex.com
Captures frontend, middleware, and backend events with structured logging
"""

import logging
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os


class DiagnosticLogger:
    """
    Multi-layer diagnostic logger with structured output
    Captures frontend JS errors, API calls, backend operations, and database queries
    """
    
    def __init__(self, log_dir: str = "logs/diagnostics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup structured logging
        self.setup_loggers()
        
        # Performance tracking
        self.request_times = {}
        self.error_counts = {}
        
    def setup_loggers(self):
        """Setup different loggers for different components"""
        
        # Main application logger
        self.app_logger = self._create_logger('app', 'app.log')
        
        # API request/response logger
        self.api_logger = self._create_logger('api', 'api.log')
        
        # Frontend error logger
        self.frontend_logger = self._create_logger('frontend', 'frontend.log')
        
        # Database operation logger
        self.db_logger = self._create_logger('database', 'database.log')
        
        # Yahoo Finance specific logger
        self.yahoo_logger = self._create_logger('yahoo_finance', 'yahoo_finance.log')
        
        # Performance logger
        self.perf_logger = self._create_logger('performance', 'performance.log')
        
        # Error aggregation logger
        self.error_logger = self._create_logger('errors', 'errors.log')
        
    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """Create a structured logger with JSON formatting"""
        logger = logging.getLogger(f'rex.{name}')
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler with JSON formatting
        file_handler = logging.FileHandler(self.log_dir / filename)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # JSON formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_api_request(self, method: str, endpoint: str, params: Dict = None, 
                       headers: Dict = None, request_id: str = None):
        """Log incoming API request"""
        log_data = {
            'event': 'api_request',
            'method': method,
            'endpoint': endpoint,
            'params': params or {},
            'headers': {k: v for k, v in (headers or {}).items() if k.lower() not in ['authorization', 'cookie']},
            'request_id': request_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.api_logger.info(json.dumps(log_data))
        
        # Start performance tracking
        if request_id:
            self.request_times[request_id] = datetime.now()
    
    def log_api_response(self, endpoint: str, status_code: int, response_size: int = None,
                        error: str = None, request_id: str = None):
        """Log API response"""
        duration = None
        if request_id and request_id in self.request_times:
            duration = (datetime.now() - self.request_times[request_id]).total_seconds()
            del self.request_times[request_id]
        
        log_data = {
            'event': 'api_response',
            'endpoint': endpoint,
            'status_code': status_code,
            'response_size': response_size,
            'duration_seconds': duration,
            'error': error,
            'request_id': request_id,
            'timestamp': datetime.now().isoformat()
        }
        
        if status_code >= 400:
            self.api_logger.error(json.dumps(log_data))
            self._track_error(endpoint, error or f"HTTP {status_code}")
        else:
            self.api_logger.info(json.dumps(log_data))
        
        # Log performance metrics
        if duration:
            self.log_performance('api_response_time', duration, {'endpoint': endpoint})
    
    def log_yahoo_finance_operation(self, operation: str, symbol: str, 
                                   success: bool, data: Dict = None, error: str = None):
        """Log Yahoo Finance specific operations"""
        log_data = {
            'event': 'yahoo_finance_operation',
            'operation': operation,
            'symbol': symbol,
            'success': success,
            'data_size': len(str(data)) if data else 0,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            self.yahoo_logger.info(json.dumps(log_data))
        else:
            self.yahoo_logger.error(json.dumps(log_data))
            self._track_error('yahoo_finance', error or 'Unknown error')
    
    def log_frontend_error(self, error_type: str, message: str, stack_trace: str = None,
                          url: str = None, user_agent: str = None):
        """Log frontend JavaScript errors"""
        log_data = {
            'event': 'frontend_error',
            'error_type': error_type,
            'message': message,
            'stack_trace': stack_trace,
            'url': url,
            'user_agent': user_agent,
            'timestamp': datetime.now().isoformat()
        }
        
        self.frontend_logger.error(json.dumps(log_data))
        self._track_error('frontend', f"{error_type}: {message}")
    
    def log_database_operation(self, operation: str, table: str = None, 
                              success: bool = True, error: str = None, 
                              duration: float = None):
        """Log database operations"""
        log_data = {
            'event': 'database_operation',
            'operation': operation,
            'table': table,
            'success': success,
            'duration_seconds': duration,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            self.db_logger.info(json.dumps(log_data))
        else:
            self.db_logger.error(json.dumps(log_data))
            self._track_error('database', error or 'Database operation failed')
    
    def log_performance(self, metric_name: str, value: float, tags: Dict = None):
        """Log performance metrics"""
        log_data = {
            'event': 'performance_metric',
            'metric': metric_name,
            'value': value,
            'tags': tags or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.perf_logger.info(json.dumps(log_data))
    
    def log_exception(self, exception: Exception, context: str = None, 
                     additional_data: Dict = None):
        """Log exceptions with full context"""
        log_data = {
            'event': 'exception',
            'exception_type': type(exception).__name__,
            'message': str(exception),
            'traceback': traceback.format_exc(),
            'context': context,
            'additional_data': additional_data or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.error_logger.error(json.dumps(log_data))
        self._track_error('exception', f"{type(exception).__name__}: {str(exception)}")
    
    def _track_error(self, component: str, error: str):
        """Track error frequency for analysis"""
        key = f"{component}:{error}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors for analysis"""
        return dict(self.error_counts)
    
    def log_system_state(self):
        """Log current system state for diagnostics"""
        import psutil
        
        try:
            log_data = {
                'event': 'system_state',
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'active_connections': len(psutil.net_connections()),
                'python_version': sys.version,
                'timestamp': datetime.now().isoformat()
            }
            
            self.app_logger.info(json.dumps(log_data))
        except Exception as e:
            self.log_exception(e, "Failed to log system state")


# Global diagnostic logger instance
diagnostic_logger = DiagnosticLogger()
