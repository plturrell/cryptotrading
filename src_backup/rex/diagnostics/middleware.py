"""
Flask middleware integration for rex.com diagnostics
Automatically captures API requests, responses, and errors
"""

import time
import uuid
from functools import wraps
from flask import request, g
from .logger import diagnostic_logger
from .tracer import request_tracer


class DiagnosticMiddleware:
    """
    Flask middleware that automatically captures diagnostic information
    """
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_appcontext(self.teardown_request)
        
        # Register error handlers
        app.errorhandler(Exception)(self.handle_exception)
    
    def before_request(self):
        """Capture request start"""
        g.request_id = str(uuid.uuid4())
        g.start_time = time.time()
        
        # Start trace
        g.trace_id = request_tracer.start_trace(
            f"{request.method} {request.endpoint or request.path}",
            {
                'method': request.method,
                'path': request.path,
                'endpoint': request.endpoint,
                'remote_addr': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', '')
            }
        )
        
        # Log request
        diagnostic_logger.log_api_request(
            method=request.method,
            endpoint=request.path,
            params=dict(request.args),
            headers=dict(request.headers),
            request_id=g.request_id
        )
    
    def after_request(self, response):
        """Capture request completion"""
        if hasattr(g, 'request_id'):
            # Calculate response size
            response_size = len(response.get_data()) if response.get_data() else 0
            
            # Log response
            diagnostic_logger.log_api_response(
                endpoint=request.path,
                status_code=response.status_code,
                response_size=response_size,
                request_id=g.request_id
            )
            
            # End trace
            if hasattr(g, 'trace_id'):
                status = 'success' if response.status_code < 400 else 'error'
                request_tracer.end_trace(g.trace_id, status)
        
        return response
    
    def teardown_request(self, exception):
        """Handle request teardown"""
        if exception:
            self.handle_exception(exception)
    
    def handle_exception(self, exception):
        """Handle exceptions during request processing"""
        if hasattr(g, 'request_id'):
            diagnostic_logger.log_exception(
                exception,
                context=f"Request {request.method} {request.path}",
                additional_data={
                    'request_id': g.request_id,
                    'endpoint': request.path,
                    'method': request.method
                }
            )
            
            # End trace with error
            if hasattr(g, 'trace_id'):
                request_tracer.end_trace(g.trace_id, 'error', str(exception))
        
        return exception


def trace_yahoo_finance_operation(operation_name: str):
    """Decorator to trace Yahoo Finance operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error = None
            result = None
            
            # Extract symbol if available
            symbol = kwargs.get('symbol') or (args[1] if len(args) > 1 else 'unknown')
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                # Log operation
                diagnostic_logger.log_yahoo_finance_operation(
                    operation=operation_name,
                    symbol=symbol,
                    success=success,
                    data=result if success else None,
                    error=error
                )
                
                # Add trace span if we're in a trace
                if hasattr(g, 'trace_id'):
                    end_time = time.time()
                    request_tracer.add_span(
                        g.trace_id,
                        f"yahoo_finance_{operation_name}",
                        start_time,
                        end_time,
                        {
                            'symbol': symbol,
                            'success': success,
                            'error': error
                        }
                    )
        
        return wrapper
    return decorator


# Global middleware instance
diagnostic_middleware = DiagnosticMiddleware()
