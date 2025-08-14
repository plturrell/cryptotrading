"""
Observability Integration for A2A Agents
Integrates tracing, logging, and metrics with the A2A agent system
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps

from .tracer import get_tracer, trace_function
from .logger import get_logger, log_function_call  
from .error_tracker import get_error_tracker, track_errors, ErrorSeverity, ErrorCategory
from .metrics import get_metrics, get_business_metrics
from .context import (
    TraceContext, A2AContextEnhancer, WorkflowContext, 
    get_current_trace, with_trace_context
)

logger = get_logger(__name__)
tracer = get_tracer()
metrics = get_metrics()
business_metrics = get_business_metrics()
error_tracker = get_error_tracker()

class ObservableA2AAgent:
    """Mixin class to add observability to A2A agents"""
    
    def __init__(self, agent_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.logger = get_logger(f"agent.{agent_id}")
        self.metrics = get_metrics()
        
    def _wrap_tool_with_observability(self, tool_func: Callable, tool_name: str) -> Callable:
        """Wrap agent tool with observability"""
        
        @wraps(tool_func)
        def sync_wrapper(*args, **kwargs):
            # Create child context for tool execution
            current_ctx = get_current_trace()
            if current_ctx:
                tool_ctx = current_ctx.create_child_context(
                    operation=f"tool:{tool_name}",
                    agent_id=self.agent_id
                )
            else:
                from .context import create_trace_context
                tool_ctx = create_trace_context(
                    operation=f"tool:{tool_name}",
                    agent_id=self.agent_id
                )
            
            with with_trace_context(tool_ctx):
                start_time = time.time()
                success = False
                
                try:
                    self.logger.info(f"Executing tool: {tool_name}")
                    result = tool_func(*args, **kwargs)
                    success = True
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Tool execution failed: {tool_name}", error=e)
                    error_tracker.track_error(
                        e, 
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.BUSINESS_LOGIC_ERROR,
                        context=error_tracker._get_current_context()
                    )
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    business_metrics.track_agent_operation(
                        agent_id=self.agent_id,
                        operation=tool_name,
                        success=success,
                        duration_ms=duration_ms
                    )
        
        @wraps(tool_func)
        async def async_wrapper(*args, **kwargs):
            # Create child context for tool execution
            current_ctx = get_current_trace()
            if current_ctx:
                tool_ctx = current_ctx.create_child_context(
                    operation=f"tool:{tool_name}",
                    agent_id=self.agent_id
                )
            else:
                from .context import create_trace_context
                tool_ctx = create_trace_context(
                    operation=f"tool:{tool_name}",
                    agent_id=self.agent_id
                )
            
            with with_trace_context(tool_ctx):
                start_time = time.time()
                success = False
                
                try:
                    self.logger.info(f"Executing async tool: {tool_name}")
                    result = await tool_func(*args, **kwargs)
                    success = True
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Async tool execution failed: {tool_name}", error=e)
                    error_tracker.track_error(
                        e,
                        severity=ErrorSeverity.MEDIUM, 
                        category=ErrorCategory.BUSINESS_LOGIC_ERROR,
                        context=error_tracker._get_current_context()
                    )
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    business_metrics.track_agent_operation(
                        agent_id=self.agent_id,
                        operation=tool_name,
                        success=success,
                        duration_ms=duration_ms
                    )
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(tool_func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def process_a2a_message_with_observability(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process A2A message with full observability"""
        
        # Extract or create trace context
        context = A2AContextEnhancer.extract_context(message)
        if not context:
            from .context import create_trace_context
            context = create_trace_context(
                operation=f"process_message:{message.get('type', 'unknown')}",
                agent_id=self.agent_id,
                metadata={'message_type': message.get('type')}
            )
        else:
            context = context.create_child_context(
                operation=f"process_message:{message.get('type', 'unknown')}",
                agent_id=self.agent_id
            )
        
        with with_trace_context(context):
            start_time = time.time()
            success = False
            
            try:
                self.logger.info(f"Processing A2A message", extra={
                    'message_type': message.get('type'),
                    'sender_id': message.get('sender_id'),
                    'message_id': message.get('message_id')
                })
                
                # Process the message (implement in subclass)
                result = self._process_message_impl(message)
                
                # Enhance response with trace context
                if isinstance(result, dict):
                    result = A2AContextEnhancer.enhance_message(result, context)
                
                success = True
                return result
                
            except Exception as e:
                self.logger.error(f"A2A message processing failed", error=e, extra={
                    'message_type': message.get('type'),
                    'sender_id': message.get('sender_id')
                })
                
                error_tracker.track_error(
                    e,
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.INTEGRATION_ERROR,
                    context=error_tracker._get_current_context()
                )
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_agent_operation(
                    agent_id=self.agent_id,
                    operation=f"process_message:{message.get('type', 'unknown')}",
                    success=success,
                    duration_ms=duration_ms
                )
    
    def _process_message_impl(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Implement message processing logic in subclass"""
        raise NotImplementedError("Subclass must implement _process_message_impl")

class ObservableWorkflow:
    """Observable workflow executor for multi-agent operations"""
    
    def __init__(self, workflow_name: str, user_id: Optional[int] = None):
        self.workflow_context = WorkflowContext(workflow_name, user_id)
        self.logger = get_logger(f"workflow.{workflow_name}")
        
    async def execute_step(self, step_name: str, agent_id: str, 
                          operation: Callable, *args, **kwargs) -> Any:
        """Execute workflow step with observability"""
        step_context = self.workflow_context.create_step_context(step_name, agent_id)
        
        with with_trace_context(step_context):
            start_time = time.time()
            success = False
            
            try:
                self.logger.info(f"Starting workflow step: {step_name}", extra={
                    'workflow_id': self.workflow_context.workflow_id,
                    'step_name': step_name,
                    'agent_id': agent_id
                })
                
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                success = True
                self.logger.info(f"Completed workflow step: {step_name}", extra={
                    'workflow_id': self.workflow_context.workflow_id,
                    'step_name': step_name,
                    'success': True
                })
                
                return result
                
            except Exception as e:
                self.logger.error(f"Workflow step failed: {step_name}", error=e, extra={
                    'workflow_id': self.workflow_context.workflow_id,
                    'step_name': step_name,
                    'agent_id': agent_id
                })
                
                error_tracker.track_error(
                    e,
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS_LOGIC_ERROR,
                    context=error_tracker._get_current_context()
                )
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_agent_operation(
                    agent_id=agent_id,
                    operation=f"workflow_step:{step_name}",
                    success=success,
                    duration_ms=duration_ms
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary"""
        return self.workflow_context.get_workflow_summary()

# Decorators for easy integration
def observable_agent_method(agent_id: str, operation_name: Optional[str] = None):
    """Decorator to add observability to agent methods"""
    def decorator(func):
        method_name = operation_name or func.__name__
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Create trace context
            current_ctx = get_current_trace()
            if current_ctx:
                ctx = current_ctx.create_child_context(
                    operation=f"{agent_id}:{method_name}"
                )
            else:
                from .context import create_trace_context
                ctx = create_trace_context(
                    operation=f"{agent_id}:{method_name}",
                    agent_id=agent_id
                )
            
            with with_trace_context(ctx):
                start_time = time.time()
                success = False
                
                try:
                    result = func(self, *args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    error_tracker.track_error(e, severity=ErrorSeverity.MEDIUM)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    business_metrics.track_agent_operation(
                        agent_id=agent_id,
                        operation=method_name,
                        success=success,
                        duration_ms=duration_ms
                    )
        
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Create trace context
            current_ctx = get_current_trace()
            if current_ctx:
                ctx = current_ctx.create_child_context(
                    operation=f"{agent_id}:{method_name}"
                )
            else:
                from .context import create_trace_context
                ctx = create_trace_context(
                    operation=f"{agent_id}:{method_name}",
                    agent_id=agent_id
                )
            
            with with_trace_context(ctx):
                start_time = time.time()
                success = False
                
                try:
                    result = await func(self, *args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    error_tracker.track_error(e, severity=ErrorSeverity.MEDIUM)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    business_metrics.track_agent_operation(
                        agent_id=agent_id,
                        operation=method_name,
                        success=success,
                        duration_ms=duration_ms
                    )
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def observable_api_endpoint(endpoint_path: str):
    """Decorator to add observability to API endpoints"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request
            
            start_time = time.time()
            status_code = 200
            
            try:
                # Create trace context for API request
                from .context import create_trace_context
                ctx = create_trace_context(
                    operation=f"api:{request.method}:{endpoint_path}",
                    metadata={
                        'endpoint': endpoint_path,
                        'method': request.method,
                        'user_agent': request.headers.get('User-Agent', ''),
                        'remote_addr': request.remote_addr
                    }
                )
                
                with with_trace_context(ctx):
                    result = func(*args, **kwargs)
                    return result
                    
            except Exception as e:
                status_code = getattr(e, 'code', 500)
                error_tracker.track_error(
                    e,
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.API_ERROR
                )
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                business_metrics.track_api_request(
                    endpoint=endpoint_path,
                    method=request.method,
                    status_code=status_code,
                    duration_ms=duration_ms
                )
        
        return wrapper
    return decorator

# Health check endpoint for observability
def get_observability_health() -> Dict[str, Any]:
    """Get health status of observability components"""
    return {
        'tracer': {
            'status': 'healthy',
            'service_name': tracer.service_name
        },
        'error_tracker': {
            'status': 'healthy',
            'total_errors_tracked': len(error_tracker.errors)
        },
        'metrics': {
            'status': 'healthy',
            'total_metrics': len(metrics.metrics)
        },
        'timestamp': time.time()
    }