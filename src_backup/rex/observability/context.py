"""
Trace Context Management for A2A Communication
Provides context propagation across distributed agents
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import contextvars
from dataclasses import dataclass, asdict

# Context variables for thread-safe context propagation
current_trace_context = contextvars.ContextVar('trace_context', default=None)

@dataclass
class TraceContext:
    """Trace context information"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    user_id: Optional[int] = None
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: Optional[str] = None
    operation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceContext':
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def create_child_context(self, operation: Optional[str] = None, 
                           agent_id: Optional[str] = None) -> 'TraceContext':
        """Create child context for sub-operations"""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            parent_span_id=self.span_id,
            user_id=self.user_id,
            agent_id=agent_id or self.agent_id,
            workflow_id=self.workflow_id,
            request_id=self.request_id,
            service_name=self.service_name,
            operation=operation,
            metadata=self.metadata.copy() if self.metadata else {}
        )

def generate_trace_id() -> str:
    """Generate unique trace ID"""
    return f"trace_{uuid.uuid4().hex[:16]}"

def generate_span_id() -> str:
    """Generate unique span ID"""
    return f"span_{uuid.uuid4().hex[:8]}"

def create_trace_context(operation: str, agent_id: Optional[str] = None,
                        user_id: Optional[int] = None, 
                        service_name: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> TraceContext:
    """Create new trace context"""
    return TraceContext(
        trace_id=generate_trace_id(),
        span_id=generate_span_id(),
        user_id=user_id,
        agent_id=agent_id,
        service_name=service_name or "rex-trading",
        operation=operation,
        metadata=metadata or {}
    )

def get_current_trace() -> Optional[TraceContext]:
    """Get current trace context"""
    return current_trace_context.get()

def set_trace_context(context: TraceContext) -> contextvars.Token:
    """Set trace context for current execution"""
    return current_trace_context.set(context)

def clear_trace_context():
    """Clear current trace context"""
    current_trace_context.set(None)

class TraceContextManager:
    """Context manager for trace context"""
    
    def __init__(self, context: TraceContext):
        self.context = context
        self.token = None
    
    def __enter__(self) -> TraceContext:
        self.token = set_trace_context(self.context)
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            current_trace_context.reset(self.token)

def with_trace_context(context: TraceContext):
    """Context manager decorator"""
    return TraceContextManager(context)

# A2A Message Context Enhancement
class A2AContextEnhancer:
    """Enhances A2A messages with trace context"""
    
    @staticmethod
    def enhance_message(message: Dict[str, Any], 
                       context: Optional[TraceContext] = None) -> Dict[str, Any]:
        """Add trace context to A2A message"""
        if context is None:
            context = get_current_trace()
        
        if context:
            if 'metadata' not in message:
                message['metadata'] = {}
            
            message['metadata']['trace_context'] = context.to_dict()
            message['metadata']['correlation_id'] = context.trace_id
        
        return message
    
    @staticmethod
    def extract_context(message: Dict[str, Any]) -> Optional[TraceContext]:
        """Extract trace context from A2A message"""
        if ('metadata' in message and 
            'trace_context' in message['metadata']):
            try:
                return TraceContext.from_dict(message['metadata']['trace_context'])
            except Exception:
                return None
        return None
    
    @staticmethod
    def create_child_from_message(message: Dict[str, Any], 
                                 operation: str,
                                 agent_id: str) -> TraceContext:
        """Create child context from incoming A2A message"""
        parent_context = A2AContextEnhancer.extract_context(message)
        
        if parent_context:
            return parent_context.create_child_context(
                operation=operation,
                agent_id=agent_id
            )
        else:
            # Create new trace if no parent context
            return create_trace_context(
                operation=operation,
                agent_id=agent_id,
                metadata={'source': 'a2a_message'}
            )

# Workflow Context Manager
class WorkflowContext:
    """Manages context for multi-step workflows"""
    
    def __init__(self, workflow_name: str, user_id: Optional[int] = None):
        self.workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        self.workflow_name = workflow_name
        self.user_id = user_id
        self.steps = []
        self.root_context = create_trace_context(
            operation=f"workflow:{workflow_name}",
            user_id=user_id,
            metadata={'workflow_id': self.workflow_id}
        )
    
    def create_step_context(self, step_name: str, 
                           agent_id: Optional[str] = None) -> TraceContext:
        """Create context for workflow step"""
        step_context = self.root_context.create_child_context(
            operation=f"step:{step_name}",
            agent_id=agent_id
        )
        step_context.workflow_id = self.workflow_id
        
        self.steps.append({
            'step_name': step_name,
            'trace_id': step_context.trace_id,
            'span_id': step_context.span_id,
            'agent_id': agent_id,
            'started_at': datetime.utcnow().isoformat()
        })
        
        return step_context
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary"""
        return {
            'workflow_id': self.workflow_id,
            'workflow_name': self.workflow_name,
            'user_id': self.user_id,
            'root_trace_id': self.root_context.trace_id,
            'steps': self.steps,
            'total_steps': len(self.steps),
            'started_at': self.root_context.created_at.isoformat()
        }