"""
Request tracer for rex.com
Provides distributed tracing across frontend, middleware, and backend
"""

import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import threading
from .logger import diagnostic_logger


class RequestTracer:
    """
    Distributed request tracer that follows requests through the entire stack
    """
    
    def __init__(self):
        self.active_traces = {}
        self.completed_traces = []
        self.lock = threading.Lock()
        
    def start_trace(self, operation: str, metadata: Dict = None) -> str:
        """Start a new trace"""
        trace_id = str(uuid.uuid4())
        
        with self.lock:
            self.active_traces[trace_id] = {
                'trace_id': trace_id,
                'operation': operation,
                'start_time': time.time(),
                'metadata': metadata or {},
                'spans': [],
                'status': 'active'
            }
        
        diagnostic_logger.app_logger.info(f"TRACE_START: {trace_id} - {operation}")
        return trace_id
    
    def add_span(self, trace_id: str, span_name: str, start_time: float = None, 
                 end_time: float = None, metadata: Dict = None):
        """Add a span to an existing trace"""
        if trace_id not in self.active_traces:
            return
        
        span = {
            'span_name': span_name,
            'start_time': start_time or time.time(),
            'end_time': end_time,
            'duration': (end_time - start_time) if (start_time and end_time) else None,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        with self.lock:
            self.active_traces[trace_id]['spans'].append(span)
        
        diagnostic_logger.app_logger.info(f"TRACE_SPAN: {trace_id} - {span_name}")
    
    def end_trace(self, trace_id: str, status: str = 'success', error: str = None):
        """End a trace"""
        if trace_id not in self.active_traces:
            return
        
        with self.lock:
            trace = self.active_traces[trace_id]
            trace['end_time'] = time.time()
            trace['duration'] = trace['end_time'] - trace['start_time']
            trace['status'] = status
            trace['error'] = error
            
            # Move to completed traces
            self.completed_traces.append(trace)
            del self.active_traces[trace_id]
        
        diagnostic_logger.app_logger.info(f"TRACE_END: {trace_id} - {status} - {trace['duration']:.3f}s")
        
        # Log performance
        diagnostic_logger.log_performance(
            'trace_duration',
            trace['duration'],
            {'operation': trace['operation'], 'status': status}
        )
    
    @contextmanager
    def trace_operation(self, operation: str, metadata: Dict = None):
        """Context manager for tracing operations"""
        trace_id = self.start_trace(operation, metadata)
        try:
            yield trace_id
            self.end_trace(trace_id, 'success')
        except Exception as e:
            self.end_trace(trace_id, 'error', str(e))
            raise
    
    @contextmanager
    def trace_span(self, trace_id: str, span_name: str, metadata: Dict = None):
        """Context manager for tracing spans within an operation"""
        start_time = time.time()
        try:
            yield
            end_time = time.time()
            self.add_span(trace_id, span_name, start_time, end_time, metadata)
        except Exception as e:
            end_time = time.time()
            metadata = metadata or {}
            metadata['error'] = str(e)
            self.add_span(trace_id, span_name, start_time, end_time, metadata)
            raise
    
    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get a specific trace"""
        # Check active traces
        if trace_id in self.active_traces:
            return self.active_traces[trace_id].copy()
        
        # Check completed traces
        for trace in self.completed_traces:
            if trace['trace_id'] == trace_id:
                return trace.copy()
        
        return None
    
    def get_active_traces(self) -> List[Dict]:
        """Get all active traces"""
        with self.lock:
            return list(self.active_traces.values())
    
    def get_recent_traces(self, limit: int = 100) -> List[Dict]:
        """Get recent completed traces"""
        return self.completed_traces[-limit:]
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get trace statistics for analysis"""
        with self.lock:
            active_count = len(self.active_traces)
            completed_count = len(self.completed_traces)
            
            # Calculate average durations by operation
            operation_stats = {}
            for trace in self.completed_traces:
                op = trace['operation']
                if op not in operation_stats:
                    operation_stats[op] = {'count': 0, 'total_duration': 0, 'errors': 0}
                
                operation_stats[op]['count'] += 1
                if trace.get('duration'):
                    operation_stats[op]['total_duration'] += trace['duration']
                if trace.get('status') == 'error':
                    operation_stats[op]['errors'] += 1
            
            # Calculate averages
            for op, stats in operation_stats.items():
                if stats['count'] > 0:
                    stats['avg_duration'] = stats['total_duration'] / stats['count']
                    stats['error_rate'] = stats['errors'] / stats['count']
            
            return {
                'active_traces': active_count,
                'completed_traces': completed_count,
                'operation_statistics': operation_stats,
                'timestamp': datetime.now().isoformat()
            }


# Global request tracer instance
request_tracer = RequestTracer()
