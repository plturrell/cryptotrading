"""
CDS Integration Monitoring
Advanced monitoring and metrics for CDS-Agent integration
"""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from .metrics import MetricsCollector, MetricType, get_metrics
from .logger import get_logger
from .error_tracker import get_error_tracker, ErrorCategory, ErrorSeverity

logger = get_logger(__name__)
metrics = get_metrics()
error_tracker = get_error_tracker()


class CDSOperationType(Enum):
    """Types of CDS operations to monitor"""
    AGENT_REGISTRATION = "agent_registration"
    DATA_ANALYSIS = "data_analysis" 
    ML_REQUEST = "ml_request"
    MESSAGE_SEND = "message_send"
    TRANSACTION = "transaction"
    CONNECTION = "connection"
    HEARTBEAT = "heartbeat"


class CDSIntegrationStatus(Enum):
    """CDS integration status states"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    FALLBACK = "fallback"


@dataclass
class CDSOperationMetrics:
    """Metrics for a CDS operation"""
    operation_type: CDSOperationType
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = False
    method_used: str = "unknown"  # CDS, Local, Fallback
    error_message: Optional[str] = None
    payload_size: int = 0
    response_size: int = 0
    retry_count: int = 0
    transaction_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_type": self.operation_type.value,
            "agent_id": self.agent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "method_used": self.method_used,
            "error_message": self.error_message,
            "payload_size": self.payload_size,
            "response_size": self.response_size,
            "retry_count": self.retry_count,
            "transaction_id": self.transaction_id
        }


@dataclass 
class CDSAgentStats:
    """Statistics for a CDS agent"""
    agent_id: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    cds_operations: int = 0
    local_operations: int = 0
    fallback_operations: int = 0
    avg_response_time_ms: float = 0.0
    last_operation: Optional[datetime] = None
    connection_status: CDSIntegrationStatus = CDSIntegrationStatus.DISCONNECTED
    error_count: int = 0
    last_error: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
    
    @property
    def cds_utilization_rate(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return self.cds_operations / self.total_operations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "cds_operations": self.cds_operations,
            "local_operations": self.local_operations,
            "fallback_operations": self.fallback_operations,
            "avg_response_time_ms": self.avg_response_time_ms,
            "success_rate": self.success_rate,
            "cds_utilization_rate": self.cds_utilization_rate,
            "last_operation": self.last_operation.isoformat() if self.last_operation else None,
            "connection_status": self.connection_status.value,
            "error_count": self.error_count,
            "last_error": self.last_error
        }


class CDSIntegrationMonitor:
    """Advanced monitoring for CDS-Agent integration"""
    
    def __init__(self):
        self.agent_stats: Dict[str, CDSAgentStats] = {}
        self.operation_history: List[CDSOperationMetrics] = []
        self.max_history_size = 10000
        self.system_health = {
            "total_agents": 0,
            "connected_agents": 0,
            "healthy_agents": 0,
            "overall_success_rate": 0.0,
            "overall_cds_utilization": 0.0,
            "last_updated": datetime.now()
        }
        
    def register_agent(self, agent_id: str) -> None:
        """Register an agent for monitoring"""
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = CDSAgentStats(agent_id=agent_id)
            logger.info(f"Registered agent {agent_id} for CDS monitoring")
            
            # Update system metrics
            metrics.increment(
                "cds.agent.registered",
                tags={"agent_id": agent_id}
            )
    
    def update_agent_status(self, agent_id: str, status: CDSIntegrationStatus) -> None:
        """Update agent connection status"""
        self.register_agent(agent_id)
        old_status = self.agent_stats[agent_id].connection_status
        self.agent_stats[agent_id].connection_status = status
        
        if old_status != status:
            logger.info(f"Agent {agent_id} status changed: {old_status.value} -> {status.value}")
            metrics.set_gauge(
                "cds.agent.status",
                1 if status == CDSIntegrationStatus.CONNECTED else 0,
                tags={"agent_id": agent_id, "status": status.value}
            )
    
    @asynccontextmanager
    async def track_operation(
        self,
        agent_id: str,
        operation_type: CDSOperationType,
        transaction_id: Optional[str] = None
    ):
        """Context manager to track CDS operations"""
        operation = CDSOperationMetrics(
            operation_type=operation_type,
            agent_id=agent_id,
            start_time=datetime.now(),
            transaction_id=transaction_id
        )
        
        # Register agent if not already registered
        self.register_agent(agent_id)
        
        try:
            yield operation
            
            # Mark as successful if no exception
            operation.success = True
            operation.end_time = datetime.now()
            operation.duration_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            
        except Exception as e:
            operation.success = False
            operation.end_time = datetime.now()
            operation.duration_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            operation.error_message = str(e)
            
            # Track error
            error_tracker.track_error(
                error=e,
                category=ErrorCategory.INTEGRATION,
                severity=ErrorSeverity.MEDIUM,
                context={
                    "agent_id": agent_id,
                    "operation_type": operation_type.value,
                    "transaction_id": transaction_id
                }
            )
            
            raise
        
        finally:
            # Record operation metrics
            self._record_operation(operation)
    
    def _record_operation(self, operation: CDSOperationMetrics) -> None:
        """Record operation metrics and update agent stats"""
        # Add to history
        self.operation_history.append(operation)
        if len(self.operation_history) > self.max_history_size:
            self.operation_history.pop(0)
        
        # Update agent stats
        agent_stats = self.agent_stats[operation.agent_id]
        agent_stats.total_operations += 1
        agent_stats.last_operation = operation.end_time
        
        if operation.success:
            agent_stats.successful_operations += 1
        else:
            agent_stats.failed_operations += 1
            agent_stats.error_count += 1
            agent_stats.last_error = operation.error_message
        
        # Track method used
        if operation.method_used == "CDS":
            agent_stats.cds_operations += 1
        elif operation.method_used == "Local":
            agent_stats.local_operations += 1
        elif operation.method_used == "Fallback":
            agent_stats.fallback_operations += 1
        
        # Update average response time
        if operation.duration_ms is not None:
            total_time = agent_stats.avg_response_time_ms * (agent_stats.total_operations - 1)
            agent_stats.avg_response_time_ms = (total_time + operation.duration_ms) / agent_stats.total_operations
        
        # Record metrics
        tags = {
            "agent_id": operation.agent_id,
            "operation": operation.operation_type.value,
            "method": operation.method_used,
            "success": str(operation.success)
        }
        
        metrics.increment("cds.operation.count", tags=tags)
        
        if operation.duration_ms is not None:
            metrics.record_histogram("cds.operation.duration_ms", operation.duration_ms, tags=tags)
        
        if operation.payload_size > 0:
            metrics.record_histogram("cds.operation.payload_size", operation.payload_size, tags=tags)
        
        if operation.response_size > 0:
            metrics.record_histogram("cds.operation.response_size", operation.response_size, tags=tags)
        
        # Update system health
        self._update_system_health()
    
    def _update_system_health(self) -> None:
        """Update overall system health metrics"""
        if not self.agent_stats:
            return
        
        total_agents = len(self.agent_stats)
        connected_agents = sum(
            1 for stats in self.agent_stats.values()
            if stats.connection_status == CDSIntegrationStatus.CONNECTED
        )
        healthy_agents = sum(
            1 for stats in self.agent_stats.values()
            if stats.success_rate >= 0.9 and stats.connection_status == CDSIntegrationStatus.CONNECTED
        )
        
        # Calculate overall success rate
        total_operations = sum(stats.total_operations for stats in self.agent_stats.values())
        successful_operations = sum(stats.successful_operations for stats in self.agent_stats.values())
        overall_success_rate = successful_operations / total_operations if total_operations > 0 else 0.0
        
        # Calculate overall CDS utilization
        total_cds_operations = sum(stats.cds_operations for stats in self.agent_stats.values())
        overall_cds_utilization = total_cds_operations / total_operations if total_operations > 0 else 0.0
        
        self.system_health = {
            "total_agents": total_agents,
            "connected_agents": connected_agents,
            "healthy_agents": healthy_agents,
            "overall_success_rate": overall_success_rate,
            "overall_cds_utilization": overall_cds_utilization,
            "last_updated": datetime.now()
        }
        
        # Record system-level metrics
        metrics.set_gauge("cds.system.total_agents", total_agents)
        metrics.set_gauge("cds.system.connected_agents", connected_agents)
        metrics.set_gauge("cds.system.healthy_agents", healthy_agents)
        metrics.set_gauge("cds.system.success_rate", overall_success_rate)
        metrics.set_gauge("cds.system.cds_utilization", overall_cds_utilization)
    
    def get_agent_stats(self, agent_id: str) -> Optional[CDSAgentStats]:
        """Get statistics for a specific agent"""
        return self.agent_stats.get(agent_id)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        return self.system_health.copy()
    
    def get_recent_operations(self, agent_id: Optional[str] = None, limit: int = 100) -> List[CDSOperationMetrics]:
        """Get recent operations, optionally filtered by agent"""
        operations = self.operation_history
        
        if agent_id:
            operations = [op for op in operations if op.agent_id == agent_id]
        
        # Return most recent operations
        return sorted(operations, key=lambda x: x.start_time, reverse=True)[:limit]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": self.get_system_health(),
            "agent_statistics": {
                agent_id: stats.to_dict() 
                for agent_id, stats in self.agent_stats.items()
            },
            "top_performers": self._get_top_performers(),
            "performance_issues": self._identify_performance_issues(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _get_top_performers(self) -> List[Dict[str, Any]]:
        """Identify top performing agents"""
        performers = []
        
        for agent_id, stats in self.agent_stats.items():
            if stats.total_operations >= 10:  # Minimum operations for consideration
                score = (
                    stats.success_rate * 0.4 +
                    stats.cds_utilization_rate * 0.3 +
                    (1.0 / (stats.avg_response_time_ms / 1000 + 1)) * 0.3
                )
                
                performers.append({
                    "agent_id": agent_id,
                    "performance_score": score,
                    "success_rate": stats.success_rate,
                    "cds_utilization": stats.cds_utilization_rate,
                    "avg_response_time_ms": stats.avg_response_time_ms
                })
        
        return sorted(performers, key=lambda x: x["performance_score"], reverse=True)[:5]
    
    def _identify_performance_issues(self) -> List[Dict[str, Any]]:
        """Identify agents with performance issues"""
        issues = []
        
        for agent_id, stats in self.agent_stats.items():
            if stats.total_operations >= 5:  # Minimum operations for analysis
                if stats.success_rate < 0.8:
                    issues.append({
                        "agent_id": agent_id,
                        "issue_type": "low_success_rate",
                        "severity": "high",
                        "value": stats.success_rate,
                        "description": f"Success rate {stats.success_rate:.2%} below threshold"
                    })
                
                if stats.avg_response_time_ms > 5000:
                    issues.append({
                        "agent_id": agent_id,
                        "issue_type": "high_latency",
                        "severity": "medium",
                        "value": stats.avg_response_time_ms,
                        "description": f"Average response time {stats.avg_response_time_ms:.0f}ms above threshold"
                    })
                
                if stats.cds_utilization_rate < 0.5 and stats.connection_status == CDSIntegrationStatus.CONNECTED:
                    issues.append({
                        "agent_id": agent_id,
                        "issue_type": "low_cds_utilization",
                        "severity": "low",
                        "value": stats.cds_utilization_rate,
                        "description": f"CDS utilization {stats.cds_utilization_rate:.2%} below expected"
                    })
        
        return sorted(issues, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["severity"]], reverse=True)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze system health
        health = self.system_health
        
        if health["connected_agents"] < health["total_agents"] * 0.8:
            recommendations.append(
                "Consider investigating CDS connection issues - "
                f"{health['total_agents'] - health['connected_agents']} agents disconnected"
            )
        
        if health["overall_success_rate"] < 0.9:
            recommendations.append(
                f"Overall success rate ({health['overall_success_rate']:.2%}) could be improved - "
                "review error logs and consider fallback strategy optimization"
            )
        
        if health["overall_cds_utilization"] < 0.7:
            recommendations.append(
                f"CDS utilization ({health['overall_cds_utilization']:.2%}) is low - "
                "consider optimizing CDS connection reliability"
            )
        
        # Agent-specific recommendations
        slow_agents = [
            agent_id for agent_id, stats in self.agent_stats.items()
            if stats.avg_response_time_ms > 3000
        ]
        if slow_agents:
            recommendations.append(
                f"Optimize performance for slow agents: {', '.join(slow_agents[:3])}"
            )
        
        return recommendations


# Global monitor instance
_cds_monitor = None


def get_cds_monitor() -> CDSIntegrationMonitor:
    """Get the global CDS integration monitor"""
    global _cds_monitor
    if _cds_monitor is None:
        _cds_monitor = CDSIntegrationMonitor()
    return _cds_monitor


# Convenience decorators and functions

def track_cds_operation(operation_type: CDSOperationType):
    """Decorator to track CDS operations"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            agent_id = getattr(self, 'agent_id', 'unknown')
            monitor = get_cds_monitor()
            
            async with monitor.track_operation(agent_id, operation_type) as operation:
                try:
                    result = await func(self, *args, **kwargs)
                    
                    # Determine method used from result
                    if isinstance(result, dict):
                        operation.method_used = result.get('method', 'unknown')
                        
                        # Track payload/response sizes if available
                        if 'payload_size' in result:
                            operation.payload_size = result['payload_size']
                        if 'response_size' in result:
                            operation.response_size = result['response_size']
                    
                    return result
                    
                except Exception as e:
                    # Error will be tracked by context manager
                    raise
        
        return wrapper
    return decorator