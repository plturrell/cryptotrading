"""
MCP Monitoring and Audit System

Provides comprehensive monitoring, audit trails, and compliance reporting
for multi-tenant MCP agent segregation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
from pathlib import Path

from .mcp_agent_segregation import AgentContext, ResourceType, AccessLog
from .mcp_auth_middleware import AuthenticationRequest, AuthenticationResponse

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    agent_id: str
    tenant_id: str
    resource_type: Optional[ResourceType]
    description: str
    metadata: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False

@dataclass
class PerformanceMetric:
    """Performance metric for monitoring"""
    metric_name: str
    agent_id: str
    tenant_id: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None

@dataclass
class ComplianceReport:
    """Compliance report for tenant isolation"""
    report_id: str
    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    security_violations: int
    cross_tenant_attempts: int
    quota_violations: int
    compliance_score: float
    recommendations: List[str]

class MCPAuditLogger:
    """Comprehensive audit logging for MCP operations"""
    
    def __init__(self, db_path: str = "data/mcp_audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self.event_queue = deque(maxlen=10000)
        self.batch_size = 100
        self._background_task = None
        
    def _init_database(self):
        """Initialize audit database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    reason TEXT,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    resource_type TEXT,
                    description TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                );
                
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    tags TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_access_logs_tenant ON access_logs(tenant_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity, timestamp);
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_tenant ON performance_metrics(tenant_id, metric_name, timestamp);
            """)
    
    async def log_access(self, access_log: AccessLog):
        """Log access attempt"""
        self.event_queue.append(('access_log', access_log))
        await self._process_queue_if_needed()
    
    async def log_security_event(self, event: SecurityEvent):
        """Log security event"""
        self.event_queue.append(('security_event', event))
        await self._process_queue_if_needed()
        
        # Log critical events immediately
        if event.severity == "CRITICAL":
            await self._flush_queue()
    
    async def log_performance_metric(self, metric: PerformanceMetric):
        """Log performance metric"""
        self.event_queue.append(('performance_metric', metric))
        await self._process_queue_if_needed()
    
    async def _process_queue_if_needed(self):
        """Process queue if batch size reached"""
        if len(self.event_queue) >= self.batch_size:
            await self._flush_queue()
    
    async def _flush_queue(self):
        """Flush all queued events to database"""
        if not self.event_queue:
            return
        
        events_to_process = list(self.event_queue)
        self.event_queue.clear()
        
        with sqlite3.connect(self.db_path) as conn:
            for event_type, event_data in events_to_process:
                try:
                    if event_type == 'access_log':
                        self._insert_access_log(conn, event_data)
                    elif event_type == 'security_event':
                        self._insert_security_event(conn, event_data)
                    elif event_type == 'performance_metric':
                        self._insert_performance_metric(conn, event_data)
                except Exception as e:
                    logger.error("Failed to insert %s: %s", event_type, e)
    
    def _insert_access_log(self, conn: sqlite3.Connection, log: AccessLog):
        """Insert access log into database"""
        conn.execute("""
            INSERT INTO access_logs (agent_id, tenant_id, resource_type, action, success, reason, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log.agent_id, log.tenant_id, log.resource_type.value, log.action,
            log.success, log.reason, log.timestamp, json.dumps(log.metadata or {})
        ))
    
    def _insert_security_event(self, conn: sqlite3.Connection, event: SecurityEvent):
        """Insert security event into database"""
        conn.execute("""
            INSERT OR REPLACE INTO security_events 
            (event_id, event_type, severity, agent_id, tenant_id, resource_type, description, metadata, timestamp, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id, event.event_type, event.severity, event.agent_id, event.tenant_id,
            event.resource_type.value if event.resource_type else None,
            event.description, json.dumps(event.metadata), event.timestamp, event.resolved
        ))
    
    def _insert_performance_metric(self, conn: sqlite3.Connection, metric: PerformanceMetric):
        """Insert performance metric into database"""
        conn.execute("""
            INSERT INTO performance_metrics (metric_name, agent_id, tenant_id, value, unit, timestamp, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.metric_name, metric.agent_id, metric.tenant_id, metric.value,
            metric.unit, metric.timestamp, json.dumps(metric.tags or {})
        ))
    
    async def get_tenant_access_summary(self, tenant_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get access summary for tenant"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get access statistics
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests,
                    COUNT(DISTINCT agent_id) as unique_agents,
                    COUNT(DISTINCT resource_type) as resources_accessed
                FROM access_logs 
                WHERE tenant_id = ? AND timestamp > ?
            """, (tenant_id, cutoff)).fetchone()
            
            # Get resource usage breakdown
            resources = conn.execute("""
                SELECT resource_type, COUNT(*) as requests
                FROM access_logs 
                WHERE tenant_id = ? AND timestamp > ?
                GROUP BY resource_type
                ORDER BY requests DESC
            """, (tenant_id, cutoff)).fetchall()
            
            return {
                "tenant_id": tenant_id,
                "period_hours": hours,
                "statistics": dict(stats),
                "resource_usage": [dict(row) for row in resources]
            }
    
    async def get_security_events(self, tenant_id: str = None, severity: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get security events with optional filtering"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        query = "SELECT * FROM security_events WHERE timestamp > ?"
        params = [cutoff]
        
        if tenant_id:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            events = conn.execute(query, params).fetchall()
            
            return [dict(row) for row in events]

class MCPPerformanceMonitor:
    """Performance monitoring for MCP operations"""
    
    def __init__(self, audit_logger: MCPAuditLogger):
        self.audit_logger = audit_logger
        self.metrics_buffer = defaultdict(list)
        self.alert_thresholds = {
            "response_time_ms": 5000,  # 5 seconds
            "memory_usage_mb": 1024,   # 1GB
            "error_rate_percent": 10,  # 10%
            "concurrent_operations": 50
        }
    
    async def record_operation_metrics(self, agent_context: AgentContext, tool_name: str, 
                                     execution_time: float, memory_used: float, success: bool):
        """Record metrics for tool operation"""
        timestamp = datetime.utcnow()
        
        # Record execution time
        await self.audit_logger.log_performance_metric(PerformanceMetric(
            metric_name="execution_time_ms",
            agent_id=agent_context.agent_id,
            tenant_id=agent_context.tenant_id,
            value=execution_time * 1000,
            unit="milliseconds",
            timestamp=timestamp,
            tags={"tool": tool_name, "success": str(success)}
        ))
        
        # Record memory usage
        await self.audit_logger.log_performance_metric(PerformanceMetric(
            metric_name="memory_usage_mb",
            agent_id=agent_context.agent_id,
            tenant_id=agent_context.tenant_id,
            value=memory_used,
            unit="megabytes",
            timestamp=timestamp,
            tags={"tool": tool_name}
        ))
        
        # Check for performance alerts
        await self._check_performance_alerts(agent_context, tool_name, execution_time * 1000, memory_used)
    
    async def _check_performance_alerts(self, agent_context: AgentContext, tool_name: str, 
                                      execution_time_ms: float, memory_mb: float):
        """Check if performance metrics exceed thresholds"""
        alerts = []
        
        if execution_time_ms > self.alert_thresholds["response_time_ms"]:
            alerts.append(f"High response time: {execution_time_ms:.1f}ms")
        
        if memory_mb > self.alert_thresholds["memory_usage_mb"]:
            alerts.append(f"High memory usage: {memory_mb:.1f}MB")
        
        for alert_msg in alerts:
            await self.audit_logger.log_security_event(SecurityEvent(
                event_id=f"perf_{agent_context.agent_id}_{int(time.time())}",
                event_type="PERFORMANCE_ALERT",
                severity="MEDIUM",
                agent_id=agent_context.agent_id,
                tenant_id=agent_context.tenant_id,
                resource_type=None,
                description=f"Performance threshold exceeded for {tool_name}: {alert_msg}",
                metadata={"tool": tool_name, "execution_time_ms": execution_time_ms, "memory_mb": memory_mb},
                timestamp=datetime.utcnow()
            ))
    
    async def get_tenant_performance_summary(self, tenant_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for tenant"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.audit_logger.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Average response times by tool
            response_times = conn.execute("""
                SELECT 
                    JSON_EXTRACT(tags, '$.tool') as tool,
                    AVG(value) as avg_response_time,
                    MAX(value) as max_response_time,
                    COUNT(*) as operations
                FROM performance_metrics 
                WHERE tenant_id = ? AND metric_name = 'execution_time_ms' AND timestamp > ?
                GROUP BY JSON_EXTRACT(tags, '$.tool')
                ORDER BY avg_response_time DESC
            """, (tenant_id, cutoff)).fetchall()
            
            # Memory usage statistics
            memory_stats = conn.execute("""
                SELECT 
                    AVG(value) as avg_memory_mb,
                    MAX(value) as max_memory_mb,
                    COUNT(*) as measurements
                FROM performance_metrics 
                WHERE tenant_id = ? AND metric_name = 'memory_usage_mb' AND timestamp > ?
            """, (tenant_id, cutoff)).fetchone()
            
            return {
                "tenant_id": tenant_id,
                "period_hours": hours,
                "response_times": [dict(row) for row in response_times],
                "memory_usage": dict(memory_stats) if memory_stats else {}
            }

class MCPComplianceReporter:
    """Compliance reporting for multi-tenant segregation"""
    
    def __init__(self, audit_logger: MCPAuditLogger):
        self.audit_logger = audit_logger
    
    async def generate_compliance_report(self, tenant_id: str, period_days: int = 30) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        period_start = datetime.utcnow() - timedelta(days=period_days)
        period_end = datetime.utcnow()
        
        with sqlite3.connect(self.audit_logger.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Access statistics
            access_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests
                FROM access_logs 
                WHERE tenant_id = ? AND timestamp BETWEEN ? AND ?
            """, (tenant_id, period_start, period_end)).fetchone()
            
            # Security violations
            security_violations = conn.execute("""
                SELECT COUNT(*) as violations
                FROM security_events 
                WHERE tenant_id = ? AND severity IN ('HIGH', 'CRITICAL') AND timestamp BETWEEN ? AND ?
            """, (tenant_id, period_start, period_end)).fetchone()
            
            # Cross-tenant access attempts
            cross_tenant_attempts = conn.execute("""
                SELECT COUNT(*) as attempts
                FROM access_logs 
                WHERE agent_id LIKE ? AND success = 0 AND reason = 'CROSS_TENANT_ACCESS' AND timestamp BETWEEN ? AND ?
            """, (f"%{tenant_id}%", period_start, period_end)).fetchone()
            
            # Quota violations
            quota_violations = conn.execute("""
                SELECT COUNT(*) as violations
                FROM access_logs 
                WHERE tenant_id = ? AND success = 0 AND reason LIKE '%QUOTA%' AND timestamp BETWEEN ? AND ?
            """, (tenant_id, period_start, period_end)).fetchone()
        
        # Calculate compliance score
        total_requests = access_stats["total_requests"] or 1
        successful_requests = access_stats["successful_requests"] or 0
        violations = security_violations["violations"] or 0
        cross_tenant = cross_tenant_attempts["attempts"] or 0
        quota_viol = quota_violations["violations"] or 0
        
        compliance_score = max(0, 100 - (violations * 10) - (cross_tenant * 5) - (quota_viol * 2))
        compliance_score *= (successful_requests / total_requests)
        
        # Generate recommendations
        recommendations = []
        if violations > 0:
            recommendations.append("Review and address security violations")
        if cross_tenant > 0:
            recommendations.append("Strengthen tenant isolation controls")
        if quota_viol > 5:
            recommendations.append("Review and adjust resource quotas")
        if compliance_score < 90:
            recommendations.append("Implement additional monitoring and controls")
        
        return ComplianceReport(
            report_id=f"compliance_{tenant_id}_{int(time.time())}",
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=access_stats["failed_requests"] or 0,
            security_violations=violations,
            cross_tenant_attempts=cross_tenant,
            quota_violations=quota_viol,
            compliance_score=compliance_score,
            recommendations=recommendations
        )
    
    async def export_audit_trail(self, tenant_id: str, period_days: int = 30) -> Dict[str, Any]:
        """Export complete audit trail for tenant"""
        period_start = datetime.utcnow() - timedelta(days=period_days)
        
        with sqlite3.connect(self.audit_logger.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Access logs
            access_logs = conn.execute("""
                SELECT * FROM access_logs 
                WHERE tenant_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (tenant_id, period_start)).fetchall()
            
            # Security events
            security_events = conn.execute("""
                SELECT * FROM security_events 
                WHERE tenant_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (tenant_id, period_start)).fetchall()
            
            # Performance metrics
            performance_metrics = conn.execute("""
                SELECT * FROM performance_metrics 
                WHERE tenant_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (tenant_id, period_start)).fetchall()
        
        return {
            "tenant_id": tenant_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "period_start": period_start.isoformat(),
            "access_logs": [dict(row) for row in access_logs],
            "security_events": [dict(row) for row in security_events],
            "performance_metrics": [dict(row) for row in performance_metrics]
        }

# Global instances
_audit_logger = None
_performance_monitor = None
_compliance_reporter = None

def get_audit_logger() -> MCPAuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = MCPAuditLogger()
    return _audit_logger

def get_performance_monitor() -> MCPPerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = MCPPerformanceMonitor(get_audit_logger())
    return _performance_monitor

def get_compliance_reporter() -> MCPComplianceReporter:
    """Get global compliance reporter instance"""
    global _compliance_reporter
    if _compliance_reporter is None:
        _compliance_reporter = MCPComplianceReporter(get_audit_logger())
    return _compliance_reporter
