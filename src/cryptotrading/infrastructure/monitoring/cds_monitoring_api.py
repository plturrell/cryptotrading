"""
CDS Monitoring API
Provides REST API endpoints for monitoring CDS integration
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import asyncio

from .cds_integration_monitor import (
    get_cds_monitor, 
    CDSOperationType, 
    CDSIntegrationStatus,
    CDSOperationMetrics
)
from .logger import get_logger

logger = get_logger(__name__)


class AgentStatsResponse(BaseModel):
    """Response model for agent statistics"""
    agent_id: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    cds_operations: int
    local_operations: int
    fallback_operations: int
    avg_response_time_ms: float
    success_rate: float
    cds_utilization_rate: float
    last_operation: Optional[str]
    connection_status: str
    error_count: int
    last_error: Optional[str]


class SystemHealthResponse(BaseModel):
    """Response model for system health"""
    total_agents: int
    connected_agents: int
    healthy_agents: int
    overall_success_rate: float
    overall_cds_utilization: float
    last_updated: str


class OperationResponse(BaseModel):
    """Response model for operation details"""
    operation_type: str
    agent_id: str
    start_time: str
    end_time: Optional[str]
    duration_ms: Optional[float]
    success: bool
    method_used: str
    error_message: Optional[str]
    payload_size: int
    response_size: int
    retry_count: int
    transaction_id: Optional[str]


class PerformanceReportResponse(BaseModel):
    """Response model for performance report"""
    timestamp: str
    system_health: SystemHealthResponse
    agent_statistics: Dict[str, AgentStatsResponse]
    top_performers: List[Dict[str, Any]]
    performance_issues: List[Dict[str, Any]]
    recommendations: List[str]


class CDSMonitoringAPI:
    """API for CDS integration monitoring"""
    
    def __init__(self):
        self.app = FastAPI(title="CDS Integration Monitoring API")
        self.monitor = get_cds_monitor()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/monitoring/health", response_model=SystemHealthResponse)
        async def get_system_health():
            """Get overall system health"""
            health = self.monitor.get_system_health()
            return SystemHealthResponse(
                total_agents=health["total_agents"],
                connected_agents=health["connected_agents"],
                healthy_agents=health["healthy_agents"],
                overall_success_rate=health["overall_success_rate"],
                overall_cds_utilization=health["overall_cds_utilization"],
                last_updated=health["last_updated"].isoformat()
            )
        
        @self.app.get("/monitoring/agents", response_model=List[str])
        async def list_agents():
            """Get list of registered agents"""
            return list(self.monitor.agent_stats.keys())
        
        @self.app.get("/monitoring/agents/{agent_id}", response_model=AgentStatsResponse)
        async def get_agent_stats(agent_id: str):
            """Get statistics for specific agent"""
            stats = self.monitor.get_agent_stats(agent_id)
            if not stats:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            return AgentStatsResponse(
                agent_id=stats.agent_id,
                total_operations=stats.total_operations,
                successful_operations=stats.successful_operations,
                failed_operations=stats.failed_operations,
                cds_operations=stats.cds_operations,
                local_operations=stats.local_operations,
                fallback_operations=stats.fallback_operations,
                avg_response_time_ms=stats.avg_response_time_ms,
                success_rate=stats.success_rate,
                cds_utilization_rate=stats.cds_utilization_rate,
                last_operation=stats.last_operation.isoformat() if stats.last_operation else None,
                connection_status=stats.connection_status.value,
                error_count=stats.error_count,
                last_error=stats.last_error
            )
        
        @self.app.get("/monitoring/operations", response_model=List[OperationResponse])
        async def get_operations(
            agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
            limit: int = Query(100, ge=1, le=1000, description="Number of operations to return"),
            operation_type: Optional[str] = Query(None, description="Filter by operation type")
        ):
            """Get recent operations"""
            operations = self.monitor.get_recent_operations(agent_id, limit)
            
            # Filter by operation type if specified
            if operation_type:
                try:
                    op_type = CDSOperationType(operation_type)
                    operations = [op for op in operations if op.operation_type == op_type]
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid operation type: {operation_type}")
            
            return [
                OperationResponse(
                    operation_type=op.operation_type.value,
                    agent_id=op.agent_id,
                    start_time=op.start_time.isoformat(),
                    end_time=op.end_time.isoformat() if op.end_time else None,
                    duration_ms=op.duration_ms,
                    success=op.success,
                    method_used=op.method_used,
                    error_message=op.error_message,
                    payload_size=op.payload_size,
                    response_size=op.response_size,
                    retry_count=op.retry_count,
                    transaction_id=op.transaction_id
                )
                for op in operations
            ]
        
        @self.app.get("/monitoring/report", response_model=PerformanceReportResponse)
        async def get_performance_report():
            """Get comprehensive performance report"""
            report = self.monitor.get_performance_report()
            
            return PerformanceReportResponse(
                timestamp=report["timestamp"],
                system_health=SystemHealthResponse(**report["system_health"]),
                agent_statistics={
                    agent_id: AgentStatsResponse(**stats_dict)
                    for agent_id, stats_dict in report["agent_statistics"].items()
                },
                top_performers=report["top_performers"],
                performance_issues=report["performance_issues"],
                recommendations=report["recommendations"]
            )
        
        @self.app.get("/monitoring/metrics/summary")
        async def get_metrics_summary():
            """Get summary metrics"""
            health = self.monitor.get_system_health()
            
            # Calculate additional metrics
            total_operations = sum(
                stats.total_operations for stats in self.monitor.agent_stats.values()
            )
            
            avg_response_time = sum(
                stats.avg_response_time_ms * stats.total_operations 
                for stats in self.monitor.agent_stats.values()
                if stats.total_operations > 0
            ) / max(total_operations, 1)
            
            return {
                "total_operations": total_operations,
                "average_response_time_ms": avg_response_time,
                "system_health": health,
                "operation_types": [op_type.value for op_type in CDSOperationType],
                "agent_statuses": [status.value for status in CDSIntegrationStatus]
            }
        
        @self.app.post("/monitoring/agents/{agent_id}/status")
        async def update_agent_status(agent_id: str, status: str):
            """Update agent status (admin endpoint)"""
            try:
                status_enum = CDSIntegrationStatus(status)
                self.monitor.update_agent_status(agent_id, status_enum)
                return {"message": f"Agent {agent_id} status updated to {status}"}
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        @self.app.get("/monitoring/alerts")
        async def get_alerts():
            """Get current performance alerts"""
            issues = self.monitor._identify_performance_issues()
            
            # Convert to alerts format
            alerts = []
            for issue in issues:
                alert_level = {
                    "high": "critical",
                    "medium": "warning", 
                    "low": "info"
                }.get(issue["severity"], "info")
                
                alerts.append({
                    "level": alert_level,
                    "agent_id": issue["agent_id"],
                    "type": issue["issue_type"],
                    "message": issue["description"],
                    "value": issue["value"],
                    "timestamp": datetime.now().isoformat()
                })
            
            return {"alerts": alerts}
        
        @self.app.get("/monitoring/trends")
        async def get_trends(
            hours: int = Query(24, ge=1, le=168, description="Hours of history to analyze")
        ):
            """Get performance trends over time"""
            # For now, return basic trend analysis
            # In a full implementation, this would analyze historical data
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_operations = [
                op for op in self.monitor.operation_history
                if op.start_time >= cutoff_time
            ]
            
            if not recent_operations:
                return {
                    "timespan_hours": hours,
                    "total_operations": 0,
                    "trends": {}
                }
            
            # Calculate trends
            hourly_buckets = {}
            for op in recent_operations:
                hour_key = op.start_time.strftime("%Y-%m-%d %H:00")
                if hour_key not in hourly_buckets:
                    hourly_buckets[hour_key] = {
                        "total": 0, "successful": 0, "cds": 0, "local": 0, 
                        "avg_duration": 0, "durations": []
                    }
                
                bucket = hourly_buckets[hour_key]
                bucket["total"] += 1
                if op.success:
                    bucket["successful"] += 1
                if op.method_used == "CDS":
                    bucket["cds"] += 1
                elif op.method_used == "Local":
                    bucket["local"] += 1
                
                if op.duration_ms is not None:
                    bucket["durations"].append(op.duration_ms)
            
            # Calculate averages
            for bucket in hourly_buckets.values():
                if bucket["durations"]:
                    bucket["avg_duration"] = sum(bucket["durations"]) / len(bucket["durations"])
                del bucket["durations"]  # Remove raw data
            
            return {
                "timespan_hours": hours,
                "total_operations": len(recent_operations),
                "hourly_data": hourly_buckets,
                "trends": {
                    "operations_per_hour": len(recent_operations) / hours,
                    "success_rate": sum(1 for op in recent_operations if op.success) / len(recent_operations),
                    "cds_utilization": sum(1 for op in recent_operations if op.method_used == "CDS") / len(recent_operations)
                }
            }


# Global API instance
_monitoring_api = None


def get_monitoring_api() -> CDSMonitoringAPI:
    """Get the global monitoring API instance"""
    global _monitoring_api
    if _monitoring_api is None:
        _monitoring_api = CDSMonitoringAPI()
    return _monitoring_api


def create_monitoring_endpoints(app: FastAPI):
    """Add monitoring endpoints to existing FastAPI app"""
    monitoring_api = get_monitoring_api()
    
    # Mount all monitoring routes
    app.mount("/api/cds", monitoring_api.app)


# Health check function for external monitoring
async def check_cds_system_health() -> Dict[str, Any]:
    """Check CDS system health for external monitoring systems"""
    monitor = get_cds_monitor()
    health = monitor.get_system_health()
    
    # Determine overall status
    if health["connected_agents"] == 0:
        status = "critical"
    elif health["overall_success_rate"] < 0.8:
        status = "warning"
    elif health["connected_agents"] < health["total_agents"] * 0.8:
        status = "warning"
    else:
        status = "healthy"
    
    return {
        "status": status,
        "details": health,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }