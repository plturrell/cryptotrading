"""
MCP Agent Segregation and Multi-Tenancy Implementation

Ensures strict isolation between agents and their tools in the MCP server.
Implements tenant-specific authentication, resource isolation, and access control.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

import jwt

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles with different permission levels"""

    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class ResourceType(Enum):
    """Types of resources that can be accessed"""

    CLRS_ALGORITHMS = "clrs_algorithms"
    TREE_OPERATIONS = "tree_operations"
    CODE_ANALYSIS = "code_analysis"
    DEPENDENCY_GRAPH = "dependency_graph"
    CONFIGURATION = "configuration"
    OPTIMIZATION = "optimization"
    PORTFOLIO_DATA = "portfolio_data"
    MARKET_DATA = "market_data"
    TRADING_OPERATIONS = "trading_operations"
    WALLET_OPERATIONS = "wallet_operations"
    PRICE_ALERTS = "price_alerts"
    TRANSACTION_HISTORY = "transaction_history"
    RISK_ANALYSIS = "risk_analysis"
    COMPLIANCE_REPORTING = "compliance_reporting"
    STORAGE = "storage"  # S3 and other storage operations
    LOGGING = "logging"  # Agent logging operations


@dataclass
class AgentContext:
    """Context information for an agent"""

    agent_id: str
    tenant_id: str
    role: AgentRole
    permissions: Set[ResourceType] = field(default_factory=set)
    resource_quotas: Dict[str, int] = field(default_factory=dict)
    session_token: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_access: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    def __post_init__(self):
        """Initialize default permissions based on role"""
        if self.role == AgentRole.ADMIN:
            self.resource_quotas = {
                "requests_per_hour": 1000,
                "max_file_size_mb": 100,
                "max_concurrent_operations": 10,
            }
        elif self.role == AgentRole.ANALYST:
            self.resource_quotas = {
                "requests_per_hour": 500,
                "max_file_size_mb": 50,
                "max_concurrent_operations": 5,
            }
        elif self.role == AgentRole.BASIC_USER:
            self.resource_quotas = {
                "requests_per_hour": 200,
                "max_file_size_mb": 20,
                "max_concurrent_operations": 3,
            }
        elif self.role == AgentRole.VIEWER:
            self.resource_quotas = {
                "requests_per_hour": 100,
                "max_file_size_mb": 10,
                "max_concurrent_operations": 2,
            }
        else:  # GUEST
            self.resource_quotas = {
                "requests_per_hour": 50,
                "max_file_size_mb": 5,
                "max_concurrent_operations": 1,
            }


@dataclass
class AccessLog:
    """Log entry for agent access"""

    timestamp: datetime
    agent_id: str
    tenant_id: str
    tool_name: str
    resource_type: ResourceType
    action: str
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    resource_usage: Optional[Dict[str, Any]] = None


class AgentSegregationManager:
    """Manages agent segregation and multi-tenancy for MCP tools"""

    def __init__(self, jwt_secret: str = "default-secret"):
        self.jwt_secret = jwt_secret
        self.agent_contexts: Dict[str, AgentContext] = {}
        self.access_logs: List[AccessLog] = []
        self.active_sessions: Dict[str, str] = {}  # session_token -> agent_id
        self.resource_usage: Dict[str, Dict[str, int]] = {}  # agent_id -> resource -> count
        self.tenant_isolation: Dict[str, Set[str]] = {}  # tenant_id -> set of agent_ids

    def generate_session_token(self, agent_context: AgentContext) -> str:
        """Generate a secure session token for an agent"""
        payload = {
            "agent_id": agent_context.agent_id,
            "tenant_id": agent_context.tenant_id,
            "role": agent_context.role.value,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,  # 1 hour expiry
        }
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        agent_context.session_token = token
        self.active_sessions[token] = agent_context.agent_id
        return token

    def validate_session_token(self, token: str) -> Optional[AgentContext]:
        """Validate a session token and return agent context"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            agent_id = payload.get("agent_id")

            if agent_id in self.agent_contexts:
                context = self.agent_contexts[agent_id]
                context.last_access = datetime.utcnow()
                context.access_count += 1
                return context

        except jwt.ExpiredSignatureError:
            logger.warning(f"Expired session token: {str(token)[:20]}...")
        except jwt.InvalidTokenError:
            logger.warning(f"Invalid session token: {str(token)[:20]}...")

        return None

    def register_agent(
        self,
        agent_id: str,
        tenant_id: str,
        role: AgentRole,
        custom_permissions: Optional[Set[ResourceType]] = None,
        custom_quotas: Optional[Dict[str, int]] = None,
    ) -> AgentContext:
        """Register a new agent with the segregation manager"""

        # Ensure tenant isolation
        if tenant_id not in self.tenant_isolation:
            self.tenant_isolation[tenant_id] = set()

        # Check if agent already exists in different tenant
        for existing_tenant, agents in self.tenant_isolation.items():
            if agent_id in agents and existing_tenant != tenant_id:
                raise ValueError(f"Agent {agent_id} already exists in tenant {existing_tenant}")

        context = AgentContext(agent_id=agent_id, tenant_id=tenant_id, role=role)

        # Apply custom permissions and quotas if provided
        if custom_permissions:
            context.permissions = custom_permissions
        if custom_quotas:
            context.resource_quotas.update(custom_quotas)

        # Generate session token
        self.generate_session_token(context)

        # Store context and update tenant isolation
        self.agent_contexts[agent_id] = context
        self.tenant_isolation[tenant_id].add(agent_id)
        self.resource_usage[agent_id] = {}

        logger.info(f"Registered agent {agent_id} for tenant {tenant_id} with role {role.value}")
        return context

    def check_permission(
        self, agent_context: AgentContext, resource_type: ResourceType, action: str = "access"
    ) -> bool:
        """Check if agent has permission for a resource"""
        if resource_type not in agent_context.permissions:
            self._log_access(
                agent_context,
                "permission_check",
                resource_type,
                action,
                success=False,
                error_message="Permission denied",
            )
            return False

        return True

    def check_resource_quota(
        self, agent_context: AgentContext, resource_key: str, requested_amount: int = 1
    ) -> bool:
        """Check if agent is within resource quota limits"""
        quota_limit = agent_context.resource_quotas.get(resource_key, 0)
        current_usage = self.resource_usage[agent_context.agent_id].get(resource_key, 0)

        if current_usage + requested_amount > quota_limit:
            self._log_access(
                agent_context,
                "quota_check",
                ResourceType.CODE_ANALYSIS,
                "quota_exceeded",
                success=False,
                error_message=f"Quota exceeded for {resource_key}",
            )
            return False

        return True

    def consume_resource(self, agent_context: AgentContext, resource_key: str, amount: int = 1):
        """Consume resources from agent's quota"""
        if agent_context.agent_id not in self.resource_usage:
            self.resource_usage[agent_context.agent_id] = {}

        current = self.resource_usage[agent_context.agent_id].get(resource_key, 0)
        self.resource_usage[agent_context.agent_id][resource_key] = current + amount

    def reset_hourly_quotas(self):
        """Reset hourly quotas for all agents"""
        for agent_id in self.resource_usage:
            if "requests_per_hour" in self.resource_usage[agent_id]:
                self.resource_usage[agent_id]["requests_per_hour"] = 0

        logger.info("Reset hourly quotas for all agents")

    def get_tenant_agents(self, tenant_id: str) -> Set[str]:
        """Get all agent IDs for a specific tenant"""
        return self.tenant_isolation.get(tenant_id, set())

    def is_cross_tenant_access(
        self, agent_context: AgentContext, target_data: Dict[str, Any]
    ) -> bool:
        """Check if agent is trying to access data from another tenant"""
        target_tenant = target_data.get("tenant_id")
        if target_tenant and target_tenant != agent_context.tenant_id:
            self._log_access(
                agent_context,
                "cross_tenant_check",
                ResourceType.CODE_ANALYSIS,
                "blocked",
                success=False,
                error_message="Cross-tenant access denied",
            )
            return True
        return False

    def _log_access(
        self,
        agent_context: AgentContext,
        tool_name: str,
        resource_type: ResourceType,
        action: str,
        success: bool,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        resource_usage: Optional[Dict[str, Any]] = None,
    ):
        """Log agent access for audit trail"""
        log_entry = AccessLog(
            timestamp=datetime.utcnow(),
            agent_id=agent_context.agent_id,
            tenant_id=agent_context.tenant_id,
            tool_name=tool_name,
            resource_type=resource_type,
            action=action,
            success=success,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            resource_usage=resource_usage,
        )

        self.access_logs.append(log_entry)

        # Keep only last 10000 log entries to prevent memory issues
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-5000:]

    def get_access_logs(
        self,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AccessLog]:
        """Get filtered access logs"""
        logs = self.access_logs

        if agent_id:
            logs = [log for log in logs if log.agent_id == agent_id]

        if tenant_id:
            logs = [log for log in logs if log.tenant_id == tenant_id]

        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]

        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]

        return logs

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for a specific agent"""
        if agent_id not in self.agent_contexts:
            return {}

        context = self.agent_contexts[agent_id]
        usage = self.resource_usage.get(agent_id, {})
        recent_logs = [log for log in self.access_logs if log.agent_id == agent_id]

        return {
            "agent_id": agent_id,
            "tenant_id": context.tenant_id,
            "role": context.role.value,
            "permissions": [p.value for p in context.permissions],
            "resource_quotas": context.resource_quotas,
            "current_usage": usage,
            "total_access_count": context.access_count,
            "last_access": context.last_access.isoformat(),
            "recent_activity_count": len(
                [
                    log
                    for log in recent_logs
                    if log.timestamp > datetime.utcnow() - timedelta(hours=1)
                ]
            ),
            "success_rate": len([log for log in recent_logs if log.success])
            / max(len(recent_logs), 1)
            * 100,
        }


class SecureToolWrapper:
    """Wrapper for MCP tools that enforces agent segregation"""

    def __init__(
        self,
        tool_instance: Any,
        resource_type: ResourceType,
        segregation_manager: AgentSegregationManager,
    ):
        self.tool_instance = tool_instance
        self.resource_type = resource_type
        self.segregation_manager = segregation_manager
        self.tool_name = tool_instance.__class__.__name__

    async def execute_with_segregation(
        self, session_token: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool with proper agent segregation checks"""
        start_time = time.time()

        # Validate session token
        agent_context = self.segregation_manager.validate_session_token(session_token)
        if not agent_context:
            return {"error": "Invalid or expired session token", "code": "AUTHENTICATION_FAILED"}

        try:
            # Check permissions
            if not self.segregation_manager.check_permission(agent_context, self.resource_type):
                return {
                    "error": f"Permission denied for {self.resource_type.value}",
                    "code": "PERMISSION_DENIED",
                }

            # Check resource quotas
            if not self.segregation_manager.check_resource_quota(
                agent_context, "requests_per_hour"
            ):
                return {"error": "Request quota exceeded", "code": "QUOTA_EXCEEDED"}

            # Check for cross-tenant access
            if self.segregation_manager.is_cross_tenant_access(agent_context, parameters):
                return {"error": "Cross-tenant access denied", "code": "CROSS_TENANT_ACCESS"}

            # Add tenant context to parameters
            secure_parameters = {
                **parameters,
                "tenant_id": agent_context.tenant_id,
                "agent_id": agent_context.agent_id,
            }

            # Execute the actual tool
            result = await self.tool_instance.execute(secure_parameters)

            # Consume resources
            self.segregation_manager.consume_resource(agent_context, "requests_per_hour")

            execution_time = int((time.time() - start_time) * 1000)

            # Log successful access
            self.segregation_manager._log_access(
                agent_context,
                self.tool_name,
                self.resource_type,
                "execute",
                success=True,
                execution_time_ms=execution_time,
                resource_usage={"requests": 1},
            )

            return {
                "success": True,
                "result": result,
                "execution_time_ms": execution_time,
                "agent_id": agent_context.agent_id,
                "tenant_id": agent_context.tenant_id,
            }

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            error_message = str(e)

            # Log failed access
            self.segregation_manager._log_access(
                agent_context,
                self.tool_name,
                self.resource_type,
                "execute",
                success=False,
                error_message=error_message,
                execution_time_ms=execution_time,
            )

            return {
                "error": error_message,
                "code": "EXECUTION_FAILED",
                "execution_time_ms": execution_time,
            }


# Global segregation manager instance
_segregation_manager = None


def get_segregation_manager() -> AgentSegregationManager:
    """Get the global segregation manager instance"""
    global _segregation_manager
    if _segregation_manager is None:
        import os

        jwt_secret = os.getenv("JWT_SECRET", "default-mcp-secret-2024")
        _segregation_manager = AgentSegregationManager(jwt_secret)
    return _segregation_manager


def require_agent_auth(resource_type: ResourceType):
    """Decorator to enforce agent authentication and authorization"""

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Extract session token from kwargs or args
            session_token = kwargs.get("session_token") or (args[0] if args else None)
            if not session_token:
                return {"error": "Session token required", "code": "MISSING_TOKEN"}

            segregation_manager = get_segregation_manager()
            agent_context = segregation_manager.validate_session_token(session_token)

            if not agent_context:
                return {"error": "Invalid session token", "code": "INVALID_TOKEN"}

            if not segregation_manager.check_permission(agent_context, resource_type):
                return {"error": "Permission denied", "code": "PERMISSION_DENIED"}

            # Add agent context to kwargs
            kwargs["agent_context"] = agent_context
            return await func(*args, **kwargs)

        return wrapper

    return decorator
