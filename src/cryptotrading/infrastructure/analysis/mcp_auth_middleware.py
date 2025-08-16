"""
MCP Authentication Middleware

Provides comprehensive authentication, authorization, and context isolation
for MCP tools with multi-tenant support and audit logging.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, asdict
from functools import wraps
import jwt
from datetime import datetime, timedelta

from .mcp_agent_segregation import (
    AgentContext,
    AgentRole,
    ResourceType,
    AccessLog,
    get_segregation_manager
)

logger = logging.getLogger(__name__)

@dataclass
class AuthenticationRequest:
    """Authentication request structure"""
    agent_id: str
    tenant_id: str
    session_token: Optional[str] = None
    requested_resources: List[ResourceType] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None

@dataclass
class AuthenticationResponse:
    """Authentication response structure"""
    success: bool
    agent_context: Optional[AgentContext] = None
    session_token: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    expires_at: Optional[datetime] = None

class MCPAuthenticationMiddleware:
    """Authentication middleware for MCP tools"""
    
    def __init__(self, jwt_secret: str = "production-jwt-secret-rex-crypto-2024"):
        self.jwt_secret = jwt_secret
        self.active_sessions: Dict[str, AgentContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.segregation_manager = get_segregation_manager()
        
    async def authenticate_agent(self, auth_request: AuthenticationRequest) -> AuthenticationResponse:
        """Authenticate agent and create session"""
        try:
            # Check for brute force attempts
            if self._is_rate_limited(auth_request.agent_id):
                return AuthenticationResponse(
                    success=False,
                    error_code="RATE_LIMITED",
                    error_message="Too many authentication attempts"
                )
            
            # Validate session token if provided
            if auth_request.session_token:
                agent_context = await self._validate_session_token(auth_request.session_token)
                if agent_context:
                    # Verify tenant match
                    if agent_context.tenant_id != auth_request.tenant_id:
                        self._log_failed_attempt(auth_request.agent_id)
                        return AuthenticationResponse(
                            success=False,
                            error_code="TENANT_MISMATCH",
                            error_message="Session tenant does not match request"
                        )
                    
                    return AuthenticationResponse(
                        success=True,
                        agent_context=agent_context,
                        session_token=auth_request.session_token
                    )
            
            # Create new session
            agent_context = await self._create_agent_session(auth_request)
            if not agent_context:
                self._log_failed_attempt(auth_request.agent_id)
                return AuthenticationResponse(
                    success=False,
                    error_code="AUTHENTICATION_FAILED",
                    error_message="Invalid agent credentials"
                )
            
            # Generate session token
            session_token = self._generate_session_token(agent_context)
            expires_at = datetime.utcnow() + timedelta(hours=24)
            
            # Store active session
            self.active_sessions[session_token] = agent_context
            
            return AuthenticationResponse(
                success=True,
                agent_context=agent_context,
                session_token=session_token,
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.error("Authentication failed for agent %s: %s", auth_request.agent_id, e)
            return AuthenticationResponse(
                success=False,
                error_code="INTERNAL_ERROR",
                error_message="Authentication service error"
            )
    
    async def authorize_resource_access(self, agent_context: AgentContext, resource_type: ResourceType) -> bool:
        """Authorize agent access to specific resource"""
        try:
            # Check if agent has permission for resource
            if not self.segregation_manager.check_permission(agent_context, resource_type):
                self._log_access_attempt(agent_context, resource_type, False, "PERMISSION_DENIED")
                return False
            
            # Check resource quotas
            if not self.segregation_manager.check_resource_quota(agent_context, "requests_per_hour"):
                self._log_access_attempt(agent_context, resource_type, False, "QUOTA_EXCEEDED")
                return False
            
            self._log_access_attempt(agent_context, resource_type, True, "AUTHORIZED")
            return True
            
        except Exception as e:
            logger.error("Authorization failed for agent %s: %s", agent_context.agent_id, e)
            return False
    
    async def create_isolated_context(self, agent_context: AgentContext, tool_name: str) -> Dict[str, Any]:
        """Create isolated execution context for agent"""
        return {
            "agent_id": agent_context.agent_id,
            "tenant_id": agent_context.tenant_id,
            "role": agent_context.role.value,
            "tool_name": tool_name,
            "session_id": f"{agent_context.agent_id}_{int(time.time())}",
            "isolation_boundary": f"tenant_{agent_context.tenant_id}",
            "resource_limits": agent_context.resource_quotas,
            "permissions": [perm.value for perm in agent_context.permissions],
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _is_rate_limited(self, agent_id: str) -> bool:
        """Check if agent is rate limited due to failed attempts"""
        if agent_id not in self.failed_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_attempts[agent_id]
            if attempt > datetime.utcnow() - timedelta(minutes=15)
        ]
        
        return len(recent_attempts) >= 5
    
    def _log_failed_attempt(self, agent_id: str):
        """Log failed authentication attempt"""
        if agent_id not in self.failed_attempts:
            self.failed_attempts[agent_id] = []
        
        self.failed_attempts[agent_id].append(datetime.utcnow())
        
        # Keep only recent attempts
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.failed_attempts[agent_id] = [
            attempt for attempt in self.failed_attempts[agent_id]
            if attempt > cutoff
        ]
    
    async def _validate_session_token(self, session_token: str) -> Optional[AgentContext]:
        """Validate and decode session token"""
        try:
            # Check active sessions first
            if session_token in self.active_sessions:
                return self.active_sessions[session_token]
            
            # Decode JWT token
            payload = jwt.decode(session_token, self.jwt_secret, algorithms=["HS256"])
            
            # Reconstruct agent context
            agent_context = AgentContext(
                agent_id=payload["agent_id"],
                tenant_id=payload["tenant_id"],
                role=AgentRole(payload["role"]),
                permissions=[ResourceType(perm) for perm in payload["permissions"]],
                resource_quotas=payload["resource_quotas"],
                session_token=session_token
            )
            
            # Store in active sessions
            self.active_sessions[session_token] = agent_context
            return agent_context
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired session token")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid session token")
            return None
        except Exception as e:
            logger.error("Token validation error: %s", e)
            return None
    
    async def _create_agent_session(self, auth_request: AuthenticationRequest) -> Optional[AgentContext]:
        """Create new agent session"""
        # Mock implementation - would integrate with actual user/agent database
        if auth_request.agent_id.startswith("agent_") and auth_request.tenant_id.startswith("tenant_"):
            # Determine role based on agent_id pattern
            if "admin" in auth_request.agent_id:
                role = AgentRole.ADMIN
            elif "analyst" in auth_request.agent_id:
                role = AgentRole.ANALYST
            else:
                role = AgentRole.BASIC_USER
            
            return self.segregation_manager.create_agent_context(
                agent_id=auth_request.agent_id,
                tenant_id=auth_request.tenant_id,
                role=role
            )
        
        return None
    
    def _generate_session_token(self, agent_context: AgentContext) -> str:
        """Generate JWT session token"""
        payload = {
            "agent_id": agent_context.agent_id,
            "tenant_id": agent_context.tenant_id,
            "role": agent_context.role.value,
            "permissions": [perm.value for perm in agent_context.permissions],
            "resource_quotas": agent_context.resource_quotas,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def _log_access_attempt(self, agent_context: AgentContext, resource_type: ResourceType, success: bool, reason: str):
        """Log resource access attempt"""
        access_log = AccessLog(
            agent_id=agent_context.agent_id,
            tenant_id=agent_context.tenant_id,
            resource_type=resource_type,
            action="ACCESS_ATTEMPT",
            success=success,
            reason=reason,
            timestamp=datetime.utcnow()
        )
        
        self.segregation_manager.log_access(access_log)

class ContextIsolationManager:
    """Manages execution context isolation for agents"""
    
    def __init__(self):
        self.active_contexts: Dict[str, Dict[str, Any]] = {}
        self.context_locks: Dict[str, asyncio.Lock] = {}
    
    async def create_isolated_execution_context(self, agent_context: AgentContext, tool_name: str) -> str:
        """Create isolated execution context"""
        context_id = f"{agent_context.tenant_id}_{agent_context.agent_id}_{tool_name}_{int(time.time())}"
        
        # Create context lock
        self.context_locks[context_id] = asyncio.Lock()
        
        # Create isolated context
        isolated_context = {
            "context_id": context_id,
            "agent_id": agent_context.agent_id,
            "tenant_id": agent_context.tenant_id,
            "tool_name": tool_name,
            "working_directory": f"/tmp/tenant_{agent_context.tenant_id}",
            "environment_variables": {
                "TENANT_ID": agent_context.tenant_id,
                "AGENT_ID": agent_context.agent_id,
                "ISOLATION_LEVEL": "STRICT"
            },
            "resource_limits": {
                "max_memory_mb": agent_context.resource_quotas.get("max_memory_mb", 512),
                "max_cpu_time_seconds": agent_context.resource_quotas.get("max_cpu_time_seconds", 30),
                "max_file_operations": agent_context.resource_quotas.get("max_file_operations", 100)
            },
            "created_at": datetime.utcnow(),
            "status": "ACTIVE"
        }
        
        self.active_contexts[context_id] = isolated_context
        return context_id
    
    async def execute_in_context(self, context_id: str, operation: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute operation in isolated context"""
        if context_id not in self.active_contexts:
            raise ValueError(f"Context {context_id} not found")
        
        context = self.active_contexts[context_id]
        
        async with self.context_locks[context_id]:
            try:
                # Set context-specific environment
                original_env = {}
                for key, value in context["environment_variables"].items():
                    import os
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value
                
                # Execute operation with resource monitoring
                start_time = time.time()
                result = await operation(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Update context metrics
                context["last_execution_time"] = execution_time
                context["total_executions"] = context.get("total_executions", 0) + 1
                
                return result
                
            finally:
                # Restore original environment
                for key, original_value in original_env.items():
                    import os
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
    
    async def cleanup_context(self, context_id: str):
        """Clean up isolated context"""
        if context_id in self.active_contexts:
            context = self.active_contexts[context_id]
            context["status"] = "CLEANED_UP"
            context["cleaned_up_at"] = datetime.utcnow()
            
            # Remove from active contexts after delay
            await asyncio.sleep(1)
            self.active_contexts.pop(context_id, None)
            self.context_locks.pop(context_id, None)

# Decorator for enforcing context isolation
def with_context_isolation(tool_name: str):
    """Decorator to enforce context isolation for tool execution"""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(self, parameters: Dict[str, Any], agent_context: AgentContext = None, *args, **kwargs):
            if not agent_context:
                return {"error": "Agent context required", "code": "NO_CONTEXT"}
            
            isolation_manager = ContextIsolationManager()
            context_id = await isolation_manager.create_isolated_execution_context(agent_context, tool_name)
            
            try:
                # Execute in isolated context
                result = await isolation_manager.execute_in_context(
                    context_id, func, self, parameters, agent_context, *args, **kwargs
                )
                return result
                
            finally:
                await isolation_manager.cleanup_context(context_id)
        
        return wrapper
    return decorator

# Global instances
_auth_middleware = None
_isolation_manager = None

def get_auth_middleware() -> MCPAuthenticationMiddleware:
    """Get global authentication middleware instance"""
    global _auth_middleware
    if _auth_middleware is None:
        _auth_middleware = MCPAuthenticationMiddleware()
    return _auth_middleware

def get_isolation_manager() -> ContextIsolationManager:
    """Get global context isolation manager instance"""
    global _isolation_manager
    if _isolation_manager is None:
        _isolation_manager = ContextIsolationManager()
    return _isolation_manager
