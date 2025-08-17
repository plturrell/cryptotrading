"""
Unified base agent class for the cryptotrading platform.
Consolidates functionality from multiple previous agent base classes.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
import time
import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import json

class AgentStatus(Enum):
    """Agent status enumeration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"
    DEGRADED = "degraded"

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    messages_processed: int = 0
    errors_count: int = 0
    average_response_time: float = 0.0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
class BaseAgent(ABC):
    """
    Enhanced unified base agent class with MCP protocol support, monitoring, and enterprise features.
    
    Features:
    - MCP tool discovery and execution
    - Performance monitoring and metrics
    - Health checks and circuit breaker patterns
    - Structured logging and observability
    - Enterprise-grade error handling
    - Agent lifecycle management
    """
    
    def __init__(self, agent_id: str, agent_type: str, **kwargs):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.created_at = datetime.utcnow()
        self.status = AgentStatus.INITIALIZING
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.agent_id}]")
        
        # Performance monitoring
        self.metrics = AgentMetrics()
        self.start_time = time.time()
        
        # MCP protocol support
        self.mcp_tools = {}
        self.mcp_capabilities = []
        
        # Circuit breaker for resilience
        self.error_threshold = kwargs.get('error_threshold', 10)
        self.circuit_open = False
        self.circuit_reset_time = None
        
        # Initialize agent-specific features
        self._initialize(**kwargs)
        self.status = AgentStatus.ACTIVE
    
    def _initialize(self, **kwargs):
        """Initialize agent-specific configuration"""
        self.logger.info(f"Initializing {self.agent_type} agent with ID: {self.agent_id}")
        
        # Register default MCP tools
        self._register_default_mcp_tools()
        
        # Load agent-specific configuration
        self.config = kwargs.get('config', {})
        
        # Setup health check interval
        self.health_check_interval = kwargs.get('health_check_interval', 60)
        
    def _register_default_mcp_tools(self):
        """Register default MCP tools available to all agents"""
        self.mcp_tools.update({
            "get_agent_status": {
                "name": "get_agent_status",
                "description": "Get current agent status and metrics",
                "parameters": {},
                "handler": self._handle_get_status
            },
            "get_agent_metrics": {
                "name": "get_agent_metrics", 
                "description": "Get agent performance metrics",
                "parameters": {},
                "handler": self._handle_get_metrics
            },
            "health_check": {
                "name": "health_check",
                "description": "Perform agent health check",
                "parameters": {},
                "handler": self._handle_health_check
            }
        })
        
        self.mcp_capabilities.extend([
            "agent_monitoring",
            "health_checks", 
            "metrics_collection"
        ])
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response"""
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if self.circuit_open:
                if time.time() - self.circuit_reset_time > 60:  # 1 minute reset
                    self.circuit_open = False
                    self.logger.info("Circuit breaker reset")
                else:
                    raise Exception("Circuit breaker is open")
            
            # Update status
            self.status = AgentStatus.BUSY
            self.metrics.last_activity = datetime.utcnow()
            
            # Process the message (implemented by subclasses)
            result = await self._process_message_impl(message)
            
            # Update metrics
            self.metrics.messages_processed += 1
            response_time = time.time() - start_time
            self._update_response_time(response_time)
            
            self.status = AgentStatus.ACTIVE
            return result
            
        except Exception as e:
            self.metrics.errors_count += 1
            self.status = AgentStatus.ERROR
            
            # Circuit breaker logic
            if self.metrics.errors_count >= self.error_threshold:
                self.circuit_open = True
                self.circuit_reset_time = time.time()
                self.logger.error("Circuit breaker opened due to excessive errors")
            
            self.logger.error(f"Message processing failed: {e}", exc_info=True)
            raise
    
    @abstractmethod 
    async def _process_message_impl(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Actual message processing implementation (to be overridden by subclasses)"""
        pass
    
    async def start(self):
        """Start the agent with enhanced lifecycle management"""
        try:
            self.logger.info(f"Starting {self.agent_type} agent {self.agent_id}")
            self.status = AgentStatus.ACTIVE
            self.start_time = time.time()
            
            # Start background tasks
            asyncio.create_task(self._health_monitor())
            
            # Agent-specific startup
            await self._on_start()
            
            self.logger.info(f"Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Failed to start agent {self.agent_id}: {e}")
            raise
    
    async def _on_start(self):
        """Override for agent-specific startup logic"""
        pass
    
    async def stop(self):
        """Stop the agent gracefully"""
        try:
            self.logger.info(f"Stopping agent {self.agent_id}")
            self.status = AgentStatus.STOPPED
            
            # Agent-specific cleanup
            await self._on_stop()
            
            self.logger.info(f"Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping agent {self.agent_id}: {e}")
            raise
    
    async def _on_stop(self):
        """Override for agent-specific cleanup logic"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        uptime = time.time() - self.start_time
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "uptime_seconds": uptime,
            "metrics": asdict(self.metrics),
            "mcp_tools_count": len(self.mcp_tools),
            "mcp_capabilities": self.mcp_capabilities,
            "circuit_breaker": {
                "open": self.circuit_open,
                "error_count": self.metrics.errors_count,
                "threshold": self.error_threshold
            }
        }
    
    # MCP Protocol Methods
    async def list_mcp_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools"""
        return [{
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"]
        } for tool in self.mcp_tools.values()]
    
    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an MCP tool"""
        if tool_name not in self.mcp_tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool = self.mcp_tools[tool_name]
        handler = tool["handler"]
        
        try:
            result = await handler(parameters or {})
            return {"success": True, "result": result}
        except Exception as e:
            self.logger.error(f"MCP tool '{tool_name}' failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Tool Handlers
    async def _handle_get_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_agent_status MCP tool"""
        return self.get_status()
    
    async def _handle_get_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_agent_metrics MCP tool"""
        return asdict(self.metrics)
    
    async def _handle_health_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health_check MCP tool"""
        return await self.health_check()
    
    # Health and Monitoring
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = "healthy"
        issues = []
        
        # Check error rate
        if self.metrics.messages_processed > 0:
            error_rate = self.metrics.errors_count / self.metrics.messages_processed
            if error_rate > 0.1:  # 10% error rate threshold
                health_status = "degraded"
                issues.append(f"High error rate: {error_rate:.2%}")
        
        # Check circuit breaker
        if self.circuit_open:
            health_status = "unhealthy"
            issues.append("Circuit breaker is open")
        
        # Check status
        if self.status == AgentStatus.ERROR:
            health_status = "unhealthy"
            issues.append("Agent is in error state")
        
        return {
            "status": health_status,
            "issues": issues,
            "last_check": datetime.utcnow().isoformat(),
            "uptime": time.time() - self.start_time
        }
    
    async def _health_monitor(self):
        """Background health monitoring task"""
        while self.status != AgentStatus.STOPPED:
            try:
                health = await self.health_check()
                if health["status"] != "healthy":
                    self.logger.warning(f"Health check issues: {health['issues']}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)  # Shorter retry interval on error
    
    def _update_response_time(self, response_time: float):
        """Update average response time with exponential moving average"""
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = response_time
        else:
            # EMA with alpha = 0.1
            self.metrics.average_response_time = (0.1 * response_time + 
                                                0.9 * self.metrics.average_response_time)
