"""
Modular Strands Agent - Replacement for God Object EnhancedStrandsAgent
Uses focused components with Single Responsibility Principle
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import uuid

from .memory import MemoryAgent
from .components import (
    ToolManager, WorkflowEngine, ContextManager
)
from .secure_code_sandbox import SecureCodeExecutor, SecurityLevel
from ..interfaces import (
    ISecurityManager, ICommunicationManager, ILogger,
    IMetricsCollector, IHealthChecker
)
from ..di_container import resolve, get_container

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for modular Strands agent"""
    agent_id: str
    agent_type: str = "modular_strands"
    
    # Component configurations
    max_concurrent_tools: int = 10
    max_concurrent_workflows: int = 5
    max_contexts: int = 100
    
    # Security settings
    security_level: SecurityLevel = SecurityLevel.STRICT
    enable_authentication: bool = True
    enable_code_execution: bool = True
    
    # Performance settings
    tool_timeout: float = 30.0
    workflow_timeout: float = 300.0
    context_cleanup_interval: int = 3600
    
    # Feature flags
    enable_communication: bool = True
    enable_database: bool = True
    enable_observability: bool = True


class ModularStrandsAgent(MemoryAgent):
    """
    Modular Strands Agent - Enterprise Grade
    
    Replaces the monolithic EnhancedStrandsAgent with focused components:
    - ToolManager: Tool registration and execution
    - WorkflowEngine: Workflow orchestration
    - ContextManager: Context and memory management
    - ObserverManager: Observability and monitoring
    - SecurityManager: Authentication and authorization
    - CommunicationManager: Agent-to-agent communication
    - DatabaseManager: Database connectivity and operations
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config.agent_id, config.agent_type)
        self.config = config
        
        # Initialize core components
        self._init_components()
        
        # Component lifecycle
        self._initialized = False
        self._shutdown_in_progress = False
        
        logger.info(f"ModularStrandsAgent created: {config.agent_id}")
    
    def _init_components(self):
        """Initialize all components using DI container"""
        # Core components - directly initialized for now
        # TODO: Move these to DI container registration
        self.tool_manager = ToolManager(
            max_concurrent_executions=self.config.max_concurrent_tools
        )
        
        # Workflow orchestration
        self.workflow_engine = WorkflowEngine(
            tool_manager=self.tool_manager,
            max_concurrent_workflows=self.config.max_concurrent_workflows
        )
        
        # Context and memory management
        self.context_manager = ContextManager(
            agent_id=self.config.agent_id,
            max_contexts=self.config.max_contexts
        )
        
        # Interface-based components - resolved via DI container
        self.security_manager: Optional[ISecurityManager] = None
        self.communication_manager: Optional[ICommunicationManager] = None
        self.logger: Optional[ILogger] = None
        self.metrics_collector: Optional[IMetricsCollector] = None
        self.health_checker: Optional[IHealthChecker] = None
        
        # Secure code execution
        self.code_executor = None
        if self.config.enable_code_execution:
            self.code_executor = SecureCodeExecutor(self.config.security_level)
        
        logger.info("All components initialized")
    
    async def initialize(self, **kwargs) -> bool:
        """Initialize the agent and all components"""
        if self._initialized:
            return True
        
        try:
            logger.info(f"Initializing ModularStrandsAgent {self.config.agent_id}")
            
            # Initialize base agent (BaseAgent doesn't have initialize method, so just continue)
            
            # Resolve components from DI container
            await self._resolve_dependencies()
            
            # Initialize components that need async setup
            init_tasks = []
            
            if self.security_manager and hasattr(self.security_manager, 'initialize'):
                init_tasks.append(self.security_manager.initialize({
                    'agent_id': self.config.agent_id,
                    'security_level': self.config.security_level.value
                }))
            
            if self.communication_manager and hasattr(self.communication_manager, 'initialize'):
                init_tasks.append(self.communication_manager.initialize({
                    'agent_id': self.config.agent_id
                }))
            
            # Wait for all components to initialize
            if init_tasks:
                await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Register core tools
            await self._register_core_tools()
            
            # Register core workflows
            await self._register_core_workflows()
            
            self._initialized = True
            self._start_time = time.time()
            
            logger.info(f"ModularStrandsAgent {self.config.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            return False
    
    async def _resolve_dependencies(self):
        """Resolve dependencies from DI container"""
        container = get_container()
        
        # Try to resolve optional dependencies
        try:
            if self.config.enable_authentication:
                self.security_manager = await resolve(ISecurityManager)
        except Exception:
            logger.warning("ISecurityManager not available in DI container")
        
        try:
            if self.config.enable_communication:
                self.communication_manager = await resolve(ICommunicationManager)
        except Exception:
            logger.warning("ICommunicationManager not available in DI container")
        
        try:
            self.logger = await resolve(ILogger)
        except Exception:
            logger.debug("ILogger not available in DI container, using standard logger")
        
        try:
            if self.config.enable_observability:
                self.metrics_collector = await resolve(IMetricsCollector)
        except Exception:
            logger.debug("IMetricsCollector not available in DI container")
        
        try:
            self.health_checker = await resolve(IHealthChecker)
        except Exception:
            logger.debug("IHealthChecker not available in DI container")
    
    async def _register_core_tools(self):
        """Register core agent tools"""
        # Memory tools
        await self.tool_manager.register_tool(
            name="store_memory",
            handler=self._tool_store_memory,
            description="Store data in agent memory",
            category="memory"
        )
        
        await self.tool_manager.register_tool(
            name="retrieve_memory", 
            handler=self._tool_retrieve_memory,
            description="Retrieve data from agent memory",
            category="memory"
        )
        
        # Context tools
        await self.tool_manager.register_tool(
            name="get_context",
            handler=self._tool_get_context,
            description="Get current context information",
            category="context"
        )
        
        # Code execution (if enabled)
        if self.code_executor:
            await self.tool_manager.register_tool(
                name="execute_code",
                handler=self._tool_execute_code,
                description="Execute code safely in sandbox",
                category="execution",
                requires_auth=True
            )
        
        # System tools
        await self.tool_manager.register_tool(
            name="get_agent_status",
            handler=self._tool_get_agent_status,
            description="Get agent status and metrics",
            category="system"
        )
        
        logger.info("Core tools registered")
    
    async def _register_core_workflows(self):
        """Register core workflows"""
        from .components.workflow_engine import WorkflowDefinition, WorkflowStep
        
        # Health check workflow
        health_workflow = WorkflowDefinition(
            id="health_check",
            name="Agent Health Check",
            description="Comprehensive agent health check",
            steps=[
                WorkflowStep(
                    id="check_components",
                    name="Check Component Health",
                    tool_name="get_agent_status",
                    parameters={"include_components": True}
                ),
                WorkflowStep(
                    id="check_memory",
                    name="Check Memory Usage",
                    tool_name="get_context",
                    parameters={"include_stats": True},
                    dependencies=["check_components"]
                )
            ]
        )
        
        await self.workflow_engine.register_workflow(health_workflow)
        logger.info("Core workflows registered")
    
    # Tool Execution Interface
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any],
                          session_id: str = None, auth_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a tool with comprehensive tracking"""
        try:
            # Get or create context
            if session_id:
                context = await self.context_manager.get_or_create_context(session_id)
                execution_context = {"session_id": session_id, "context_id": context.context_id}
            else:
                execution_context = None
            
            # Execute tool
            execution = await self.tool_manager.execute_tool(
                tool_name=tool_name,
                parameters=parameters,
                auth_context=auth_context,
                execution_context=execution_context
            )
            
            # Record in context if available
            if session_id:
                await self.context_manager.add_tool_execution(
                    session_id, tool_name, parameters, 
                    execution.result, execution.error
                )
            
            # Record metrics
            if self.metrics_collector:
                await self.metrics_collector.record_timing(
                    "tool_execution_time", execution.execution_time * 1000,
                    {"tool_name": tool_name, "status": execution.status.value}
                )
            
            # Return structured result
            return {
                "success": execution.status.value in ["completed", "success"],
                "result": execution.result,
                "error": execution.error,
                "execution_time": execution.execution_time,
                "execution_id": execution.execution_id,
                "metadata": execution.metadata
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "execution_time": 0.0
            }
    
    # Workflow Execution Interface
    async def execute_workflow(self, workflow_id: str, initial_context: Dict[str, Any] = None,
                              session_id: str = None) -> Dict[str, Any]:
        """Execute a workflow"""
        try:
            # Execute workflow
            execution = await self.workflow_engine.execute_workflow(workflow_id, initial_context)
            
            # Record in context if available
            if session_id:
                await self.context_manager.add_tool_execution(
                    session_id, f"workflow_{workflow_id}", initial_context or {},
                    execution.context, execution.error
                )
            
            # Record metrics
            if self.metrics_collector:
                await self.metrics_collector.record_timing(
                    "workflow_execution_time", execution.total_execution_time * 1000,
                    {"workflow_id": workflow_id, "status": execution.status.value}
                )
            
            return {
                "success": execution.status.value == "completed",
                "result": execution.context,
                "error": execution.error,
                "execution_time": execution.total_execution_time,
                "execution_id": execution.execution_id,
                "status": execution.status.value
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {workflow_id} - {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}"
            }
    
    # Memory Interface
    async def store_memory(self, key: str, value: Any, session_id: str = None,
                          memory_type: str = "short_term") -> bool:
        """Store data in memory"""
        if session_id:
            return await self.context_manager.store_memory(session_id, key, value, memory_type)
        else:
            # Global agent memory
            return await self.context_manager.memory_manager.store_short_term(
                f"global:{key}", value
            )
    
    async def retrieve_memory(self, key: str, session_id: str = None) -> Any:
        """Retrieve data from memory"""
        if session_id:
            return await self.context_manager.retrieve_memory(session_id, key)
        else:
            # Global agent memory
            return await self.context_manager.memory_manager.retrieve(f"global:{key}")
    
    # Context Interface
    async def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get context information"""
        context = await self.context_manager.get_context(session_id)
        if not context:
            return None
        
        return {
            "session_id": context.session_id,
            "context_id": context.context_id,
            "created_at": context.created_at.isoformat(),
            "last_activity": context.last_activity.isoformat(),
            "conversation_entries": len(context.conversation_history),
            "tool_executions": len(context.tool_executions),
            "variables": context.variables,
            "metadata": context.metadata
        }
    
    # Communication Interface
    async def send_message(self, recipient_id: str, message: Dict[str, Any]) -> bool:
        """Send message to another agent"""
        if not self.communication_manager:
            logger.warning("Communication not enabled")
            return False
        
        return await self.communication_manager.send_message(recipient_id, message)
    
    async def broadcast_message(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected agents"""
        if not self.communication_manager:
            logger.warning("Communication not enabled")
            return 0
        
        return await self.communication_manager.broadcast_message(message)
    
    # Tool Implementations
    async def _tool_store_memory(self, key: str, value: Any, 
                               session_id: str = None, memory_type: str = "short_term") -> Dict[str, Any]:
        """Tool implementation for storing memory"""
        success = await self.store_memory(key, value, session_id, memory_type)
        return {"success": success, "key": key, "memory_type": memory_type}
    
    async def _tool_retrieve_memory(self, key: str, session_id: str = None) -> Dict[str, Any]:
        """Tool implementation for retrieving memory"""
        value = await self.retrieve_memory(key, session_id)
        return {"success": value is not None, "key": key, "value": value}
    
    async def _tool_get_context(self, session_id: str, include_stats: bool = False) -> Dict[str, Any]:
        """Tool implementation for getting context"""
        context = await self.get_context(session_id)
        if not context:
            return {"success": False, "error": "Context not found"}
        
        result = {"success": True, "context": context}
        
        if include_stats:
            stats = await self.context_manager.get_context_stats()
            result["stats"] = stats
        
        return result
    
    async def _tool_execute_code(self, code: str, language: str = "python",
                               session_id: str = None) -> Dict[str, Any]:
        """Tool implementation for secure code execution"""
        if not self.code_executor:
            return {
                "success": False,
                "error": "Code execution not enabled"
            }
        
        # Only support Python for security
        if language != "python":
            return {
                "success": False,
                "error": f"Language '{language}' not supported. Only Python is allowed."
            }
        
        # Get context for code execution
        context_vars = {}
        if session_id:
            context_vars["session_id"] = session_id
            context_vars["agent_id"] = self.config.agent_id
        
        # Execute in secure sandbox
        result = await self.code_executor.execute_safe_code(
            code=code,
            tool_name="execute_code",
            context=context_vars,
            security_level=self.config.security_level
        )
        
        return {
            "success": result["success"],
            "output": result.get("output", ""),
            "error": result.get("error", ""),
            "execution_time": result.get("execution_time", 0.0),
            "security_violations": result.get("security_violations", [])
        }
    
    async def _tool_get_agent_status(self, include_components: bool = False) -> Dict[str, Any]:
        """Tool implementation for getting agent status"""
        status = {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "initialized": self._initialized,
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
            "configuration": {
                "max_concurrent_tools": self.config.max_concurrent_tools,
                "max_concurrent_workflows": self.config.max_concurrent_workflows,
                "security_level": self.config.security_level.value
            }
        }
        
        if include_components:
            components = {}
            
            # Tool manager metrics
            components["tool_manager"] = self.tool_manager.get_tool_metrics()
            
            # Workflow engine metrics
            components["workflow_engine"] = self.workflow_engine.get_metrics()
            
            # Context manager stats
            components["context_manager"] = await self.context_manager.get_context_stats()
            
            # Component availability
            components["available_components"] = {
                "security_manager": self.security_manager is not None,
                "communication_manager": self.communication_manager is not None,
                "logger": self.logger is not None,
                "metrics_collector": self.metrics_collector is not None,
                "health_checker": self.health_checker is not None,
                "code_executor": self.code_executor is not None
            }
            
            status["components"] = components
        
        return {"success": True, "status": status}
    
    # Agent Lifecycle
    async def start(self):
        """Start the agent"""
        if not self._initialized:
            await self.initialize()
        
        self._start_time = time.time()
        logger.info(f"ModularStrandsAgent {self.config.agent_id} started")
    
    async def shutdown(self):
        """Graceful shutdown"""
        if self._shutdown_in_progress:
            return
        
        self._shutdown_in_progress = True
        logger.info(f"Shutting down ModularStrandsAgent {self.config.agent_id}")
        
        # Shutdown components in reverse order
        shutdown_tasks = []
        
        if self.communication_manager and hasattr(self.communication_manager, 'shutdown'):
            shutdown_tasks.append(self.communication_manager.shutdown())
        
        if self.security_manager and hasattr(self.security_manager, 'shutdown'):
            shutdown_tasks.append(self.security_manager.shutdown())
        
        # Core components
        shutdown_tasks.extend([
            self.context_manager.shutdown(),
            self.workflow_engine.shutdown(),
            self.tool_manager.shutdown()
        ])
        
        # Wait for all shutdowns
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Base agent cleanup (BaseAgent doesn't have shutdown method)
        
        logger.info(f"ModularStrandsAgent {self.config.agent_id} shutdown complete")
    
    # Convenience Methods
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        tools = self.tool_manager.registry.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "parameters": tool.parameters,
                "requires_auth": tool.requires_auth
            }
            for tool in tools
        ]
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List available workflows"""
        workflows = self.workflow_engine.list_workflows()
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "steps": len(workflow.steps),
                "version": workflow.version
            }
            for workflow in workflows
        ]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent metrics"""
        metrics = {
            "agent": await self._tool_get_agent_status(include_components=True),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return metrics
    
    # Implement abstract methods from BaseAgent
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response"""
        # Delegate to parent class implementation
        return await super().process_message(message)
    
    async def _process_message_impl(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Actual message processing implementation"""
        # Extract message components
        message_type = message.get("type", "unknown")
        payload = message.get("payload", {})
        sender_id = message.get("sender_id")
        
        # Route message based on type
        if message_type == "tool_execution":
            tool_name = payload.get("tool_name")
            parameters = payload.get("parameters", {})
            session_id = payload.get("session_id")
            
            result = await self.execute_tool(tool_name, parameters, session_id)
            
            return {
                "type": "tool_execution_response",
                "payload": result,
                "recipient_id": sender_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif message_type == "workflow_execution":
            workflow_id = payload.get("workflow_id")
            initial_context = payload.get("context", {})
            session_id = payload.get("session_id")
            
            result = await self.execute_workflow(workflow_id, initial_context, session_id)
            
            return {
                "type": "workflow_execution_response", 
                "payload": result,
                "recipient_id": sender_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif message_type == "status_request":
            status = await self._tool_get_agent_status(include_components=True)
            
            return {
                "type": "status_response",
                "payload": status,
                "recipient_id": sender_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif message_type == "memory_operation":
            operation = payload.get("operation")
            
            if operation == "store":
                key = payload.get("key")
                value = payload.get("value")
                session_id = payload.get("session_id")
                memory_type = payload.get("memory_type", "short_term")
                
                success = await self.store_memory(key, value, session_id, memory_type)
                
                return {
                    "type": "memory_operation_response",
                    "payload": {"success": success, "operation": "store", "key": key},
                    "recipient_id": sender_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif operation == "retrieve":
                key = payload.get("key")
                session_id = payload.get("session_id")
                
                value = await self.retrieve_memory(key, session_id)
                
                return {
                    "type": "memory_operation_response",
                    "payload": {"success": value is not None, "operation": "retrieve", "key": key, "value": value},
                    "recipient_id": sender_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        else:
            # Unknown message type
            return {
                "type": "error_response",
                "payload": {
                    "error": f"Unknown message type: {message_type}",
                    "supported_types": ["tool_execution", "workflow_execution", "status_request", "memory_operation"]
                },
                "recipient_id": sender_id,
                "timestamp": datetime.utcnow().isoformat()
            }


# Factory function for easy creation
def create_modular_strands_agent(agent_id: str, **config_overrides) -> ModularStrandsAgent:
    """Create a ModularStrandsAgent with configuration"""
    config = AgentConfig(agent_id=agent_id, **config_overrides)
    return ModularStrandsAgent(config)