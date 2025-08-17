"""
Tool Management Component for Strands Framework
Handles tool registration, execution, and lifecycle management
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import inspect

logger = logging.getLogger(__name__)


class ToolPriority(Enum):
    """Tool execution priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ToolStatus(Enum):
    """Tool execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ToolDefinition:
    """Comprehensive tool definition"""
    name: str
    description: str
    handler: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: ToolPriority = ToolPriority.NORMAL
    timeout: float = 30.0
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    requires_auth: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecution:
    """Tool execution tracking"""
    execution_id: str
    tool_name: str
    parameters: Dict[str, Any]
    status: ToolStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker for tool reliability"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        """Check if execution is allowed"""
        async with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                    return True
                return False
            else:  # half-open
                return True
    
    async def record_success(self):
        """Record successful execution"""
        async with self._lock:
            self.failure_count = 0
            self.state = "closed"
    
    async def record_failure(self):
        """Record failed execution"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ToolRegistry:
    """Registry for managing tool definitions and metadata"""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
    
    async def register_tool(self, tool: ToolDefinition) -> bool:
        """Register a new tool"""
        async with self._lock:
            if tool.name in self._tools:
                logger.warning(f"Tool {tool.name} already registered, overwriting")
            
            self._tools[tool.name] = tool
            
            # Update category index
            if tool.category not in self._categories:
                self._categories[tool.category] = set()
            self._categories[tool.category].add(tool.name)
            
            # Update dependency graph
            self._dependencies[tool.name] = set(tool.dependencies)
            
            logger.info(f"Registered tool: {tool.name} (category: {tool.category})")
            return True
    
    async def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool"""
        async with self._lock:
            if tool_name not in self._tools:
                return False
            
            tool = self._tools[tool_name]
            del self._tools[tool_name]
            
            # Update category index
            if tool.category in self._categories:
                self._categories[tool.category].discard(tool_name)
                if not self._categories[tool.category]:
                    del self._categories[tool.category]
            
            # Update dependency graph
            if tool_name in self._dependencies:
                del self._dependencies[tool_name]
            
            logger.info(f"Unregistered tool: {tool_name}")
            return True
    
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool definition"""
        return self._tools.get(tool_name)
    
    def list_tools(self, category: Optional[str] = None, 
                   tags: Optional[List[str]] = None) -> List[ToolDefinition]:
        """List tools with optional filtering"""
        tools = list(self._tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]
        
        return tools
    
    def get_tool_dependencies(self, tool_name: str) -> Set[str]:
        """Get tool dependencies"""
        return self._dependencies.get(tool_name, set())
    
    def validate_dependencies(self, tool_name: str) -> List[str]:
        """Validate that all dependencies are available"""
        missing_deps = []
        dependencies = self.get_tool_dependencies(tool_name)
        
        for dep in dependencies:
            if dep not in self._tools:
                missing_deps.append(dep)
        
        return missing_deps


class ToolManager:
    """Main tool management system"""
    
    def __init__(self, max_concurrent_executions: int = 10):
        self.registry = ToolRegistry()
        self.max_concurrent_executions = max_concurrent_executions
        
        # Execution tracking
        self._active_executions: Dict[str, ToolExecution] = {}
        self._execution_history: List[ToolExecution] = []
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Concurrency control
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_executions)
        
        # Metrics
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "tools_by_category": {},
            "circuit_breaker_trips": 0
        }
        
        logger.info(f"ToolManager initialized with max {max_concurrent_executions} concurrent executions")
    
    async def register_tool(self, name: str, handler: Callable, 
                          description: str = "", category: str = "general",
                          priority: ToolPriority = ToolPriority.NORMAL,
                          timeout: float = 30.0, requires_auth: bool = False,
                          **kwargs) -> bool:
        """Register a tool with the manager"""
        # Extract parameters from function signature
        parameters = self._extract_parameters(handler)
        
        tool_definition = ToolDefinition(
            name=name,
            description=description,
            handler=handler,
            parameters=parameters,
            priority=priority,
            timeout=timeout,
            category=category,
            requires_auth=requires_auth,
            **kwargs
        )
        
        success = await self.registry.register_tool(tool_definition)
        
        if success:
            # Initialize circuit breaker
            self._circuit_breakers[name] = CircuitBreaker()
            
            # Update metrics
            if category not in self._metrics["tools_by_category"]:
                self._metrics["tools_by_category"][category] = 0
            self._metrics["tools_by_category"][category] += 1
        
        return success
    
    def _extract_parameters(self, handler: Callable) -> Dict[str, Any]:
        """Extract parameter definitions from function signature"""
        sig = inspect.signature(handler)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name not in ['self', 'cls']:
                param_info = {
                    "required": param.default == inspect.Parameter.empty,
                }
                
                # Extract type information
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation)
                
                # Extract default value
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                
                parameters[param_name] = param_info
        
        return parameters
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any],
                          auth_context: Optional[Dict[str, Any]] = None,
                          execution_context: Optional[Dict[str, Any]] = None) -> ToolExecution:
        """Execute a tool with comprehensive tracking and error handling"""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Get tool definition
        tool = self.registry.get_tool(tool_name)
        if not tool:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            return self._create_failed_execution(execution_id, tool_name, parameters, error_msg)
        
        # Check circuit breaker
        circuit_breaker = self._circuit_breakers.get(tool_name)
        if circuit_breaker and not await circuit_breaker.can_execute():
            error_msg = f"Tool '{tool_name}' circuit breaker is open"
            logger.warning(error_msg)
            self._metrics["circuit_breaker_trips"] += 1
            return self._create_failed_execution(execution_id, tool_name, parameters, error_msg)
        
        # Validate dependencies
        missing_deps = self.registry.validate_dependencies(tool_name)
        if missing_deps:
            error_msg = f"Missing dependencies for '{tool_name}': {missing_deps}"
            logger.error(error_msg)
            return self._create_failed_execution(execution_id, tool_name, parameters, error_msg)
        
        # Create execution record
        execution = ToolExecution(
            execution_id=execution_id,
            tool_name=tool_name,
            parameters=parameters,
            status=ToolStatus.RUNNING,
            started_at=datetime.utcnow(),
            metadata={
                "auth_context": auth_context,
                "execution_context": execution_context
            }
        )
        
        self._active_executions[execution_id] = execution
        
        try:
            # Execute with concurrency control
            async with self._execution_semaphore:
                logger.info(f"Executing tool: {tool_name} (id: {execution_id})")
                
                # Execute the tool handler
                if asyncio.iscoroutinefunction(tool.handler):
                    result = await asyncio.wait_for(
                        tool.handler(**parameters),
                        timeout=tool.timeout
                    )
                else:
                    # Run in thread pool for blocking functions
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self._executor, tool.handler, **parameters
                        ),
                        timeout=tool.timeout
                    )
                
                # Record success
                execution.status = ToolStatus.COMPLETED
                execution.result = result
                execution.completed_at = datetime.utcnow()
                execution.execution_time = time.time() - start_time
                
                if circuit_breaker:
                    await circuit_breaker.record_success()
                
                self._metrics["successful_executions"] += 1
                logger.info(f"Tool {tool_name} completed successfully in {execution.execution_time:.2f}s")
        
        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out after {tool.timeout}s"
            execution.status = ToolStatus.TIMEOUT
            execution.error = error_msg
            execution.completed_at = datetime.utcnow()
            execution.execution_time = time.time() - start_time
            
            if circuit_breaker:
                await circuit_breaker.record_failure()
            
            self._metrics["failed_executions"] += 1
            logger.error(error_msg)
        
        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            execution.status = ToolStatus.FAILED
            execution.error = error_msg
            execution.completed_at = datetime.utcnow()
            execution.execution_time = time.time() - start_time
            
            if circuit_breaker:
                await circuit_breaker.record_failure()
            
            self._metrics["failed_executions"] += 1
            logger.error(error_msg, exc_info=True)
        
        finally:
            # Clean up
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]
            
            # Add to history
            self._execution_history.append(execution)
            
            # Update metrics
            self._metrics["total_executions"] += 1
            self._update_average_execution_time(execution.execution_time)
            
            # Trim history if too large
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-500:]
        
        return execution
    
    def _create_failed_execution(self, execution_id: str, tool_name: str, 
                               parameters: Dict[str, Any], error: str) -> ToolExecution:
        """Create a failed execution record"""
        execution = ToolExecution(
            execution_id=execution_id,
            tool_name=tool_name,
            parameters=parameters,
            status=ToolStatus.FAILED,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            error=error,
            execution_time=0.0
        )
        
        self._execution_history.append(execution)
        self._metrics["failed_executions"] += 1
        self._metrics["total_executions"] += 1
        
        return execution
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time metric"""
        total = self._metrics["total_executions"]
        current_avg = self._metrics["average_execution_time"]
        
        # Calculate new average
        self._metrics["average_execution_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    def get_active_executions(self) -> List[ToolExecution]:
        """Get currently active executions"""
        return list(self._active_executions.values())
    
    def get_execution_history(self, limit: int = 100) -> List[ToolExecution]:
        """Get recent execution history"""
        return self._execution_history[-limit:]
    
    def get_tool_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get tool execution metrics"""
        if tool_name:
            # Tool-specific metrics
            tool_executions = [e for e in self._execution_history if e.tool_name == tool_name]
            
            if not tool_executions:
                return {"tool_name": tool_name, "executions": 0}
            
            successful = len([e for e in tool_executions if e.status == ToolStatus.COMPLETED])
            failed = len([e for e in tool_executions if e.status == ToolStatus.FAILED])
            avg_time = sum(e.execution_time for e in tool_executions) / len(tool_executions)
            
            return {
                "tool_name": tool_name,
                "total_executions": len(tool_executions),
                "successful_executions": successful,
                "failed_executions": failed,
                "success_rate": successful / len(tool_executions),
                "average_execution_time": avg_time,
                "circuit_breaker_state": self._circuit_breakers.get(tool_name, {}).state if tool_name in self._circuit_breakers else "N/A"
            }
        else:
            # Overall metrics
            return self._metrics.copy()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down ToolManager")
        
        # Wait for active executions to complete (with timeout)
        if self._active_executions:
            logger.info(f"Waiting for {len(self._active_executions)} active executions to complete")
            
            timeout = 30.0  # 30 second timeout
            start_time = time.time()
            
            while self._active_executions and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if self._active_executions:
                logger.warning(f"Forcibly terminating {len(self._active_executions)} active executions")
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        logger.info("ToolManager shutdown complete")


# Decorator for easy tool registration
def tool_registry_decorator(registry: ToolRegistry, name: str = None, 
                          description: str = "", category: str = "general",
                          priority: ToolPriority = ToolPriority.NORMAL,
                          **kwargs):
    """Decorator for registering tools with a registry"""
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        
        # Create tool definition
        tool_def = ToolDefinition(
            name=tool_name,
            description=description,
            handler=func,
            category=category,
            priority=priority,
            **kwargs
        )
        
        # Register immediately (this is synchronous)
        # In practice, you'd want to register during startup
        func._tool_definition = tool_def
        
        return func
    
    return decorator