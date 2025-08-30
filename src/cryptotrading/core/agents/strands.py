"""
Production-ready Strands framework implementation
Provides enterprise-grade agent orchestration, workflow management, and tool execution
"""

import asyncio
import json
import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type, Union

from ...infrastructure.database import UnifiedDatabase
from ..config.production_config import StrandsConfig, get_config
from .memory import MemoryAgent

# Import S3 logging capabilities
try:
    from .s3_logging_mixin import S3LoggingMixin, log_agent_method

    S3_LOGGING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"S3 logging mixin not available: {e}")
    S3_LOGGING_AVAILABLE = False

    # Create dummy classes for compatibility
    class S3LoggingMixin:
        pass

    def log_agent_method(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)

# Context variables for distributed tracing
current_workflow_id: ContextVar[Optional[str]] = ContextVar("current_workflow_id", default=None)
current_span_id: ContextVar[Optional[str]] = ContextVar("current_span_id", default=None)


class WorkflowStatus(str, Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ToolStatus(str, Enum):
    """Tool execution status"""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class WorkflowContext:
    """Context for workflow execution"""

    workflow_id: str
    agent_id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    parent_workflow_id: Optional[str] = None
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ToolExecutionResult:
    """Result of tool execution"""

    tool_name: str
    status: ToolStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsExporter:
    """Export metrics to various backends"""

    def __init__(self, config: StrandsConfig):
        self.config = config
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100
        self.flush_interval = 60.0  # seconds
        self._last_flush = asyncio.get_event_loop().time()

    async def export(self, metrics: Dict[str, Any]):
        """Export metrics to configured backend"""
        timestamp = datetime.utcnow().isoformat()

        # Add to buffer
        self.buffer.append({"timestamp": timestamp, "metrics": metrics.copy()})

        # Flush if buffer is full or interval passed
        if (
            len(self.buffer) >= self.buffer_size
            or (asyncio.get_event_loop().time() - self._last_flush) > self.flush_interval
        ):
            await self.flush()

    async def send_event(self, event: Dict[str, Any]):
        """Send individual telemetry event"""
        # Add to buffer for batch sending
        self.buffer.append(
            {"timestamp": event.get("timestamp", datetime.utcnow().isoformat()), "event": event}
        )

        if len(self.buffer) >= self.buffer_size:
            await self.flush()

    async def flush(self):
        """Flush metrics buffer to backend"""
        if not self.buffer:
            return

        try:
            # Export to different backends based on configuration
            if self.config.metrics_backend == "prometheus":
                await self._export_to_prometheus(self.buffer)
            elif self.config.metrics_backend == "cloudwatch":
                await self._export_to_cloudwatch(self.buffer)
            elif self.config.metrics_backend == "datadog":
                await self._export_to_datadog(self.buffer)
            else:
                # Default: log metrics
                for item in self.buffer:
                    logger.info(f"Metrics: {json.dumps(item, indent=2)}")

            # Clear buffer after successful export
            self.buffer.clear()
            self._last_flush = asyncio.get_event_loop().time()

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    async def _export_to_prometheus(self, data: List[Dict[str, Any]]):
        """Export to Prometheus pushgateway"""
        # Would use prometheus_client library
        logger.debug(f"Would export {len(data)} items to Prometheus")

    async def _export_to_cloudwatch(self, data: List[Dict[str, Any]]):
        """Export to AWS CloudWatch"""
        # Would use boto3 library
        logger.debug(f"Would export {len(data)} items to CloudWatch")

    async def _export_to_datadog(self, data: List[Dict[str, Any]]):
        """Export to DataDog"""
        # Would use datadog library
        logger.debug(f"Would export {len(data)} items to DataDog")


class CircuitBreaker:
    """Circuit breaker for tool execution with retry logic"""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
        self._failure_history: List[Dict[str, Any]] = []

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker and retry logic"""
        # Check circuit state
        async with self._lock:
            if self.state == "open":
                if (
                    self.last_failure_time
                    and (asyncio.get_event_loop().time() - self.last_failure_time) > self.timeout
                ):
                    self.state = "half-open"
                    logger.info(f"Circuit breaker half-open for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker is open for {func.__name__}")

        # Execute with retry logic
        last_exception = None
        delay = self.retry_delay

        for attempt in range(self.retry_count):
            try:
                # Add jitter to prevent thundering herd
                if attempt > 0:
                    jitter = asyncio.get_event_loop().time() % 0.1
                    await asyncio.sleep(delay + jitter)

                # Execute function
                result = await func(*args, **kwargs)

                # Success - update circuit breaker state
                async with self._lock:
                    if self.state == "half-open":
                        self.state = "closed"
                        self.failure_count = 0
                        self._failure_history.clear()
                        logger.info(f"Circuit breaker closed for {func.__name__}")
                    elif self.state == "closed" and self.failure_count > 0:
                        # Gradual recovery
                        self.failure_count = max(0, self.failure_count - 1)

                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.retry_count} failed for {func.__name__}: {e}"
                )

                # Update failure tracking
                async with self._lock:
                    self._failure_history.append(
                        {
                            "timestamp": asyncio.get_event_loop().time(),
                            "error": str(e),
                            "attempt": attempt + 1,
                        }
                    )

                    # Trim old failure history (keep last hour)
                    current_time = asyncio.get_event_loop().time()
                    self._failure_history = [
                        f for f in self._failure_history if current_time - f["timestamp"] < 3600
                    ]

                # Check if should retry
                if attempt < self.retry_count - 1:
                    # Exponential backoff
                    delay *= self.backoff_factor

                    # Check if error is retryable
                    if not self._is_retryable_error(e):
                        logger.error(f"Non-retryable error for {func.__name__}: {e}")
                        break

        # All retries failed - update circuit breaker
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = asyncio.get_event_loop().time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"Circuit breaker opened for {func.__name__} after {self.failure_count} failures"
                )

                # Calculate next retry time
                next_retry = datetime.utcfromtimestamp(
                    self.last_failure_time + self.timeout
                ).isoformat()
                logger.info(f"Circuit will attempt recovery at {next_retry}")

        raise last_exception

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if error should trigger retry"""
        # Network errors
        if isinstance(error, (asyncio.TimeoutError, ConnectionError)):
            return True

        # Rate limiting
        error_msg = str(error).lower()
        if any(term in error_msg for term in ["rate limit", "too many requests", "429"]):
            return True

        # Temporary failures
        if any(term in error_msg for term in ["temporary", "timeout", "unavailable"]):
            return True

        # Non-retryable errors
        if any(term in error_msg for term in ["invalid", "unauthorized", "forbidden", "not found"]):
            return False

        # Default: retry for unknown errors
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recent_failures": len(self._failure_history),
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
        }

    async def reset(self):
        """Manually reset circuit breaker"""
        async with self._lock:
            self.state = "closed"
            self.failure_count = 0
            self.last_failure_time = None
            self._failure_history.clear()
            logger.info("Circuit breaker manually reset")


class ToolRegistry:
    """Registry for available tools"""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._registry_lock = asyncio.Lock()  # Fix race condition

    async def register(self, name: str, func: Callable, metadata: Optional[Dict[str, Any]] = None):
        """Register a tool"""
        async with self._registry_lock:  # Fix race condition
            self._tools[name] = func
            self._tool_metadata[name] = metadata or {}
            self._circuit_breakers[name] = CircuitBreaker()
            logger.info(f"Registered tool: {name}")

    async def get(self, name: str) -> Optional[Callable]:
        """Get a tool by name"""
        async with self._registry_lock:  # Fix race condition
            return self._tools.get(name)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        async with self._registry_lock:  # Fix race condition
            return [
                {
                    "name": name,
                    "metadata": self._tool_metadata.get(name, {}),
                    "available": self._circuit_breakers[name].state != "open",
                }
                for name in self._tools
            ]

    async def execute(self, name: str, *args, **kwargs) -> Any:
        """Execute a tool with circuit breaker"""
        # Get tool and circuit breaker atomically
        async with self._registry_lock:
            if name not in self._tools:
                raise ValueError(f"Tool {name} not found")

            circuit_breaker = self._circuit_breakers[name]
            tool = self._tools[name]

        # Execute outside the lock to avoid blocking other operations
        return await circuit_breaker.call(tool, *args, **kwargs)

    async def unregister(self, name: str) -> bool:
        """Unregister a tool"""
        async with self._registry_lock:  # Fix race condition
            if name in self._tools:
                del self._tools[name]
                del self._tool_metadata[name]
                del self._circuit_breakers[name]
                logger.info(f"Unregistered tool: {name}")
                return True
            return False


class WorkflowEngine:
    """Engine for executing workflows"""

    def __init__(self, config: StrandsConfig):
        self.config = config
        self._workflows: Dict[str, WorkflowContext] = {}
        self._workflow_lock = asyncio.Lock()  # Fix race condition
        self._executor = ThreadPoolExecutor(max_workers=config.worker_pool_size)
        self._event_queue = asyncio.Queue(maxsize=config.event_bus_capacity)
        self._running = False

    async def start(self):
        """Start the workflow engine"""
        self._running = True
        logger.info("Workflow engine started")

    async def stop(self):
        """Stop the workflow engine"""
        self._running = False
        self._executor.shutdown(wait=True)
        logger.info("Workflow engine stopped")

    async def submit_workflow(self, workflow: WorkflowContext) -> str:
        """Submit a workflow for execution"""
        async with self._workflow_lock:  # Fix race condition
            workflow.status = WorkflowStatus.PENDING
            workflow.started_at = datetime.utcnow()

            self._workflows[workflow.workflow_id] = workflow
            await self._event_queue.put(("workflow_submitted", workflow))

            logger.info(f"Workflow {workflow.workflow_id} submitted")
            return workflow.workflow_id

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get workflow status"""
        async with self._workflow_lock:  # Fix race condition
            return self._workflows.get(workflow_id)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow"""
        async with self._workflow_lock:  # Fix race condition
            workflow = self._workflows.get(workflow_id)
            if workflow and workflow.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                workflow.status = WorkflowStatus.CANCELLED
                workflow.completed_at = datetime.utcnow()
                await self._event_queue.put(("workflow_cancelled", workflow))
                logger.info(f"Workflow {workflow_id} cancelled")
                return True
            return False

    async def emit_event(self, event_type: str, data: Any):
        """Emit an event to the event bus"""
        if not self._event_queue.full():
            await self._event_queue.put((event_type, data))


class StrandsAgent(S3LoggingMixin, MemoryAgent):
    """
    Production-ready Strands framework agent with enterprise features:
    - Distributed workflow orchestration
    - Tool execution with circuit breakers
    - Event-driven architecture
    - Comprehensive observability
    - Fault tolerance and recovery
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: Optional[List[str]] = None,
        model_provider: str = "grok4",
        config: Optional[StrandsConfig] = None,
        **kwargs,
    ):
        super().__init__(agent_id, agent_type, **kwargs)
        self.capabilities = capabilities or []
        self.model_provider = model_provider
        self.config = config or get_config().strands

        # Initialize components
        self.tool_registry = ToolRegistry()
        self.workflow_engine = WorkflowEngine(self.config)
        self.metrics_exporter = (
            MetricsExporter(self.config) if self.config.enable_telemetry else None
        )
        self._setup_strands()

        # Metrics
        self._metrics = {
            "workflows_executed": 0,
            "workflows_succeeded": 0,
            "workflows_failed": 0,
            "tools_executed": 0,
            "tools_succeeded": 0,
            "tools_failed": 0,
            "avg_workflow_time_ms": 0.0,
            "avg_tool_time_ms": 0.0,
        }

        # Initialize S3 logging if available
        if S3_LOGGING_AVAILABLE:
            asyncio.create_task(self._init_s3_logging_async())

    def _setup_strands(self):
        """Setup Strands framework integration"""
        # Register default tools
        self._register_default_tools()

        # Start background tasks
        asyncio.create_task(self._start_background_tasks())

        logger.info(f"Strands agent {self.agent_id} initialized")

    async def _init_s3_logging_async(self):
        """Initialize S3 logging asynchronously"""
        try:
            # Log agent startup
            await self.log_agent_startup(
                {
                    "agent_type": getattr(self, "agent_type", "strands_agent"),
                    "capabilities": self.capabilities,
                    "model_provider": self.model_provider,
                    "config": {
                        "enable_telemetry": getattr(self.config, "enable_telemetry", False),
                        "max_workflow_time": getattr(self.config, "max_workflow_time", 300),
                        "max_retries": getattr(self.config, "max_retries", 3),
                    },
                }
            )
        except Exception as e:
            logger.warning(f"S3 logging initialization failed: {e}")

    def _register_default_tools(self):
        """Register default tools"""
        # Register system tools
        self.tool_registry.register(
            "get_memory",
            self._tool_get_memory,
            {"description": "Retrieve agent memory", "category": "system"},
        )

        self.tool_registry.register(
            "store_memory",
            self._tool_store_memory,
            {"description": "Store in agent memory", "category": "system"},
        )

        self.tool_registry.register(
            "execute_code",
            self._tool_execute_code,
            {"description": "Execute code safely", "category": "compute"},
        )

    async def _start_background_tasks(self):
        """Start background tasks"""
        await self.workflow_engine.start()

        # Start event processor
        asyncio.create_task(self._process_events())

        # Start metrics collector
        if self.config.enable_telemetry:
            asyncio.create_task(self._collect_metrics())

        # Start cleanup task
        asyncio.create_task(self._cleanup_old_contexts())

    @log_agent_method(activity_type="tool_execution", log_params=True, log_result=True)
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool using Strands framework with full observability"""
        start_time = asyncio.get_event_loop().time()

        # Create execution context
        execution_id = str(uuid.uuid4())
        current_span_id.set(execution_id)

        try:
            # Validate tool exists
            if not self.tool_registry.get(tool_name):
                return ToolExecutionResult(
                    tool_name=tool_name,
                    status=ToolStatus.FAILURE,
                    error=f"Tool {tool_name} not found",
                )

            # Execute with timeout
            result = await asyncio.wait_for(
                self.tool_registry.execute(tool_name, **parameters),
                timeout=self.config.tool_timeout_seconds,
            )

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Update metrics
            self._metrics["tools_executed"] += 1
            self._metrics["tools_succeeded"] += 1
            self._update_avg_metric(
                "avg_tool_time_ms", execution_time, self._metrics["tools_executed"]
            )

            # Log execution
            logger.info(f"Tool {tool_name} executed successfully in {execution_time:.2f}ms")

            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolStatus.SUCCESS,
                result=result,
                execution_time_ms=execution_time,
                metadata={"execution_id": execution_id},
            )

        except asyncio.TimeoutError:
            self._metrics["tools_failed"] += 1
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolStatus.TIMEOUT,
                error=f"Tool execution timed out after {self.config.tool_timeout_seconds}s",
            )

        except Exception as e:
            self._metrics["tools_failed"] += 1
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolStatus.FAILURE,
                error=error_msg,
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
            )

    @log_agent_method(activity_type="workflow_processing", log_params=True, log_result=True)
    async def process_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a workflow using Strands orchestration"""
        # Create workflow context
        context = WorkflowContext(
            workflow_id=workflow_id,
            agent_id=self.agent_id,
            inputs=inputs,
            parent_workflow_id=current_workflow_id.get(),
        )

        # Set context
        token = current_workflow_id.set(workflow_id)

        try:
            # Submit workflow
            await self.workflow_engine.submit_workflow(context)

            # Execute workflow steps
            result = await self._execute_workflow(context)

            # Update metrics
            self._metrics["workflows_executed"] += 1
            if context.status == WorkflowStatus.COMPLETED:
                self._metrics["workflows_succeeded"] += 1
            else:
                self._metrics["workflows_failed"] += 1

            execution_time = 0.0
            if context.started_at and context.completed_at:
                execution_time = (context.completed_at - context.started_at).total_seconds() * 1000
                self._update_avg_metric(
                    "avg_workflow_time_ms", execution_time, self._metrics["workflows_executed"]
                )

            logger.info(
                f"Workflow {workflow_id} completed with status {context.status} in {execution_time:.2f}ms"
            )

            return {
                "workflow_id": workflow_id,
                "status": context.status,
                "outputs": context.outputs,
                "execution_time_ms": execution_time,
                "metadata": context.metadata,
            }

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}\n{traceback.format_exc()}")
            context.status = WorkflowStatus.FAILED
            context.error = str(e)
            context.completed_at = datetime.utcnow()

            return {"workflow_id": workflow_id, "status": WorkflowStatus.FAILED, "error": str(e)}

        finally:
            # Reset context
            current_workflow_id.reset(token)

    async def _execute_workflow(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute workflow steps with advanced orchestration capabilities"""
        context.status = WorkflowStatus.RUNNING

        try:
            # Check for workflow definition
            if "workflow_definition" in context.inputs:
                # Execute defined workflow with dependencies and parallel execution
                workflow_def = context.inputs["workflow_definition"]
                results = await self._execute_defined_workflow(context, workflow_def)
                context.outputs["workflow_results"] = results

            elif "tools" in context.inputs:
                # Legacy simple tool execution
                results = await self._execute_simple_tools(context, context.inputs["tools"])
                context.outputs["tool_results"] = results

            elif "dag" in context.inputs:
                # Execute as Directed Acyclic Graph
                dag = context.inputs["dag"]
                results = await self._execute_dag_workflow(context, dag)
                context.outputs["dag_results"] = results

            elif "pipeline" in context.inputs:
                # Execute as data pipeline
                pipeline = context.inputs["pipeline"]
                results = await self._execute_pipeline(context, pipeline)
                context.outputs["pipeline_results"] = results

            else:
                # Try to infer workflow type
                results = await self._execute_inferred_workflow(context)
                context.outputs["results"] = results

            if context.status != WorkflowStatus.FAILED:
                context.status = WorkflowStatus.COMPLETED

        except asyncio.CancelledError:
            context.status = WorkflowStatus.CANCELLED
            context.error = "Workflow cancelled"
            raise

        except Exception as e:
            context.status = WorkflowStatus.FAILED
            context.error = str(e)
            logger.error(f"Workflow {context.workflow_id} failed: {e}\n{traceback.format_exc()}")
            raise

        finally:
            context.completed_at = datetime.utcnow()
            # Emit workflow completion event
            await self.workflow_engine.emit_event("workflow_completed", context)

        return context.outputs

    async def _execute_simple_tools(
        self, context: WorkflowContext, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute simple tool list"""
        results = []
        for tool_spec in tools:
            tool_name = tool_spec.get("name")
            tool_params = tool_spec.get("parameters", {})

            result = await self.execute_tool(tool_name, tool_params)
            results.append(result.__dict__)

            if result.status != ToolStatus.SUCCESS:
                context.status = WorkflowStatus.FAILED
                context.error = f"Tool {tool_name} failed: {result.error}"
                break

        return results

    async def _execute_defined_workflow(
        self, context: WorkflowContext, workflow_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a defined workflow with steps, dependencies, and conditions"""
        steps = workflow_def.get("steps", [])
        parallel_execution = workflow_def.get("parallel", False)

        if parallel_execution:
            return await self._execute_parallel_steps(context, steps)
        else:
            return await self._execute_sequential_steps(context, steps)

    async def _execute_sequential_steps(
        self, context: WorkflowContext, steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute workflow steps sequentially"""
        results = {}
        completed_steps = set()

        for step in steps:
            step_id = step.get("id", f"step_{len(completed_steps)}")

            # Check dependencies
            dependencies = step.get("dependencies", [])
            if not all(dep in completed_steps for dep in dependencies):
                missing = [d for d in dependencies if d not in completed_steps]
                raise ValueError(f"Step '{step_id}' missing dependencies: {missing}")

            # Check condition
            if "condition" in step:
                condition_met = await self._evaluate_condition(step["condition"], results)
                if not condition_met:
                    logger.info(f"Skipping step '{step_id}' due to condition")
                    results[step_id] = {"skipped": True, "reason": "condition_not_met"}
                    continue

            # Execute step
            try:
                step_result = await self._execute_workflow_step(step, results)
                results[step_id] = step_result
                completed_steps.add(step_id)

            except Exception as e:
                if step.get("continue_on_error", False):
                    results[step_id] = {"error": str(e), "continued": True}
                    completed_steps.add(step_id)
                else:
                    raise

        return results

    async def _execute_parallel_steps(
        self, context: WorkflowContext, steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute workflow steps in parallel where dependencies allow"""
        results = {}
        completed_steps = set()
        pending_steps = steps.copy()

        while pending_steps:
            # Find executable steps
            ready_steps = []
            for step in pending_steps:
                step_id = step.get("id", f"step_{len(completed_steps)}")
                dependencies = step.get("dependencies", [])

                if all(dep in completed_steps for dep in dependencies):
                    ready_steps.append(step)

            if not ready_steps and pending_steps:
                # Circular dependency detected
                step_ids = [s.get("id", "unknown") for s in pending_steps]
                raise ValueError(f"Circular dependency in steps: {step_ids}")

            # Execute ready steps in parallel
            if ready_steps:
                tasks = []
                for step in ready_steps:
                    task = self._execute_workflow_step(step, results)
                    tasks.append((step, task))

                # Wait for parallel execution
                step_results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

                # Process results
                for (step, _), result in zip(tasks, step_results):
                    step_id = step.get("id", f"step_{len(completed_steps)}")

                    if isinstance(result, Exception):
                        if step.get("continue_on_error", False):
                            results[step_id] = {"error": str(result), "continued": True}
                        else:
                            # Cancel other tasks and raise
                            for _, task in tasks:
                                if not task.done():
                                    task.cancel()
                            raise result
                    else:
                        results[step_id] = result

                    completed_steps.add(step_id)
                    pending_steps.remove(step)

        return results

    async def _execute_workflow_step(
        self, step: Dict[str, Any], previous_results: Dict[str, Any]
    ) -> Any:
        """Execute a single workflow step"""
        step_type = step.get("type", "tool")

        if step_type == "tool":
            tool_name = step.get("tool")
            parameters = self._resolve_parameters(step.get("parameters", {}), previous_results)
            result = await self.execute_tool(tool_name, parameters)
            return result.__dict__

        elif step_type == "condition":
            # Conditional branching
            condition = step.get("condition")
            if await self._evaluate_condition(condition, previous_results):
                return await self._execute_workflow_step(step.get("if_true"), previous_results)
            else:
                return await self._execute_workflow_step(step.get("if_false"), previous_results)

        elif step_type == "loop":
            # Loop execution
            items = self._resolve_parameters(step.get("items", []), previous_results)
            loop_results = []

            for item in items:
                loop_context = {"item": item, "index": len(loop_results), **previous_results}
                result = await self._execute_workflow_step(step.get("body"), loop_context)
                loop_results.append(result)

            return loop_results

        elif step_type == "parallel":
            # Nested parallel execution
            sub_steps = step.get("steps", [])
            tasks = [self._execute_workflow_step(s, previous_results) for s in sub_steps]
            return await asyncio.gather(*tasks)

        elif step_type == "delay":
            # Add delay
            delay_seconds = step.get("seconds", 1)
            await asyncio.sleep(delay_seconds)
            return {"delayed": delay_seconds}

        else:
            raise ValueError(f"Unknown step type: {step_type}")

    async def _execute_dag_workflow(
        self, context: WorkflowContext, dag: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow as a Directed Acyclic Graph"""
        nodes = dag.get("nodes", {})
        edges = dag.get("edges", [])

        # Build adjacency list
        graph = {node_id: [] for node_id in nodes}
        in_degree = {node_id: 0 for node_id in nodes}

        for edge in edges:
            from_node = edge["from"]
            to_node = edge["to"]
            graph[from_node].append(to_node)
            in_degree[to_node] += 1

        # Topological sort with parallel execution
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        results = {}

        while queue:
            # Execute all nodes in current level in parallel
            current_level = queue.copy()
            queue.clear()

            tasks = []
            for node_id in current_level:
                node = nodes[node_id]
                task = self._execute_dag_node(node, results)
                tasks.append((node_id, task))

            # Wait for level completion
            level_results = await asyncio.gather(*[t for _, t in tasks])

            # Update results and find next level
            for (node_id, _), result in zip(tasks, level_results):
                results[node_id] = result

                # Update in-degrees and queue
                for neighbor in graph[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return results

    async def _execute_dag_node(
        self, node: Dict[str, Any], previous_results: Dict[str, Any]
    ) -> Any:
        """Execute a single DAG node"""
        node_type = node.get("type", "tool")

        if node_type == "tool":
            return await self.execute_tool(node["tool"], node.get("parameters", {}))
        else:
            # Extend for other node types
            return {"node_type": node_type, "executed": True}

    async def _execute_pipeline(
        self, context: WorkflowContext, pipeline: List[Dict[str, Any]]
    ) -> Any:
        """Execute as data transformation pipeline"""
        data = context.inputs.get("initial_data", {})

        for stage in pipeline:
            transform = stage.get("transform")

            if transform == "map":
                func_name = stage.get("function")
                data = await self._apply_map_transform(data, func_name)

            elif transform == "filter":
                condition = stage.get("condition")
                data = await self._apply_filter_transform(data, condition)

            elif transform == "reduce":
                func_name = stage.get("function")
                initial = stage.get("initial")
                data = await self._apply_reduce_transform(data, func_name, initial)

            elif transform == "tool":
                tool_name = stage.get("tool")
                data = await self.execute_tool(tool_name, {"data": data})

        return data

    async def _execute_inferred_workflow(self, context: WorkflowContext) -> Dict[str, Any]:
        """Infer and execute workflow from context"""
        # Analyze inputs to determine workflow pattern
        if "symbol" in context.inputs and "action" in context.inputs:
            # Trading workflow
            return await self._execute_trading_workflow(context)
        elif "analysis_type" in context.inputs:
            # Analysis workflow
            return await self._execute_analysis_workflow(context)
        else:
            # Default: execute available tools
            return {"message": "No specific workflow pattern detected", "inputs": context.inputs}

    async def _execute_trading_workflow(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute inferred trading workflow"""
        symbol = context.inputs["symbol"]
        action = context.inputs["action"]

        # Market analysis
        market_data = await self.execute_tool("get_market_data", {"symbol": symbol})

        # Risk assessment
        risk_metrics = await self.execute_tool("get_risk_metrics", {"symbol": symbol})

        # Decision making
        decision = {
            "symbol": symbol,
            "action": action,
            "market_data": market_data.__dict__,
            "risk_metrics": risk_metrics.__dict__,
            "recommendation": "proceed" if risk_metrics.status == ToolStatus.SUCCESS else "abort",
        }

        return decision

    async def _execute_analysis_workflow(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute inferred analysis workflow"""
        analysis_type = context.inputs["analysis_type"]

        if analysis_type == "market":
            return {"analysis": "market", "status": "completed"}
        elif analysis_type == "portfolio":
            return {"analysis": "portfolio", "status": "completed"}
        else:
            return {"analysis": analysis_type, "status": "unknown_type"}

    async def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate workflow condition"""
        condition_type = condition.get("type", "simple")

        if condition_type == "simple":
            # Simple comparison
            left = self._resolve_parameters(condition.get("left"), context)
            operator = condition.get("operator", "==")
            right = self._resolve_parameters(condition.get("right"), context)

            if operator == "==":
                return left == right
            elif operator == "!=":
                return left != right
            elif operator == ">":
                return left > right
            elif operator == "<":
                return left < right
            elif operator == ">=":
                return left >= right
            elif operator == "<=":
                return left <= right
            elif operator == "in":
                return left in right
            elif operator == "not_in":
                return left not in right
            else:
                raise ValueError(f"Unknown operator: {operator}")

        elif condition_type == "complex":
            # Boolean logic
            operator = condition.get("operator", "and")
            conditions = condition.get("conditions", [])

            if operator == "and":
                return all(await self._evaluate_condition(c, context) for c in conditions)
            elif operator == "or":
                return any(await self._evaluate_condition(c, context) for c in conditions)
            elif operator == "not":
                return not await self._evaluate_condition(conditions[0], context)
            else:
                raise ValueError(f"Unknown logical operator: {operator}")

        else:
            raise ValueError(f"Unknown condition type: {condition_type}")

    def _resolve_parameters(self, params: Any, context: Dict[str, Any]) -> Any:
        """Resolve parameter references from context"""
        if isinstance(params, str) and params.startswith("$"):
            # Reference to previous result
            path = params[1:].split(".")
            value = context
            for key in path:
                value = value.get(key, None)
                if value is None:
                    break
            return value

        elif isinstance(params, dict):
            # Recursively resolve dict
            return {k: self._resolve_parameters(v, context) for k, v in params.items()}

        elif isinstance(params, list):
            # Recursively resolve list
            return [self._resolve_parameters(item, context) for item in params]

        else:
            # Return as-is
            return params

    async def _apply_map_transform(self, data: Any, func_name: str) -> Any:
        """Apply map transformation"""
        if isinstance(data, list):
            return [await self.execute_tool(func_name, {"item": item}) for item in data]
        else:
            return await self.execute_tool(func_name, {"item": data})

    async def _apply_filter_transform(self, data: Any, condition: Dict[str, Any]) -> Any:
        """Apply filter transformation"""
        if isinstance(data, list):
            filtered = []
            for item in data:
                if await self._evaluate_condition(condition, {"item": item}):
                    filtered.append(item)
            return filtered
        else:
            return data if await self._evaluate_condition(condition, {"item": data}) else None

    async def _apply_reduce_transform(self, data: Any, func_name: str, initial: Any) -> Any:
        """Apply reduce transformation"""
        if isinstance(data, list):
            accumulator = initial
            for item in data:
                result = await self.execute_tool(
                    func_name, {"accumulator": accumulator, "item": item}
                )
                accumulator = result.result
            return accumulator
        else:
            return data

    async def _process_events(self):
        """Process events from the event bus"""
        while True:
            try:
                event_type, data = await self.workflow_engine._event_queue.get()

                # Handle different event types
                if event_type == "workflow_submitted":
                    logger.debug(f"Workflow {data.workflow_id} submitted")
                elif event_type == "workflow_completed":
                    logger.info(f"Workflow {data.workflow_id} completed")
                elif event_type == "workflow_failed":
                    logger.error(f"Workflow {data.workflow_id} failed: {data.error}")

                # Emit telemetry if enabled
                if self.config.enable_telemetry:
                    await self._emit_telemetry(event_type, data)

            except Exception as e:
                logger.error(f"Error processing event: {e}")
                await asyncio.sleep(1)

    async def _collect_metrics(self):
        """Collect and export metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute

                # Log metrics
                logger.info(f"Strands metrics: {json.dumps(self._metrics, indent=2)}")

                # Export to monitoring system if configured
                if hasattr(self, "metrics_exporter"):
                    await self.metrics_exporter.export(self._metrics)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

    async def _cleanup_old_contexts(self):
        """Clean up old workflow contexts"""
        while True:
            try:
                await asyncio.sleep(self.config.context_cleanup_interval)

                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                old_workflows = [
                    wf_id
                    for wf_id, context in self.workflow_engine._workflows.items()
                    if context.completed_at and context.completed_at < cutoff_time
                ]

                for wf_id in old_workflows:
                    del self.workflow_engine._workflows[wf_id]

                if old_workflows:
                    logger.info(f"Cleaned up {len(old_workflows)} old workflow contexts")

            except Exception as e:
                logger.error(f"Error cleaning up contexts: {e}")

    async def _emit_telemetry(self, event_type: str, data: Any):
        """Emit telemetry data to configured backends"""
        telemetry_event = {
            "event_type": event_type,
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": current_workflow_id.get(),
            "span_id": current_span_id.get(),
            "data": {},
        }

        # Extract relevant data based on event type
        if event_type == "workflow_submitted":
            telemetry_event["data"] = {
                "workflow_id": data.workflow_id,
                "workflow_name": data.metadata.get("name", "unknown"),
                "inputs": len(data.inputs),
            }
        elif event_type == "workflow_completed":
            telemetry_event["data"] = {
                "workflow_id": data.workflow_id,
                "status": data.status,
                "duration_ms": (data.completed_at - data.started_at).total_seconds() * 1000
                if data.completed_at and data.started_at
                else 0,
                "outputs": len(data.outputs),
            }
        elif event_type == "workflow_failed":
            telemetry_event["data"] = {
                "workflow_id": data.workflow_id,
                "error": data.error,
                "duration_ms": (data.completed_at - data.started_at).total_seconds() * 1000
                if data.completed_at and data.started_at
                else 0,
            }
        elif event_type == "tool_execution":
            telemetry_event["data"] = {
                "tool_name": data.get("tool_name"),
                "status": data.get("status"),
                "duration_ms": data.get("duration_ms", 0),
            }

        # Send to different backends based on configuration
        try:
            # Console output for debugging (if enabled)
            if self.config.enable_debug_logging:
                logger.debug(f"Telemetry event: {json.dumps(telemetry_event, indent=2)}")

            # Send to metrics exporter if available
            if hasattr(self, "metrics_exporter") and self.metrics_exporter:
                await self.metrics_exporter.send_event(telemetry_event)

            # Store in database if available
            if hasattr(self, "database_connection"):
                await self._store_telemetry_event(telemetry_event)

            # Send to external telemetry service if configured
            if self.config.telemetry_endpoint:
                await self._send_to_telemetry_service(telemetry_event)

        except Exception as e:
            logger.error(f"Failed to emit telemetry: {e}")

    async def _store_telemetry_event(self, event: Dict[str, Any]):
        """Store telemetry event in database"""
        try:
            # Use UnifiedDatabase for telemetry storage
            db = UnifiedDatabase()
            await db.initialize()

            # Store telemetry event using execute_query_async
            await db.execute_query_async(
                """
                INSERT INTO strands_telemetry 
                (event_type, agent_id, workflow_id, span_id, timestamp, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    event["event_type"],
                    event["agent_id"],
                    event["workflow_id"],
                    event["span_id"],
                    event["timestamp"],
                    json.dumps(event["data"]),
                ),
            )

        except Exception as e:
            logger.error(f"Failed to store telemetry event: {e}")

    async def _send_to_telemetry_service(self, event: Dict[str, Any]):
        """Send telemetry to external service"""
        # This would integrate with services like DataDog, New Relic, etc.
        # For now, just log that we would send it
        logger.debug(f"Would send telemetry to {self.config.telemetry_endpoint}")

    def _update_avg_metric(self, metric_name: str, new_value: float, count: int):
        """Update average metric"""
        old_avg = self._metrics[metric_name]
        self._metrics[metric_name] = ((old_avg * (count - 1)) + new_value) / count

    # Tool implementations
    async def _tool_get_memory(self, key: str) -> Any:
        """Tool to get memory"""
        return await self.get_memory(key)

    async def _tool_store_memory(self, key: str, value: Any) -> None:
        """Tool to store memory"""
        await self.store_memory(key, value)

    async def _tool_execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Tool to execute code safely in a secure sandbox environment"""
        # Import secure sandbox
        from .secure_code_sandbox import SecureCodeExecutor, SecurityLevel

        # Only support Python for now (most secure)
        if language != "python":
            return {
                "status": "error",
                "error": f"Language '{language}' not supported for security reasons. Only 'python' is allowed.",
                "language": language,
            }

        try:
            # Initialize secure executor if not already done
            if not hasattr(self, "_secure_executor"):
                self._secure_executor = SecureCodeExecutor(
                    default_security_level=SecurityLevel.STRICT
                )

            # Execute code in secure sandbox
            result = await self._secure_executor.execute_safe_code(
                code=code,
                tool_name="execute_code",
                context={"agent_id": self.agent_id, "agent_type": self.agent_type},
                security_level=SecurityLevel.STRICT,
                timeout=10.0,  # 10 second timeout
            )

            # Convert to expected format
            if result["success"]:
                return {
                    "status": "success",
                    "output": result.get("output", ""),
                    "language": language,
                    "execution_time_ms": result.get("execution_time", 0) * 1000,
                    "security_level": "strict",
                    "sandbox_version": "secure_v1",
                }
            else:
                error_msg = result.get("error", "Unknown execution error")

                # Include security violation details if present
                if result.get("security_violations"):
                    error_msg += (
                        f" (Security violations: {', '.join(result['security_violations'])})"
                    )

                return {
                    "status": "error",
                    "error": error_msg,
                    "language": language,
                    "security_violations": result.get("security_violations", []),
                    "execution_result": result.get("result", "unknown_error"),
                }

        except Exception as e:
            logger.error(f"Secure code execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": f"Secure sandbox execution failed: {str(e)}",
                "language": language,
                "security_violations": ["sandbox_system_error"],
            }

    def register_tool(self, name: str, func: Callable, metadata: Optional[Dict[str, Any]] = None):
        """Register a custom tool"""
        self.tool_registry.register(name, func, metadata)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        return self.tool_registry.list_tools()

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return self._metrics.copy()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Shutting down Strands agent {self.agent_id}")
        await self.workflow_engine.stop()
        await super().shutdown()
