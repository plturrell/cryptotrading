"""
Workflow Engine Component for Strands Framework
Handles workflow orchestration, execution, and state management
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Workflow step status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""

    id: str
    name: str
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 60.0
    retry_on_failure: bool = True
    max_retries: int = 3
    condition: Optional[str] = None  # JSON logic condition
    on_success: Optional[str] = None  # Next step on success
    on_failure: Optional[str] = None  # Next step on failure
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""

    id: str
    name: str
    description: str
    version: str = "1.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    parallel_steps: List[List[str]] = field(default_factory=list)  # Steps that can run in parallel
    global_timeout: float = 300.0
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)  # Workflow variables
    triggers: List[str] = field(default_factory=list)  # Event triggers
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepExecution:
    """Step execution tracking"""

    step_id: str
    execution_id: str
    status: StepStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0


@dataclass
class WorkflowExecution:
    """Workflow execution tracking"""

    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    step_executions: Dict[str, StepExecution] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    total_execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConditionEvaluator:
    """Evaluates workflow step conditions"""

    @staticmethod
    def evaluate(condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string against context"""
        if not condition:
            return True

        try:
            # Simple condition evaluation
            # In production, use a proper JSON logic library
            condition = condition.strip()

            # Handle simple comparisons
            if " == " in condition:
                left, right = condition.split(" == ", 1)
                left_val = ConditionEvaluator._resolve_value(left.strip(), context)
                right_val = ConditionEvaluator._resolve_value(right.strip(), context)
                return left_val == right_val

            elif " != " in condition:
                left, right = condition.split(" != ", 1)
                left_val = ConditionEvaluator._resolve_value(left.strip(), context)
                right_val = ConditionEvaluator._resolve_value(right.strip(), context)
                return left_val != right_val

            elif " > " in condition:
                left, right = condition.split(" > ", 1)
                left_val = ConditionEvaluator._resolve_value(left.strip(), context)
                right_val = ConditionEvaluator._resolve_value(right.strip(), context)
                return float(left_val) > float(right_val)

            elif " < " in condition:
                left, right = condition.split(" < ", 1)
                left_val = ConditionEvaluator._resolve_value(left.strip(), context)
                right_val = ConditionEvaluator._resolve_value(right.strip(), context)
                return float(left_val) < float(right_val)

            # Handle boolean values
            elif condition.lower() in ["true", "false"]:
                return condition.lower() == "true"

            # Handle variable references
            else:
                return bool(ConditionEvaluator._resolve_value(condition, context))

        except Exception as e:
            logger.error(f"Condition evaluation failed: {condition} - {e}")
            return False

    @staticmethod
    def _resolve_value(value_str: str, context: Dict[str, Any]) -> Any:
        """Resolve a value string to actual value from context"""
        value_str = value_str.strip()

        # Handle quoted strings
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]

        if value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]

        # Handle numbers
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # Handle variable references
        if "." in value_str:
            # Nested property access (e.g., "result.data.value")
            parts = value_str.split(".")
            current = context
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        else:
            # Simple variable reference
            return context.get(value_str)


class WorkflowExecutor:
    """Executes individual workflows"""

    def __init__(self, tool_manager, max_parallel_steps: int = 5):
        self.tool_manager = tool_manager
        self.max_parallel_steps = max_parallel_steps
        self.condition_evaluator = ConditionEvaluator()
        self._execution_semaphore = asyncio.Semaphore(max_parallel_steps)

    async def execute_workflow(
        self, workflow: WorkflowDefinition, initial_context: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Execute a complete workflow"""
        execution_id = str(uuid.uuid4())
        start_time = time.time()

        # Create execution record
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow.id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.utcnow(),
            context=initial_context or {},
            variables=workflow.variables.copy(),
        )

        logger.info(f"Starting workflow execution: {workflow.name} (id: {execution_id})")

        try:
            # Build step dependency graph
            dependency_graph = self._build_dependency_graph(workflow.steps)

            # Execute steps in dependency order
            completed_steps = set()
            failed_steps = set()

            while len(completed_steps) < len(workflow.steps):
                # Find steps ready to execute
                ready_steps = self._find_ready_steps(
                    workflow.steps, completed_steps, failed_steps, dependency_graph
                )

                if not ready_steps:
                    # Check if we have parallel steps defined
                    parallel_ready = self._find_parallel_ready_steps(
                        workflow, completed_steps, failed_steps
                    )

                    if parallel_ready:
                        # Execute parallel steps
                        await self._execute_parallel_steps(workflow, execution, parallel_ready)
                        completed_steps.update(step.id for step in parallel_ready)
                    else:
                        # No more steps can execute - check for failures
                        remaining_steps = set(step.id for step in workflow.steps) - completed_steps
                        if remaining_steps:
                            execution.status = WorkflowStatus.FAILED
                            execution.error = f"Cannot execute remaining steps due to dependencies: {remaining_steps}"
                            break
                        else:
                            break
                else:
                    # Execute ready steps sequentially
                    for step in ready_steps:
                        step_success = await self._execute_step(workflow, execution, step)

                        if step_success:
                            completed_steps.add(step.id)
                        else:
                            failed_steps.add(step.id)

                            # Check if workflow should continue on failure
                            if not step.retry_on_failure:
                                execution.status = WorkflowStatus.FAILED
                                execution.error = f"Step {step.id} failed and cannot be retried"
                                break

                # Check global timeout
                if time.time() - start_time > workflow.global_timeout:
                    execution.status = WorkflowStatus.FAILED
                    execution.error = f"Workflow timed out after {workflow.global_timeout}s"
                    break

            # Set final status if not already set
            if execution.status == WorkflowStatus.RUNNING:
                if failed_steps:
                    execution.status = WorkflowStatus.FAILED
                    execution.error = f"Steps failed: {failed_steps}"
                else:
                    execution.status = WorkflowStatus.COMPLETED

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = f"Workflow execution error: {str(e)}"
            logger.error(f"Workflow execution failed: {e}", exc_info=True)

        finally:
            execution.completed_at = datetime.utcnow()
            execution.total_execution_time = time.time() - start_time

            logger.info(
                f"Workflow {workflow.name} completed with status {execution.status} "
                f"in {execution.total_execution_time:.2f}s"
            )

        return execution

    async def _execute_step(
        self, workflow: WorkflowDefinition, execution: WorkflowExecution, step: WorkflowStep
    ) -> bool:
        """Execute a single workflow step"""
        execution.current_step = step.id

        # Evaluate condition
        if step.condition and not self.condition_evaluator.evaluate(
            step.condition, execution.context
        ):
            logger.info(f"Skipping step {step.id} due to condition: {step.condition}")

            step_execution = StepExecution(
                step_id=step.id,
                execution_id=execution.execution_id,
                status=StepStatus.SKIPPED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )
            execution.step_executions[step.id] = step_execution
            return True

        # Create step execution record
        step_execution = StepExecution(
            step_id=step.id,
            execution_id=execution.execution_id,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        execution.step_executions[step.id] = step_execution

        logger.info(f"Executing step: {step.name} (tool: {step.tool_name})")

        # Prepare parameters with context substitution
        resolved_parameters = self._resolve_parameters(step.parameters, execution.context)

        # Execute with retries
        retry_count = 0
        max_retries = step.max_retries if step.retry_on_failure else 1

        while retry_count < max_retries:
            try:
                start_time = time.time()

                # Execute tool
                tool_execution = await self.tool_manager.execute_tool(
                    step.tool_name,
                    resolved_parameters,
                    execution_context={
                        "workflow_id": workflow.id,
                        "execution_id": execution.execution_id,
                        "step_id": step.id,
                    },
                )

                step_execution.execution_time = time.time() - start_time

                if tool_execution.status.value in ["completed", "success"]:
                    step_execution.status = StepStatus.COMPLETED
                    step_execution.result = tool_execution.result
                    step_execution.completed_at = datetime.utcnow()

                    # Update execution context with results
                    if hasattr(tool_execution, "result") and tool_execution.result:
                        execution.context[f"step_{step.id}_result"] = tool_execution.result

                    logger.info(f"Step {step.id} completed successfully")
                    return True
                else:
                    raise Exception(tool_execution.error or "Tool execution failed")

            except Exception as e:
                retry_count += 1
                step_execution.retry_count = retry_count

                if retry_count < max_retries:
                    logger.warning(f"Step {step.id} attempt {retry_count} failed, retrying: {e}")
                    await asyncio.sleep(min(2**retry_count, 10))  # Exponential backoff
                else:
                    step_execution.status = StepStatus.FAILED
                    step_execution.error = str(e)
                    step_execution.completed_at = datetime.utcnow()

                    logger.error(f"Step {step.id} failed after {retry_count} attempts: {e}")
                    return False

        return False

    async def _execute_parallel_steps(
        self, workflow: WorkflowDefinition, execution: WorkflowExecution, steps: List[WorkflowStep]
    ) -> Dict[str, bool]:
        """Execute multiple steps in parallel"""
        logger.info(f"Executing {len(steps)} steps in parallel")

        # Create tasks for all steps
        tasks = []
        for step in steps:
            task = asyncio.create_task(self._execute_step(workflow, execution, step))
            tasks.append((step.id, task))

        # Wait for all tasks to complete
        results = {}
        for step_id, task in tasks:
            try:
                result = await task
                results[step_id] = result
            except Exception as e:
                logger.error(f"Parallel step {step_id} failed: {e}")
                results[step_id] = False

        return results

    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, Set[str]]:
        """Build step dependency graph"""
        graph = {}
        for step in steps:
            graph[step.id] = set(step.dependencies)
        return graph

    def _find_ready_steps(
        self,
        steps: List[WorkflowStep],
        completed: Set[str],
        failed: Set[str],
        dependency_graph: Dict[str, Set[str]],
    ) -> List[WorkflowStep]:
        """Find steps that are ready to execute"""
        ready = []
        for step in steps:
            if step.id not in completed and step.id not in failed:
                # Check if all dependencies are completed
                dependencies = dependency_graph.get(step.id, set())
                if dependencies.issubset(completed):
                    ready.append(step)
        return ready

    def _find_parallel_ready_steps(
        self, workflow: WorkflowDefinition, completed: Set[str], failed: Set[str]
    ) -> List[WorkflowStep]:
        """Find steps that can run in parallel"""
        if not workflow.parallel_steps:
            return []

        # Find first parallel group where all steps are ready
        for parallel_group in workflow.parallel_steps:
            group_steps = [s for s in workflow.steps if s.id in parallel_group]

            # Check if all steps in group are ready
            all_ready = True
            for step in group_steps:
                if step.id in completed or step.id in failed:
                    all_ready = False
                    break

                # Check dependencies
                for dep in step.dependencies:
                    if dep not in completed:
                        all_ready = False
                        break

            if all_ready:
                return group_steps

        return []

    def _resolve_parameters(
        self, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter values using context variables"""
        resolved = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Variable substitution
                var_name = value[2:-1]
                resolved[key] = context.get(var_name, value)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_parameters(value, context)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_parameters(item, context) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved[key] = value

        return resolved


class WorkflowEngine:
    """Main workflow orchestration engine"""

    def __init__(self, tool_manager, max_concurrent_workflows: int = 5):
        self.tool_manager = tool_manager
        self.max_concurrent_workflows = max_concurrent_workflows

        # Workflow registry
        self._workflows: Dict[str, WorkflowDefinition] = {}

        # Execution tracking
        self._active_executions: Dict[str, WorkflowExecution] = {}
        self._execution_history: List[WorkflowExecution] = []

        # Executors
        self._executors = [WorkflowExecutor(tool_manager) for _ in range(max_concurrent_workflows)]
        self._executor_semaphore = asyncio.Semaphore(max_concurrent_workflows)

        # Metrics
        self._metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0,
            "active_executions": 0,
        }

        logger.info(f"WorkflowEngine initialized with {max_concurrent_workflows} executors")

    async def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Register a workflow definition"""
        self._workflows[workflow.id] = workflow
        self._metrics["total_workflows"] += 1

        logger.info(f"Registered workflow: {workflow.name} (id: {workflow.id})")
        return True

    async def execute_workflow(
        self, workflow_id: str, initial_context: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Execute a workflow by ID"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        async with self._executor_semaphore:
            # Get available executor
            executor = self._executors[0]  # Simple round-robin could be improved

            # Execute workflow
            execution = await executor.execute_workflow(workflow, initial_context)

            # Track execution
            self._active_executions[execution.execution_id] = execution
            self._execution_history.append(execution)

            # Update metrics
            if execution.status == WorkflowStatus.COMPLETED:
                self._metrics["successful_workflows"] += 1
            elif execution.status == WorkflowStatus.FAILED:
                self._metrics["failed_workflows"] += 1

            self._update_average_execution_time(execution.total_execution_time)

            # Cleanup
            if execution.execution_id in self._active_executions:
                del self._active_executions[execution.execution_id]

            # Trim history
            if len(self._execution_history) > 500:
                self._execution_history = self._execution_history[-250:]

        return execution

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition"""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all registered workflows"""
        return list(self._workflows.values())

    def get_active_executions(self) -> List[WorkflowExecution]:
        """Get currently active workflow executions"""
        return list(self._active_executions.values())

    def get_execution_history(self, limit: int = 100) -> List[WorkflowExecution]:
        """Get recent execution history"""
        return self._execution_history[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get workflow engine metrics"""
        self._metrics["active_executions"] = len(self._active_executions)
        return self._metrics.copy()

    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time metric"""
        total = self._metrics["successful_workflows"] + self._metrics["failed_workflows"]
        if total <= 1:
            self._metrics["average_execution_time"] = execution_time
        else:
            current_avg = self._metrics["average_execution_time"]
            self._metrics["average_execution_time"] = (
                current_avg * (total - 1) + execution_time
            ) / total

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down WorkflowEngine")

        # Wait for active executions to complete
        if self._active_executions:
            logger.info(f"Waiting for {len(self._active_executions)} active workflows to complete")

            timeout = 60.0  # 1 minute timeout
            start_time = time.time()

            while self._active_executions and (time.time() - start_time) < timeout:
                await asyncio.sleep(1.0)

            if self._active_executions:
                logger.warning(
                    f"Forcibly terminating {len(self._active_executions)} active workflows"
                )

        logger.info("WorkflowEngine shutdown complete")
