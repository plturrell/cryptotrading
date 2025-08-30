"""
Advanced Strands Workflow Orchestration System
Production-grade workflow engine with parallel execution, dependency management, and error recovery.
"""
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .strands_enhanced import StrandsWorkflow, WorkflowStatus, WorkflowStep


class WorkflowEngine:
    """Advanced workflow orchestration engine"""

    def __init__(self, agent: "EnhancedStrandsAgent"):
        self.agent = agent
        self.logger = logging.getLogger("WorkflowEngine")
        self.execution_history: List[Dict[str, Any]] = []
        self.workflow_templates: Dict[str, StrandsWorkflow] = {}
        self._setup_enterprise_workflows()

    def _setup_enterprise_workflows(self):
        """Setup enterprise-grade workflow templates"""

        # Complete Trading Strategy Workflow
        trading_strategy = StrandsWorkflow(
            id="complete_trading_strategy",
            name="Complete Trading Strategy Execution",
            description="End-to-end trading strategy with risk management",
            steps=[
                WorkflowStep(
                    id="market_scan",
                    tool_name="advanced_market_scanner",
                    parameters={
                        "criteria": {"min_volume": 5000000},
                        "markets": ["BTC", "ETH", "ADA"],
                    },
                ),
                WorkflowStep(
                    id="multi_timeframe",
                    tool_name="multi_timeframe_analysis",
                    parameters={"symbol": "BTC"},
                    dependencies=["market_scan"],
                ),
                WorkflowStep(
                    id="portfolio_check",
                    tool_name="get_portfolio",
                    parameters={"include_history": True},
                ),
                WorkflowStep(
                    id="risk_assessment",
                    tool_name="risk_assessment_comprehensive",
                    parameters={"include_stress_test": True},
                    dependencies=["portfolio_check"],
                ),
                WorkflowStep(
                    id="position_sizing",
                    tool_name="dynamic_position_sizing",
                    parameters={"symbol": "BTC", "risk_percentage": 0.02},
                    dependencies=["risk_assessment", "multi_timeframe"],
                ),
                WorkflowStep(
                    id="trading_decision",
                    tool_name="make_trading_decision",
                    parameters={},
                    dependencies=["position_sizing", "multi_timeframe", "risk_assessment"],
                ),
                WorkflowStep(
                    id="execute_trade",
                    tool_name="execute_trade",
                    parameters={"symbol": "BTC", "side": "buy", "amount": 0.1},
                    dependencies=["trading_decision"],
                    condition=lambda state: state["step_results"]["trading_decision"]["result"][
                        "action"
                    ]
                    == "buy",
                ),
            ],
            parallel_execution=True,
            max_execution_time=180.0,
        )

        # Portfolio Management Workflow
        portfolio_management = StrandsWorkflow(
            id="portfolio_management_cycle",
            name="Complete Portfolio Management Cycle",
            description="Comprehensive portfolio analysis and rebalancing",
            steps=[
                WorkflowStep(
                    id="portfolio_analysis",
                    tool_name="get_portfolio",
                    parameters={"include_history": True},
                ),
                WorkflowStep(
                    id="risk_analysis",
                    tool_name="risk_assessment_comprehensive",
                    parameters={"include_stress_test": True},
                    dependencies=["portfolio_analysis"],
                ),
                WorkflowStep(
                    id="rebalancing_calculation",
                    tool_name="portfolio_rebalancing",
                    parameters={"threshold": 0.05},
                    dependencies=["portfolio_analysis"],
                ),
                WorkflowStep(
                    id="market_conditions",
                    tool_name="data_aggregation_engine",
                    parameters={
                        "symbols": ["BTC", "ETH", "ADA"],
                        "data_types": ["market_data", "sentiment"],
                    },
                ),
                WorkflowStep(
                    id="rebalancing_decision",
                    tool_name="make_rebalancing_decision",
                    parameters={},
                    dependencies=["rebalancing_calculation", "risk_analysis", "market_conditions"],
                ),
            ],
            parallel_execution=False,
            max_execution_time=300.0,
        )

        # Real-time Monitoring Workflow
        monitoring_workflow = StrandsWorkflow(
            id="realtime_monitoring",
            name="Real-time System Monitoring",
            description="Continuous system and market monitoring",
            steps=[
                WorkflowStep(id="system_health", tool_name="system_health_monitor", parameters={}),
                WorkflowStep(
                    id="market_monitoring",
                    tool_name="advanced_market_scanner",
                    parameters={"criteria": {"min_volume": 1000000}},
                ),
                WorkflowStep(id="portfolio_status", tool_name="get_portfolio", parameters={}),
                WorkflowStep(
                    id="risk_monitoring",
                    tool_name="get_risk_metrics",
                    parameters={"scope": "portfolio"},
                ),
                WorkflowStep(
                    id="alert_generation",
                    tool_name="generate_alerts",
                    parameters={},
                    dependencies=[
                        "system_health",
                        "market_monitoring",
                        "portfolio_status",
                        "risk_monitoring",
                    ],
                ),
            ],
            parallel_execution=True,
            max_execution_time=60.0,
        )

        # Multi-Agent Coordination Workflow
        coordination_workflow = StrandsWorkflow(
            id="multi_agent_coordination",
            name="Multi-Agent Coordination Protocol",
            description="Coordinate actions across agent network",
            steps=[
                WorkflowStep(
                    id="network_discovery", tool_name="discover_agent_network", parameters={}
                ),
                WorkflowStep(
                    id="health_check_network",
                    tool_name="broadcast_to_network",
                    parameters={"message_type": "health_check", "data": {}},
                    dependencies=["network_discovery"],
                ),
                WorkflowStep(
                    id="data_sharing",
                    tool_name="broadcast_to_network",
                    parameters={"message_type": "share_data", "data": {"market_analysis": True}},
                    dependencies=["health_check_network"],
                ),
                WorkflowStep(
                    id="coordinate_actions",
                    tool_name="coordinate_agents",
                    parameters={"action": "market_analysis"},
                    dependencies=["data_sharing"],
                ),
                WorkflowStep(
                    id="consensus_building",
                    tool_name="build_consensus",
                    parameters={},
                    dependencies=["coordinate_actions"],
                ),
            ],
            parallel_execution=False,
            max_execution_time=120.0,
        )

        # Data Processing Pipeline
        data_pipeline = StrandsWorkflow(
            id="data_processing_pipeline",
            name="Comprehensive Data Processing Pipeline",
            description="ETL pipeline for market and trading data",
            steps=[
                WorkflowStep(
                    id="data_extraction",
                    tool_name="data_aggregation_engine",
                    parameters={
                        "symbols": ["BTC", "ETH", "ADA", "DOT", "LINK"],
                        "data_types": ["market_data", "sentiment", "volume"],
                    },
                ),
                WorkflowStep(
                    id="data_validation",
                    tool_name="validate_data_quality",
                    parameters={},
                    dependencies=["data_extraction"],
                ),
                WorkflowStep(
                    id="data_transformation",
                    tool_name="transform_data",
                    parameters={
                        "transformations": ["normalize", "aggregate", "calculate_indicators"]
                    },
                    dependencies=["data_validation"],
                ),
                WorkflowStep(
                    id="data_storage",
                    tool_name="store_processed_data",
                    parameters={},
                    dependencies=["data_transformation"],
                ),
                WorkflowStep(
                    id="data_analysis",
                    tool_name="analyze_processed_data",
                    parameters={},
                    dependencies=["data_storage"],
                ),
            ],
            parallel_execution=False,
            max_execution_time=240.0,
        )

        # Store workflows
        self.workflow_templates.update(
            {
                "complete_trading_strategy": trading_strategy,
                "portfolio_management_cycle": portfolio_management,
                "realtime_monitoring": monitoring_workflow,
                "multi_agent_coordination": coordination_workflow,
                "data_processing_pipeline": data_pipeline,
            }
        )

        # Register workflows with agent
        self.agent.workflow_registry.update(self.workflow_templates)

    async def process_workflow(
        self, workflow_id: str, inputs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Enhanced workflow processing with advanced features"""
        inputs = inputs or {}

        if workflow_id not in self.workflow_templates:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        workflow = self.workflow_templates[workflow_id]
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        # Initialize execution state
        execution_state = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "status": WorkflowStatus.RUNNING,
            "start_time": start_time,
            "inputs": inputs,
            "step_results": {},
            "errors": [],
            "retry_counts": {},
            "execution_context": {
                "agent_id": self.agent.agent_id,
                "session_id": self.agent.context.session_id,
            },
        }

        self.agent.active_workflows[execution_id] = execution_state

        try:
            # Pre-execution validation
            validation_result = await self._validate_workflow(workflow, inputs)
            if not validation_result["valid"]:
                raise ValueError(f"Workflow validation failed: {validation_result['errors']}")

            # Execute workflow
            if workflow.parallel_execution:
                result = await self._execute_workflow_parallel_advanced(workflow, execution_state)
            else:
                result = await self._execute_workflow_sequential_advanced(workflow, execution_state)

            execution_state["status"] = WorkflowStatus.COMPLETED
            duration = (datetime.utcnow() - start_time).total_seconds()

            # Post-execution processing
            await self._post_execution_processing(execution_state, result)

            # Store execution history
            self.execution_history.append(
                {
                    "execution_id": execution_id,
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "duration": duration,
                    "timestamp": start_time.isoformat(),
                    "steps_executed": len(execution_state["step_results"]),
                }
            )

            await self.agent.observer.on_workflow_complete(workflow_id, duration)

            return {
                "success": True,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "completed",
                "result": result,
                "duration": duration,
                "steps_executed": len(execution_state["step_results"]),
                "metadata": execution_state["execution_context"],
            }

        except Exception as e:
            execution_state["status"] = WorkflowStatus.FAILED
            execution_state["errors"].append(str(e))

            # Store failed execution
            self.execution_history.append(
                {
                    "execution_id": execution_id,
                    "workflow_id": workflow_id,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": start_time.isoformat(),
                    "steps_completed": len(execution_state["step_results"]),
                }
            )

            return {
                "success": False,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "failed",
                "error": str(e),
                "steps_completed": len(execution_state["step_results"]),
                "retry_counts": execution_state["retry_counts"],
            }

        finally:
            # Cleanup
            if execution_id in self.agent.active_workflows:
                del self.agent.active_workflows[execution_id]

    async def _validate_workflow(
        self, workflow: StrandsWorkflow, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate workflow before execution"""
        errors = []

        # Check tool availability
        for step in workflow.steps:
            if step.tool_name not in self.agent.tool_registry:
                errors.append(f"Tool '{step.tool_name}' not available for step '{step.id}'")

        # Check dependency graph for cycles
        if self._has_dependency_cycles(workflow.steps):
            errors.append("Circular dependencies detected in workflow")

        # Validate step parameters
        for step in workflow.steps:
            if step.tool_name in self.agent.tool_registry:
                tool = self.agent.tool_registry[step.tool_name]
                for param_name, param_config in tool.parameters.items():
                    if (
                        param_config.get("required", False)
                        and param_name not in step.parameters
                        and param_name not in inputs
                    ):
                        errors.append(
                            f"Required parameter '{param_name}' missing for step '{step.id}'"
                        )

        return {"valid": len(errors) == 0, "errors": errors}

    def _has_dependency_cycles(self, steps: List[WorkflowStep]) -> bool:
        """Check for circular dependencies"""
        # Build adjacency list
        graph = {step.id: step.dependencies for step in steps}

        # DFS cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return True

        return False

    async def _execute_workflow_sequential_advanced(
        self, workflow: StrandsWorkflow, execution_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow steps sequentially with advanced error handling"""
        results = {}

        # Sort steps by dependencies (topological sort)
        sorted_steps = self._topological_sort(workflow.steps)

        for step in sorted_steps:
            # Check if step should be executed
            if not await self._should_execute_step(step, execution_state):
                continue

            # Execute step with retry logic
            step_result = await self._execute_step_with_retry(step, execution_state)

            if step_result.get("success"):
                execution_state["step_results"][step.id] = step_result
                results[step.id] = step_result
            else:
                # Handle step failure
                if step.retry_on_failure:
                    self.logger.warning(
                        f"Step {step.id} failed but will be retried: {step_result.get('error')}"
                    )
                else:
                    execution_state["errors"].append(
                        f"Step {step.id} failed: {step_result.get('error')}"
                    )
                    raise Exception(f"Critical step {step.id} failed")

        return results

    async def _execute_workflow_parallel_advanced(
        self, workflow: StrandsWorkflow, execution_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow steps in parallel with dependency management"""
        completed_steps = set()
        results = {}
        max_parallel = 5  # Limit concurrent executions

        while len(completed_steps) < len(workflow.steps):
            # Find ready steps
            ready_steps = []
            for step in workflow.steps:
                if step.id not in completed_steps and all(
                    dep in completed_steps for dep in step.dependencies
                ):
                    ready_steps.append(step)

            if not ready_steps:
                break

            # Execute ready steps in parallel (with limit)
            semaphore = asyncio.Semaphore(max_parallel)

            async def execute_with_semaphore(step):
                async with semaphore:
                    if await self._should_execute_step(step, execution_state):
                        return await self._execute_step_with_retry(step, execution_state)
                    return {"success": True, "skipped": True}

            # Create tasks for ready steps
            tasks = [
                (step.id, asyncio.create_task(execute_with_semaphore(step))) for step in ready_steps
            ]

            # Wait for completion
            for step_id, task in tasks:
                try:
                    step_result = await task
                    if step_result.get("success"):
                        execution_state["step_results"][step_id] = step_result
                        results[step_id] = step_result
                        completed_steps.add(step_id)
                    else:
                        execution_state["errors"].append(
                            f"Step {step_id} failed: {step_result.get('error')}"
                        )
                        # For parallel execution, continue with other steps
                        completed_steps.add(step_id)  # Mark as completed to avoid infinite loop

                except Exception as e:
                    execution_state["errors"].append(f"Step {step_id} exception: {str(e)}")
                    completed_steps.add(step_id)

        return results

    async def _should_execute_step(
        self, step: WorkflowStep, execution_state: Dict[str, Any]
    ) -> bool:
        """Check if step should be executed based on conditions"""
        if step.condition:
            try:
                return await step.condition(execution_state)
            except Exception as e:
                self.logger.error(f"Error evaluating condition for step {step.id}: {e}")
                return False
        return True

    async def _execute_step_with_retry(
        self, step: WorkflowStep, execution_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute step with retry logic"""
        max_retries = 3
        retry_count = execution_state["retry_counts"].get(step.id, 0)

        for attempt in range(max_retries):
            try:
                # Prepare parameters
                parameters = step.parameters.copy()

                # Add context from previous steps
                context_data = {}
                for dep_id in step.dependencies:
                    if dep_id in execution_state["step_results"]:
                        context_data[dep_id] = execution_state["step_results"][dep_id].get("result")

                if context_data:
                    parameters["_context"] = context_data

                # Execute tool
                result = await asyncio.wait_for(
                    self.agent.execute_tool(step.tool_name, parameters), timeout=step.timeout
                )

                if result.get("success"):
                    return result
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)  # Exponential backoff
                        continue
                    else:
                        return result

            except asyncio.TimeoutError:
                error_msg = f"Step {step.id} timed out after {step.timeout}s"
                if attempt < max_retries - 1:
                    self.logger.warning(f"{error_msg}, retrying...")
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    return {"success": False, "error": error_msg}

            except Exception as e:
                error_msg = f"Step {step.id} failed: {str(e)}"
                if attempt < max_retries - 1:
                    self.logger.warning(f"{error_msg}, retrying...")
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    return {"success": False, "error": error_msg}

            finally:
                execution_state["retry_counts"][step.id] = retry_count + attempt + 1

        return {"success": False, "error": f"Max retries exceeded for step {step.id}"}

    def _topological_sort(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Topological sort of workflow steps"""
        # Build dependency graph
        graph = {step.id: step for step in steps}
        in_degree = {step.id: len(step.dependencies) for step in steps}

        # Find steps with no dependencies
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        sorted_steps = []

        while queue:
            current_id = queue.pop(0)
            sorted_steps.append(graph[current_id])

            # Update in-degrees of dependent steps
            for step in steps:
                if current_id in step.dependencies:
                    in_degree[step.id] -= 1
                    if in_degree[step.id] == 0:
                        queue.append(step.id)

        return sorted_steps

    async def _post_execution_processing(
        self, execution_state: Dict[str, Any], result: Dict[str, Any]
    ):
        """Post-execution processing and cleanup"""
        # Update agent context
        workflow_summary = {
            "execution_id": execution_state["execution_id"],
            "workflow_id": execution_state["workflow_id"],
            "steps_executed": len(execution_state["step_results"]),
            "duration": (datetime.utcnow() - execution_state["start_time"]).total_seconds(),
            "success": execution_state["status"] == WorkflowStatus.COMPLETED,
        }

        self.agent.context.workflow_state[execution_state["execution_id"]] = workflow_summary

        # Trigger any follow-up workflows if configured
        await self._trigger_follow_up_workflows(execution_state, result)

    async def _trigger_follow_up_workflows(
        self, execution_state: Dict[str, Any], result: Dict[str, Any]
    ):
        """Trigger follow-up workflows based on results"""
        workflow_id = execution_state["workflow_id"]

        # Example follow-up logic
        if workflow_id == "complete_trading_strategy":
            # If a trade was executed, trigger monitoring
            if any(
                "execute_trade" in step_result.get("tool", "") for step_result in result.values()
            ):
                await self.process_workflow("realtime_monitoring")

        elif workflow_id == "portfolio_management_cycle":
            # If rebalancing is needed, trigger coordination workflow
            rebalancing_result = result.get("rebalancing_decision", {}).get("result", {})
            if rebalancing_result.get("rebalancing_needed"):
                await self.process_workflow(
                    "multi_agent_coordination",
                    {"action": "portfolio_rebalancing", "data": rebalancing_result},
                )

    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get comprehensive workflow execution metrics"""
        if not self.execution_history:
            return {"total_executions": 0}

        completed = [e for e in self.execution_history if e["status"] == "completed"]
        failed = [e for e in self.execution_history if e["status"] == "failed"]

        durations = [e.get("duration", 0) for e in completed]

        return {
            "total_executions": len(self.execution_history),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self.execution_history),
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "total_steps_executed": sum(e.get("steps_executed", 0) for e in completed),
            "workflow_types": list(set(e["workflow_id"] for e in self.execution_history)),
        }
