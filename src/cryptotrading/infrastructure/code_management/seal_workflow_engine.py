"""
SEAL Self-Editing Workflow Engine
Automated workflows for continuous code improvement using SEAL framework
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database_adapter import CodeManagementDatabaseAdapter
from .issue_backlog_tracker import IssueBacklogTracker
from .issue_lifecycle_manager import IssueLifecycleManager, IssueState
from .seal_code_adapter import (
    AdaptationStrategy,
    AdaptationType,
    CodeAdaptationRequest,
    SEALCodeAdapter,
)

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of SEAL workflows"""

    CONTINUOUS_IMPROVEMENT = "continuous_improvement"
    SPRINT_OPTIMIZATION = "sprint_optimization"
    EMERGENCY_FIXING = "emergency_fixing"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    PROACTIVE_ENHANCEMENT = "proactive_enhancement"


class WorkflowStatus(Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class SEALWorkflow:
    """SEAL workflow definition"""

    id: str
    name: str
    workflow_type: WorkflowType
    description: str
    trigger_conditions: Dict[str, Any]
    execution_schedule: str  # cron-like schedule
    priority: int
    enabled: bool
    created_at: str
    last_execution: Optional[str] = None
    execution_count: int = 0


@dataclass
class WorkflowExecution:
    """SEAL workflow execution record"""

    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    started_at: str
    completed_at: Optional[str]
    adaptations_processed: int
    successful_adaptations: int
    failed_adaptations: int
    total_confidence: float
    execution_log: List[str]
    results: Dict[str, Any]


class SEALWorkflowEngine:
    """Engine for managing and executing SEAL-based code improvement workflows"""

    def __init__(
        self,
        database_adapter: CodeManagementDatabaseAdapter,
        lifecycle_manager: IssueLifecycleManager,
        backlog_tracker: IssueBacklogTracker,
        seal_adapter: SEALCodeAdapter,
    ):
        self.database_adapter = database_adapter
        self.lifecycle_manager = lifecycle_manager
        self.backlog_tracker = backlog_tracker
        self.seal_adapter = seal_adapter

        self.workflows: List[SEALWorkflow] = []
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []

        # Initialize default workflows
        self._initialize_default_workflows()

    def _initialize_default_workflows(self) -> None:
        """Initialize default SEAL workflows"""
        default_workflows = [
            SEALWorkflow(
                id="continuous_improvement",
                name="Continuous Code Improvement",
                workflow_type=WorkflowType.CONTINUOUS_IMPROVEMENT,
                description="Continuously improve code quality using SEAL adaptations",
                trigger_conditions={"min_issues": 5, "max_severity": 8, "auto_fixable_ratio": 0.3},
                execution_schedule="0 */6 * * *",  # Every 6 hours
                priority=5,
                enabled=True,
                created_at=datetime.now().isoformat(),
            ),
            SEALWorkflow(
                id="sprint_optimization",
                name="Sprint-Based Optimization",
                workflow_type=WorkflowType.SPRINT_OPTIMIZATION,
                description="Optimize code during sprint cycles",
                trigger_conditions={"sprint_active": True, "completion_rate": 0.7},
                execution_schedule="0 9 * * 1",  # Monday mornings
                priority=7,
                enabled=True,
                created_at=datetime.now().isoformat(),
            ),
            SEALWorkflow(
                id="emergency_fixing",
                name="Emergency Issue Resolution",
                workflow_type=WorkflowType.EMERGENCY_FIXING,
                description="Rapidly address critical issues using SEAL",
                trigger_conditions={"critical_issues": 1, "severity_threshold": 9},
                execution_schedule="immediate",
                priority=10,
                enabled=True,
                created_at=datetime.now().isoformat(),
            ),
            SEALWorkflow(
                id="proactive_enhancement",
                name="Proactive Code Enhancement",
                workflow_type=WorkflowType.PROACTIVE_ENHANCEMENT,
                description="Proactively enhance code before issues arise",
                trigger_conditions={"code_age_days": 30, "complexity_threshold": 7},
                execution_schedule="0 2 * * 0",  # Sunday nights
                priority=3,
                enabled=True,
                created_at=datetime.now().isoformat(),
            ),
        ]

        self.workflows.extend(default_workflows)
        logger.info("Initialized %d default SEAL workflows", len(default_workflows))

    async def evaluate_workflow_triggers(self) -> List[str]:
        """Evaluate which workflows should be triggered"""
        triggered_workflows = []

        try:
            for workflow in self.workflows:
                if not workflow.enabled:
                    continue

                should_trigger = await self._evaluate_trigger_conditions(workflow)

                if should_trigger:
                    triggered_workflows.append(workflow.id)
                    logger.info("Workflow %s triggered", workflow.name)

            return triggered_workflows

        except Exception as e:
            logger.error("Error evaluating workflow triggers: %s", e)
            return []

    async def _evaluate_trigger_conditions(self, workflow: SEALWorkflow) -> bool:
        """Evaluate if workflow trigger conditions are met"""
        try:
            conditions = workflow.trigger_conditions

            if workflow.workflow_type == WorkflowType.CONTINUOUS_IMPROVEMENT:
                # Check issue count and severity
                issues = await self.database_adapter.get_issues()
                active_issues = []
                for i in issues:
                    metadata = (
                        json.loads(i.get("metadata", "{}"))
                        if isinstance(i.get("metadata"), str)
                        else i.get("metadata", {})
                    )
                    if metadata.get("lifecycle_state") in ["triaged", "backlog", "in_progress"]:
                        active_issues.append(i)

                if len(active_issues) < conditions.get("min_issues", 5):
                    return False

                high_severity_issues = [
                    i
                    for i in active_issues
                    if i.get("severity", 5) >= conditions.get("max_severity", 8)
                ]
                if len(high_severity_issues) > 3:
                    return True

                auto_fixable_count = len([i for i in active_issues if i.get("auto_fixable", False)])
                auto_fixable_ratio = auto_fixable_count / len(active_issues) if active_issues else 0

                return auto_fixable_ratio >= conditions.get("auto_fixable_ratio", 0.3)

            elif workflow.workflow_type == WorkflowType.EMERGENCY_FIXING:
                # Check for critical issues
                issues = await self.database_adapter.get_issues()
                critical_issues = [
                    i
                    for i in issues
                    if i.get("severity", 5) >= conditions.get("severity_threshold", 9)
                ]

                return len(critical_issues) >= conditions.get("critical_issues", 1)

            elif workflow.workflow_type == WorkflowType.SPRINT_OPTIMIZATION:
                # Check sprint status
                if self.backlog_tracker.current_sprint:
                    sprint_report = await self.backlog_tracker.get_progress_report(
                        self.backlog_tracker.current_sprint.id
                    )
                    return (
                        sprint_report.completion_percentage
                        >= conditions.get("completion_rate", 0.7) * 100
                    )

                return False

            elif workflow.workflow_type == WorkflowType.PROACTIVE_ENHANCEMENT:
                # Check code age and complexity (simplified)
                return True  # Would implement actual code analysis

            return False

        except Exception as e:
            logger.error("Error evaluating trigger conditions for %s: %s", workflow.name, e)
            return False

    async def execute_workflow(self, workflow_id: str) -> WorkflowExecution:
        """Execute a SEAL workflow"""
        try:
            workflow = next((w for w in self.workflows if w.id == workflow_id), None)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")

            execution_id = f"{workflow_id}_{int(datetime.now().timestamp())}"

            execution = WorkflowExecution(
                workflow_id=workflow_id,
                execution_id=execution_id,
                status=WorkflowStatus.RUNNING,
                started_at=datetime.now().isoformat(),
                completed_at=None,
                adaptations_processed=0,
                successful_adaptations=0,
                failed_adaptations=0,
                total_confidence=0.0,
                execution_log=[],
                results={},
            )

            self.active_executions[execution_id] = execution
            execution.execution_log.append(f"Started workflow execution: {workflow.name}")

            # Execute workflow based on type
            if workflow.workflow_type == WorkflowType.CONTINUOUS_IMPROVEMENT:
                await self._execute_continuous_improvement(execution, workflow)
            elif workflow.workflow_type == WorkflowType.EMERGENCY_FIXING:
                await self._execute_emergency_fixing(execution, workflow)
            elif workflow.workflow_type == WorkflowType.SPRINT_OPTIMIZATION:
                await self._execute_sprint_optimization(execution, workflow)
            elif workflow.workflow_type == WorkflowType.PROACTIVE_ENHANCEMENT:
                await self._execute_proactive_enhancement(execution, workflow)

            # Complete execution
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now().isoformat()

            # Update workflow statistics
            workflow.last_execution = execution.completed_at
            workflow.execution_count += 1

            # Store execution results
            await self._store_workflow_execution(execution)

            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]

            logger.info(
                "Workflow %s completed: %d/%d adaptations successful",
                workflow.name,
                execution.successful_adaptations,
                execution.adaptations_processed,
            )

            return execution

        except Exception as e:
            logger.error("Error executing workflow %s: %s", workflow_id, e)

            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = WorkflowStatus.FAILED
                execution.completed_at = datetime.now().isoformat()
                execution.execution_log.append(f"Workflow failed: {e}")

                self.execution_history.append(execution)
                del self.active_executions[execution_id]

                return execution

            raise

    async def _execute_continuous_improvement(
        self, execution: WorkflowExecution, workflow: SEALWorkflow
    ) -> None:
        """Execute continuous improvement workflow"""
        execution.execution_log.append("Starting continuous improvement cycle")

        # Get adaptation opportunities
        project_path = Path("/Users/apple/projects/cryptotrading")  # Would be configurable
        adaptation_requests = await self.seal_adapter.analyze_codebase_for_adaptation(project_path)

        # Filter for continuous improvement
        improvement_requests = [
            req
            for req in adaptation_requests
            if req.adaptation_type
            in [
                AdaptationType.CODE_QUALITY_IMPROVEMENT,
                AdaptationType.REFACTORING,
                AdaptationType.PERFORMANCE_OPTIMIZATION,
            ]
        ]

        execution.execution_log.append(
            f"Found {len(improvement_requests)} improvement opportunities"
        )

        # Process adaptations
        await self._process_adaptations(execution, improvement_requests[:10])  # Limit to 10

    async def _execute_emergency_fixing(
        self, execution: WorkflowExecution, workflow: SEALWorkflow
    ) -> None:
        """Execute emergency fixing workflow"""
        execution.execution_log.append("Starting emergency fixing cycle")

        # Get critical issues
        issues = await self.database_adapter.get_issues()
        critical_issues = [i for i in issues if i.get("severity", 5) >= 9]

        # Convert to adaptation requests
        adaptation_requests = []
        for issue in critical_issues:
            try:
                file_path = issue.get("file_path", "")
                if file_path:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code_content = f.read()
                else:
                    code_content = ""

                request = CodeAdaptationRequest(
                    issue_id=issue.get("id", ""),
                    file_path=file_path,
                    original_code=code_content,
                    issue_description=issue.get("description", ""),
                    adaptation_type=AdaptationType.BUG_FIX,
                    strategy=AdaptationStrategy.SELF_EDIT,
                    context={"severity": issue.get("severity", 5), "emergency": True},
                    priority=10,
                    created_at=datetime.now().isoformat(),
                )
                adaptation_requests.append(request)

            except Exception as e:
                execution.execution_log.append(
                    f"Could not process critical issue {issue.get('id', 'unknown')}: {e}"
                )

        execution.execution_log.append(f"Processing {len(adaptation_requests)} critical issues")

        # Process with high priority
        await self._process_adaptations(execution, adaptation_requests)

    async def _execute_sprint_optimization(
        self, execution: WorkflowExecution, workflow: SEALWorkflow
    ) -> None:
        """Execute sprint optimization workflow"""
        execution.execution_log.append("Starting sprint optimization cycle")

        if not self.backlog_tracker.current_sprint:
            execution.execution_log.append("No active sprint found")
            return

        # Get sprint issues
        sprint_id = self.backlog_tracker.current_sprint.id
        issues = await self.database_adapter.get_issues()
        sprint_issues = [
            i for i in issues if i.metadata and i.metadata.get("sprint_id") == sprint_id
        ]

        # Focus on in-progress issues
        in_progress_issues = [
            i for i in sprint_issues if i.metadata.get("lifecycle_state") == "in_progress"
        ]

        execution.execution_log.append(f"Optimizing {len(in_progress_issues)} sprint issues")

        # Convert to adaptation requests
        adaptation_requests = []
        for issue in in_progress_issues:
            try:
                with open(issue.file_path, "r", encoding="utf-8") as f:
                    code_content = f.read()

                request = CodeAdaptationRequest(
                    issue_id=issue.id,
                    file_path=issue.file_path,
                    code_content=code_content,
                    issue_description=issue.description,
                    adaptation_type=self.seal_adapter._classify_adaptation_type(issue),
                    strategy=AdaptationStrategy.FEW_SHOT_ADAPTATION,
                    context={"sprint_id": sprint_id, "sprint_optimization": True},
                    priority=7,
                    created_at=datetime.now().isoformat(),
                )
                adaptation_requests.append(request)

            except Exception as e:
                execution.execution_log.append(f"Could not process sprint issue {issue.id}: {e}")

        await self._process_adaptations(execution, adaptation_requests)

    async def _execute_proactive_enhancement(
        self, execution: WorkflowExecution, workflow: SEALWorkflow
    ) -> None:
        """Execute proactive enhancement workflow"""
        execution.execution_log.append("Starting proactive enhancement cycle")

        # Get all code files for proactive analysis
        project_path = Path("/Users/apple/projects/cryptotrading")
        python_files = list(project_path.rglob("*.py"))

        # Analyze files for enhancement opportunities
        enhancement_requests = []

        for file_path in python_files[:20]:  # Limit to 20 files
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_content = f.read()

                # Simple heuristics for enhancement opportunities
                if (
                    len(code_content) > 1000
                    and "TODO" in code_content  # Large files
                    or code_content.count("def ") > 10  # Has TODOs
                ):  # Many functions
                    request = CodeAdaptationRequest(
                        issue_id=f"proactive_{file_path.stem}_{int(datetime.now().timestamp())}",
                        file_path=str(file_path),
                        code_content=code_content,
                        issue_description=f"Proactive enhancement for {file_path.name}",
                        adaptation_type=AdaptationType.CODE_QUALITY_IMPROVEMENT,
                        strategy=AdaptationStrategy.CONTINUAL_LEARNING,
                        context={"proactive": True, "file_size": len(code_content)},
                        priority=3,
                        created_at=datetime.now().isoformat(),
                    )
                    enhancement_requests.append(request)

            except Exception as e:
                execution.execution_log.append(f"Could not analyze {file_path}: {e}")

        execution.execution_log.append(
            f"Found {len(enhancement_requests)} enhancement opportunities"
        )

        await self._process_adaptations(execution, enhancement_requests[:5])  # Limit to 5

    async def _process_adaptations(
        self, execution: WorkflowExecution, requests: List[CodeAdaptationRequest]
    ) -> None:
        """Process adaptation requests within a workflow execution"""
        execution.adaptations_processed = len(requests)

        for request in requests:
            try:
                execution.execution_log.append(f"Processing adaptation: {request.issue_id}")

                # Perform SEAL adaptation
                result = await self.seal_adapter.perform_self_edit_adaptation(request)

                if result.confidence_score >= 0.7:
                    execution.successful_adaptations += 1
                    execution.execution_log.append(
                        f"Successful adaptation: {request.issue_id} (confidence: {result.confidence_score:.2f})"
                    )
                else:
                    execution.failed_adaptations += 1
                    execution.execution_log.append(
                        f"Failed adaptation: {request.issue_id} (confidence: {result.confidence_score:.2f})"
                    )

                execution.total_confidence += result.confidence_score

            except Exception as e:
                execution.failed_adaptations += 1
                execution.execution_log.append(f"Error processing {request.issue_id}: {e}")

        # Calculate average confidence
        if execution.adaptations_processed > 0:
            execution.results["average_confidence"] = (
                execution.total_confidence / execution.adaptations_processed
            )
            execution.results["success_rate"] = (
                execution.successful_adaptations / execution.adaptations_processed
            )

    async def _store_workflow_execution(self, execution: WorkflowExecution) -> None:
        """Store workflow execution results"""
        try:
            await self.database_adapter.log_monitoring_event(
                event_type="seal_workflow_execution",
                component="seal_workflow_engine",
                severity="info",
                message=f"Workflow execution completed: {execution.workflow_id}",
                details=asdict(execution),
            )

        except Exception as e:
            logger.error("Error storing workflow execution: %s", e)

    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status"""
        try:
            return {
                "total_workflows": len(self.workflows),
                "enabled_workflows": len([w for w in self.workflows if w.enabled]),
                "active_executions": len(self.active_executions),
                "total_executions": len(self.execution_history),
                "successful_executions": len(
                    [e for e in self.execution_history if e.status == WorkflowStatus.COMPLETED]
                ),
                "workflows": [asdict(w) for w in self.workflows],
                "recent_executions": [asdict(e) for e in self.execution_history[-10:]],
            }

        except Exception as e:
            logger.error("Error getting workflow status: %s", e)
            return {"error": str(e)}

    async def run_workflow_scheduler(self) -> None:
        """Run the workflow scheduler loop"""
        logger.info("Starting SEAL workflow scheduler...")

        while True:
            try:
                # Evaluate triggers
                triggered_workflows = await self.evaluate_workflow_triggers()

                # Execute triggered workflows
                for workflow_id in triggered_workflows:
                    try:
                        await self.execute_workflow(workflow_id)
                    except Exception as e:
                        logger.error("Error executing workflow %s: %s", workflow_id, e)

                # Wait before next evaluation
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error("Error in workflow scheduler: %s", e)
                await asyncio.sleep(60)  # Wait before retrying
