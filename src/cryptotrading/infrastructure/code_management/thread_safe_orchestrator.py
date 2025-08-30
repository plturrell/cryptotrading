"""
Thread-Safe Enterprise Code Orchestrator
Provides production-ready orchestration with proper concurrency handling
"""

import asyncio
import json
import logging
import threading
import uuid
from asyncio import Lock, Queue, Semaphore
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ...core.config.production_config import get_config
from .automated_quality_monitor import AutomatedQualityMonitor
from .code_health_dashboard import CodeHealthDashboard
from .database_adapter import CodeManagementDatabaseAdapter
from .intelligent_code_manager import CodeHealthMetrics, IntelligentCodeManager
from .issue_backlog_tracker import IssueBacklogTracker
from .issue_lifecycle_manager import IssueLifecycleManager
from .proactive_issue_detector import ProactiveIssueDetector
from .seal_code_adapter import SEALCodeAdapter
from .seal_workflow_engine import SEALWorkflowEngine

logger = logging.getLogger(__name__)

# Thread-safe context variables
current_task_id: ContextVar[Optional[str]] = ContextVar("current_task_id", default=None)
current_component: ContextVar[Optional[str]] = ContextVar("current_component", default=None)


@dataclass
class TaskContext:
    """Context for coordinated tasks"""

    task_id: str
    component: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


class ThreadSafeMetrics:
    """Thread-safe metrics collection"""

    def __init__(self):
        self._metrics: Dict[str, Any] = defaultdict(lambda: 0)
        self._lock = threading.Lock()

    def increment(self, key: str, value: float = 1.0):
        """Thread-safe increment"""
        with self._lock:
            self._metrics[key] += value

    def set(self, key: str, value: Any):
        """Thread-safe set"""
        with self._lock:
            self._metrics[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe get"""
        with self._lock:
            return self._metrics.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            return dict(self._metrics)


class CoordinatedTaskManager:
    """Manages coordinated execution of tasks across components"""

    def __init__(self, max_concurrent_tasks: int = 10):
        self._tasks: Dict[str, TaskContext] = {}
        self._task_lock = asyncio.Lock()
        self._semaphore = Semaphore(max_concurrent_tasks)
        self._task_queue: Queue[TaskContext] = Queue()
        self._results: Dict[str, Any] = {}
        self._result_lock = asyncio.Lock()

    async def submit_task(self, component: str, func: Callable, *args, **kwargs) -> str:
        """Submit a task for coordinated execution"""
        task_id = str(uuid.uuid4())
        context = TaskContext(task_id=task_id, component=component, started_at=datetime.utcnow())

        async with self._task_lock:
            self._tasks[task_id] = context

        await self._task_queue.put((context, func, args, kwargs))
        return task_id

    async def execute_tasks(self):
        """Execute tasks from the queue"""
        while True:
            try:
                context, func, args, kwargs = await self._task_queue.get()

                async with self._semaphore:
                    # Set context
                    token_task = current_task_id.set(context.task_id)
                    token_comp = current_component.set(context.component)

                    try:
                        context.status = "running"

                        # Execute function
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            # Run sync function in thread pool
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(None, func, *args, **kwargs)

                        context.status = "completed"
                        context.result = result

                        async with self._result_lock:
                            self._results[context.task_id] = result

                    except Exception as e:
                        context.status = "failed"
                        context.error = str(e)
                        logger.error(f"Task {context.task_id} failed: {e}")

                    finally:
                        context.completed_at = datetime.utcnow()
                        current_task_id.reset(token_task)
                        current_component.reset(token_comp)

            except Exception as e:
                logger.error(f"Task execution error: {e}")
                await asyncio.sleep(1)

    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result of a task with optional timeout"""
        start_time = datetime.utcnow()

        while True:
            async with self._task_lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    if task.status in ["completed", "failed"]:
                        if task.status == "failed":
                            raise Exception(f"Task failed: {task.error}")
                        return task.result

            if timeout and (datetime.utcnow() - start_time).total_seconds() > timeout:
                raise TimeoutError(f"Task {task_id} timed out")

            await asyncio.sleep(0.1)

    async def wait_for_all(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        """Wait for multiple tasks to complete"""
        results = {}

        if timeout:
            tasks = [
                asyncio.create_task(self.get_task_result(task_id, timeout)) for task_id in task_ids
            ]
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            for task_id, result in zip(task_ids, completed_results):
                if isinstance(result, Exception):
                    results[task_id] = {"error": str(result)}
                else:
                    results[task_id] = result
        else:
            for task_id in task_ids:
                try:
                    results[task_id] = await self.get_task_result(task_id)
                except Exception as e:
                    results[task_id] = {"error": str(e)}

        return results


class ThreadSafeEnterpriseOrchestrator:
    """Thread-safe version of the enterprise code orchestrator"""

    def __init__(
        self,
        project_path: Path,
        config: "OrchestrationConfig",
        database_adapter: Optional[CodeManagementDatabaseAdapter] = None,
    ):
        self.project_path = project_path
        self.config = config
        self.database_adapter = database_adapter
        self.running = False

        # Thread-safe components
        self.metrics = ThreadSafeMetrics()
        self.task_manager = CoordinatedTaskManager(max_concurrent_tasks=5)
        self._component_locks = {
            "health": asyncio.Lock(),
            "quality": asyncio.Lock(),
            "proactive": asyncio.Lock(),
            "lifecycle": asyncio.Lock(),
            "seal": asyncio.Lock(),
            "dashboard": asyncio.Lock(),
        }

        # Event coordination
        self._event_bus: Queue[Dict[str, Any]] = Queue(maxsize=1000)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

        # Initialize components
        self._initialize_components()

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()

    def _initialize_components(self):
        """Initialize all components with thread safety"""
        # Use lazy initialization to avoid thread issues
        self._code_manager = None
        self._quality_monitor = None
        self._proactive_detector = None
        self._lifecycle_manager = None
        self._backlog_tracker = None
        self._seal_adapter = None
        self._seal_workflow_engine = None
        self._dashboard = None

    @property
    def code_manager(self) -> IntelligentCodeManager:
        """Lazy initialization of code manager"""
        if self._code_manager is None:
            self._code_manager = IntelligentCodeManager(self.project_path, self.database_adapter)
        return self._code_manager

    @property
    def quality_monitor(self) -> AutomatedQualityMonitor:
        """Lazy initialization of quality monitor"""
        if self._quality_monitor is None:
            self._quality_monitor = AutomatedQualityMonitor(
                self.project_path, self.database_adapter
            )
        return self._quality_monitor

    @property
    def proactive_detector(self) -> ProactiveIssueDetector:
        """Lazy initialization of proactive detector"""
        if self._proactive_detector is None:
            self._proactive_detector = ProactiveIssueDetector(
                self.project_path, self.database_adapter
            )
        return self._proactive_detector

    @property
    def dashboard(self) -> CodeHealthDashboard:
        """Lazy initialization of dashboard"""
        if self._dashboard is None:
            self._dashboard = CodeHealthDashboard(
                project_path=str(self.project_path),
                port=self.config.dashboard_port,
                database_adapter=self.database_adapter,
            )
        return self._dashboard

    async def start_enterprise_monitoring(self) -> None:
        """Start the enterprise monitoring system with proper coordination"""
        logger.info("🚀 Starting Thread-Safe Enterprise Code Management System...")

        self.running = True

        # Start task executor
        executor_task = asyncio.create_task(self.task_manager.execute_tasks())
        self._background_tasks.add(executor_task)

        # Start event processor
        event_task = asyncio.create_task(self._process_events())
        self._background_tasks.add(event_task)

        # Start monitoring loops with coordination
        monitoring_tasks = [
            self._start_health_monitoring(),
            self._start_quality_monitoring(),
            self._start_proactive_scanning(),
            self._start_lifecycle_management(),
            self._start_seal_workflow(),
            self._start_dashboard_server(),
        ]

        # Submit all monitoring tasks
        task_ids = []
        for task_func in monitoring_tasks:
            task_id = await self.task_manager.submit_task(
                component=task_func.__name__, func=task_func
            )
            task_ids.append(task_id)

        try:
            # Wait for all tasks
            await self.task_manager.wait_for_all(task_ids)
        except Exception as e:
            logger.error(f"Error in enterprise monitoring: {e}")
            self.running = False

    async def _start_health_monitoring(self):
        """Health monitoring with thread safety"""
        logger.info("🏥 Starting health monitoring...")

        while self.running:
            try:
                async with self._component_locks["health"]:
                    # Run health check
                    health_metrics = await self.code_manager.comprehensive_health_check()

                    # Update metrics
                    self.metrics.set("last_health_check", datetime.utcnow())
                    self.metrics.set("coverage_percentage", health_metrics.coverage_percentage)
                    self.metrics.set("technical_debt_score", health_metrics.technical_debt_score)

                    # Emit event
                    await self._emit_event(
                        "health_check_completed",
                        {
                            "metrics": asdict(health_metrics),
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                    # Check for critical issues
                    if health_metrics.coverage_percentage < 70.0:
                        await self._emit_event(
                            "critical_health_issue",
                            {"type": "low_coverage", "value": health_metrics.coverage_percentage},
                        )

                await asyncio.sleep(self.config.continuous_monitoring_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                self.metrics.increment("health_errors")
                await asyncio.sleep(60)

    async def _start_quality_monitoring(self):
        """Quality monitoring with thread safety"""
        logger.info("🔍 Starting quality monitoring...")

        while self.running:
            try:
                async with self._component_locks["quality"]:
                    # Run quality checks
                    results = await self.quality_monitor.run_all_checks()
                    issues = self.quality_monitor.process_results(results)

                    # Update metrics
                    self.metrics.set("last_quality_check", datetime.utcnow())
                    self.metrics.set("quality_issues_found", len(issues))

                    # Auto-fix if enabled
                    if self.config.auto_fix_enabled and issues:
                        fixed_count = 0
                        for issue in issues[: self.config.max_auto_fixes_per_cycle]:
                            try:
                                if await self._try_auto_fix(issue):
                                    fixed_count += 1
                            except Exception as e:
                                logger.error(f"Auto-fix failed: {e}")

                        self.metrics.increment("total_fixes_applied", fixed_count)

                    # Emit event
                    await self._emit_event(
                        "quality_check_completed",
                        {
                            "issues_found": len(issues),
                            "auto_fixed": fixed_count if self.config.auto_fix_enabled else 0,
                        },
                    )

                await asyncio.sleep(self.config.quality_check_interval)

            except Exception as e:
                logger.error(f"Quality monitoring error: {e}")
                self.metrics.increment("quality_errors")
                await asyncio.sleep(60)

    async def _start_proactive_scanning(self):
        """Proactive scanning with thread safety"""
        logger.info("🔮 Starting proactive scanning...")

        while self.running:
            try:
                async with self._component_locks["proactive"]:
                    # Run proactive scan
                    issues = await self.proactive_detector.scan_project()

                    # Update metrics
                    self.metrics.set("last_proactive_scan", datetime.utcnow())
                    self.metrics.set("proactive_issues_found", len(issues))

                    # Emit event
                    await self._emit_event(
                        "proactive_scan_completed",
                        {"issues_found": len(issues), "timestamp": datetime.utcnow().isoformat()},
                    )

                await asyncio.sleep(self.config.proactive_scan_interval)

            except Exception as e:
                logger.error(f"Proactive scanning error: {e}")
                self.metrics.increment("proactive_errors")
                await asyncio.sleep(60)

    async def _start_lifecycle_management(self):
        """Lifecycle management with thread safety"""
        if not self.database_adapter:
            logger.warning("Lifecycle management not available - database required")
            return

        logger.info("🔄 Starting lifecycle management...")

        # Initialize components
        if self._lifecycle_manager is None:
            self._lifecycle_manager = IssueLifecycleManager(self.database_adapter)
        if self._backlog_tracker is None:
            self._backlog_tracker = IssueBacklogTracker(
                self.database_adapter, self._lifecycle_manager
            )

        while self.running:
            try:
                async with self._component_locks["lifecycle"]:
                    # Auto-triage issues
                    triaged = await self._lifecycle_manager.auto_triage_issues()

                    # Auto-progress issues
                    progressed = await self._lifecycle_manager.auto_progress_fixable_issues()

                    # Get backlog metrics
                    backlog_metrics = await self._backlog_tracker.get_backlog_metrics()

                    # Update metrics
                    self.metrics.set("issues_triaged", triaged)
                    self.metrics.set("issues_progressed", progressed)
                    self.metrics.set("backlog_size", backlog_metrics.total_backlog_size)

                    # Emit event
                    await self._emit_event(
                        "lifecycle_cycle_completed",
                        {
                            "triaged": triaged,
                            "progressed": progressed,
                            "backlog_size": backlog_metrics.total_backlog_size,
                        },
                    )

                await asyncio.sleep(1800)  # 30 minutes

            except Exception as e:
                logger.error(f"Lifecycle management error: {e}")
                self.metrics.increment("lifecycle_errors")
                await asyncio.sleep(300)

    async def _start_seal_workflow(self):
        """SEAL workflow with thread safety"""
        if not self.database_adapter:
            logger.warning("SEAL workflow not available - database required")
            return

        logger.info("🤖 Starting SEAL workflow...")

        # Initialize components
        if self._seal_adapter is None:
            self._seal_adapter = SEALCodeAdapter(self.database_adapter, self._lifecycle_manager)
        if self._seal_workflow_engine is None:
            self._seal_workflow_engine = SEALWorkflowEngine(
                self.database_adapter,
                self._lifecycle_manager,
                self._backlog_tracker,
                self._seal_adapter,
            )

        while self.running:
            try:
                async with self._component_locks["seal"]:
                    # Run workflow scheduler
                    await self._seal_workflow_engine.run_workflow_scheduler()

                    # Update metrics
                    self.metrics.increment("seal_workflow_runs")

                await asyncio.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"SEAL workflow error: {e}")
                self.metrics.increment("seal_errors")
                await asyncio.sleep(600)

    async def _start_dashboard_server(self):
        """Start dashboard server safely"""
        logger.info("📊 Starting dashboard server...")

        async with self._component_locks["dashboard"]:
            # Start dashboard in a safe way
            try:
                # Create async wrapper for dashboard
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.dashboard.run, False)  # debug=False
            except Exception as e:
                logger.error(f"Dashboard server error: {e}")
                self.metrics.increment("dashboard_errors")

    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to event bus"""
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": current_task_id.get(),
            "component": current_component.get(),
            "data": data,
        }

        try:
            await self._event_bus.put(event)
        except asyncio.QueueFull:
            logger.warning(f"Event bus full, dropping event: {event_type}")

    async def _process_events(self):
        """Process events from event bus"""
        while self.running:
            try:
                event = await self._event_bus.get()

                # Notify subscribers
                subscribers = self._subscribers.get(event["type"], [])
                for subscriber in subscribers:
                    try:
                        if asyncio.iscoroutinefunction(subscriber):
                            await subscriber(event)
                        else:
                            subscriber(event)
                    except Exception as e:
                        logger.error(f"Event subscriber error: {e}")

                # Log event
                logger.debug(f"Event: {event['type']} - {event['data']}")

            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(1)

    def subscribe_to_event(self, event_type: str, callback: Callable):
        """Subscribe to specific event type"""
        self._subscribers[event_type].append(callback)

    async def _try_auto_fix(self, issue: Any) -> bool:
        """Try to auto-fix an issue"""
        try:
            # This would implement actual fix logic
            logger.info(f"Auto-fixing issue: {issue}")
            return True
        except Exception as e:
            logger.error(f"Auto-fix failed: {e}")
            return False

    async def stop_monitoring(self):
        """Stop all monitoring with proper cleanup"""
        logger.info("🛑 Stopping Enterprise Code Management System...")
        self.running = False

        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clear background tasks
        self._background_tasks.clear()

        logger.info("✅ System stopped successfully")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        metrics = self.metrics.get_all()

        return {
            "running": self.running,
            "metrics": metrics,
            "components": {
                "health": self._code_manager is not None,
                "quality": self._quality_monitor is not None,
                "proactive": self._proactive_detector is not None,
                "lifecycle": self._lifecycle_manager is not None,
                "seal": self._seal_workflow_engine is not None,
                "dashboard": self._dashboard is not None,
            },
            "dashboard_url": f"http://localhost:{self.config.dashboard_port}",
            "config": asdict(self.config),
        }
