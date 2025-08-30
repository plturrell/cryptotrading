"""
Strands Framework Lifecycle Management
Comprehensive lifecycle management for Strands agents, tools, and workflows
"""

import asyncio
import json
import logging
import signal
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """Lifecycle states for Strands components"""

    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"


class ComponentType(Enum):
    """Types of Strands components"""

    AGENT = "agent"
    TOOL = "tool"
    WORKFLOW = "workflow"
    DATABASE = "database"
    API_CLIENT = "api_client"
    OBSERVER = "observer"


@dataclass
class ComponentHealth:
    """Health status of a component"""

    component_id: str
    component_type: ComponentType
    state: LifecycleState
    health_score: float  # 0.0 to 1.0
    last_check: datetime
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class LifecycleEvent:
    """Lifecycle event for tracking state changes"""

    timestamp: datetime
    component_id: str
    event_type: str
    from_state: Optional[LifecycleState]
    to_state: LifecycleState
    metadata: Dict[str, Any] = field(default_factory=dict)


class LifecycleManager:
    """Comprehensive lifecycle management for Strands framework"""

    def __init__(
        self,
        health_check_interval: float = 30.0,
        cleanup_interval: float = 300.0,
        max_event_history: int = 1000,
    ):
        self.health_check_interval = health_check_interval
        self.cleanup_interval = cleanup_interval
        self.max_event_history = max_event_history

        # Component tracking
        self.components: Dict[str, ComponentHealth] = {}
        self.component_refs: Dict[str, weakref.ref] = {}  # Weak references to avoid memory leaks
        self.event_history: List[LifecycleEvent] = []

        # Lifecycle hooks
        self.lifecycle_hooks: Dict[str, List[Callable]] = {
            "before_start": [],
            "after_start": [],
            "before_stop": [],
            "after_stop": [],
            "on_error": [],
            "on_health_check": [],
        }

        # State management
        self.manager_state = LifecycleState.STOPPED
        self.shutdown_requested = False
        self._background_tasks: List[asyncio.Task] = []

        # Health thresholds
        self.health_thresholds = {
            "error_rate": 0.1,  # 10% error rate threshold
            "response_time": 5.0,  # 5 second response time threshold
            "memory_usage": 0.8,  # 80% memory usage threshold
        }

        # Graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except Exception as e:
            logger.warning(f"Could not setup signal handlers: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_requested = True

        # Schedule shutdown
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.shutdown())

    async def start(self):
        """Start the lifecycle manager"""
        logger.info("Starting Strands Lifecycle Manager")

        await self._transition_state(LifecycleState.STARTING)

        try:
            # Start background tasks
            self._background_tasks = [
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._cleanup_loop()),
                asyncio.create_task(self._event_processor()),
            ]

            await self._transition_state(LifecycleState.RUNNING)
            logger.info("Lifecycle Manager started successfully")

        except Exception as e:
            logger.error(f"Failed to start Lifecycle Manager: {e}")
            await self._transition_state(LifecycleState.ERROR)
            raise

    async def shutdown(self):
        """Graceful shutdown of lifecycle manager"""
        logger.info("Shutting down Strands Lifecycle Manager")

        await self._transition_state(LifecycleState.STOPPING)

        try:
            # Execute before_stop hooks
            await self._execute_hooks("before_stop")

            # Stop all managed components
            await self._stop_all_components()

            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            # Execute after_stop hooks
            await self._execute_hooks("after_stop")

            await self._transition_state(LifecycleState.STOPPED)
            logger.info("Lifecycle Manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            await self._transition_state(LifecycleState.ERROR)

    def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        component_ref: Any,
        dependencies: List[str] = None,
    ) -> ComponentHealth:
        """Register a component for lifecycle management"""
        health = ComponentHealth(
            component_id=component_id,
            component_type=component_type,
            state=LifecycleState.INITIALIZING,
            health_score=1.0,
            last_check=datetime.utcnow(),
            dependencies=dependencies or [],
        )

        self.components[component_id] = health
        self.component_refs[component_id] = weakref.ref(component_ref)

        logger.info(f"Registered component: {component_id} ({component_type.value})")

        # Record event
        self._record_event(component_id, "registered", None, LifecycleState.INITIALIZING)

        return health

    def unregister_component(self, component_id: str):
        """Unregister a component"""
        if component_id in self.components:
            self._record_event(
                component_id,
                "unregistered",
                self.components[component_id].state,
                LifecycleState.STOPPED,
            )

            del self.components[component_id]
            if component_id in self.component_refs:
                del self.component_refs[component_id]

            logger.info(f"Unregistered component: {component_id}")

    async def start_component(self, component_id: str) -> bool:
        """Start a specific component"""
        if component_id not in self.components:
            logger.error(f"Component {component_id} not registered")
            return False

        health = self.components[component_id]

        try:
            await self._transition_component_state(component_id, LifecycleState.STARTING)

            # Check dependencies
            if not await self._check_dependencies(component_id):
                logger.error(f"Dependencies not met for component {component_id}")
                await self._transition_component_state(component_id, LifecycleState.ERROR)
                return False

            # Get component reference
            component_ref = self.component_refs.get(component_id)
            if component_ref is None:
                logger.error(f"Component reference lost for {component_id}")
                await self._transition_component_state(component_id, LifecycleState.ERROR)
                return False

            component = component_ref()
            if component is None:
                logger.error(f"Component {component_id} was garbage collected")
                self.unregister_component(component_id)
                return False

            # Start the component
            if hasattr(component, "start"):
                await component.start()
            elif hasattr(component, "initialize"):
                await component.initialize()

            await self._transition_component_state(component_id, LifecycleState.RUNNING)
            logger.info(f"Started component: {component_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to start component {component_id}: {e}")
            await self._transition_component_state(component_id, LifecycleState.ERROR)
            health.errors.append(f"Start failed: {str(e)}")
            return False

    async def stop_component(self, component_id: str) -> bool:
        """Stop a specific component"""
        if component_id not in self.components:
            logger.error(f"Component {component_id} not registered")
            return False

        health = self.components[component_id]

        try:
            await self._transition_component_state(component_id, LifecycleState.STOPPING)

            # Get component reference
            component_ref = self.component_refs.get(component_id)
            if component_ref:
                component = component_ref()
                if component and hasattr(component, "stop"):
                    await component.stop()
                elif component and hasattr(component, "shutdown"):
                    await component.shutdown()

            await self._transition_component_state(component_id, LifecycleState.STOPPED)
            logger.info(f"Stopped component: {component_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to stop component {component_id}: {e}")
            await self._transition_component_state(component_id, LifecycleState.ERROR)
            health.errors.append(f"Stop failed: {str(e)}")
            return False

    async def check_component_health(self, component_id: str) -> float:
        """Check health of a specific component"""
        if component_id not in self.components:
            return 0.0

        health = self.components[component_id]

        try:
            # Get component reference
            component_ref = self.component_refs.get(component_id)
            if component_ref is None:
                health.health_score = 0.0
                return 0.0

            component = component_ref()
            if component is None:
                health.health_score = 0.0
                self.unregister_component(component_id)
                return 0.0

            # Calculate health score
            health_score = 1.0

            # Check if component has health method
            if hasattr(component, "get_health"):
                component_health = await component.get_health()
                if isinstance(component_health, dict):
                    health_score = component_health.get("health_score", 1.0)
                    health.metrics.update(component_health.get("metrics", {}))
                elif isinstance(component_health, (int, float)):
                    health_score = float(component_health)

            # Check error rate
            if hasattr(component, "get_metrics"):
                metrics = await component.get_metrics()
                if isinstance(metrics, dict):
                    error_rate = metrics.get("error_rate", 0.0)
                    if error_rate > self.health_thresholds["error_rate"]:
                        health_score *= 0.5

                    response_time = metrics.get("avg_response_time", 0.0)
                    if response_time > self.health_thresholds["response_time"]:
                        health_score *= 0.7

            health.health_score = health_score
            health.last_check = datetime.utcnow()

            # Update state based on health
            if health_score < 0.3:
                await self._transition_component_state(component_id, LifecycleState.ERROR)
            elif health_score < 0.7:
                await self._transition_component_state(component_id, LifecycleState.DEGRADED)
            elif health.state in [LifecycleState.ERROR, LifecycleState.DEGRADED]:
                await self._transition_component_state(component_id, LifecycleState.RUNNING)

            return health_score

        except Exception as e:
            logger.error(f"Health check failed for component {component_id}: {e}")
            health.errors.append(f"Health check failed: {str(e)}")
            health.health_score = 0.0
            await self._transition_component_state(component_id, LifecycleState.ERROR)
            return 0.0

    def add_lifecycle_hook(self, hook_name: str, callback: Callable):
        """Add a lifecycle hook"""
        if hook_name in self.lifecycle_hooks:
            self.lifecycle_hooks[hook_name].append(callback)
            logger.debug(f"Added lifecycle hook: {hook_name}")
        else:
            logger.warning(f"Unknown hook name: {hook_name}")

    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        total_components = len(self.components)
        if total_components == 0:
            return {
                "overall_health": 1.0,
                "total_components": 0,
                "healthy_components": 0,
                "degraded_components": 0,
                "error_components": 0,
                "timestamp": datetime.utcnow().isoformat(),
            }

        healthy = sum(1 for h in self.components.values() if h.health_score >= 0.7)
        degraded = sum(1 for h in self.components.values() if 0.3 <= h.health_score < 0.7)
        error = sum(1 for h in self.components.values() if h.health_score < 0.3)

        overall_health = sum(h.health_score for h in self.components.values()) / total_components

        return {
            "overall_health": overall_health,
            "total_components": total_components,
            "healthy_components": healthy,
            "degraded_components": degraded,
            "error_components": error,
            "manager_state": self.manager_state.value,
            "components": {
                comp_id: {
                    "type": health.component_type.value,
                    "state": health.state.value,
                    "health_score": health.health_score,
                    "last_check": health.last_check.isoformat(),
                    "error_count": len(health.errors),
                }
                for comp_id, health in self.components.items()
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _health_check_loop(self):
        """Background health check loop"""
        while not self.shutdown_requested:
            try:
                # Check all components
                for component_id in list(self.components.keys()):
                    await self.check_component_health(component_id)

                # Execute health check hooks
                await self._execute_hooks("on_health_check")

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self.shutdown_requested:
            try:
                # Clean up old events
                if len(self.event_history) > self.max_event_history:
                    self.event_history = self.event_history[-self.max_event_history :]

                # Clean up old errors
                for health in self.components.values():
                    if len(health.errors) > 10:  # Keep last 10 errors
                        health.errors = health.errors[-10:]

                # Clean up dead component references
                dead_components = []
                for comp_id, ref in self.component_refs.items():
                    if ref() is None:
                        dead_components.append(comp_id)

                for comp_id in dead_components:
                    logger.warning(f"Component {comp_id} was garbage collected")
                    self.unregister_component(comp_id)

                await asyncio.sleep(self.cleanup_interval)

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.cleanup_interval)

    async def _event_processor(self):
        """Process lifecycle events"""
        while not self.shutdown_requested:
            try:
                # Process events (placeholder for future event processing)
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(1.0)

    async def _transition_state(self, new_state: LifecycleState):
        """Transition manager state"""
        old_state = self.manager_state
        self.manager_state = new_state

        event = LifecycleEvent(
            timestamp=datetime.utcnow(),
            component_id="lifecycle_manager",
            event_type="state_transition",
            from_state=old_state,
            to_state=new_state,
        )
        self.event_history.append(event)

        logger.info(f"Lifecycle Manager: {old_state.value} -> {new_state.value}")

    async def _transition_component_state(self, component_id: str, new_state: LifecycleState):
        """Transition component state"""
        if component_id not in self.components:
            return

        health = self.components[component_id]
        old_state = health.state
        health.state = new_state

        self._record_event(component_id, "state_transition", old_state, new_state)

        logger.debug(f"Component {component_id}: {old_state.value} -> {new_state.value}")

    def _record_event(
        self,
        component_id: str,
        event_type: str,
        from_state: Optional[LifecycleState],
        to_state: LifecycleState,
        metadata: Dict[str, Any] = None,
    ):
        """Record a lifecycle event"""
        event = LifecycleEvent(
            timestamp=datetime.utcnow(),
            component_id=component_id,
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            metadata=metadata or {},
        )

        self.event_history.append(event)

    async def _check_dependencies(self, component_id: str) -> bool:
        """Check if component dependencies are met"""
        if component_id not in self.components:
            return False

        dependencies = self.components[component_id].dependencies

        for dep_id in dependencies:
            if dep_id not in self.components:
                logger.error(f"Dependency {dep_id} not found for component {component_id}")
                return False

            dep_health = self.components[dep_id]
            if dep_health.state not in [LifecycleState.RUNNING, LifecycleState.DEGRADED]:
                logger.error(f"Dependency {dep_id} not running for component {component_id}")
                return False

        return True

    async def _stop_all_components(self):
        """Stop all managed components"""
        # Stop in reverse dependency order
        stop_order = self._calculate_stop_order()

        for component_id in stop_order:
            try:
                await self.stop_component(component_id)
            except Exception as e:
                logger.error(f"Error stopping component {component_id}: {e}")

    def _calculate_stop_order(self) -> List[str]:
        """Calculate optimal stop order based on dependencies"""
        # Simple topological sort for dependencies
        component_ids = list(self.components.keys())
        ordered = []
        remaining = set(component_ids)

        while remaining:
            # Find components with no remaining dependencies
            ready = []
            for comp_id in remaining:
                deps = set(self.components[comp_id].dependencies)
                if not (deps & remaining):  # No dependencies in remaining
                    ready.append(comp_id)

            if not ready:
                # Circular dependency or other issue, just take remaining
                ready = list(remaining)

            # Sort by dependency count (fewer dependencies first)
            ready.sort(key=lambda x: len(self.components[x].dependencies))

            ordered.extend(ready)
            remaining -= set(ready)

        # Reverse for stop order (most dependent first)
        return list(reversed(ordered))

    async def _execute_hooks(self, hook_name: str):
        """Execute lifecycle hooks"""
        if hook_name in self.lifecycle_hooks:
            for callback in self.lifecycle_hooks[hook_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self)
                    else:
                        callback(self)
                except Exception as e:
                    logger.error(f"Error executing {hook_name} hook: {e}")


# Global lifecycle manager instance
_lifecycle_manager: Optional[LifecycleManager] = None


def get_lifecycle_manager() -> LifecycleManager:
    """Get the global lifecycle manager instance"""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = LifecycleManager()
    return _lifecycle_manager


@asynccontextmanager
async def managed_component(
    component_id: str, component_type: ComponentType, component: Any, dependencies: List[str] = None
):
    """Context manager for lifecycle-managed components"""
    manager = get_lifecycle_manager()

    # Register component
    health = manager.register_component(component_id, component_type, component, dependencies)

    try:
        # Start component
        success = await manager.start_component(component_id)
        if not success:
            raise Exception(f"Failed to start component {component_id}")

        yield health

    finally:
        # Stop component
        await manager.stop_component(component_id)
        manager.unregister_component(component_id)
