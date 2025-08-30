"""
MCP Progress Notifications Implementation
Implements progress tracking and notifications for long-running operations
"""

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ProgressState(Enum):
    """Progress states"""

    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    """Progress update information"""

    token: str
    progress: Union[int, float]
    total: Optional[Union[int, float]] = None
    message: Optional[str] = None
    state: ProgressState = ProgressState.RUNNING
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    @property
    def percentage(self) -> Optional[float]:
        """Calculate percentage complete"""
        if self.total and self.total > 0:
            return min(100.0, (self.progress / self.total) * 100)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP notification format"""
        result = {
            "method": "notifications/progress",
            "params": {
                "progressToken": self.token,
                "progress": self.progress,
                "timestamp": self.timestamp.isoformat(),
                "state": self.state.value,
            },
        }

        if self.total is not None:
            result["params"]["total"] = self.total
        if self.message:
            result["params"]["message"] = self.message
        if self.percentage is not None:
            result["params"]["percentage"] = self.percentage

        return result


class ProgressTracker:
    """Tracks progress for a specific operation"""

    def __init__(
        self,
        token: str,
        total: Optional[Union[int, float]] = None,
        description: Optional[str] = None,
    ):
        self.token = token
        self.total = total
        self.description = description
        self.current_progress = 0
        self.state = ProgressState.STARTING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.cancelled = False

        # Event handlers
        self.update_handlers: List[Callable[[ProgressUpdate], None]] = []

        # Sub-operations
        self.sub_operations: Dict[str, "ProgressTracker"] = {}
        self.sub_operation_weights: Dict[str, float] = {}

    def add_update_handler(self, handler: Callable[[ProgressUpdate], None]):
        """Add progress update handler"""
        self.update_handlers.append(handler)

    def start(self):
        """Start tracking progress"""
        if self.state == ProgressState.STARTING:
            self.state = ProgressState.RUNNING
            self.started_at = datetime.utcnow()
            self._notify_update("Started operation")

    def update(
        self, progress: Union[int, float], message: Optional[str] = None, force_notify: bool = False
    ):
        """Update progress"""
        if self.state not in [ProgressState.RUNNING, ProgressState.PAUSED]:
            return

        old_progress = self.current_progress
        self.current_progress = progress

        # Only notify if progress changed significantly or forced
        if force_notify or abs(progress - old_progress) >= 1:
            self._notify_update(message)

    def increment(self, amount: Union[int, float] = 1, message: Optional[str] = None):
        """Increment progress by amount"""
        new_progress = self.current_progress + amount
        if self.total:
            new_progress = min(new_progress, self.total)
        self.update(new_progress, message)

    def set_total(self, total: Union[int, float]):
        """Update total progress amount"""
        self.total = total
        self._notify_update("Updated total progress")

    def pause(self, message: Optional[str] = None):
        """Pause progress tracking"""
        if self.state == ProgressState.RUNNING:
            self.state = ProgressState.PAUSED
            self._notify_update(message or "Operation paused")

    def resume(self, message: Optional[str] = None):
        """Resume progress tracking"""
        if self.state == ProgressState.PAUSED:
            self.state = ProgressState.RUNNING
            self._notify_update(message or "Operation resumed")

    def complete(self, message: Optional[str] = None):
        """Mark operation as completed"""
        if self.state in [ProgressState.RUNNING, ProgressState.PAUSED]:
            self.state = ProgressState.COMPLETED
            self.completed_at = datetime.utcnow()
            if self.total:
                self.current_progress = self.total
            self._notify_update(message or "Operation completed")

    def fail(self, error: str, message: Optional[str] = None):
        """Mark operation as failed"""
        if self.state in [ProgressState.RUNNING, ProgressState.PAUSED]:
            self.state = ProgressState.FAILED
            self.error = error
            self.completed_at = datetime.utcnow()
            self._notify_update(message or f"Operation failed: {error}")

    def cancel(self, message: Optional[str] = None):
        """Cancel operation"""
        if self.state in [ProgressState.RUNNING, ProgressState.PAUSED]:
            self.state = ProgressState.CANCELLED
            self.cancelled = True
            self.completed_at = datetime.utcnow()
            self._notify_update(message or "Operation cancelled")

    def add_sub_operation(
        self, name: str, weight: float = 1.0, total: Optional[Union[int, float]] = None
    ) -> "ProgressTracker":
        """Add sub-operation with weighted contribution to overall progress"""
        token = f"{self.token}:{name}"
        sub_tracker = ProgressTracker(token, total, f"{self.description} - {name}")

        # Forward updates to parent
        def handle_sub_update(update: ProgressUpdate):
            self._update_from_sub_operations()

        sub_tracker.add_update_handler(handle_sub_update)

        self.sub_operations[name] = sub_tracker
        self.sub_operation_weights[name] = weight

        return sub_tracker

    def _update_from_sub_operations(self):
        """Update progress based on sub-operations"""
        if not self.sub_operations:
            return

        total_weight = sum(self.sub_operation_weights.values())
        if total_weight == 0:
            return

        weighted_progress = 0
        for name, sub_tracker in self.sub_operations.items():
            weight = self.sub_operation_weights[name]

            if sub_tracker.total and sub_tracker.total > 0:
                sub_percentage = sub_tracker.current_progress / sub_tracker.total
            else:
                # For operations without known total, consider state
                if sub_tracker.state == ProgressState.COMPLETED:
                    sub_percentage = 1.0
                elif sub_tracker.state in [ProgressState.FAILED, ProgressState.CANCELLED]:
                    sub_percentage = 1.0  # Consider failed/cancelled as "done"
                else:
                    sub_percentage = 0.0

            weighted_progress += (weight / total_weight) * sub_percentage

        # Update parent progress
        if self.total:
            new_progress = weighted_progress * self.total
        else:
            new_progress = weighted_progress * 100  # Percentage

        self.current_progress = new_progress
        self._notify_update("Sub-operation progress updated")

    def _notify_update(self, message: Optional[str] = None):
        """Notify handlers of progress update"""
        update = ProgressUpdate(
            token=self.token,
            progress=self.current_progress,
            total=self.total,
            message=message,
            state=self.state,
        )

        for handler in self.update_handlers:
            try:
                handler(update)
            except Exception as e:
                logger.error(f"Error in progress update handler: {e}")

    @property
    def duration(self) -> Optional[timedelta]:
        """Get operation duration"""
        if not self.started_at:
            return None

        end_time = self.completed_at or datetime.utcnow()
        return end_time - self.started_at

    @property
    def estimated_remaining(self) -> Optional[timedelta]:
        """Estimate remaining time"""
        if (
            not self.started_at
            or not self.total
            or self.current_progress <= 0
            or self.state != ProgressState.RUNNING
        ):
            return None

        elapsed = datetime.utcnow() - self.started_at
        progress_ratio = self.current_progress / self.total

        if progress_ratio >= 1.0:
            return timedelta(0)

        estimated_total = elapsed / progress_ratio
        return estimated_total - elapsed

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        status = {
            "token": self.token,
            "description": self.description,
            "progress": self.current_progress,
            "total": self.total,
            "state": self.state.value,
            "createdAt": self.created_at.isoformat(),
            "error": self.error,
            "cancelled": self.cancelled,
        }

        if self.started_at:
            status["startedAt"] = self.started_at.isoformat()
        if self.completed_at:
            status["completedAt"] = self.completed_at.isoformat()

        duration = self.duration
        if duration:
            status["duration"] = duration.total_seconds()

        remaining = self.estimated_remaining
        if remaining:
            status["estimatedRemaining"] = remaining.total_seconds()

        if self.sub_operations:
            status["subOperations"] = {
                name: tracker.get_status() for name, tracker in self.sub_operations.items()
            }

        percentage = ProgressUpdate(self.token, self.current_progress, self.total).percentage
        if percentage is not None:
            status["percentage"] = percentage

        return status


class ProgressManager:
    """Manager for progress tracking operations"""

    def __init__(self):
        self.trackers: Dict[str, ProgressTracker] = {}
        self.notification_handlers: List[Callable[[ProgressUpdate], None]] = []
        self.cleanup_interval = 3600  # 1 hour
        self._cleanup_task: Optional[asyncio.Task] = None

    def add_notification_handler(self, handler: Callable[[ProgressUpdate], None]):
        """Add global progress notification handler"""
        self.notification_handlers.append(handler)

    def create_tracker(
        self,
        total: Optional[Union[int, float]] = None,
        description: Optional[str] = None,
        token: Optional[str] = None,
    ) -> ProgressTracker:
        """Create new progress tracker"""
        if not token:
            token = str(uuid.uuid4())

        tracker = ProgressTracker(token, total, description)

        # Add global notification handler
        def handle_update(update: ProgressUpdate):
            for handler in self.notification_handlers:
                try:
                    handler(update)
                except Exception as e:
                    logger.error(f"Error in progress notification handler: {e}")

        tracker.add_update_handler(handle_update)

        self.trackers[token] = tracker

        # Start cleanup task if needed
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_completed())

        logger.info(f"Created progress tracker: {token}")
        return tracker

    def get_tracker(self, token: str) -> Optional[ProgressTracker]:
        """Get tracker by token"""
        return self.trackers.get(token)

    def cancel_tracker(self, token: str, message: Optional[str] = None):
        """Cancel tracker by token"""
        tracker = self.get_tracker(token)
        if tracker:
            tracker.cancel(message)

    def list_trackers(
        self, active_only: bool = False, include_sub_operations: bool = False
    ) -> List[Dict[str, Any]]:
        """List all trackers"""
        trackers = list(self.trackers.values())

        if active_only:
            active_states = [ProgressState.STARTING, ProgressState.RUNNING, ProgressState.PAUSED]
            trackers = [t for t in trackers if t.state in active_states]

        # Filter out sub-operation trackers unless requested
        if not include_sub_operations:
            trackers = [t for t in trackers if ":" not in t.token]

        return [t.get_status() for t in trackers]

    async def _cleanup_completed(self):
        """Periodically clean up completed trackers"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Keep for 24 hours
                completed_states = [
                    ProgressState.COMPLETED,
                    ProgressState.FAILED,
                    ProgressState.CANCELLED,
                ]

                to_remove = []
                for token, tracker in self.trackers.items():
                    if (
                        tracker.state in completed_states
                        and tracker.completed_at
                        and tracker.completed_at < cutoff_time
                    ):
                        to_remove.append(token)

                for token in to_remove:
                    del self.trackers[token]
                    logger.info(f"Cleaned up completed tracker: {token}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in progress cleanup: {e}")


# Global progress manager
progress_manager = ProgressManager()


# Context manager for progress tracking
class progress_tracker:
    """Context manager for automatic progress tracking"""

    def __init__(
        self,
        total: Optional[Union[int, float]] = None,
        description: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.total = total
        self.description = description
        self.token = token
        self.tracker: Optional[ProgressTracker] = None

    def __enter__(self) -> ProgressTracker:
        self.tracker = progress_manager.create_tracker(self.total, self.description, self.token)
        self.tracker.start()
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker:
            if exc_type is None:
                self.tracker.complete()
            else:
                self.tracker.fail(str(exc_val))


# Async context manager
class async_progress_tracker:
    """Async context manager for progress tracking"""

    def __init__(
        self,
        total: Optional[Union[int, float]] = None,
        description: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.total = total
        self.description = description
        self.token = token
        self.tracker: Optional[ProgressTracker] = None

    async def __aenter__(self) -> ProgressTracker:
        self.tracker = progress_manager.create_tracker(self.total, self.description, self.token)
        self.tracker.start()
        return self.tracker

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.tracker:
            if exc_type is None:
                self.tracker.complete()
            else:
                self.tracker.fail(str(exc_val))


# Decorator for automatic progress tracking
def track_progress(total: Optional[Union[int, float]] = None, description: Optional[str] = None):
    """Decorator to automatically track function progress"""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                async with async_progress_tracker(total, description or func.__name__) as tracker:
                    return await func(tracker, *args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with progress_tracker(total, description or func.__name__) as tracker:
                    return func(tracker, *args, **kwargs)

            return sync_wrapper

    return decorator


# Helper functions
def create_progress_notification(
    token: str,
    progress: Union[int, float],
    total: Optional[Union[int, float]] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Create progress notification in MCP format"""
    update = ProgressUpdate(token, progress, total, message)
    return update.to_dict()
