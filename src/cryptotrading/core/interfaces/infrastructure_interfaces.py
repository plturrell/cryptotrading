"""
Infrastructure Interface Definitions
Abstract interfaces for infrastructure components to prevent circular dependencies
"""
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class ServiceStatus(Enum):
    """Service status enumeration"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class IConfigProvider(ABC):
    """Configuration provider interface"""

    @abstractmethod
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass

    @abstractmethod
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        pass

    @abstractmethod
    async def get_config_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        pass

    @abstractmethod
    async def reload_config(self) -> bool:
        """Reload configuration from source"""
        pass

    @abstractmethod
    def watch_config(self, key: str, callback: Callable[[Any], None]):
        """Watch configuration key for changes"""
        pass


class IServiceRegistry(ABC):
    """Service registry interface"""

    @abstractmethod
    async def register_service(
        self, service_name: str, service_instance: Any, metadata: Dict[str, Any] = None
    ) -> bool:
        """Register service instance"""
        pass

    @abstractmethod
    async def unregister_service(self, service_name: str, instance_id: str = None) -> bool:
        """Unregister service instance"""
        pass

    @abstractmethod
    async def get_service(self, service_name: str) -> Optional[Any]:
        """Get service instance"""
        pass

    @abstractmethod
    async def list_services(self, service_type: str = None) -> List[Dict[str, Any]]:
        """List registered services"""
        pass

    @abstractmethod
    async def service_exists(self, service_name: str) -> bool:
        """Check if service is registered"""
        pass


class IHealthChecker(ABC):
    """Health checker interface"""

    @abstractmethod
    async def check_health(self, service_name: str = None) -> Dict[str, Any]:
        """Perform health check"""
        pass

    @abstractmethod
    async def register_health_check(
        self, service_name: str, check_func: Callable[[], Dict[str, Any]]
    ):
        """Register health check function"""
        pass

    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        pass

    @abstractmethod
    def get_health_status(self, service_name: str) -> ServiceStatus:
        """Get health status for specific service"""
        pass


class IMetricsCollector(ABC):
    """Metrics collection interface"""

    @abstractmethod
    async def record_metric(
        self,
        name: str,
        value: Union[int, float],
        tags: Dict[str, str] = None,
        timestamp: datetime = None,
    ):
        """Record a metric"""
        pass

    @abstractmethod
    async def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        pass

    @abstractmethod
    async def record_timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record timing metric"""
        pass

    @abstractmethod
    async def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Set gauge metric"""
        pass

    @abstractmethod
    async def get_metrics(self, name_pattern: str = None) -> Dict[str, Any]:
        """Get collected metrics"""
        pass


class ILogger(ABC):
    """Logging interface"""

    @abstractmethod
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        pass

    @abstractmethod
    def info(self, message: str, **kwargs):
        """Log info message"""
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        pass

    @abstractmethod
    def error(self, message: str, **kwargs):
        """Log error message"""
        pass

    @abstractmethod
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        pass

    @abstractmethod
    def set_level(self, level: str):
        """Set logging level"""
        pass


class ICache(ABC):
    """Cache interface"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    async def clear(self, pattern: str = None) -> int:
        """Clear cache entries"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class ITaskScheduler(ABC):
    """Task scheduler interface"""

    @abstractmethod
    async def schedule_task(
        self, task_id: str, task_func: Callable, schedule: str, **kwargs
    ) -> bool:
        """Schedule a task"""
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel scheduled task"""
        pass

    @abstractmethod
    async def run_task_now(self, task_id: str) -> Any:
        """Run task immediately"""
        pass

    @abstractmethod
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List scheduled tasks"""
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        pass


class IResourceManager(ABC):
    """Resource manager interface"""

    @abstractmethod
    async def acquire_resource(self, resource_type: str, resource_id: str = None) -> Optional[str]:
        """Acquire a resource"""
        pass

    @abstractmethod
    async def release_resource(self, resource_id: str) -> bool:
        """Release a resource"""
        pass

    @abstractmethod
    async def get_resource_status(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get resource status"""
        pass

    @abstractmethod
    async def list_resources(self, resource_type: str = None) -> List[Dict[str, Any]]:
        """List available resources"""
        pass

    @abstractmethod
    async def set_resource_limit(self, resource_type: str, limit: int):
        """Set resource limit"""
        pass


class ILockManager(ABC):
    """Distributed lock manager interface"""

    @abstractmethod
    async def acquire_lock(self, lock_name: str, timeout: float = 30.0) -> Optional[str]:
        """Acquire distributed lock"""
        pass

    @abstractmethod
    async def release_lock(self, lock_name: str, lock_token: str) -> bool:
        """Release distributed lock"""
        pass

    @abstractmethod
    async def extend_lock(self, lock_name: str, lock_token: str, additional_time: float) -> bool:
        """Extend lock duration"""
        pass

    @abstractmethod
    async def is_locked(self, lock_name: str) -> bool:
        """Check if resource is locked"""
        pass


class IFileStorage(ABC):
    """File storage interface"""

    @abstractmethod
    async def store_file(
        self, file_path: str, content: bytes, metadata: Dict[str, Any] = None
    ) -> bool:
        """Store file"""
        pass

    @abstractmethod
    async def retrieve_file(self, file_path: str) -> Optional[bytes]:
        """Retrieve file content"""
        pass

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file"""
        pass

    @abstractmethod
    async def list_files(self, prefix: str = None) -> List[Dict[str, Any]]:
        """List files"""
        pass

    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists"""
        pass

    @abstractmethod
    async def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata"""
        pass
