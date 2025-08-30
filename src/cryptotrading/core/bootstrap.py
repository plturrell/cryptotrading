"""
Bootstrap module for dependency injection container
Registers all services and breaks circular dependencies
"""
import logging
from typing import Dict, Any, Optional

from .di_container import DIContainer, get_container, reset_container
from .interfaces import (
    ILogger,
    IMetricsCollector,
    IHealthChecker,
    IConfigProvider,
    IServiceRegistry,
    ICache,
    ISecurityManager,
    ICommunicationManager,
)
from .infrastructure import (
    EnterpriseLogger,
    EnterpriseMetricsCollector,
    EnterpriseHealthChecker,
    EnterpriseConfigProvider,
    EnterpriseServiceRegistry,
    EnterpriseInMemoryCache,
    SimpleSecurityManager,
    SimpleCommunicationManager,
)

logger = logging.getLogger(__name__)


class ApplicationBootstrap:
    """
    Application bootstrap class that configures dependency injection
    and initializes all enterprise services
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.container: Optional[DIContainer] = None
        self._initialized = False

    async def initialize(self) -> DIContainer:
        """Initialize the application with dependency injection"""
        if self._initialized:
            return self.container

        logger.info("Initializing application bootstrap")

        # Reset container for clean start
        reset_container()
        self.container = get_container()

        # Register core infrastructure services
        await self._register_infrastructure_services()

        # Register business services
        await self._register_business_services()

        # Initialize all singleton services
        logger.info("Initializing all registered services")
        initialization_results = await self.container.initialize_all()

        # Log initialization results
        for service_name, success in initialization_results.items():
            if success:
                logger.info(f"Successfully initialized: {service_name}")
            else:
                logger.error(f"Failed to initialize: {service_name}")

        # Perform initial health checks
        await self._perform_initial_health_checks()

        self._initialized = True
        logger.info("Application bootstrap completed successfully")

        return self.container

    async def _register_infrastructure_services(self):
        """Register core infrastructure services in DI container"""
        logger.info("Registering infrastructure services")

        # Logging service
        self.container.register_singleton(
            ILogger,
            factory=lambda: EnterpriseLogger(
                name=self.config.get("logger_name", "cryptotrading"),
                level=self.config.get("log_level", "INFO"),
            ),
        )

        # Metrics collection service
        self.container.register_singleton(
            IMetricsCollector,
            factory=lambda: EnterpriseMetricsCollector(
                max_entries=self.config.get("metrics_max_entries", 10000)
            ),
        )

        # Health checking service
        self.container.register_singleton(IHealthChecker, EnterpriseHealthChecker)

        # Configuration service
        config_file = self.config.get("config_file")
        if config_file:
            self.container.register_singleton(
                IConfigProvider, factory=lambda: EnterpriseConfigProvider(config_file)
            )
        else:
            self.container.register_singleton(IConfigProvider, EnterpriseConfigProvider)

        # Service registry
        self.container.register_singleton(IServiceRegistry, EnterpriseServiceRegistry)

        # Cache service
        self.container.register_singleton(
            ICache,
            factory=lambda: EnterpriseInMemoryCache(
                default_ttl=self.config.get("cache_default_ttl", 3600)
            ),
        )

        # Security services
        self.container.register_singleton(ISecurityManager, SimpleSecurityManager)

        # Communication services
        self.container.register_singleton(ICommunicationManager, SimpleCommunicationManager)

        logger.info("Infrastructure services registered successfully")

    async def _register_business_services(self):
        """Register business logic services"""
        logger.info("Registering business services")

        # Import business services here to avoid circular imports
        try:
            from .agents.components import ToolManager, WorkflowEngine, ContextManager

            # Register component types (but not as singletons since they're per-agent)
            self.container.register_transient(ToolManager, ToolManager)
            self.container.register_transient(WorkflowEngine, WorkflowEngine)
            self.container.register_transient(ContextManager, ContextManager)

            logger.info("Business services registered successfully")

        except ImportError as e:
            logger.warning(f"Some business services not available: {e}")

    async def _perform_initial_health_checks(self):
        """Perform initial health checks on all services"""
        try:
            health_checker = await self.container.resolve(IHealthChecker)

            # Register health checks for all services
            await self._register_service_health_checks(health_checker)

            # Perform initial system health check
            system_health = await health_checker.get_system_health()

            logger.info(
                f"System health: {system_health['overall_status']} "
                f"({system_health['healthy_services']}/{system_health['total_services']} "
                f"services healthy)"
            )

            if system_health["overall_status"] != "healthy":
                logger.warning("System is not fully healthy after initialization")
                for service_name, health in system_health["services"].items():
                    if not health.get("healthy", False):
                        logger.warning(
                            f"Unhealthy service: {service_name} - "
                            f"{health.get('error', 'Unknown error')}"
                        )

        except Exception as e:
            logger.error(f"Failed to perform initial health checks: {e}")

    async def _register_service_health_checks(self, health_checker: IHealthChecker):
        """Register health check functions for all services"""

        # Health check for logger
        async def logger_health_check():
            try:
                logger = await self.container.resolve(ILogger)
                logger.debug("Health check test")
                return {"healthy": True, "response_time_ms": 1}
            except Exception as e:
                return {"healthy": False, "error": str(e)}

        await health_checker.register_health_check("logger", logger_health_check)

        # Health check for metrics collector
        async def metrics_health_check():
            try:
                metrics = await self.container.resolve(IMetricsCollector)
                await metrics.record_metric("health_check", 1.0)
                return {"healthy": True, "response_time_ms": 2}
            except Exception as e:
                return {"healthy": False, "error": str(e)}

        await health_checker.register_health_check("metrics_collector", metrics_health_check)

        # Health check for cache
        async def cache_health_check():
            try:
                cache = await self.container.resolve(ICache)
                test_key = "health_check_test"
                await cache.set(test_key, "test_value", ttl=10)
                value = await cache.get(test_key)
                await cache.delete(test_key)

                healthy = value == "test_value"
                return {"healthy": healthy, "response_time_ms": 3}
            except Exception as e:
                return {"healthy": False, "error": str(e)}

        await health_checker.register_health_check("cache", cache_health_check)

        # Health check for service registry
        async def registry_health_check():
            try:
                registry = await self.container.resolve(IServiceRegistry)
                services = await registry.list_services()
                return {"healthy": True, "response_time_ms": 1, "service_count": len(services)}
            except Exception as e:
                return {"healthy": False, "error": str(e)}

        await health_checker.register_health_check("service_registry", registry_health_check)

    async def shutdown(self):
        """Graceful shutdown of all services"""
        if not self._initialized or not self.container:
            return

        logger.info("Shutting down application")

        try:
            await self.container.shutdown_all()
            logger.info("Application shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self._initialized = False

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all services"""
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            health_checker = await self.container.resolve(IHealthChecker)
            return await health_checker.get_system_health()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            metrics_collector = await self.container.resolve(IMetricsCollector)
            return await metrics_collector.get_metrics()
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global bootstrap instance
_bootstrap: Optional[ApplicationBootstrap] = None


async def initialize_application(config: Dict[str, Any] = None) -> DIContainer:
    """Initialize the application with dependency injection"""
    global _bootstrap

    if _bootstrap is None:
        _bootstrap = ApplicationBootstrap(config)

    return await _bootstrap.initialize()


async def shutdown_application():
    """Shutdown the application"""
    global _bootstrap

    if _bootstrap:
        await _bootstrap.shutdown()
        _bootstrap = None


async def get_application_health() -> Dict[str, Any]:
    """Get application health status"""
    global _bootstrap

    if _bootstrap:
        return await _bootstrap.get_health_status()
    else:
        return {"status": "not_initialized"}


async def get_application_metrics() -> Dict[str, Any]:
    """Get application metrics"""
    global _bootstrap

    if _bootstrap:
        return await _bootstrap.get_metrics()
    else:
        return {"status": "not_initialized"}


# Convenience function for getting the initialized container
def get_initialized_container() -> Optional[DIContainer]:
    """Get the initialized DI container"""
    return get_container() if _bootstrap and _bootstrap._initialized else None
