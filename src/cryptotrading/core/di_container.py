"""
Dependency Injection Container
Manages component dependencies and breaks circular import cycles
"""
import asyncio
import inspect
import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ServiceRegistration:
    """Service registration information"""

    service_type: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    singleton: bool = True
    initialized: bool = False
    dependencies: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class CircularDependencyError(Exception):
    """Raised when circular dependency is detected"""

    pass


class ServiceNotFoundError(Exception):
    """Raised when requested service is not registered"""

    pass


class DIContainer:
    """
    Dependency Injection Container

    Manages service registration, resolution, and lifecycle.
    Breaks circular dependencies by using interfaces and lazy loading.
    """

    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._resolving: set = set()  # Track services currently being resolved
        self._initialized = False
        self._lock = asyncio.Lock()

        logger.info("DIContainer initialized")

    def register_singleton(
        self,
        service_type: Type[T],
        implementation: Type[T] = None,
        factory: Callable[[], T] = None,
        instance: T = None,
    ) -> "DIContainer":
        """Register singleton service"""
        return self._register(service_type, implementation, factory, instance, singleton=True)

    def register_transient(
        self, service_type: Type[T], implementation: Type[T] = None, factory: Callable[[], T] = None
    ) -> "DIContainer":
        """Register transient service (new instance each time)"""
        return self._register(service_type, implementation, factory, None, singleton=False)

    def register_instance(self, service_type: Type[T], instance: T) -> "DIContainer":
        """Register specific instance"""
        return self._register(service_type, None, None, instance, singleton=True)

    def _register(
        self,
        service_type: Type[T],
        implementation: Type[T] = None,
        factory: Callable[[], T] = None,
        instance: T = None,
        singleton: bool = True,
    ) -> "DIContainer":
        """Internal registration method"""

        # Validate registration
        if sum([implementation is not None, factory is not None, instance is not None]) != 1:
            raise ValueError("Exactly one of implementation, factory, or instance must be provided")

        # Extract dependencies from constructor if implementation provided
        dependencies = []
        if implementation:
            dependencies = self._extract_dependencies(implementation)

        registration = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            instance=instance,
            singleton=singleton,
            dependencies=dependencies,
        )

        self._services[service_type] = registration

        logger.debug(
            f"Registered service: {service_type.__name__} -> "
            f"{implementation.__name__ if implementation else 'factory/instance'}"
        )

        return self

    def _extract_dependencies(self, implementation: Type) -> list:
        """Extract constructor dependencies from type annotations"""
        try:
            sig = inspect.signature(implementation.__init__)
            dependencies = []

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)

            return dependencies
        except Exception as e:
            logger.warning(f"Could not extract dependencies for {implementation}: {e}")
            return []

    async def resolve(self, service_type: Type[T]) -> T:
        """Resolve service instance"""
        async with self._lock:
            return await self._resolve_internal(service_type)

    async def _resolve_internal(self, service_type: Type[T]) -> T:
        """Internal resolve method"""
        # Check for circular dependency
        if service_type in self._resolving:
            cycle = " -> ".join([t.__name__ for t in self._resolving] + [service_type.__name__])
            raise CircularDependencyError(f"Circular dependency detected: {cycle}")

        # Check if service is registered
        if service_type not in self._services:
            raise ServiceNotFoundError(f"Service {service_type.__name__} is not registered")

        registration = self._services[service_type]

        # Return existing instance if singleton and already created
        if (
            registration.singleton
            and registration.instance is not None
            and registration.initialized
        ):
            return registration.instance

        # Mark as resolving
        self._resolving.add(service_type)

        try:
            instance = await self._create_instance(registration)

            # Store instance if singleton
            if registration.singleton:
                registration.instance = instance
                registration.initialized = True

            return instance

        finally:
            # Remove from resolving set
            self._resolving.discard(service_type)

    async def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create service instance"""
        if registration.instance is not None:
            return registration.instance

        if registration.factory is not None:
            # Use factory function
            if asyncio.iscoroutinefunction(registration.factory):
                return await registration.factory()
            else:
                return registration.factory()

        if registration.implementation is not None:
            # Create instance with dependency injection
            return await self._create_with_dependencies(
                registration.implementation, registration.dependencies
            )

        raise ValueError(f"Cannot create instance for {registration.service_type}")

    async def _create_with_dependencies(self, implementation: Type, dependencies: list) -> Any:
        """Create instance with resolved dependencies"""
        resolved_deps = []

        # Resolve all dependencies
        for dep_type in dependencies:
            try:
                dep_instance = await self._resolve_internal(dep_type)
                resolved_deps.append(dep_instance)
            except ServiceNotFoundError:
                logger.warning(f"Dependency {dep_type.__name__} not registered, skipping")
                # For optional dependencies, we could provide None or default
                # For now, we'll try to create without this dependency
                pass

        # Create instance
        try:
            if asyncio.iscoroutinefunction(implementation.__init__):
                instance = implementation(*resolved_deps)
                if hasattr(instance, "__aenter__"):  # Async context manager
                    await instance.__aenter__()
                return instance
            else:
                return implementation(*resolved_deps)
        except TypeError as e:
            # Fallback: try without dependencies (some constructors might be flexible)
            logger.warning(f"Failed to inject dependencies into {implementation.__name__}: {e}")
            try:
                return implementation()
            except Exception as fallback_error:
                logger.error(f"Failed to create {implementation.__name__}: {fallback_error}")
                raise

    def is_registered(self, service_type: Type) -> bool:
        """Check if service type is registered"""
        return service_type in self._services

    def get_registration(self, service_type: Type) -> Optional[ServiceRegistration]:
        """Get service registration info"""
        return self._services.get(service_type)

    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all registered services"""
        services = {}
        for service_type, registration in self._services.items():
            services[service_type.__name__] = {
                "service_type": service_type.__name__,
                "implementation": registration.implementation.__name__
                if registration.implementation
                else None,
                "has_factory": registration.factory is not None,
                "has_instance": registration.instance is not None,
                "singleton": registration.singleton,
                "initialized": registration.initialized,
                "dependencies": [dep.__name__ for dep in registration.dependencies],
                "created_at": registration.created_at.isoformat(),
            }
        return services

    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered singleton services"""
        if self._initialized:
            return {}

        results = {}

        # First pass: create all instances
        for service_type, registration in self._services.items():
            if registration.singleton and not registration.initialized:
                try:
                    await self.resolve(service_type)
                    results[service_type.__name__] = True
                    logger.info(f"Initialized service: {service_type.__name__}")
                except Exception as e:
                    results[service_type.__name__] = False
                    logger.error(f"Failed to initialize service {service_type.__name__}: {e}")

        # Second pass: call initialize methods if they exist
        for service_type, registration in self._services.items():
            if (
                registration.singleton
                and registration.instance
                and hasattr(registration.instance, "initialize")
            ):
                try:
                    init_method = getattr(registration.instance, "initialize")
                    if asyncio.iscoroutinefunction(init_method):
                        await init_method()
                    else:
                        init_method()
                    logger.debug(f"Called initialize on {service_type.__name__}")
                except Exception as e:
                    logger.error(f"Failed to call initialize on {service_type.__name__}: {e}")

        self._initialized = True
        return results

    async def shutdown_all(self):
        """Shutdown all services gracefully"""
        logger.info("Shutting down DIContainer")

        # Call shutdown methods in reverse order of creation
        services_by_creation = sorted(
            self._services.items(), key=lambda x: x[1].created_at, reverse=True
        )

        for service_type, registration in services_by_creation:
            if registration.instance and hasattr(registration.instance, "shutdown"):
                try:
                    shutdown_method = getattr(registration.instance, "shutdown")
                    if asyncio.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                    logger.debug(f"Called shutdown on {service_type.__name__}")
                except Exception as e:
                    logger.error(f"Failed to shutdown {service_type.__name__}: {e}")

        # Clear all instances
        for registration in self._services.values():
            registration.instance = None
            registration.initialized = False

        self._initialized = False
        logger.info("DIContainer shutdown complete")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on container and services"""
        health = {
            "container_initialized": self._initialized,
            "total_services": len(self._services),
            "initialized_services": 0,
            "healthy_services": 0,
            "services": {},
        }

        for service_type, registration in self._services.items():
            service_name = service_type.__name__
            service_health = {
                "registered": True,
                "initialized": registration.initialized,
                "has_instance": registration.instance is not None,
                "healthy": False,
            }

            if registration.initialized:
                health["initialized_services"] += 1

            # Check if service has health check method
            if registration.instance and hasattr(registration.instance, "health_check"):
                try:
                    health_method = getattr(registration.instance, "health_check")
                    if asyncio.iscoroutinefunction(health_method):
                        service_health_result = await health_method()
                    else:
                        service_health_result = health_method()

                    service_health["healthy"] = service_health_result.get("healthy", True)
                    service_health["health_details"] = service_health_result

                    if service_health["healthy"]:
                        health["healthy_services"] += 1

                except Exception as e:
                    service_health["health_error"] = str(e)
            else:
                # If no health check method, assume healthy if initialized
                service_health["healthy"] = registration.initialized
                if service_health["healthy"]:
                    health["healthy_services"] += 1

            health["services"][service_name] = service_health

        return health


# Global container instance
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get global DI container instance"""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def reset_container():
    """Reset global container (for testing)"""
    global _container
    _container = None


# Convenience functions
def register_singleton(
    service_type: Type[T], implementation: Type[T] = None, **kwargs
) -> DIContainer:
    """Register singleton service"""
    return get_container().register_singleton(service_type, implementation, **kwargs)


def register_transient(
    service_type: Type[T], implementation: Type[T] = None, **kwargs
) -> DIContainer:
    """Register transient service"""
    return get_container().register_transient(service_type, implementation, **kwargs)


def register_instance(service_type: Type[T], instance: T) -> DIContainer:
    """Register instance"""
    return get_container().register_instance(service_type, instance)


async def resolve(service_type: Type[T]) -> T:
    """Resolve service"""
    return await get_container().resolve(service_type)


async def initialize_container() -> Dict[str, bool]:
    """Initialize all services in container"""
    return await get_container().initialize_all()


async def shutdown_container():
    """Shutdown container"""
    if _container:
        await _container.shutdown_all()
