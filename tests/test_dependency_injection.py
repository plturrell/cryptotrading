"""
Test Suite for Dependency Injection Container
Demonstrates how DI container breaks circular dependencies and manages component lifecycle
"""
import pytest
import asyncio
from typing import Optional
from abc import ABC, abstractmethod

from src.cryptotrading.core.di_container import (
    DIContainer,
    CircularDependencyError,
    ServiceNotFoundError,
    register_singleton,
    register_transient,
    register_instance,
    resolve,
    get_container,
    reset_container
)


# Test interfaces to demonstrate dependency inversion
class IRepository(ABC):
    @abstractmethod
    async def save(self, data: dict) -> str:
        pass
    
    @abstractmethod
    async def get(self, id: str) -> Optional[dict]:
        pass


class INotificationService(ABC):
    @abstractmethod
    async def send_notification(self, message: str) -> bool:
        pass


class ILogger(ABC):
    @abstractmethod
    def log(self, message: str):
        pass


# Test implementations
class DatabaseRepository(IRepository):
    def __init__(self, logger: ILogger):
        self.logger = logger
        self.data = {}
    
    async def save(self, data: dict) -> str:
        id = f"id_{len(self.data)}"
        self.data[id] = data
        self.logger.log(f"Saved data with id: {id}")
        return id
    
    async def get(self, id: str) -> Optional[dict]:
        self.logger.log(f"Retrieved data for id: {id}")
        return self.data.get(id)


class EmailNotificationService(INotificationService):
    def __init__(self, logger: ILogger):
        self.logger = logger
    
    async def send_notification(self, message: str) -> bool:
        self.logger.log(f"Sending email: {message}")
        return True


class ConsoleLogger(ILogger):
    def __init__(self):
        self.messages = []
    
    def log(self, message: str):
        self.messages.append(message)
        print(f"LOG: {message}")


class UserService:
    def __init__(self, repository: IRepository, notification_service: INotificationService):
        self.repository = repository
        self.notification_service = notification_service
    
    async def create_user(self, user_data: dict) -> str:
        user_id = await self.repository.save(user_data)
        await self.notification_service.send_notification(f"User {user_id} created")
        return user_id
    
    async def get_user(self, user_id: str) -> Optional[dict]:
        return await self.repository.get(user_id)


# Test circular dependency scenario (without DI this would fail)
class ServiceA:
    def __init__(self, service_b: 'ServiceB'):
        self.service_b = service_b
        self.name = "ServiceA"
    
    def do_work(self) -> str:
        return f"{self.name} -> {self.service_b.do_work()}"


class ServiceB:
    def __init__(self, service_c: 'ServiceC'):
        self.service_c = service_c
        self.name = "ServiceB"
    
    def do_work(self) -> str:
        return f"{self.name} -> {self.service_c.do_work()}"


class ServiceC:
    def __init__(self):
        self.name = "ServiceC"
    
    def do_work(self) -> str:
        return f"{self.name} (end)"


class TestDIContainer:
    """Test the DI container functionality"""
    
    @pytest.fixture
    def container(self):
        """Create fresh container for each test"""
        reset_container()
        return DIContainer()
    
    @pytest.mark.asyncio
    async def test_basic_registration_and_resolution(self, container):
        """Test basic service registration and resolution"""
        # Register services
        container.register_singleton(ILogger, ConsoleLogger)
        container.register_singleton(IRepository, DatabaseRepository)
        container.register_singleton(INotificationService, EmailNotificationService)
        container.register_singleton(UserService, UserService)
        
        # Resolve user service (should auto-inject dependencies)
        user_service = await container.resolve(UserService)
        
        assert isinstance(user_service, UserService)
        assert isinstance(user_service.repository, DatabaseRepository)
        assert isinstance(user_service.notification_service, EmailNotificationService)
    
    @pytest.mark.asyncio
    async def test_singleton_behavior(self, container):
        """Test that singletons return same instance"""
        container.register_singleton(ILogger, ConsoleLogger)
        
        # Resolve same service twice
        logger1 = await container.resolve(ILogger)
        logger2 = await container.resolve(ILogger)
        
        # Should be same instance
        assert logger1 is logger2
    
    @pytest.mark.asyncio
    async def test_transient_behavior(self, container):
        """Test that transients return new instances"""
        container.register_transient(ILogger, ConsoleLogger)
        
        # Resolve same service twice
        logger1 = await container.resolve(ILogger)
        logger2 = await container.resolve(ILogger)
        
        # Should be different instances
        assert logger1 is not logger2
        assert isinstance(logger1, ConsoleLogger)
        assert isinstance(logger2, ConsoleLogger)
    
    @pytest.mark.asyncio
    async def test_instance_registration(self, container):
        """Test registering specific instances"""
        logger_instance = ConsoleLogger()
        logger_instance.log("Test message")
        
        container.register_instance(ILogger, logger_instance)
        
        resolved_logger = await container.resolve(ILogger)
        
        # Should be same instance
        assert resolved_logger is logger_instance
        assert len(resolved_logger.messages) == 1
        assert resolved_logger.messages[0] == "Test message"
    
    @pytest.mark.asyncio
    async def test_factory_registration(self, container):
        """Test factory function registration"""
        def create_logger():
            logger = ConsoleLogger()
            logger.log("Created by factory")
            return logger
        
        container.register_singleton(ILogger, factory=create_logger)
        
        logger = await container.resolve(ILogger)
        
        assert isinstance(logger, ConsoleLogger)
        assert len(logger.messages) == 1
        assert logger.messages[0] == "Created by factory"
    
    @pytest.mark.asyncio
    async def test_dependency_injection(self, container):
        """Test automatic dependency injection"""
        container.register_singleton(ILogger, ConsoleLogger)
        container.register_singleton(IRepository, DatabaseRepository)
        
        # Repository should automatically get logger injected
        repository = await container.resolve(IRepository)
        
        assert isinstance(repository, DatabaseRepository)
        assert isinstance(repository.logger, ConsoleLogger)
        
        # Test the injected dependency works
        await repository.save({"name": "test"})
        assert len(repository.logger.messages) > 0
    
    @pytest.mark.asyncio
    async def test_service_not_found_error(self, container):
        """Test error when service not registered"""
        with pytest.raises(ServiceNotFoundError):
            await container.resolve(ILogger)
    
    @pytest.mark.asyncio
    async def test_complex_dependency_chain(self, container):
        """Test complex dependency chains are resolved correctly"""
        container.register_singleton(ILogger, ConsoleLogger)
        container.register_singleton(IRepository, DatabaseRepository)
        container.register_singleton(INotificationService, EmailNotificationService)
        container.register_singleton(UserService, UserService)
        
        # Resolve the top-level service
        user_service = await container.resolve(UserService)
        
        # Test the full dependency chain works
        user_id = await user_service.create_user({"name": "John", "email": "john@example.com"})
        user = await user_service.get_user(user_id)
        
        assert user is not None
        assert user["name"] == "John"
        assert user["email"] == "john@example.com"
    
    @pytest.mark.asyncio
    async def test_circular_dependency_prevention(self, container):
        """Test that circular dependencies are detected and prevented"""
        # This would create A -> B -> C chain (no circle)
        container.register_singleton(ServiceC, ServiceC)
        container.register_singleton(ServiceB, ServiceB)
        container.register_singleton(ServiceA, ServiceA)
        
        # Should resolve successfully
        service_a = await container.resolve(ServiceA)
        result = service_a.do_work()
        
        assert "ServiceA -> ServiceB -> ServiceC (end)" == result
    
    def test_service_listing(self, container):
        """Test listing registered services"""
        container.register_singleton(ILogger, ConsoleLogger)
        container.register_transient(IRepository, DatabaseRepository)
        
        services = container.list_services()
        
        assert len(services) == 2
        assert "ILogger" in services
        assert "IRepository" in services
        
        logger_info = services["ILogger"]
        assert logger_info["singleton"] is True
        assert logger_info["implementation"] == "ConsoleLogger"
        
        repo_info = services["IRepository"]
        assert repo_info["singleton"] is False
        assert repo_info["implementation"] == "DatabaseRepository"
    
    @pytest.mark.asyncio
    async def test_container_initialization(self, container):
        """Test container initialization of all services"""
        container.register_singleton(ILogger, ConsoleLogger)
        container.register_singleton(IRepository, DatabaseRepository)
        
        # Initialize all services
        results = await container.initialize_all()
        
        assert len(results) == 2
        assert all(results.values())  # All should succeed
        
        # Services should now be initialized
        logger_reg = container.get_registration(ILogger)
        repo_reg = container.get_registration(IRepository)
        
        assert logger_reg.initialized is True
        assert repo_reg.initialized is True
        assert logger_reg.instance is not None
        assert repo_reg.instance is not None
    
    @pytest.mark.asyncio
    async def test_container_health_check(self, container):
        """Test container health checking"""
        container.register_singleton(ILogger, ConsoleLogger)
        await container.initialize_all()
        
        health = await container.health_check()
        
        assert health["container_initialized"] is True
        assert health["total_services"] == 1
        assert health["initialized_services"] == 1
        assert health["healthy_services"] == 1
        assert "ILogger" in health["services"]
        assert health["services"]["ILogger"]["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_container_shutdown(self, container):
        """Test graceful container shutdown"""
        container.register_singleton(ILogger, ConsoleLogger)
        await container.initialize_all()
        
        # Verify services are initialized
        logger_reg = container.get_registration(ILogger)
        assert logger_reg.initialized is True
        
        # Shutdown container
        await container.shutdown_all()
        
        # Verify services are cleaned up
        assert logger_reg.initialized is False
        assert logger_reg.instance is None


class TestGlobalContainer:
    """Test global container functions"""
    
    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before each test"""
        reset_container()
        yield
        reset_container()
    
    @pytest.mark.asyncio
    async def test_global_container_functions(self):
        """Test global container convenience functions"""
        # Register services using global functions
        register_singleton(ILogger, ConsoleLogger)
        register_transient(IRepository, DatabaseRepository)
        
        # Resolve using global function
        logger = await resolve(ILogger)
        repository = await resolve(IRepository)
        
        assert isinstance(logger, ConsoleLogger)
        assert isinstance(repository, DatabaseRepository)
        assert isinstance(repository.logger, ConsoleLogger)
    
    @pytest.mark.asyncio
    async def test_global_container_lifecycle(self):
        """Test global container initialization and shutdown"""
        from src.cryptotrading.core.di_container import initialize_container, shutdown_container
        
        register_singleton(ILogger, ConsoleLogger)
        
        # Initialize container
        results = await initialize_container()
        assert results["ILogger"] is True
        
        # Verify service is available
        logger = await resolve(ILogger)
        assert isinstance(logger, ConsoleLogger)
        
        # Shutdown container
        await shutdown_container()


class TestRealWorldScenario:
    """Test real-world dependency injection scenarios"""
    
    @pytest.fixture
    def container(self):
        reset_container()
        return DIContainer()
    
    @pytest.mark.asyncio
    async def test_service_layer_pattern(self, container):
        """Test complete service layer with DI"""
        # Register all dependencies
        container.register_singleton(ILogger, ConsoleLogger)
        container.register_singleton(IRepository, DatabaseRepository)
        container.register_singleton(INotificationService, EmailNotificationService)
        container.register_singleton(UserService, UserService)
        
        # Simulate application startup
        await container.initialize_all()
        
        # Use the service layer
        user_service = await container.resolve(UserService)
        
        # Create multiple users
        user1_id = await user_service.create_user({"name": "Alice", "role": "admin"})
        user2_id = await user_service.create_user({"name": "Bob", "role": "user"})
        
        # Retrieve users
        user1 = await user_service.get_user(user1_id)
        user2 = await user_service.get_user(user2_id)
        
        assert user1["name"] == "Alice"
        assert user2["name"] == "Bob"
        
        # Verify all services worked together
        logger = await container.resolve(ILogger)
        assert len(logger.messages) >= 4  # 2 saves + 2 notifications + 2 gets
        
        # Cleanup
        await container.shutdown_all()
    
    @pytest.mark.asyncio
    async def test_interface_segregation(self, container):
        """Test that interfaces properly segregate dependencies"""
        # Service that only needs logger
        class SimpleService:
            def __init__(self, logger: ILogger):
                self.logger = logger
            
            def do_work(self):
                self.logger.log("Simple work done")
                return "completed"
        
        # Service that needs everything
        class ComplexService:
            def __init__(self, logger: ILogger, repository: IRepository, 
                        notification_service: INotificationService):
                self.logger = logger
                self.repository = repository
                self.notification_service = notification_service
            
            async def do_complex_work(self):
                self.logger.log("Starting complex work")
                data_id = await self.repository.save({"work": "complex"})
                await self.notification_service.send_notification(f"Work {data_id} completed")
                return data_id
        
        # Register services
        container.register_singleton(ILogger, ConsoleLogger)
        container.register_singleton(IRepository, DatabaseRepository)
        container.register_singleton(INotificationService, EmailNotificationService)
        container.register_singleton(SimpleService, SimpleService)
        container.register_singleton(ComplexService, ComplexService)
        
        # Resolve both services
        simple_service = await container.resolve(SimpleService)
        complex_service = await container.resolve(ComplexService)
        
        # Simple service only gets logger
        assert hasattr(simple_service, 'logger')
        assert not hasattr(simple_service, 'repository')
        assert not hasattr(simple_service, 'notification_service')
        
        # Complex service gets all dependencies
        assert hasattr(complex_service, 'logger')
        assert hasattr(complex_service, 'repository')
        assert hasattr(complex_service, 'notification_service')
        
        # Both should share the same logger instance (singleton)
        assert simple_service.logger is complex_service.logger


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])