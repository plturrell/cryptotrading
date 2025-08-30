"""
Interface Definitions for Dependency Inversion
Breaks circular dependencies by defining abstract interfaces
"""

from .agent_interfaces import (
    IAgent,
    IContextManager,
    IMemoryAgent,
    IToolExecutor,
    IWorkflowExecutor,
)
from .communication_interfaces import (
    ICircuitBreaker,
    ICommunicationManager,
    IEventBus,
    ILoadBalancer,
    IMessage,
    IMessageHandler,
    IProtocolHandler,
    IServiceDiscovery,
    ITransport,
    MessagePriority,
    MessageType,
)
from .database_interfaces import IConnectionPool, IDatabaseClient, IDatabaseManager
from .infrastructure_interfaces import (
    ICache,
    IConfigProvider,
    IFileStorage,
    IHealthChecker,
    ILockManager,
    ILogger,
    IMetricsCollector,
    IResourceManager,
    IServiceRegistry,
    ITaskScheduler,
    ServiceStatus,
)
from .security_interfaces import (
    IAuthenticator,
    ICryptoProvider,
    IInputValidator,
    IPermissionChecker,
    IRateLimiter,
    ISecurityAuditor,
    ISecurityManager,
    SecurityLevel,
)

__all__ = [
    # Agent interfaces
    "IAgent",
    "IMemoryAgent",
    "IToolExecutor",
    "IWorkflowExecutor",
    "IContextManager",
    # Database interfaces
    "IDatabaseClient",
    "IDatabaseManager",
    "IConnectionPool",
    # Communication interfaces
    "IMessage",
    "IMessageHandler",
    "ITransport",
    "ICommunicationManager",
    "IProtocolHandler",
    "IServiceDiscovery",
    "ILoadBalancer",
    "ICircuitBreaker",
    "IEventBus",
    "MessageType",
    "MessagePriority",
    # Security interfaces
    "IAuthenticator",
    "ISecurityManager",
    "IPermissionChecker",
    "IRateLimiter",
    "IInputValidator",
    "ICryptoProvider",
    "ISecurityAuditor",
    "SecurityLevel",
    # Infrastructure interfaces
    "IConfigProvider",
    "IServiceRegistry",
    "IHealthChecker",
    "IMetricsCollector",
    "ILogger",
    "ICache",
    "ITaskScheduler",
    "IResourceManager",
    "ILockManager",
    "IFileStorage",
    "ServiceStatus",
]
