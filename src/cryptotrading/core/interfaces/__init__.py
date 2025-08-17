"""
Interface Definitions for Dependency Inversion
Breaks circular dependencies by defining abstract interfaces
"""

from .agent_interfaces import (
    IAgent,
    IMemoryAgent,
    IToolExecutor,
    IWorkflowExecutor,
    IContextManager
)

from .database_interfaces import (
    IDatabaseClient,
    IDatabaseManager,
    IConnectionPool
)

from .communication_interfaces import (
    IMessage,
    IMessageHandler,
    ITransport,
    ICommunicationManager,
    IProtocolHandler,
    IServiceDiscovery,
    ILoadBalancer,
    ICircuitBreaker,
    IEventBus,
    MessageType,
    MessagePriority
)

from .security_interfaces import (
    IAuthenticator,
    ISecurityManager,
    IPermissionChecker,
    IRateLimiter,
    IInputValidator,
    ICryptoProvider,
    ISecurityAuditor,
    SecurityLevel
)

from .infrastructure_interfaces import (
    IConfigProvider,
    IServiceRegistry,
    IHealthChecker,
    IMetricsCollector,
    ILogger,
    ICache,
    ITaskScheduler,
    IResourceManager,
    ILockManager,
    IFileStorage,
    ServiceStatus
)

__all__ = [
    # Agent interfaces
    'IAgent',
    'IMemoryAgent', 
    'IToolExecutor',
    'IWorkflowExecutor',
    'IContextManager',
    
    # Database interfaces
    'IDatabaseClient',
    'IDatabaseManager',
    'IConnectionPool',
    
    # Communication interfaces
    'IMessage',
    'IMessageHandler',
    'ITransport',
    'ICommunicationManager',
    'IProtocolHandler',
    'IServiceDiscovery',
    'ILoadBalancer',
    'ICircuitBreaker',
    'IEventBus',
    'MessageType',
    'MessagePriority',
    
    # Security interfaces
    'IAuthenticator',
    'ISecurityManager',
    'IPermissionChecker',
    'IRateLimiter',
    'IInputValidator',
    'ICryptoProvider',
    'ISecurityAuditor',
    'SecurityLevel',
    
    # Infrastructure interfaces
    'IConfigProvider',
    'IServiceRegistry',
    'IHealthChecker',
    'IMetricsCollector',
    'ILogger',
    'ICache',
    'ITaskScheduler',
    'IResourceManager',
    'ILockManager',
    'IFileStorage',
    'ServiceStatus'
]