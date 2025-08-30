"""
Dependency Injection Configuration for Modular Strands Agent
Registers all components and their dependencies in the DI container
"""
import logging
from typing import Dict, Any, Optional
from ..di_container import get_container, DIContainer
from ..interfaces import (
    ISecurityManager, ICommunicationManager, ILogger,
    IMetricsCollector, IHealthChecker
)

logger = logging.getLogger(__name__)

class StrandsAgentDIConfiguration:
    """DI configuration for modular strands agent components"""
    
    def __init__(self, container: Optional[DIContainer] = None):
        self.container = container or get_container()
        self._configured = False
    
    def configure_core_components(self) -> 'StrandsAgentDIConfiguration':
        """Configure core agent components"""
        from .components import ToolManager, WorkflowEngine, ContextManager
        from .secure_code_sandbox import SecureCodeExecutor
        
        # Register core components as singletons
        self.container.register_singleton(
            ToolManager,
            implementation=ToolManager
        )
        
        self.container.register_singleton(
            WorkflowEngine,
            implementation=WorkflowEngine
        )
        
        self.container.register_singleton(
            ContextManager,
            implementation=ContextManager
        )
        
        # Register secure code executor factory
        self.container.register_singleton(
            SecureCodeExecutor,
            factory=lambda: SecureCodeExecutor()
        )
        
        logger.info("Core components registered in DI container")
        return self
    
    def configure_interface_implementations(self) -> 'StrandsAgentDIConfiguration':
        """Configure interface implementations"""
        # Register default implementations for interfaces
        
        # Security Manager - use default implementation
        try:
            from .implementations.default_security_manager import DefaultSecurityManager
            self.container.register_singleton(
                ISecurityManager,
                implementation=DefaultSecurityManager
            )
        except ImportError:
            logger.warning("DefaultSecurityManager not available")
        
        # Communication Manager - use default implementation
        try:
            from .implementations.default_communication_manager import DefaultCommunicationManager
            self.container.register_singleton(
                ICommunicationManager,
                implementation=DefaultCommunicationManager
            )
        except ImportError:
            logger.warning("DefaultCommunicationManager not available")
        
        # Logger - use structured logger implementation
        try:
            from .implementations.structured_logger import StructuredLogger
            self.container.register_singleton(
                ILogger,
                implementation=StructuredLogger
            )
        except ImportError:
            logger.warning("StructuredLogger not available")
        
        # Metrics Collector - use default implementation
        try:
            from .implementations.default_metrics_collector import DefaultMetricsCollector
            self.container.register_singleton(
                IMetricsCollector,
                implementation=DefaultMetricsCollector
            )
        except ImportError:
            logger.warning("DefaultMetricsCollector not available")
        
        # Health Checker - use default implementation
        try:
            from .implementations.default_health_checker import DefaultHealthChecker
            self.container.register_singleton(
                IHealthChecker,
                implementation=DefaultHealthChecker
            )
        except ImportError:
            logger.warning("DefaultHealthChecker not available")
        
        logger.info("Interface implementations registered in DI container")
        return self
    
    def configure_external_services(self) -> 'StrandsAgentDIConfiguration':
        """Configure external service integrations"""
        # Database services
        try:
            from ...infrastructure.database.unified_database import UnifiedDatabase
            self.container.register_singleton(
                UnifiedDatabase,
                factory=lambda: UnifiedDatabase()
            )
        except ImportError:
            logger.warning("UnifiedDatabase not available")
        
        # Market data services
        try:
            from ...infrastructure.data.market_data_service import MarketDataService
            self.container.register_singleton(
                MarketDataService,
                implementation=MarketDataService
            )
        except ImportError:
            logger.warning("MarketDataService not available")
        
        # DEX services
        try:
            from ...infrastructure.defi.dex_service import DEXService
            self.container.register_singleton(
                DEXService,
                implementation=DEXService
            )
        except ImportError:
            logger.warning("DEXService not available")
        
        # Web3 services
        try:
            from ...infrastructure.blockchain.web3_service import Web3Service
            self.container.register_singleton(
                Web3Service,
                implementation=Web3Service
            )
        except ImportError:
            logger.warning("Web3Service not available")
        
        # API rate monitoring
        try:
            from ...infrastructure.monitoring.api_rate_monitor import APIRateMonitor
            self.container.register_singleton(
                APIRateMonitor,
                implementation=APIRateMonitor
            )
        except ImportError:
            logger.warning("APIRateMonitor not available")
        
        logger.info("External services registered in DI container")
        return self
    
    def configure_mcp_integration(self) -> 'StrandsAgentDIConfiguration':
        """Configure MCP (Model Context Protocol) integration"""
        try:
            from ...infrastructure.analysis.all_segregated_mcp_tools import MCPTools
            self.container.register_singleton(
                MCPTools,
                implementation=MCPTools
            )
        except ImportError:
            logger.warning("MCPTools not available")
        
        # MCP agent segregation
        try:
            from ...infrastructure.analysis.mcp_agent_segregation import MCPAgentSegregationManager
            self.container.register_singleton(
                MCPAgentSegregationManager,
                implementation=MCPAgentSegregationManager
            )
        except ImportError:
            logger.warning("MCPAgentSegregationManager not available")
        
        logger.info("MCP integration registered in DI container")
        return self
    
    def configure_all(self) -> 'StrandsAgentDIConfiguration':
        """Configure all components"""
        if self._configured:
            return self
        
        self.configure_core_components()
        self.configure_interface_implementations()
        self.configure_external_services()
        self.configure_mcp_integration()
        
        self._configured = True
        logger.info("All DI components configured")
        return self
    
    async def initialize_container(self) -> Dict[str, bool]:
        """Initialize all registered services"""
        if not self._configured:
            self.configure_all()
        
        logger.info("Initializing DI container services")
        results = await self.container.initialize_all()
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        logger.info(f"DI container initialization complete: {success_count}/{total_count} services initialized")
        return results
    
    def get_container(self) -> DIContainer:
        """Get the configured container"""
        return self.container

# Global configuration instance
_di_config: Optional[StrandsAgentDIConfiguration] = None

def get_strands_di_config() -> StrandsAgentDIConfiguration:
    """Get global DI configuration instance"""
    global _di_config
    if _di_config is None:
        _di_config = StrandsAgentDIConfiguration()
    return _di_config

async def initialize_strands_di() -> Dict[str, bool]:
    """Initialize strands agent DI configuration"""
    config = get_strands_di_config()
    return await config.configure_all().initialize_container()

def reset_strands_di():
    """Reset DI configuration (for testing)"""
    global _di_config
    _di_config = None
