"""
MCP Server Lifecycle Manager
Provides production-ready lifecycle management for MCP servers
"""

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

from ....infrastructure.database.connection_pool import close_all_pools, get_connection_pool
from ...config.production_config import MCPConfig, get_config
from .server import MCPServer
from .tools import MCPTool

logger = logging.getLogger(__name__)


class ServerPhase(str, Enum):
    """Server lifecycle phases"""

    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class HealthStatus:
    """Server health status"""

    healthy: bool
    phase: ServerPhase
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    active_connections: int
    errors_last_hour: int
    last_error: Optional[str] = None
    checks_passed: Dict[str, bool] = field(default_factory=dict)


class LifecycleHook:
    """Lifecycle hook for extensibility"""

    def __init__(self, name: str):
        self.name = name

    async def on_initialize(self, server: MCPServer):
        """Called during server initialization"""
        pass

    async def on_start(self, server: MCPServer):
        """Called when server starts"""
        pass

    async def on_stop(self, server: MCPServer):
        """Called when server stops"""
        pass

    async def on_error(self, server: MCPServer, error: Exception):
        """Called on server error"""
        pass

    async def on_health_check(self, server: MCPServer) -> bool:
        """Called during health checks, return True if healthy"""
        return True


class DependencyContainer:
    """Enhanced dependency injection container with lifecycle support"""

    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initializers: Dict[str, Callable] = {}
        self._destructors: Dict[str, Callable] = {}
        self._initialized: Set[str] = set()

    def register_singleton(
        self,
        name: str,
        instance: Any,
        initializer: Optional[Callable] = None,
        destructor: Optional[Callable] = None,
    ):
        """Register a singleton instance"""
        self._instances[name] = instance
        if initializer:
            self._initializers[name] = initializer
        if destructor:
            self._destructors[name] = destructor

    def register_factory(
        self,
        name: str,
        factory: Callable,
        initializer: Optional[Callable] = None,
        destructor: Optional[Callable] = None,
    ):
        """Register a factory function"""
        self._factories[name] = factory
        if initializer:
            self._initializers[name] = initializer
        if destructor:
            self._destructors[name] = destructor

    async def get(self, name: str) -> Any:
        """Get a dependency, creating and initializing if needed"""
        # Return existing instance
        if name in self._instances and name in self._initialized:
            return self._instances[name]

        # Create from factory if needed
        if name not in self._instances and name in self._factories:
            self._instances[name] = await self._factories[name]()

        # Initialize if needed
        if name in self._instances and name not in self._initialized:
            if name in self._initializers:
                await self._initializers[name](self._instances[name])
            self._initialized.add(name)

        if name in self._instances:
            return self._instances[name]

        raise KeyError(f"Dependency '{name}' not found")

    async def initialize_all(self):
        """Initialize all registered dependencies"""
        for name in list(self._instances.keys()):
            await self.get(name)

    async def cleanup(self):
        """Clean up all dependencies in reverse order"""
        for name in reversed(list(self._initialized)):
            if name in self._destructors:
                try:
                    await self._destructors[name](self._instances[name])
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}")

        self._instances.clear()
        self._initialized.clear()


class MCPLifecycleManager:
    """
    Production-ready lifecycle manager for MCP servers

    Features:
    - Graceful startup and shutdown
    - Health monitoring
    - Dependency injection
    - Resource cleanup
    - Signal handling
    - Error recovery
    - Extensible hooks
    """

    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or get_config().mcp
        self.container = DependencyContainer()
        self.phase = ServerPhase.CREATED
        self.start_time: Optional[datetime] = None
        self.server: Optional[MCPServer] = None
        self.hooks: List[LifecycleHook] = []
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._process = psutil.Process()
        self._error_count = 0
        self._last_error: Optional[str] = None

    def add_hook(self, hook: LifecycleHook):
        """Add a lifecycle hook"""
        self.hooks.append(hook)
        logger.info(f"Added lifecycle hook: {hook.name}")

    async def _run_hooks(self, method_name: str, *args):
        """Run all hooks for a given method"""
        for hook in self.hooks:
            try:
                method = getattr(hook, method_name)
                await method(*args)
            except Exception as e:
                logger.error(f"Hook {hook.name}.{method_name} failed: {e}")

    async def initialize(self):
        """Initialize server and dependencies"""
        try:
            self.phase = ServerPhase.INITIALIZING
            logger.info("Initializing MCP lifecycle manager...")

            # Register core dependencies
            await self._register_dependencies()

            # Initialize all dependencies
            await self.container.initialize_all()

            # Create MCP server
            self.server = MCPServer(
                name=self.config.server_name or "cryptotrading-mcp",
                version=self.config.server_version or "1.0.0",
            )

            # Register tools from container
            await self._register_tools()

            # Run initialization hooks
            await self._run_hooks("on_initialize", self.server)

            # Setup signal handlers
            self._setup_signals()

            self.phase = ServerPhase.INITIALIZED
            logger.info("MCP lifecycle manager initialized")

        except Exception as e:
            self.phase = ServerPhase.ERROR
            self._last_error = str(e)
            logger.error(f"Initialization failed: {e}")
            raise

    async def _register_dependencies(self):
        """Register core dependencies"""
        # Database connection pool
        if hasattr(self.config, "database_url") and self.config.database_url:
            self.container.register_factory(
                "db_pool",
                lambda: get_connection_pool(self.config.database_url),
                destructor=lambda pool: pool.close(),
            )

        # Tool providers
        from .tools import CryptoTradingTools

        self.container.register_singleton("crypto_tools", CryptoTradingTools())

        # Add custom dependencies here

    async def _register_tools(self):
        """Register tools with the server"""
        # Get crypto tools
        crypto_tools = await self.container.get("crypto_tools")

        # Register actual working historical data tools
        tools = [
            crypto_tools.get_yahoo_finance_data_tool(),
            crypto_tools.get_fred_economic_data_tool(),
            crypto_tools.get_crypto_relevant_indicators_tool(),
            crypto_tools.get_comprehensive_trading_dataset_tool(),
        ]

        for tool in tools:
            self.server.add_tool(tool)

        logger.info(f"Registered {len(tools)} historical data tools with MCP server")

    def _setup_signals(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Handle Windows signals if on Windows
        if os.name == "nt":
            signal.signal(signal.SIGBREAK, signal_handler)

    async def start(self):
        """Start the MCP server with monitoring"""
        if self.phase != ServerPhase.INITIALIZED:
            raise RuntimeError(f"Cannot start from phase {self.phase}")

        try:
            self.phase = ServerPhase.STARTING
            self.start_time = datetime.utcnow()

            # Run start hooks
            await self._run_hooks("on_start", self.server)

            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())

            # Start server
            self.phase = ServerPhase.RUNNING
            logger.info("Starting MCP server...")

            # Run server (this blocks)
            server_task = asyncio.create_task(self.server.start())

            # Wait for shutdown signal
            await self._shutdown_event.wait()

            # Cancel server task
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            self.phase = ServerPhase.ERROR
            self._last_error = str(e)
            self._error_count += 1
            await self._run_hooks("on_error", self.server, e)
            logger.error(f"Server start failed: {e}")
            raise

    async def stop(self):
        """Stop the server gracefully"""
        if self.phase not in [ServerPhase.RUNNING, ServerPhase.ERROR]:
            return

        try:
            self.phase = ServerPhase.STOPPING
            logger.info("Stopping MCP server...")

            # Cancel monitoring tasks
            for task in [self._health_check_task, self._monitor_task]:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Run stop hooks
            await self._run_hooks("on_stop", self.server)

            # Stop server
            if self.server:
                await self.server.stop()

            # Cleanup dependencies
            await self.container.cleanup()

            # Close database pools
            await close_all_pools()

            self.phase = ServerPhase.STOPPED
            self._shutdown_event.set()
            logger.info("MCP server stopped")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.phase = ServerPhase.ERROR

    async def restart(self):
        """Restart the server"""
        logger.info("Restarting MCP server...")
        await self.stop()
        await asyncio.sleep(1)  # Brief pause
        await self.initialize()
        await self.start()

    async def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

        # Run health check hooks
        checks_passed = {}
        for hook in self.hooks:
            try:
                checks_passed[hook.name] = await hook.on_health_check(self.server)
            except Exception as e:
                checks_passed[hook.name] = False
                logger.error(f"Health check {hook.name} failed: {e}")

        # Get system metrics
        memory_info = self._process.memory_info()

        return HealthStatus(
            healthy=self.phase == ServerPhase.RUNNING and all(checks_passed.values()),
            phase=self.phase,
            uptime_seconds=uptime,
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_percent=self._process.cpu_percent(interval=0.1),
            active_connections=len(self.server.sessions) if self.server else 0,
            errors_last_hour=self._error_count,
            last_error=self._last_error,
            checks_passed=checks_passed,
        )

    async def _health_check_loop(self):
        """Periodic health checks"""
        while self.phase == ServerPhase.RUNNING:
            try:
                await asyncio.sleep(self.config.health_check_interval or 30)

                health = await self.get_health_status()

                if not health.healthy:
                    logger.warning(f"Health check failed: {health}")

                    # Attempt recovery if configured
                    if self.config.auto_recovery and self._error_count < 3:
                        logger.info("Attempting auto-recovery...")
                        await self.restart()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _monitor_loop(self):
        """Resource monitoring loop"""
        while self.phase == ServerPhase.RUNNING:
            try:
                await asyncio.sleep(60)  # Every minute

                # Log metrics
                health = await self.get_health_status()
                logger.info(
                    f"Server metrics - Phase: {health.phase}, "
                    f"Memory: {health.memory_usage_mb:.1f}MB, "
                    f"CPU: {health.cpu_percent:.1f}%, "
                    f"Connections: {health.active_connections}"
                )

                # Check resource limits
                if health.memory_usage_mb > (self.config.max_memory_mb or 1024):
                    logger.warning(f"Memory usage high: {health.memory_usage_mb}MB")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")


@asynccontextmanager
async def managed_mcp_server(config: Optional[MCPConfig] = None):
    """Context manager for MCP server with automatic lifecycle management"""
    manager = MCPLifecycleManager(config)

    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.stop()


# Example lifecycle hooks
class LoggingHook(LifecycleHook):
    """Example hook that logs lifecycle events"""

    def __init__(self):
        super().__init__("logging")

    async def on_initialize(self, server: MCPServer):
        logger.info(f"LoggingHook: Server {server.name} initializing")

    async def on_start(self, server: MCPServer):
        logger.info(f"LoggingHook: Server {server.name} starting")

    async def on_stop(self, server: MCPServer):
        logger.info(f"LoggingHook: Server {server.name} stopping")

    async def on_error(self, server: MCPServer, error: Exception):
        logger.error(f"LoggingHook: Server error - {error}")

    async def on_health_check(self, server: MCPServer) -> bool:
        return server.is_initialized


class DatabaseHealthHook(LifecycleHook):
    """Hook that checks database health"""

    def __init__(self, container: DependencyContainer):
        super().__init__("database_health")
        self.container = container

    async def on_health_check(self, server: MCPServer) -> bool:
        try:
            pool = await self.container.get("db_pool")
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False
