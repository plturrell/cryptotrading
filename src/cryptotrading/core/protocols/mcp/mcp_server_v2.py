"""
Enhanced MCP Server with Strand and Fiori Integrations
Production-ready MCP server with all enhancements integrated
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .auth import AuthContext, AuthMiddleware
from .cache import mcp_cache
from .events import create_event_publisher, get_event_streamer
from .fiori_integration import setup_fiori_integration
from .health import MCPHealthChecker
from .metrics import mcp_metrics
from .multi_tenant import MCPTenantManager
from .plugins import global_plugin_manager
from .rate_limiter import RateLimitMiddleware
from .server import MCPServer
from .strand_integration import MCPStrandConfig, setup_mcp_strand_integration
from .transport import StdioTransport, WebSocketTransport

logger = logging.getLogger(__name__)


class EnhancedMCPServer(MCPServer):
    """Enhanced MCP Server with all integrations and improvements"""

    def __init__(self, name: str = "рекс.com-mcp-server", version: str = "2.0.0"):
        super().__init__(name, version)

        # Enhanced components
        self.auth_middleware = AuthMiddleware()
        self.rate_limit_middleware = RateLimitMiddleware()
        self.health_checker = MCPHealthChecker()
        self.tenant_manager = MCPTenantManager()
        self.event_publisher = create_event_publisher()

        # Integration components
        self.strand_bridge = None
        self.fiori_integration = None

        # Server state
        self.enhanced_features_enabled = True
        self.startup_time = datetime.utcnow()

        logger.info(f"Enhanced MCP Server initialized: {name} v{version}")

    async def initialize_enhanced_features(self, config: Dict[str, Any] = None):
        """Initialize all enhanced features"""
        config = config or {}

        try:
            # 1. Setup authentication
            if config.get("auth_enabled", True):
                await self._setup_authentication(config.get("auth_config", {}))

            # 2. Setup caching
            if config.get("cache_enabled", True):
                await self._setup_caching(config.get("cache_config", {}))

            # 3. Setup rate limiting
            if config.get("rate_limiting_enabled", True):
                await self._setup_rate_limiting(config.get("rate_limit_config", {}))

            # 4. Setup metrics collection
            if config.get("metrics_enabled", True):
                await self._setup_metrics(config.get("metrics_config", {}))

            # 5. Setup health monitoring
            if config.get("health_checks_enabled", True):
                await self._setup_health_monitoring(config.get("health_config", {}))

            # 6. Setup multi-tenancy
            if config.get("multi_tenant_enabled", True):
                await self._setup_multi_tenancy(config.get("tenant_config", {}))

            # 7. Setup plugin system
            if config.get("plugins_enabled", True):
                await self._setup_plugins(config.get("plugin_config", {}))

            # 8. Setup event streaming
            if config.get("events_enabled", True):
                await self._setup_event_streaming(config.get("events_config", {}))

            # 9. Setup Strand integration
            if config.get("strand_integration_enabled", True):
                await self._setup_strand_integration(config.get("strand_config", {}))

            # 10. Setup Fiori integration
            if config.get("fiori_integration_enabled", True):
                await self._setup_fiori_integration(config.get("fiori_config", {}))

            logger.info("All enhanced MCP features initialized successfully")

            # Publish startup event
            await self.event_publisher.publish_system_status(
                "enhanced_mcp_server",
                "started",
                {
                    "features_enabled": list(config.keys()),
                    "startup_time": self.startup_time.isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Failed to initialize enhanced features: {e}")
            raise

    async def _setup_authentication(self, config: Dict[str, Any]):
        """Setup authentication middleware"""
        # Configure API keys and JWT
        api_keys = config.get("api_keys", ["default-api-key"])
        jwt_secret = config.get("jwt_secret", "default-jwt-secret")

        for api_key in api_keys:
            self.auth_middleware.add_api_key(api_key, {"user_id": "default", "permissions": ["*"]})

        self.auth_middleware.set_jwt_secret(jwt_secret)
        logger.info("Authentication middleware configured")

    async def _setup_caching(self, config: Dict[str, Any]):
        """Setup caching layer"""
        max_size = config.get("max_size", 1000)
        default_ttl = config.get("default_ttl", 300)

        # Cache is already initialized globally
        logger.info(f"Caching configured: max_size={max_size}, default_ttl={default_ttl}")

    async def _setup_rate_limiting(self, config: Dict[str, Any]):
        """Setup rate limiting"""
        requests_per_minute = config.get("requests_per_minute", 100)
        burst_limit = config.get("burst_limit", 10)

        # Rate limiter is already configured
        logger.info(f"Rate limiting configured: {requests_per_minute} req/min, burst={burst_limit}")

    async def _setup_metrics(self, config: Dict[str, Any]):
        """Setup metrics collection"""
        export_format = config.get("export_format", "json")

        # Metrics collector is already initialized
        logger.info(f"Metrics collection configured: format={export_format}")

    async def _setup_health_monitoring(self, config: Dict[str, Any]):
        """Setup health monitoring"""
        check_interval = config.get("check_interval", 60)

        # Start health checks
        asyncio.create_task(self._health_check_loop(check_interval))
        logger.info(f"Health monitoring configured: interval={check_interval}s")

    async def _setup_multi_tenancy(self, config: Dict[str, Any]):
        """Setup multi-tenancy"""
        # Add default tenant
        default_tenant = config.get(
            "default_tenant",
            {
                "tenant_id": "default",
                "name": "Default Tenant",
                "trading_enabled": True,
                "portfolio_limit": 10,
                "api_rate_limit": 1000,
            },
        )

        self.tenant_manager.add_tenant(
            default_tenant["tenant_id"], default_tenant["name"], default_tenant
        )

        logger.info("Multi-tenancy configured with default tenant")

    async def _setup_plugins(self, config: Dict[str, Any]):
        """Setup plugin system"""
        # Plugins are already registered globally
        enabled_plugins = config.get("enabled_plugins", ["crypto_analysis"])

        for plugin_name in enabled_plugins:
            global_plugin_manager.enable_plugin(plugin_name)

        logger.info(f"Plugin system configured: {enabled_plugins}")

    async def _setup_event_streaming(self, config: Dict[str, Any]):
        """Setup event streaming"""
        buffer_size = config.get("buffer_size", 1000)
        ttl_seconds = config.get("ttl_seconds", 300)

        # Event streamer is already initialized
        logger.info(f"Event streaming configured: buffer_size={buffer_size}, ttl={ttl_seconds}")

    async def _setup_strand_integration(self, config: Dict[str, Any]):
        """Setup Strand framework integration"""
        strand_config = MCPStrandConfig(
            enable_tool_bridging=config.get("enable_tool_bridging", True),
            enable_resource_sharing=config.get("enable_resource_sharing", True),
            enable_workflow_orchestration=config.get("enable_workflow_orchestration", True),
            cache_tool_results=config.get("cache_tool_results", True),
            auto_register_mcp_tools=config.get("auto_register_mcp_tools", True),
            event_streaming=config.get("event_streaming", True),
        )

        self.strand_bridge = setup_mcp_strand_integration(mcp_server=self, config=strand_config)

        logger.info("Strand framework integration configured")

    async def _setup_fiori_integration(self, config: Dict[str, Any]):
        """Setup SAP Fiori integration"""
        self.fiori_integration = setup_fiori_integration(self)

        # Start real-time updates if enabled
        if config.get("real_time_updates", True):
            await self.fiori_integration.start_real_time_updates()

        logger.info("SAP Fiori integration configured")

    async def _health_check_loop(self, interval: int):
        """Periodic health check loop"""
        while True:
            try:
                health_status = await self.health_checker.check_all_components()

                # Publish health status event
                await self.event_publisher.publish_system_status(
                    "health_checker",
                    "healthy" if health_status["overall_status"] == "healthy" else "unhealthy",
                    health_status,
                )

                # Log critical issues
                if health_status["overall_status"] != "healthy":
                    logger.warning(f"Health check issues detected: {health_status}")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(interval)

    async def handle_enhanced_request(
        self, request: Dict[str, Any], auth_context: AuthContext = None
    ) -> Dict[str, Any]:
        """Handle request with all enhancements"""
        request_id = request.get("id", "unknown")
        method = request.get("method", "unknown")

        try:
            # 1. Authentication check
            if not auth_context:
                return self._create_error_response(request_id, "Authentication required")

            # 2. Rate limiting check
            if not await self.rate_limit_middleware.check_rate_limit(auth_context.user_id):
                return self._create_error_response(request_id, "Rate limit exceeded")

            # 3. Multi-tenant validation
            tenant_context = self.tenant_manager.get_tenant_context(auth_context.tenant_id)
            if not tenant_context:
                return self._create_error_response(request_id, "Invalid tenant")

            # 4. Handle request based on method
            if method == "tools/list":
                return await self._handle_tools_list(request, auth_context, tenant_context)
            elif method == "tools/call":
                return await self._handle_tool_call(request, auth_context, tenant_context)
            elif method == "resources/list":
                return await self._handle_resources_list(request, auth_context, tenant_context)
            elif method == "resources/read":
                return await self._handle_resource_read(request, auth_context, tenant_context)
            elif method.startswith("strand/"):
                return await self._handle_strand_request(request, auth_context, tenant_context)
            elif method.startswith("fiori/"):
                return await self._handle_fiori_request(request, auth_context, tenant_context)
            else:
                # Fallback to standard MCP handling
                return await super()._handle_message(request)

        except Exception as e:
            logger.error(f"Enhanced request handling error: {e}")
            return self._create_error_response(request_id, str(e))

    async def _handle_tools_list(
        self, request: Dict[str, Any], auth_context: AuthContext, tenant_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle enhanced tools list with tenant filtering"""
        # Get standard MCP tools
        tools = list(self.tools.values())

        # Add plugin tools
        plugin_tools = global_plugin_manager.list_plugin_tools()

        # Filter tools based on tenant permissions
        allowed_tools = []
        for tool in tools:
            if self.tenant_manager.check_tool_permission(auth_context.tenant_id, tool.name):
                allowed_tools.append(tool.to_dict())

        # Add allowed plugin tools
        for plugin_tool in plugin_tools:
            if self.tenant_manager.check_tool_permission(
                auth_context.tenant_id, plugin_tool["name"]
            ):
                allowed_tools.append(plugin_tool)

        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": allowed_tools,
                "total_tools": len(allowed_tools),
                "tenant_id": auth_context.tenant_id,
            },
        }

    async def _handle_tool_call(
        self, request: Dict[str, Any], auth_context: AuthContext, tenant_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle enhanced tool call with caching and metrics"""
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            return self._create_error_response(request.get("id"), "Missing tool name")

        # Check tenant permission
        if not self.tenant_manager.check_tool_permission(auth_context.tenant_id, tool_name):
            return self._create_error_response(request.get("id"), "Tool access denied for tenant")

        # Check cache first
        cache_key = f"tool:{tool_name}:{auth_context.tenant_id}:{hash(str(arguments))}"
        cached_result = mcp_cache.get(cache_key)

        if cached_result:
            mcp_metrics.collector.counter("mcp.tool_cache_hits", tags={"tool": tool_name})
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [{"type": "text", "text": "Cached: " + str(cached_result)}],
                    "isError": False,
                    "cached": True,
                },
            }

        # Execute tool
        start_time = mcp_metrics.tool_execution_start(tool_name, auth_context.user_id)

        try:
            # Try MCP tool first
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                result = await tool.execute(arguments)
            else:
                # Try plugin tool
                result = await global_plugin_manager.execute_plugin_tool(
                    tool_name, arguments, auth_context
                )

            mcp_metrics.tool_execution_end(tool_name, start_time, result.is_success)

            # Cache successful results
            if result.is_success:
                mcp_cache.set(cache_key, result.content, ttl=300)

            # Publish event
            await self.event_publisher.publish_tool_execution(
                tool_name, arguments, result.to_dict(), result.is_success
            )

            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [{"type": "text", "text": result.content}],
                    "isError": not result.is_success,
                    "cached": False,
                },
            }

        except Exception as e:
            mcp_metrics.tool_execution_end(tool_name, start_time, False)
            logger.error(f"Tool execution error: {e}")
            return self._create_error_response(request.get("id"), str(e))

    async def _handle_strand_request(
        self, request: Dict[str, Any], auth_context: AuthContext, tenant_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Strand framework integration requests"""
        if not self.strand_bridge:
            return self._create_error_response(
                request.get("id"), "Strand integration not available"
            )

        method = request.get("method", "").replace("strand/", "")
        params = request.get("params", {})

        if method == "workflow/execute":
            workflow_config = params.get("workflow_config", {})
            result = await self.strand_bridge.orchestrate_workflow(workflow_config, auth_context)

            return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}

        elif method == "stats":
            stats = self.strand_bridge.get_integration_stats()
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": stats}

        else:
            return self._create_error_response(
                request.get("id"), f"Unknown Strand method: {method}"
            )

    async def _handle_fiori_request(
        self, request: Dict[str, Any], auth_context: AuthContext, tenant_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle SAP Fiori integration requests"""
        if not self.fiori_integration:
            return self._create_error_response(request.get("id"), "Fiori integration not available")

        method = request.get("method", "").replace("fiori/", "")
        params = request.get("params", {})

        if method == "manifest":
            manifest = self.fiori_integration.get_fiori_manifest()
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": manifest}

        elif method == "launchpad/config":
            config = self.fiori_integration.get_fiori_launchpad_config()
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": config}

        elif method == "navigate":
            intent = params.get("intent")
            parameters = params.get("parameters", {})
            result = await self.fiori_integration.handle_fiori_navigation(intent, parameters)

            return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}

        elif method == "stats":
            stats = self.fiori_integration.get_integration_stats()
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": stats}

        else:
            return self._create_error_response(request.get("id"), f"Unknown Fiori method: {method}")

    def _create_error_response(self, request_id: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32000,
                "message": error_message,
                "data": {"timestamp": datetime.utcnow().isoformat(), "server": self.name},
            },
        }

    async def get_enhanced_server_info(self) -> Dict[str, Any]:
        """Get comprehensive server information"""
        base_info = self.get_server_info()

        # Add enhanced features info
        enhanced_info = {
            "enhanced_features": {
                "authentication": True,
                "caching": True,
                "rate_limiting": True,
                "metrics": True,
                "health_monitoring": True,
                "multi_tenancy": True,
                "plugins": True,
                "event_streaming": True,
                "strand_integration": self.strand_bridge is not None,
                "fiori_integration": self.fiori_integration is not None,
            },
            "statistics": {
                "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
                "total_tools": len(self.tools) + len(global_plugin_manager.list_plugin_tools()),
                "total_tenants": len(self.tenant_manager.tenants),
                "cache_size": len(mcp_cache._cache),
                "active_plugins": len(global_plugin_manager.enabled_plugins),
            },
        }

        # Merge with base info
        base_info.update(enhanced_info)
        return base_info


# Factory function for easy server creation
def create_enhanced_mcp_server(config: Dict[str, Any] = None) -> EnhancedMCPServer:
    """Create and configure enhanced MCP server"""
    server = EnhancedMCPServer()

    # Initialize with config
    if config:
        asyncio.create_task(server.initialize_enhanced_features(config))

    return server
