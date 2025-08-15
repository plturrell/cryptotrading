"""
MCP Plugin System
Lightweight plugin architecture for custom tools and extensions
"""
import importlib
import inspect
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from .tools import MCPTool, ToolResult
from .auth import AuthContext
from .metrics import mcp_metrics

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class MCPPlugin(ABC):
    """Base class for MCP plugins"""
    
    def __init__(self):
        self.metadata = self.get_metadata()
        self.tools: Dict[str, MCPTool] = {}
        self.enabled = False
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the plugin"""
        pass
    
    def register_tool(self, tool: MCPTool):
        """Register a tool with the plugin"""
        self.tools[tool.name] = tool
        logger.debug(f"Plugin {self.metadata.name} registered tool: {tool.name}")
    
    def get_tools(self) -> List[MCPTool]:
        """Get all tools provided by this plugin"""
        return list(self.tools.values())
    
    def validate_dependencies(self) -> bool:
        """Validate plugin dependencies"""
        for dep in self.metadata.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                logger.error(f"Plugin {self.metadata.name} missing dependency: {dep}")
                return False
        return True


class CryptoAnalysisPlugin(MCPPlugin):
    """Example plugin for crypto analysis tools"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="crypto_analysis",
            version="1.0.0",
            description="Advanced crypto analysis tools",
            author="рекс.com",
            dependencies=["pandas", "numpy"]
        )
    
    def initialize(self) -> bool:
        """Initialize crypto analysis plugin"""
        try:
            # Register custom tools
            self.register_tool(self._create_technical_analysis_tool())
            self.register_tool(self._create_correlation_analysis_tool())
            self.register_tool(self._create_volatility_analysis_tool())
            
            self.enabled = True
            logger.info(f"Initialized plugin: {self.metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin {self.metadata.name}: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown plugin"""
        self.tools.clear()
        self.enabled = False
        logger.info(f"Shutdown plugin: {self.metadata.name}")
    
    def _create_technical_analysis_tool(self) -> MCPTool:
        """Create technical analysis tool"""
        return MCPTool(
            name="technical_analysis",
            description="Perform technical analysis on crypto price data",
            parameters={
                "symbol": {
                    "type": "string",
                    "description": "Crypto symbol (e.g., BTC, ETH)"
                },
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe for analysis (1h, 4h, 1d)",
                    "default": "1h"
                },
                "indicators": {
                    "type": "array",
                    "description": "Technical indicators to calculate",
                    "items": {"type": "string"},
                    "default": ["RSI", "MACD", "SMA"]
                }
            },
            function=self._technical_analysis
        )
    
    def _create_correlation_analysis_tool(self) -> MCPTool:
        """Create correlation analysis tool"""
        return MCPTool(
            name="correlation_analysis",
            description="Analyze correlation between crypto assets",
            parameters={
                "symbols": {
                    "type": "array",
                    "description": "List of crypto symbols to analyze",
                    "items": {"type": "string"}
                },
                "period": {
                    "type": "string",
                    "description": "Analysis period (7d, 30d, 90d)",
                    "default": "30d"
                }
            },
            function=self._correlation_analysis
        )
    
    def _create_volatility_analysis_tool(self) -> MCPTool:
        """Create volatility analysis tool"""
        return MCPTool(
            name="volatility_analysis",
            description="Analyze volatility patterns and risk metrics",
            parameters={
                "symbol": {
                    "type": "string",
                    "description": "Crypto symbol to analyze"
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Number of days to look back",
                    "default": 30
                }
            },
            function=self._volatility_analysis
        )
    
    async def _technical_analysis(self, symbol: str, timeframe: str = "1h", 
                                 indicators: List[str] = None) -> Dict[str, Any]:
        """Perform technical analysis - requires real implementation"""
        raise NotImplementedError("Technical analysis requires real market data integration")
    
    async def _correlation_analysis(self, symbols: List[str], 
                                   period: str = "30d") -> Dict[str, Any]:
        """Analyze correlation between assets - requires real implementation"""
        raise NotImplementedError("Correlation analysis requires real market data integration")
    
    async def _volatility_analysis(self, symbol: str, 
                                  lookback_days: int = 30) -> Dict[str, Any]:
        """Analyze volatility patterns - requires real implementation"""
        raise NotImplementedError("Volatility analysis requires real market data integration")


class PluginManager:
    """Manages MCP plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, MCPPlugin] = {}
        self.plugin_tools: Dict[str, str] = {}  # tool_name -> plugin_name
        self.enabled_plugins: set = set()
    
    def register_plugin(self, plugin: MCPPlugin) -> bool:
        """Register a plugin"""
        if plugin.metadata.name in self.plugins:
            logger.warning(f"Plugin {plugin.metadata.name} already registered")
            return False
        
        # Validate dependencies
        if not plugin.validate_dependencies():
            logger.error(f"Plugin {plugin.metadata.name} has missing dependencies")
            return False
        
        self.plugins[plugin.metadata.name] = plugin
        logger.info(f"Registered plugin: {plugin.metadata.name}")
        return True
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} not found")
            return False
        
        plugin = self.plugins[plugin_name]
        
        if plugin_name in self.enabled_plugins:
            logger.warning(f"Plugin {plugin_name} already enabled")
            return True
        
        # Initialize plugin
        if not plugin.initialize():
            logger.error(f"Failed to initialize plugin {plugin_name}")
            return False
        
        # Register plugin tools
        for tool in plugin.get_tools():
            if tool.name in self.plugin_tools:
                logger.error(f"Tool {tool.name} already registered by another plugin")
                plugin.shutdown()
                return False
            
            self.plugin_tools[tool.name] = plugin_name
        
        self.enabled_plugins.add(plugin_name)
        logger.info(f"Enabled plugin: {plugin_name}")
        return True
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        if plugin_name not in self.enabled_plugins:
            logger.warning(f"Plugin {plugin_name} not enabled")
            return True
        
        plugin = self.plugins[plugin_name]
        
        # Remove plugin tools
        tools_to_remove = [
            tool_name for tool_name, pname in self.plugin_tools.items()
            if pname == plugin_name
        ]
        
        for tool_name in tools_to_remove:
            del self.plugin_tools[tool_name]
        
        # Shutdown plugin
        plugin.shutdown()
        self.enabled_plugins.remove(plugin_name)
        
        logger.info(f"Disabled plugin: {plugin_name}")
        return True
    
    def get_plugin_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get tool from plugin"""
        if tool_name not in self.plugin_tools:
            return None
        
        plugin_name = self.plugin_tools[tool_name]
        if plugin_name not in self.enabled_plugins:
            return None
        
        plugin = self.plugins[plugin_name]
        return plugin.tools.get(tool_name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins"""
        return [
            {
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "description": plugin.metadata.description,
                "author": plugin.metadata.author,
                "enabled": plugin.metadata.name in self.enabled_plugins,
                "tools": len(plugin.tools)
            }
            for plugin in self.plugins.values()
        ]
    
    def list_plugin_tools(self) -> List[Dict[str, Any]]:
        """List all tools provided by plugins"""
        tools = []
        
        for tool_name, plugin_name in self.plugin_tools.items():
            if plugin_name in self.enabled_plugins:
                plugin = self.plugins[plugin_name]
                tool = plugin.tools.get(tool_name)
                if tool:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "plugin": plugin_name,
                        "parameters": tool.parameters
                    })
        
        return tools
    
    async def execute_plugin_tool(self, tool_name: str, arguments: Dict[str, Any],
                                 auth_context: AuthContext = None) -> ToolResult:
        """Execute a plugin tool"""
        tool = self.get_plugin_tool(tool_name)
        if not tool:
            return ToolResult.error_result(f"Plugin tool '{tool_name}' not found")
        
        plugin_name = self.plugin_tools[tool_name]
        
        # Record metrics
        start_time = mcp_metrics.tool_execution_start(
            tool_name, 
            auth_context.user_id if auth_context else None
        )
        
        try:
            result = await tool.execute(arguments)
            
            mcp_metrics.tool_execution_end(tool_name, start_time, True)
            mcp_metrics.collector.counter(
                "mcp.plugin.tool_executions",
                tags={"plugin": plugin_name, "tool": tool_name, "status": "success"}
            )
            
            return result
            
        except Exception as e:
            mcp_metrics.tool_execution_end(tool_name, start_time, False)
            mcp_metrics.collector.counter(
                "mcp.plugin.tool_executions",
                tags={"plugin": plugin_name, "tool": tool_name, "status": "error"}
            )
            
            logger.error(f"Plugin tool execution failed: {e}")
            return ToolResult.error_result(f"Tool execution failed: {str(e)}")
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics"""
        return {
            "total_plugins": len(self.plugins),
            "enabled_plugins": len(self.enabled_plugins),
            "total_tools": len(self.plugin_tools),
            "plugins": self.list_plugins()
        }


# Global plugin manager
global_plugin_manager = PluginManager()

# Register default plugins
crypto_analysis_plugin = CryptoAnalysisPlugin()
global_plugin_manager.register_plugin(crypto_analysis_plugin)
global_plugin_manager.enable_plugin("crypto_analysis")


def get_plugin_tool(tool_name: str) -> Optional[MCPTool]:
    """Get plugin tool by name"""
    return global_plugin_manager.get_plugin_tool(tool_name)


async def execute_plugin_tool(tool_name: str, arguments: Dict[str, Any],
                             auth_context: AuthContext = None) -> ToolResult:
    """Execute plugin tool"""
    return await global_plugin_manager.execute_plugin_tool(tool_name, arguments, auth_context)


def list_available_plugins() -> List[Dict[str, Any]]:
    """List all available plugins"""
    return global_plugin_manager.list_plugins()


def list_plugin_tools() -> List[Dict[str, Any]]:
    """List all plugin tools"""
    return global_plugin_manager.list_plugin_tools()


def enable_plugin(plugin_name: str) -> bool:
    """Enable a plugin"""
    return global_plugin_manager.enable_plugin(plugin_name)


def disable_plugin(plugin_name: str) -> bool:
    """Disable a plugin"""
    return global_plugin_manager.disable_plugin(plugin_name)
