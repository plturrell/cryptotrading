"""
Cryptotrading Platform - Advanced Trading and Analytics System

A comprehensive platform for cryptocurrency trading, analysis, and automation
with enterprise-grade code management and AI-powered decision making.

Key Features:
- Multi-agent trading system with MCP protocol support
- Historical data analysis and real-time market monitoring
- Enterprise code management and quality monitoring
- Strands framework integration for advanced agent capabilities
- Production-ready database infrastructure with unified abstraction

Core Modules:
    core: Unified abstractions (storage, monitoring, config) and agent implementations
    data: Data management, historical data, and database operations
    infrastructure: Enterprise infrastructure and monitoring
    
Version: 1.0.0
Author: Cryptotrading Platform Team
"""

__version__ = "1.0.0"
__author__ = "Cryptotrading Platform Team"


# Core exports - lazy imports to avoid circular dependencies
def _lazy_import():
    """Lazy import to avoid circular dependencies during initialization"""
    # Import unified abstractions
    from .core import (
        get_bootstrap,
        get_feature_flags,
        get_monitor,
        get_storage,
        get_sync_storage,
        setup_flask_app,
    )
    from .core.agents import BaseAgent, MemoryAgent
    from .core.agents.modular_strands_agent import ModularStrandsAgent
    from .core.protocols.mcp import MCPServer, MCPTool
    from .infrastructure.database.unified_database import UnifiedDatabase

    return {
        "BaseAgent": BaseAgent,
        "MemoryAgent": MemoryAgent,
        "ModularStrandsAgent": ModularStrandsAgent,
        "MCPServer": MCPServer,
        "MCPTool": MCPTool,
        "UnifiedDatabase": UnifiedDatabase,
        # Unified abstractions
        "get_storage": get_storage,
        "get_sync_storage": get_sync_storage,
        "get_monitor": get_monitor,
        "get_feature_flags": get_feature_flags,
        "setup_flask_app": setup_flask_app,
        "get_bootstrap": get_bootstrap,
    }


# Expose lazy imports via module-level getattr
_lazy_imports = None


def __getattr__(name):
    global _lazy_imports
    if _lazy_imports is None:
        _lazy_imports = _lazy_import()

    if name in _lazy_imports:
        return _lazy_imports[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core agents
    "BaseAgent",
    "MemoryAgent",
    "ModularStrandsAgent",
    # MCP Protocol
    "MCPServer",
    "MCPTool",
    # Database
    "UnifiedDatabase",
]

# Configuration
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
