"""
Strands Agent Components Package
Modular components that replace the monolithic EnhancedStrandsAgent
"""

from .context_manager import ContextManager, StrandsContext

# Import only available components
from .tool_manager import ToolManager, ToolRegistry
from .workflow_engine import WorkflowEngine, WorkflowExecutor

# Import optional components with fallbacks
try:
    from .observer_manager import ObserverManager, StrandsObserver

    observer_available = True
except ImportError:
    observer_available = False
    ObserverManager = None
    StrandsObserver = None

try:
    from .security_manager import AgentSecurityManager

    security_available = True
except ImportError:
    security_available = False
    AgentSecurityManager = None

try:
    from .communication_manager import CommunicationManager

    communication_available = True
except ImportError:
    communication_available = False
    CommunicationManager = None

try:
    from .database_manager import DatabaseManager

    database_available = True
except ImportError:
    database_available = False
    DatabaseManager = None

__all__ = [
    # Core components (always available)
    "ToolManager",
    "ToolRegistry",
    "WorkflowEngine",
    "WorkflowExecutor",
    "ContextManager",
    "StrandsContext",
    # Optional components (may be None)
    "ObserverManager",
    "StrandsObserver",
    "AgentSecurityManager",
    "CommunicationManager",
    "DatabaseManager",
    # Availability flags
    "observer_available",
    "security_available",
    "communication_available",
    "database_available",
]
