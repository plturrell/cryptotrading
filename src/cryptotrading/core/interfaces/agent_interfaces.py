"""
Agent Interface Definitions
Abstract interfaces for agent components to prevent circular dependencies
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class IAgent(ABC):
    """Base agent interface"""

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Get agent ID"""
        pass

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Get agent type"""
        pass

    @abstractmethod
    async def initialize(self, **kwargs) -> bool:
        """Initialize the agent"""
        pass

    @abstractmethod
    async def shutdown(self):
        """Shutdown the agent"""
        pass


class IMemoryAgent(IAgent):
    """Memory-enabled agent interface"""

    @abstractmethod
    async def store_memory(self, key: str, value: Any, **kwargs) -> bool:
        """Store data in memory"""
        pass

    @abstractmethod
    async def retrieve_memory(self, key: str, **kwargs) -> Any:
        """Retrieve data from memory"""
        pass

    @abstractmethod
    async def clear_memory(self, pattern: str = None) -> bool:
        """Clear memory entries"""
        pass


class IToolExecutor(ABC):
    """Tool execution interface"""

    @abstractmethod
    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Execute a tool"""
        pass

    @abstractmethod
    async def register_tool(self, name: str, handler, **kwargs) -> bool:
        """Register a new tool"""
        pass

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        pass


class IWorkflowExecutor(ABC):
    """Workflow execution interface"""

    @abstractmethod
    async def execute_workflow(
        self, workflow_id: str, initial_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a workflow"""
        pass

    @abstractmethod
    async def register_workflow(self, workflow_definition: Any) -> bool:
        """Register a new workflow"""
        pass

    @abstractmethod
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List available workflows"""
        pass


class IContextManager(ABC):
    """Context management interface"""

    @abstractmethod
    async def create_context(self, session_id: str, **kwargs) -> Any:
        """Create a new context"""
        pass

    @abstractmethod
    async def get_context(self, session_id: str) -> Optional[Any]:
        """Get existing context"""
        pass

    @abstractmethod
    async def update_context(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update context"""
        pass

    @abstractmethod
    async def delete_context(self, session_id: str) -> bool:
        """Delete context"""
        pass


class IObserver(ABC):
    """Observer interface for monitoring"""

    @abstractmethod
    async def on_event(self, event_type: str, event_data: Dict[str, Any]):
        """Handle an event"""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get observer metrics"""
        pass


class IComponentLifecycle(ABC):
    """Component lifecycle interface"""

    @abstractmethod
    async def start(self):
        """Start the component"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the component"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        pass
