"""
MCP Capabilities Management
Defines server and client capabilities for MCP protocol
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolCapability:
    """Tool capability definition"""

    listChanged: Optional[bool] = None


@dataclass
class ResourceCapability:
    """Resource capability definition"""

    subscribe: Optional[bool] = None
    listChanged: Optional[bool] = None


@dataclass
class PromptCapability:
    """Prompt capability definition"""

    listChanged: Optional[bool] = None


@dataclass
class LoggingCapability:
    """Logging capability definition"""

    enabled: bool = True


@dataclass
class ExperimentalCapability:
    """Experimental capabilities"""

    pass


@dataclass
class ServerCapabilities:
    """MCP server capabilities"""

    def __init__(self):
        self.experimental: Dict[str, Any] = {}
        self.logging: LoggingCapability = LoggingCapability()
        self.tools: Optional[ToolCapability] = ToolCapability(listChanged=True)
        self.resources: Optional[ResourceCapability] = ResourceCapability(
            subscribe=True, listChanged=True
        )
        self.prompts: Optional[PromptCapability] = PromptCapability(listChanged=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities to dictionary"""
        result = {
            "experimental": self.experimental,
            "logging": asdict(self.logging) if self.logging else {},
        }

        if self.tools:
            result["tools"] = asdict(self.tools)

        if self.resources:
            result["resources"] = asdict(self.resources)

        if self.prompts:
            result["prompts"] = asdict(self.prompts)

        return result

    def enable_tools(self, list_changed: bool = True):
        """Enable tools capability"""
        self.tools = ToolCapability(listChanged=list_changed)

    def enable_resources(self, subscribe: bool = True, list_changed: bool = True):
        """Enable resources capability"""
        self.resources = ResourceCapability(subscribe=subscribe, listChanged=list_changed)

    def enable_prompts(self, list_changed: bool = True):
        """Enable prompts capability"""
        self.prompts = PromptCapability(listChanged=list_changed)

    def enable_logging(self, enabled: bool = True):
        """Enable logging capability"""
        self.logging = LoggingCapability(enabled=enabled)

    def add_experimental(self, name: str, config: Any):
        """Add experimental capability"""
        self.experimental[name] = config


@dataclass
class ClientCapabilities:
    """MCP client capabilities"""

    def __init__(self):
        self.experimental: Dict[str, Any] = {}
        self.sampling: Dict[str, Any] = {}
        self.roots: Dict[str, Any] = {"listChanged": True}

    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities to dictionary"""
        return {"experimental": self.experimental, "sampling": self.sampling, "roots": self.roots}

    def enable_sampling(self):
        """Enable sampling capability"""
        self.sampling = {"enabled": True}

    def enable_roots_changed(self, enabled: bool = True):
        """Enable roots list changed notifications"""
        self.roots = {"listChanged": enabled}

    def add_experimental(self, name: str, config: Any):
        """Add experimental capability"""
        self.experimental[name] = config


class CapabilityNegotiator:
    """Handles capability negotiation between client and server"""

    def __init__(self):
        self.server_capabilities = ServerCapabilities()
        self.client_capabilities = ClientCapabilities()
        self.negotiated_capabilities: Dict[str, Any] = {}

    def negotiate(self, client_caps: Dict[str, Any], server_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Negotiate capabilities between client and server"""
        negotiated = {}

        # Negotiate tools
        if "tools" in server_caps and "tools" in client_caps:
            negotiated["tools"] = self._negotiate_tools(client_caps["tools"], server_caps["tools"])
        elif "tools" in server_caps:
            negotiated["tools"] = server_caps["tools"]

        # Negotiate resources
        if "resources" in server_caps and "resources" in client_caps:
            negotiated["resources"] = self._negotiate_resources(
                client_caps["resources"], server_caps["resources"]
            )
        elif "resources" in server_caps:
            negotiated["resources"] = server_caps["resources"]

        # Negotiate logging
        if "logging" in server_caps:
            negotiated["logging"] = server_caps["logging"]

        # Merge experimental capabilities
        if "experimental" in server_caps or "experimental" in client_caps:
            negotiated["experimental"] = {}
            if "experimental" in server_caps:
                negotiated["experimental"].update(server_caps["experimental"])
            if "experimental" in client_caps:
                # Only include client experimental capabilities that server supports
                for key, value in client_caps["experimental"].items():
                    if key in server_caps.get("experimental", {}):
                        negotiated["experimental"][key] = value

        self.negotiated_capabilities = negotiated
        return negotiated

    def _negotiate_tools(
        self, client_tools: Dict[str, Any], server_tools: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Negotiate tools capabilities"""
        return {"listChanged": server_tools.get("listChanged", False)}

    def _negotiate_resources(
        self, client_resources: Dict[str, Any], server_resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Negotiate resources capabilities"""
        return {
            "subscribe": server_resources.get("subscribe", False),
            "listChanged": server_resources.get("listChanged", False),
        }

    def supports_tool_list_changed(self) -> bool:
        """Check if negotiated capabilities support tool list changed notifications"""
        return self.negotiated_capabilities.get("tools", {}).get("listChanged", False)

    def supports_resource_subscription(self) -> bool:
        """Check if negotiated capabilities support resource subscriptions"""
        return self.negotiated_capabilities.get("resources", {}).get("subscribe", False)

    def supports_logging(self) -> bool:
        """Check if negotiated capabilities support logging"""
        return self.negotiated_capabilities.get("logging", {}).get("enabled", False)
