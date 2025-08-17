"""
MCP Quick Start - One-Command Server Setup
Provides simple, opinionated defaults for immediate MCP server deployment
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import secrets
import tempfile

from .server import MCPServer
from .tools import MCPTool
from .resources import MCPResource
from .enhanced_server import EnhancedMCPServer
from .security.middleware import create_secure_middleware, SecurityConfig
from .transport import StdioTransport, WebSocketTransport, HTTPTransport

logger = logging.getLogger(__name__)


class MCPQuickStart:
    """One-command MCP server setup with sensible defaults"""
    
    def __init__(self, name: str = "quick-mcp-server"):
        self.name = name
        self.config = self._create_default_config()
        self.server: Optional[EnhancedMCPServer] = None
        
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration with secure settings"""
        return {
            "server": {
                "name": self.name,
                "version": "1.0.0",
                "description": "Quick-start MCP server"
            },
            "transport": {
                "type": "stdio",  # Safe default
                "host": "localhost",
                "port": 8080
            },
            "security": {
                "enabled": True,
                "jwt_secret": secrets.token_urlsafe(32),
                "require_auth": False,  # Start permissive, can tighten later
                "rate_limiting": True,
                "max_requests_per_minute": 60
            },
            "features": {
                "tools": True,
                "resources": True,
                "prompts": True,
                "sampling": True,
                "subscriptions": True,
                "progress": True
            },
            "development": {
                "auto_reload": True,
                "debug_logging": True,
                "cors_enabled": True
            }
        }
    
    def with_security(self, enabled: bool = True, require_auth: bool = True) -> 'MCPQuickStart':
        """Configure security settings"""
        self.config["security"]["enabled"] = enabled
        self.config["security"]["require_auth"] = require_auth
        return self
    
    def with_transport(self, transport_type: str, **kwargs) -> 'MCPQuickStart':
        """Configure transport (stdio, websocket, http)"""
        self.config["transport"]["type"] = transport_type
        self.config["transport"].update(kwargs)
        return self
    
    def with_features(self, **features) -> 'MCPQuickStart':
        """Enable/disable features"""
        self.config["features"].update(features)
        return self
    
    def add_simple_tool(
        self, 
        name: str, 
        description: str, 
        func: callable,
        parameters: Optional[Dict[str, Any]] = None
    ) -> 'MCPQuickStart':
        """Add a simple tool with minimal configuration"""
        if "tools" not in self.config:
            self.config["tools"] = []
        
        self.config["tools"].append({
            "name": name,
            "description": description,
            "function": func,
            "parameters": parameters or {}
        })
        return self
    
    def add_file_resource(
        self, 
        name: str, 
        file_path: str, 
        description: str = None
    ) -> 'MCPQuickStart':
        """Add a file resource"""
        if "resources" not in self.config:
            self.config["resources"] = []
        
        self.config["resources"].append({
            "type": "file",
            "name": name,
            "path": file_path,
            "description": description or f"File resource: {name}"
        })
        return self
    
    def add_directory_root(self, path: str, name: str = None) -> 'MCPQuickStart':
        """Add a directory as root for file access"""
        if "roots" not in self.config:
            self.config["roots"] = []
        
        abs_path = os.path.abspath(path)
        self.config["roots"].append({
            "path": abs_path,
            "name": name or os.path.basename(abs_path)
        })
        return self
    
    async def start(self) -> EnhancedMCPServer:
        """Start the MCP server with current configuration"""
        print(f"🚀 Starting MCP Server: {self.name}")
        
        # Create enhanced server
        self.server = EnhancedMCPServer(
            self.config["server"]["name"],
            self.config["server"]["version"]
        )
        
        # Configure security if enabled
        if self.config["security"]["enabled"]:
            print("🔒 Configuring security...")
            security_config = SecurityConfig(
                require_auth=self.config["security"]["require_auth"],
                rate_limiting_enabled=self.config["security"]["rate_limiting"],
                global_rate_limit=self.config["security"]["max_requests_per_minute"]
            )
            
            if self.config["security"]["require_auth"]:
                # Create secure middleware
                jwt_secret = self.config["security"]["jwt_secret"]
                self.server.security_middleware = create_secure_middleware(jwt_secret)
                
                # Generate admin token for immediate use
                admin_token = self.server.security_middleware.create_admin_token("admin")
                print(f"🔑 Admin token: {admin_token}")
                print("   Save this token - you'll need it for authenticated requests!")
        
        # Setup transport
        await self._setup_transport()
        
        # Add tools
        await self._setup_tools()
        
        # Add resources  
        await self._setup_resources()
        
        # Add roots
        self._setup_roots()
        
        # Initialize enhanced features
        await self.server.initialize_enhanced_features(self.config["features"])
        
        print(f"✅ MCP Server ready!")
        print(f"   Transport: {self.config['transport']['type']}")
        print(f"   Security: {'enabled' if self.config['security']['enabled'] else 'disabled'}")
        print(f"   Tools: {len(self.config.get('tools', []))}")
        print(f"   Resources: {len(self.config.get('resources', []))}")
        
        return self.server
    
    async def _setup_transport(self):
        """Setup transport based on configuration"""
        transport_type = self.config["transport"]["type"]
        
        if transport_type == "stdio":
            self.server.transport = StdioTransport()
        elif transport_type == "websocket":
            from .enhanced_transport import create_websocket_transport
            port = self.config["transport"].get("port", 8080)
            uri = f"ws://{self.config['transport'].get('host', 'localhost')}:{port}"
            self.server.transport = create_websocket_transport(uri)
        elif transport_type == "http":
            from .enhanced_transport import create_http_transport
            host = self.config["transport"].get("host", "localhost")
            port = self.config["transport"].get("port", 8080)
            self.server.transport = create_http_transport(host, port)
            print(f"🌐 HTTP server will start on http://{host}:{port}")
    
    async def _setup_tools(self):
        """Setup tools from configuration"""
        for tool_config in self.config.get("tools", []):
            tool = MCPTool(
                name=tool_config["name"],
                description=tool_config["description"],
                parameters=tool_config["parameters"],
                function=tool_config["function"]
            )
            self.server.add_tool(tool)
            print(f"🔧 Added tool: {tool.name}")
    
    async def _setup_resources(self):
        """Setup resources from configuration"""
        for resource_config in self.config.get("resources", []):
            if resource_config["type"] == "file":
                async def read_file(path=resource_config["path"]):
                    with open(path, 'r') as f:
                        return f.read()
                
                resource = MCPResource(
                    uri=f"file://{resource_config['path']}",
                    name=resource_config["name"],
                    description=resource_config["description"],
                    mime_type="text/plain",
                    read_func=read_file
                )
                self.server.add_resource(resource)
                print(f"📄 Added resource: {resource.name}")
    
    def _setup_roots(self):
        """Setup root directories"""
        for root_config in self.config.get("roots", []):
            self.server.add_root_directory(
                root_config["path"],
                root_config["name"]
            )
            print(f"📁 Added root: {root_config['name']} -> {root_config['path']}")
    
    def save_config(self, file_path: str = None):
        """Save current configuration to file"""
        if file_path is None:
            file_path = f"{self.name}-config.json"
        
        # Create serializable config (remove functions)
        save_config = self.config.copy()
        if "tools" in save_config:
            for tool in save_config["tools"]:
                if "function" in tool:
                    tool["function"] = "<function>"  # Can't serialize functions
        
        with open(file_path, 'w') as f:
            json.dump(save_config, f, indent=2)
        
        print(f"💾 Configuration saved to: {file_path}")
        return file_path
    
    @classmethod
    def from_config_file(cls, file_path: str) -> 'MCPQuickStart':
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        instance = cls(config["server"]["name"])
        instance.config = config
        return instance


# Pre-configured server templates
class MCPTemplates:
    """Pre-configured MCP server templates for common use cases"""
    
    @staticmethod
    def development_server(name: str = "dev-mcp") -> MCPQuickStart:
        """Development server with permissive settings"""
        return (MCPQuickStart(name)
                .with_security(enabled=True, require_auth=False)
                .with_transport("http", port=8080)
                .with_features(debug_logging=True, auto_reload=True))
    
    @staticmethod
    def production_server(name: str = "prod-mcp") -> MCPQuickStart:
        """Production server with strict security"""
        return (MCPQuickStart(name)
                .with_security(enabled=True, require_auth=True)
                .with_transport("stdio")  # Most secure
                .with_features(debug_logging=False))
    
    @staticmethod
    def api_server(name: str = "api-mcp", port: int = 8080) -> MCPQuickStart:
        """HTTP API server for web integration"""
        return (MCPQuickStart(name)
                .with_security(enabled=True, require_auth=True)
                .with_transport("http", port=port)
                .with_features(cors_enabled=True))
    
    @staticmethod
    def file_server(name: str = "file-mcp", root_dir: str = ".") -> MCPQuickStart:
        """File server for document/config access"""
        return (MCPQuickStart(name)
                .with_security(enabled=True, require_auth=False)
                .with_transport("http", port=8080)
                .add_directory_root(root_dir, "Files")
                .with_features(subscriptions=True))  # For file watching


# Simple function-based API for one-liners
def create_mcp_server(
    name: str = "quick-mcp",
    tools: List[Dict[str, Any]] = None,
    security: bool = True,
    transport: str = "stdio"
) -> MCPQuickStart:
    """Create MCP server with minimal configuration"""
    server = MCPQuickStart(name)
    
    if transport == "http":
        server.with_transport("http", port=8080)
    elif transport == "websocket":
        server.with_transport("websocket", port=8080)
    
    server.with_security(enabled=security)
    
    for tool_def in tools or []:
        server.add_simple_tool(**tool_def)
    
    return server


async def run_mcp_server(
    name: str = "quick-mcp",
    tools: List[Dict[str, Any]] = None,
    **kwargs
):
    """One-line MCP server startup"""
    server_builder = create_mcp_server(name, tools, **kwargs)
    server = await server_builder.start()
    
    try:
        # Keep server running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n👋 Shutting down MCP server...")
    finally:
        await server.stop()


# Example usage demonstrations
async def demo_simple_server():
    """Demonstrate simple server setup"""
    
    # Simple echo tool
    def echo_tool(message: str) -> str:
        return f"Echo: {message}"
    
    # Simple calculator
    def calc_tool(expression: str) -> str:
        try:
            result = eval(expression)  # Note: eval is dangerous in production
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create and start server
    server = (MCPQuickStart("demo-server")
              .add_simple_tool("echo", "Echo a message", echo_tool, 
                              {"message": {"type": "string"}})
              .add_simple_tool("calc", "Calculate expression", calc_tool,
                              {"expression": {"type": "string"}})
              .with_transport("http", port=8080)
              .with_security(enabled=True, require_auth=False))
    
    return await server.start()


async def demo_file_server():
    """Demonstrate file server setup"""
    
    # Create temp file for demo
    temp_dir = tempfile.mkdtemp()
    demo_file = os.path.join(temp_dir, "demo.txt")
    
    with open(demo_file, 'w') as f:
        f.write("Hello from MCP file server!")
    
    server = (MCPTemplates.file_server("demo-files")
              .add_file_resource("demo", demo_file, "Demo text file")
              .add_directory_root(temp_dir, "Demo Directory"))
    
    return await server.start()


# CLI interface for quick setup
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick MCP Server Setup")
    parser.add_argument("--name", default="quick-mcp", help="Server name")
    parser.add_argument("--transport", choices=["stdio", "http", "websocket"], 
                       default="stdio", help="Transport type")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP/WebSocket")
    parser.add_argument("--secure", action="store_true", help="Enable security")
    parser.add_argument("--template", choices=["dev", "prod", "api", "file"],
                       help="Use predefined template")
    parser.add_argument("--demo", action="store_true", help="Run demo server")
    
    args = parser.parse_args()
    
    async def main():
        if args.demo:
            print("🎮 Starting demo server...")
            server = await demo_simple_server()
            print("Demo server running! Try these tools: echo, calc")
        elif args.template:
            if args.template == "dev":
                server_builder = MCPTemplates.development_server(args.name)
            elif args.template == "prod":
                server_builder = MCPTemplates.production_server(args.name)
            elif args.template == "api":
                server_builder = MCPTemplates.api_server(args.name, args.port)
            elif args.template == "file":
                server_builder = MCPTemplates.file_server(args.name)
            
            server = await server_builder.start()
        else:
            # Custom server
            server_builder = (MCPQuickStart(args.name)
                             .with_transport(args.transport, port=args.port)
                             .with_security(enabled=args.secure))
            server = await server_builder.start()
        
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    
    asyncio.run(main())