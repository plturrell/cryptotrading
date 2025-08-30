#!/usr/bin/env python3
"""
Local Development Script for MCP Server
Provides hot reload, easy configuration, and development-friendly features
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cryptotrading.core.protocols.mcp.quick_start import MCPQuickStart, MCPTemplates


class MCPDevServer:
    """Development server with hot reload and easy configuration"""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or "dev-config.json"
        self.config = self._load_or_create_config()
        self.server = None
        self.server_task = None
        self.observer = None

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if self.config.get("debug", True) else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load config or create default development config"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                print(f"📄 Loaded config from {self.config_file}")
                return config
            except Exception as e:
                print(f"⚠️ Error loading config: {e}")

        # Create default development config
        config = {
            "server": {
                "name": "dev-mcp-server",
                "host": "localhost",
                "port": 8080,
                "transport": "http",
            },
            "security": {
                "enabled": True,
                "require_auth": False,  # Permissive for development
                "rate_limiting": False,
            },
            "features": {
                "hot_reload": True,
                "debug": True,
                "cors": True,
                "tools": True,
                "resources": True,
                "prompts": True,
                "sampling": False,  # Requires API keys
            },
            "tools": [
                {"name": "echo", "description": "Echo a message", "enabled": True},
                {
                    "name": "calculator",
                    "description": "Calculate mathematical expressions",
                    "enabled": True,
                },
                {"name": "current_time", "description": "Get current time", "enabled": True},
            ],
            "resources": [
                {
                    "name": "dev-config",
                    "type": "file",
                    "path": self.config_file,
                    "description": "Development configuration",
                    "enabled": True,
                }
            ],
            "watch_dirs": ["src/cryptotrading/core/protocols/mcp", "api"],
        }

        # Save default config
        self._save_config(config)
        print(f"📄 Created default config at {self.config_file}")
        return config

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

    async def start(self):
        """Start development server"""
        print("🚀 Starting MCP Development Server")
        print("=" * 50)

        # Create server from config
        await self._create_server()

        # Setup hot reload if enabled
        if self.config["features"].get("hot_reload", True):
            self._setup_hot_reload()

        # Start server
        await self._start_server()

        print("\n✅ MCP Server running!")
        print(f"🌐 URL: http://{self.config['server']['host']}:{self.config['server']['port']}")
        print(f"🔧 Config: {self.config_file}")
        host = self.config["server"]["host"]
        port = self.config["server"]["port"]
        print(f"📚 Docs: http://{host}:{port}/api/mcp (GET)")

        if not self.config["security"]["require_auth"]:
            print("🔓 Authentication: DISABLED (development mode)")

        print("\n📋 Available endpoints:")
        print("  POST /api/mcp - Main MCP endpoint")
        print("  GET  /api/mcp - Status and documentation")

        print("\n🛠️ Try these tools:")
        for tool in self.config.get("tools", []):
            if tool.get("enabled", True):
                print(f"  • {tool['name']}: {tool['description']}")

        print("\n💡 Tips:")
        print("  • Edit tools in the config file and they'll auto-reload")
        print("  • Check logs below for request details")
        print("  • Press Ctrl+C to stop")
        print("\n" + "=" * 50)

    async def _create_server(self):
        """Create MCP server from configuration"""
        server_config = self.config["server"]

        if server_config.get("template") == "dev":
            self.server = MCPTemplates.development_server(server_config["name"])
        else:
            self.server = (
                MCPQuickStart(server_config["name"])
                .with_transport(
                    server_config["transport"],
                    host=server_config["host"],
                    port=server_config["port"],
                )
                .with_security(
                    enabled=self.config["security"]["enabled"],
                    require_auth=self.config["security"]["require_auth"],
                )
                .with_features(**self.config["features"])
            )

        # Add tools from config
        await self._add_configured_tools()

        # Add resources from config
        await self._add_configured_resources()

    async def _add_configured_tools(self):
        """Add tools from configuration"""
        for tool_config in self.config.get("tools", []):
            if not tool_config.get("enabled", True):
                continue

            tool_name = tool_config["name"]

            # Built-in tools
            if tool_name == "echo":

                def echo(message: str) -> str:
                    return f"Echo: {message}"

                self.server.add_simple_tool(
                    "echo", "Echo a message", echo, {"message": {"type": "string"}}
                )

            elif tool_name == "calculator":

                def calculator(expression: str) -> str:
                    try:
                        # Safe eval
                        allowed = set("0123456789+-*/.() ")
                        if all(c in allowed for c in expression):
                            result = eval(expression, {"__builtins__": {}}, {})
                            return str(result)
                        else:
                            return "Error: Only basic math allowed"
                    except Exception as e:
                        return f"Error: {str(e)}"

                self.server.add_simple_tool(
                    "calculator",
                    "Calculate expressions",
                    calculator,
                    {"expression": {"type": "string"}},
                )

            elif tool_name == "current_time":
                from datetime import datetime

                def current_time() -> str:
                    return datetime.now().isoformat()

                self.server.add_simple_tool("current_time", "Get current time", current_time)

            # Custom tools from file
            elif "file" in tool_config:
                await self._load_tool_from_file(tool_config)

            print(f"🔧 Added tool: {tool_name}")

    async def _add_configured_resources(self):
        """Add resources from configuration"""
        for resource_config in self.config.get("resources", []):
            if not resource_config.get("enabled", True):
                continue

            if resource_config["type"] == "file":
                self.server.add_file_resource(
                    resource_config["name"],
                    resource_config["path"],
                    resource_config.get("description", ""),
                )
                print(f"📄 Added resource: {resource_config['name']}")

    async def _load_tool_from_file(self, tool_config: Dict[str, Any]):
        """Load custom tool from Python file"""
        # This would implement dynamic tool loading
        # For now, just log the intent
        self.logger.info(f"Custom tool loading not yet implemented: {tool_config}")

    async def _start_server(self):
        """Start the MCP server"""
        self.server_instance = await self.server.start()

        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down development server...")
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            await self.server_instance.stop()

    def _setup_hot_reload(self):
        """Setup file watching for hot reload"""

        class ReloadHandler(FileSystemEventHandler):
            def __init__(self, dev_server):
                self.dev_server = dev_server
                self.last_reload = 0

            def on_modified(self, event):
                if event.is_directory:
                    return

                # Debounce rapid changes
                now = time.time()
                if now - self.last_reload < 1:
                    return
                self.last_reload = now

                # Check if it's a Python file
                if event.src_path.endswith(".py"):
                    print(f"\n🔄 File changed: {event.src_path}")
                    print("⏳ Hot reload triggered...")
                    asyncio.create_task(self.dev_server._hot_reload())

        self.observer = Observer()
        handler = ReloadHandler(self)

        for watch_dir in self.config.get("watch_dirs", []):
            if os.path.exists(watch_dir):
                self.observer.schedule(handler, watch_dir, recursive=True)
                print(f"👁️ Watching: {watch_dir}")

        self.observer.start()

    async def _hot_reload(self):
        """Perform hot reload of the server"""
        try:
            print("🔄 Performing hot reload...")

            # Stop current server instance
            if hasattr(self, "server_instance") and self.server_instance:
                await self.server_instance.stop()
                print("⏹️ Stopped current server")

            # Clear module cache for watched directories
            self._clear_module_cache()

            # Recreate server with updated code
            await self._create_server()
            print("🔧 Recreated server with updated code")

            # Restart server
            self.server_instance = await self.server.start()
            print("✅ Hot reload complete!")

        except Exception as e:
            print(f"❌ Hot reload failed: {e}")
            self.logger.error("Hot reload error", exc_info=True)

    def _clear_module_cache(self):
        """Clear Python module cache for watched directories"""
        import sys

        modules_to_remove = []
        for module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "__file__") and module.__file__:
                for watch_dir in self.config.get("watch_dirs", []):
                    if watch_dir in module.__file__:
                        modules_to_remove.append(module_name)
                        break

        for module_name in modules_to_remove:
            del sys.modules[module_name]
            print(f"🗑️ Cleared module cache: {module_name}")


class MCPDevCLI:
    """Command line interface for development server"""

    def __init__(self):
        self.parser = self._setup_parser()

    def _setup_parser(self):
        parser = argparse.ArgumentParser(description="MCP Development Server")

        subparsers = parser.add_subparsers(dest="command", help="Commands")

        # Start command
        start_parser = subparsers.add_parser("start", help="Start development server")
        start_parser.add_argument("--config", default="dev-config.json", help="Config file")
        start_parser.add_argument("--port", type=int, help="Override port")
        start_parser.add_argument("--host", default="localhost", help="Override host")
        start_parser.add_argument("--no-auth", action="store_true", help="Disable authentication")
        start_parser.add_argument("--no-reload", action="store_true", help="Disable hot reload")

        # Init command
        init_parser = subparsers.add_parser("init", help="Initialize new MCP project")
        init_parser.add_argument("name", help="Project name")
        init_parser.add_argument(
            "--template", choices=["basic", "crypto", "files"], default="basic"
        )

        # Test command
        test_parser = subparsers.add_parser("test", help="Test MCP server")
        test_parser.add_argument("--url", default="http://localhost:8080/api/mcp")

        # Deploy command
        deploy_parser = subparsers.add_parser("deploy", help="Deploy to Vercel")
        deploy_parser.add_argument("--env", choices=["preview", "production"], default="preview")

        return parser

    async def run(self, args=None):
        """Run CLI with arguments"""
        parsed = self.parser.parse_args(args)

        if not parsed.command:
            self.parser.print_help()
            return

        if parsed.command == "start":
            await self._start_command(parsed)
        elif parsed.command == "init":
            await self._init_command(parsed)
        elif parsed.command == "test":
            await self._test_command(parsed)
        elif parsed.command == "deploy":
            await self._deploy_command(parsed)

    async def _start_command(self, args):
        """Start development server"""
        dev_server = MCPDevServer(args.config)

        # Apply CLI overrides
        if args.port:
            dev_server.config["server"]["port"] = args.port
        if args.host:
            dev_server.config["server"]["host"] = args.host
        if args.no_auth:
            dev_server.config["security"]["require_auth"] = False
        if args.no_reload:
            dev_server.config["features"]["hot_reload"] = False

        await dev_server.start()

    async def _init_command(self, args):
        """Initialize new MCP project"""
        project_name = args.name
        template = args.template

        print(f"🚀 Initializing MCP project: {project_name}")

        # Create project directory
        os.makedirs(project_name, exist_ok=True)
        os.chdir(project_name)

        # Create basic structure
        os.makedirs("tools", exist_ok=True)
        os.makedirs("resources", exist_ok=True)

        # Create config based on template
        if template == "crypto":
            config = self._create_crypto_config(project_name)
        elif template == "files":
            config = self._create_file_config(project_name)
        else:
            config = self._create_basic_config(project_name)

        with open("dev-config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Create example tool
        self._create_example_tool(template)

        # Create README
        self._create_readme(project_name, template)

        print(f"✅ Project {project_name} created!")
        print(f"📁 Directory: {os.getcwd()}")
        print("🚀 Start with: python -m scripts.dev start")

    def _create_basic_config(self, name: str) -> Dict[str, Any]:
        return {
            "server": {"name": name, "host": "localhost", "port": 8080, "transport": "http"},
            "security": {"enabled": True, "require_auth": False},
            "features": {"hot_reload": True, "debug": True, "tools": True},
            "tools": [{"name": "hello", "file": "tools/hello.py", "enabled": True}],
            "resources": [],
            "watch_dirs": ["tools", "resources"],
        }

    def _create_crypto_config(self, name: str) -> Dict[str, Any]:
        config = self._create_basic_config(name)
        config["tools"].extend(
            [
                {"name": "portfolio", "file": "tools/portfolio.py", "enabled": True},
                {"name": "market_data", "file": "tools/market.py", "enabled": True},
            ]
        )
        return config

    def _create_file_config(self, name: str) -> Dict[str, Any]:
        config = self._create_basic_config(name)
        config["resources"] = [
            {"name": "docs", "type": "directory", "path": "./docs", "enabled": True}
        ]
        config["features"]["subscriptions"] = True
        return config

    def _create_example_tool(self, template: str):
        if template == "crypto":
            tool_code = '''
def get_portfolio() -> dict:
    """Get portfolio summary"""
    return {
        "total_value": 10000.0,
        "positions": [
            {"symbol": "BTC", "amount": 0.5},
            {"symbol": "ETH", "amount": 3.0}
        ]
    }
'''
            with open("tools/portfolio.py", "w") as f:
                f.write(tool_code)

        # Always create hello tool
        hello_code = '''
def hello(name: str = "World") -> str:
    """Say hello to someone"""
    return f"Hello, {name}!"
'''
        with open("tools/hello.py", "w") as f:
            f.write(hello_code)

    def _create_readme(self, name: str, template: str):
        readme = f"""# {name}

MCP Server project created with template: {template}

## Quick Start

```bash
# Start development server
python -m scripts.dev start

# Test the server
curl http://localhost:8080/api/mcp

# List tools
curl -X POST http://localhost:8080/api/mcp \\
  -H "Content-Type: application/json" \\
  -d '{{"jsonrpc":"2.0","method":"tools/list","id":"1"}}'
```

## Configuration

Edit `dev-config.json` to modify server settings, add tools, and configure resources.

## Tools

Add custom tools in the `tools/` directory. Each tool should export functions
that match the MCP tool interface.

## Resources

Add resources in the `resources/` directory or configure file/API resources in the config.
"""
        with open("README.md", "w") as f:
            f.write(readme)

    async def _test_command(self, args):
        """Test MCP server"""
        import aiohttp

        url = args.url
        print(f"🧪 Testing MCP server at {url}")

        async with aiohttp.ClientSession() as session:
            # Test status
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Status: {data.get('status', 'unknown')}")
                    else:
                        print(f"❌ Status check failed: {response.status}")
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                return

            # Test tools list
            try:
                payload = {"jsonrpc": "2.0", "method": "tools/list", "id": "test"}
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        tools = data.get("result", {}).get("tools", [])
                        print(f"✅ Found {len(tools)} tools:")
                        for tool in tools:
                            print(f"  • {tool['name']}: {tool['description']}")
                    else:
                        print(f"❌ Tools list failed: {response.status}")
            except Exception as e:
                print(f"❌ Tools test failed: {e}")

    async def _deploy_command(self, args):
        """Deploy to Vercel"""
        env = args.env
        print(f"🚀 Deploying to Vercel ({env})...")

        # Check if vercel CLI is installed
        try:
            result = subprocess.run(["vercel", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Vercel CLI not found. Install with: npm i -g vercel")
                return
        except FileNotFoundError:
            print("❌ Vercel CLI not found. Install with: npm i -g vercel")
            return

        # Deploy
        cmd = ["vercel"]
        if env == "production":
            cmd.append("--prod")

        try:
            result = subprocess.run(cmd, cwd=".", text=True)
            if result.returncode == 0:
                print("✅ Deployment successful!")
            else:
                print("❌ Deployment failed")
        except Exception as e:
            print(f"❌ Deployment error: {e}")


async def main():
    """Main entry point"""
    cli = MCPDevCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
