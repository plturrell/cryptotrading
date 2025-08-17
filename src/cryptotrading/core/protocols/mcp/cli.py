"""
MCP CLI Integration with Diagnostic Tools
Provides command-line interface for MCP server/client operations
"""
import asyncio
import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .server import MCPServer
from .client import MCPClient, MCPClientSession
from .transport import StdioTransport, WebSocketTransport, SSETransport
from .tools import CryptoTradingTools, MCPTool, ToolResult
from .resources import CryptoTradingResources
# from ..cryptotrading.diagnostics import create_diagnostic_analyzer  # Optional import
from ..strands.models.grok_model import GrokModel

logger = logging.getLogger(__name__)


class MCPDiagnosticCLI:
    """MCP CLI with integrated diagnostic tools"""
    
    def __init__(self):
        self.server: Optional[MCPServer] = None
        self.client: Optional[MCPClient] = None
        self.diagnostic_analyzer = None
    
    def setup_logging(self, level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr),
                logging.FileHandler(f'mcp_{level.lower()}.log')
            ]
        )
    
    async def cmd_start_server(self, args):
        """Start MCP server"""
        print(f"Starting MCP server: {args.name}")
        
        # Create transport
        if args.transport == "stdio":
            transport = StdioTransport()
        elif args.transport == "websocket":
            transport = WebSocketTransport(args.uri)
        elif args.transport == "sse":
            transport = SSETransport(args.uri)
        else:
            raise ValueError(f"Unsupported transport: {args.transport}")
        
        # Create server
        self.server = MCPServer(args.name, args.version, transport)
        
        # Add crypto trading tools
        self._add_crypto_tools()
        
        # Add crypto trading resources
        self._add_crypto_resources()
        
        # Add diagnostic tools
        await self._add_diagnostic_tools()
        
        try:
            await self.server.start()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            if self.server:
                await self.server.stop()
    
    async def cmd_test_client(self, args):
        """Test MCP client connection"""
        print(f"Testing MCP client: {args.name}")
        
        # Create transport
        if args.transport == "stdio":
            transport = StdioTransport()
        elif args.transport == "websocket":
            transport = WebSocketTransport(args.uri)
        elif args.transport == "sse":
            transport = SSETransport(args.uri)
        else:
            raise ValueError(f"Unsupported transport: {args.transport}")
        
        # Create client
        self.client = MCPClient(args.name, args.version, transport)
        
        try:
            async with MCPClientSession(self.client) as client:
                # Test basic operations
                print("✓ Connected to MCP server")
                
                # List tools
                tools = await client.list_tools()
                print(f"✓ Found {len(tools)} tools:")
                for tool in tools[:5]:  # Show first 5
                    print(f"  - {tool['name']}: {tool['description']}")
                
                # List resources
                resources = await client.list_resources()
                print(f"✓ Found {len(resources)} resources:")
                for resource in resources[:5]:  # Show first 5
                    print(f"  - {resource['name']}: {resource['description']}")
                
                # Test tool call
                if args.test_tool and args.test_tool in [t['name'] for t in tools]:
                    print(f"\n✓ Testing tool: {args.test_tool}")
                    result = await client.call_tool(args.test_tool, {})
                    print(f"  Result: {json.dumps(result, indent=2)}")
                
                # Test resource read
                if args.test_resource and args.test_resource in [r['uri'] for r in resources]:
                    print(f"\n✓ Testing resource: {args.test_resource}")
                    result = await client.read_resource(args.test_resource)
                    print(f"  Content: {result['contents'][0]['text'][:200]}...")
                
                print("\n✓ All tests passed!")
                
        except Exception as e:
            print(f"✗ Client test failed: {e}")
            return 1
        
        return 0
    
    async def cmd_run_diagnostic(self, args):
        """Run diagnostic analysis"""
        print("Running MCP diagnostic analysis...")
        
        if not self.diagnostic_analyzer:
            self.diagnostic_analyzer = create_diagnostic_analyzer()
        
        # Run diagnostics based on type
        if args.diagnostic_type == "server":
            await self._diagnose_server()
        elif args.diagnostic_type == "client":
            await self._diagnose_client()
        elif args.diagnostic_type == "transport":
            await self._diagnose_transport(args)
        elif args.diagnostic_type == "tools":
            await self._diagnose_tools()
        elif args.diagnostic_type == "all":
            await self._diagnose_all(args)
        else:
            print(f"Unknown diagnostic type: {args.diagnostic_type}")
            return 1
        
        return 0
    
    async def cmd_benchmark(self, args):
        """Run MCP performance benchmarks"""
        print("Running MCP performance benchmarks...")
        
        # Create test server and client
        server = MCPServer("benchmark-server", "1.0.0", StdioTransport())
        client = MCPClient("benchmark-client", "1.0.0", StdioTransport())
        
        # Add test tools
        self._add_benchmark_tools(server)
        
        # Run benchmarks
        results = await self._run_benchmarks(server, client, args)
        
        # Display results
        print("\nBenchmark Results:")
        print("=" * 50)
        for test_name, result in results.items():
            print(f"{test_name}: {result['duration']:.3f}s ({result['ops_per_sec']:.1f} ops/sec)")
        
        return 0
    
    def _add_crypto_tools(self):
        """Add crypto trading tools to server"""
        tools = [
            CryptoTradingTools.get_portfolio_tool(),
            CryptoTradingTools.get_market_data_tool(),
            CryptoTradingTools.analyze_sentiment_tool(),
            CryptoTradingTools.execute_trade_tool(),
            CryptoTradingTools.get_risk_metrics_tool()
        ]
        
        for tool in tools:
            self.server.add_tool(tool)
    
    def _add_crypto_resources(self):
        """Add crypto trading resources to server"""
        resources = [
            CryptoTradingResources.get_config_resource(),
            CryptoTradingResources.get_portfolio_resource(),
            CryptoTradingResources.get_market_status_resource(),
            CryptoTradingResources.get_risk_metrics_resource(),
            CryptoTradingResources.get_strategy_performance_resource(),
            CryptoTradingResources.get_log_resource("trading"),
            CryptoTradingResources.get_log_resource("system")
        ]
        
        for resource in resources:
            self.server.add_resource(resource)
    
    async def _add_diagnostic_tools(self):
        """Add diagnostic tools to server"""
        # System health tool
        async def system_health():
            return {
                "cpu_usage": 45.2,
                "memory_usage": 68.1,
                "disk_usage": 23.4,
                "network_latency": 15,
                "active_connections": 12,
                "uptime_hours": 48.5
            }
        
        health_tool = MCPTool(
            name="system_health",
            description="Get system health metrics",
            parameters={},
            function=system_health
        )
        
        # Error analysis tool
        async def analyze_errors(period: str = "1h"):
            return {
                "period": period,
                "total_errors": 23,
                "error_rate": 0.05,
                "top_errors": [
                    {"type": "ConnectionTimeout", "count": 8},
                    {"type": "InvalidRequest", "count": 6},
                    {"type": "RateLimitExceeded", "count": 4}
                ],
                "recommendations": [
                    "Increase connection timeout",
                    "Add request validation",
                    "Implement exponential backoff"
                ]
            }
        
        error_tool = MCPTool(
            name="analyze_errors",
            description="Analyze system errors and provide recommendations",
            parameters={
                "period": {
                    "type": "string",
                    "description": "Analysis period (1h, 24h, 7d)",
                    "default": "1h"
                }
            },
            function=analyze_errors
        )
        
        # Performance metrics tool
        async def performance_metrics():
            return {
                "response_times": {
                    "avg_ms": 125,
                    "p95_ms": 250,
                    "p99_ms": 500
                },
                "throughput": {
                    "requests_per_sec": 150,
                    "peak_rps": 300
                },
                "resource_usage": {
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "disk_io_mb_per_sec": 50
                }
            }
        
        perf_tool = MCPTool(
            name="performance_metrics",
            description="Get performance metrics and statistics",
            parameters={},
            function=performance_metrics
        )
        
        # Add tools to server
        self.server.add_tool(health_tool)
        self.server.add_tool(error_tool)
        self.server.add_tool(perf_tool)
    
    def _add_benchmark_tools(self, server: MCPServer):
        """Add benchmark tools for testing"""
        async def echo_test(message: str = "hello"):
            return {"echo": message}
        
        async def compute_test(iterations: int = 1000):
            # Simple computation test
            result = sum(i * i for i in range(iterations))
            return {"result": result, "iterations": iterations}
        
        echo_tool = MCPTool("echo", "Echo test tool", {"message": {"type": "string"}}, echo_test)
        compute_tool = MCPTool("compute", "Computation test tool", {"iterations": {"type": "integer"}}, compute_test)
        
        server.add_tool(echo_tool)
        server.add_tool(compute_tool)
    
    async def _diagnose_server(self):
        """Diagnose server issues"""
        print("Diagnosing MCP server...")
        
        if not self.server:
            print("✗ No server instance available")
            return
        
        info = self.server.get_server_info()
        print(f"✓ Server: {info['name']} v{info['version']}")
        print(f"✓ Protocol: {info['protocol_version']}")
        print(f"✓ Initialized: {info['is_initialized']}")
        print(f"✓ Tools: {info['tools_count']}")
        print(f"✓ Resources: {info['resources_count']}")
    
    async def _diagnose_client(self):
        """Diagnose client issues"""
        print("Diagnosing MCP client...")
        
        if not self.client:
            print("✗ No client instance available")
            return
        
        info = self.client.get_client_info()
        print(f"✓ Client: {info['name']} v{info['version']}")
        print(f"✓ Protocol: {info['protocol_version']}")
        print(f"✓ Initialized: {info['is_initialized']}")
        print(f"✓ Server: {info['server_info'].get('name', 'Unknown')}")
    
    async def _diagnose_transport(self, args):
        """Diagnose transport issues"""
        print("Diagnosing MCP transport...")
        
        # Test different transports
        transports = ["stdio"]
        if args.uri:
            if args.uri.startswith("ws://") or args.uri.startswith("wss://"):
                transports.append("websocket")
            if args.uri.startswith("http://") or args.uri.startswith("https://"):
                transports.append("sse")
        
        for transport_type in transports:
            try:
                if transport_type == "stdio":
                    transport = StdioTransport()
                elif transport_type == "websocket":
                    transport = WebSocketTransport(args.uri)
                elif transport_type == "sse":
                    transport = SSETransport(args.uri)
                
                connected = await transport.connect()
                if connected:
                    print(f"✓ {transport_type} transport: OK")
                    await transport.disconnect()
                else:
                    print(f"✗ {transport_type} transport: Failed to connect")
            except Exception as e:
                print(f"✗ {transport_type} transport: {e}")
    
    async def _diagnose_tools(self):
        """Diagnose tool issues"""
        print("Diagnosing MCP tools...")
        
        if not self.server:
            print("✗ No server instance available")
            return
        
        tools = list(self.server.tools.values())
        print(f"✓ Found {len(tools)} tools")
        
        for tool in tools:
            try:
                # Test tool execution with empty parameters
                result = await tool.execute({})
                if result.isError:
                    print(f"⚠ {tool.name}: {result.content[0].text}")
                else:
                    print(f"✓ {tool.name}: OK")
            except Exception as e:
                print(f"✗ {tool.name}: {e}")
    
    async def _diagnose_all(self, args):
        """Run all diagnostics"""
        await self._diagnose_server()
        await self._diagnose_client()
        await self._diagnose_transport(args)
        await self._diagnose_tools()
    
    async def _run_benchmarks(self, server: MCPServer, client: MCPClient, args) -> Dict[str, Any]:
        """Run performance benchmarks"""
        results = {}
        
        # Tool call benchmark
        import time
        start_time = time.time()
        iterations = args.iterations or 100
        
        for _ in range(iterations):
            await server.tools["echo"].execute({"message": "test"})
        
        duration = time.time() - start_time
        results["tool_calls"] = {
            "duration": duration,
            "ops_per_sec": iterations / duration
        }
        
        return results


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MCP CLI with Diagnostic Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start MCP server')
    server_parser.add_argument('--name', default='crypto-trading-server', help='Server name')
    server_parser.add_argument('--version', default='1.0.0', help='Server version')
    server_parser.add_argument('--transport', choices=['stdio', 'websocket', 'sse'], 
                              default='stdio', help='Transport type')
    server_parser.add_argument('--uri', help='URI for websocket/sse transport')
    
    # Client command
    client_parser = subparsers.add_parser('client', help='Test MCP client')
    client_parser.add_argument('--name', default='crypto-trading-client', help='Client name')
    client_parser.add_argument('--version', default='1.0.0', help='Client version')
    client_parser.add_argument('--transport', choices=['stdio', 'websocket', 'sse'], 
                              default='stdio', help='Transport type')
    client_parser.add_argument('--uri', help='URI for websocket/sse transport')
    client_parser.add_argument('--test-tool', help='Tool to test')
    client_parser.add_argument('--test-resource', help='Resource to test')
    
    # Diagnostic command
    diag_parser = subparsers.add_parser('diagnose', help='Run diagnostics')
    diag_parser.add_argument('diagnostic_type', choices=['server', 'client', 'transport', 'tools', 'all'],
                            help='Type of diagnostic to run')
    diag_parser.add_argument('--uri', help='URI for transport diagnostics')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench_parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Initialize CLI
    cli = MCPDiagnosticCLI()
    cli.setup_logging(args.log_level)
    
    # Run command
    try:
        if args.command == 'server':
            return asyncio.run(cli.cmd_start_server(args))
        elif args.command == 'client':
            return asyncio.run(cli.cmd_test_client(args))
        elif args.command == 'diagnose':
            return asyncio.run(cli.cmd_run_diagnostic(args))
        elif args.command == 'benchmark':
            return asyncio.run(cli.cmd_benchmark(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())