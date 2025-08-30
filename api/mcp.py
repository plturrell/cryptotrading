"""
Vercel Serverless Function for MCP Server
Handles MCP requests in Vercel's serverless environment
"""

import json
import os
import asyncio
from typing import Dict, Any
from datetime import datetime

# Import MCP components
from cryptotrading.core.protocols.mcp.quick_start import MCPQuickStart
from cryptotrading.core.protocols.mcp.security.middleware import VercelSecurityMiddleware


class VercelMCPServer:
    """MCP Server optimized for Vercel serverless functions"""

    def __init__(self):
        self.server = None
        self._initialized = False

    async def initialize(self):
        """Initialize server (cached between requests)"""
        if self._initialized and self.server:
            return self.server

        # Create MCP server with Vercel-optimized config
        self.server = (
            MCPQuickStart("vercel-mcp-server")
            .with_transport("http")
            .with_security(
                enabled=os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true",
                require_auth=os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true",
            )
            .with_features(
                tools=True,
                resources=True,
                prompts=True,
                sampling=True,
                subscriptions=False,  # Disabled for serverless
                progress=False,  # Disabled for serverless
            )
        )

        # Add built-in tools
        await self._add_builtin_tools()

        # Add custom tools from environment
        await self._add_custom_tools()

        # Initialize server
        await self.server.start()

        self._initialized = True
        return self.server

    async def _add_builtin_tools(self):
        """Add built-in utility tools"""

        # Echo tool
        def echo(message: str) -> str:
            return f"Echo: {message}"

        # Current time tool
        def current_time() -> str:
            return datetime.utcnow().isoformat() + "Z"

        # Environment info tool
        def env_info() -> dict:
            return {
                "platform": "vercel",
                "python_version": os.sys.version,
                "environment": os.getenv("VERCEL_ENV", "unknown"),
                "region": os.getenv("VERCEL_REGION", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Calculator tool (safe eval)
        def calculator(expression: str) -> str:
            try:
                # Safe mathematical evaluation
                allowed_chars = set("0123456789+-*/.() ")
                if not all(c in allowed_chars for c in expression):
                    return "Error: Only basic mathematical operations allowed"

                # Simple eval with limited scope
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"

        # Add tools to server
        self.server.add_simple_tool("echo", "Echo a message", echo, {"message": {"type": "string"}})
        self.server.add_simple_tool("current_time", "Get current UTC time", current_time)
        self.server.add_simple_tool("env_info", "Get environment information", env_info)
        self.server.add_simple_tool(
            "calculator",
            "Calculate mathematical expressions",
            calculator,
            {"expression": {"type": "string"}},
        )

    async def _add_custom_tools(self):
        """Add custom tools from specialized agents"""
        try:
            # Import all specialized agent MCP tools
            from cryptotrading.infrastructure.mcp.technical_analysis_mcp_tools import (
                technical_analysis_mcp_tools,
            )
            from cryptotrading.infrastructure.mcp.mcts_calculation_mcp_tools import (
                mcts_calculation_mcp_tools,
            )
            from cryptotrading.infrastructure.mcp.historical_data_mcp_tools import (
                historical_data_mcp_tools,
            )
            from cryptotrading.infrastructure.mcp.database_mcp_tools import database_mcp_tools
            from cryptotrading.infrastructure.mcp.agent_manager_mcp_tools import (
                agent_manager_mcp_tools,
            )
            from cryptotrading.infrastructure.mcp.glean_agent_mcp_tools import glean_agent_mcp_tools
            from cryptotrading.infrastructure.mcp.ml_agent_mcp_tools import ml_agent_mcp_tools
            from src.cryptotrading.infrastructure.mcp.feature_store_mcp_tools import (
                FeatureStoreMCPTools,
            )
            from src.cryptotrading.infrastructure.mcp.data_analysis_mcp_tools import (
                DataAnalysisMCPTools,
            )
            from src.cryptotrading.infrastructure.mcp.clrs_algorithms_mcp_tools import (
                CLRSAlgorithmsMCPTools,
            )
            from src.cryptotrading.infrastructure.mcp.technical_analysis_skills_mcp_tools import (
                TechnicalAnalysisSkillsMCPTools,
            )
            from src.cryptotrading.infrastructure.mcp.ml_models_mcp_tools import MLModelsMCPTools
            from src.cryptotrading.infrastructure.mcp.code_quality_mcp_tools import (
                CodeQualityMCPTools,
            )
            from src.cryptotrading.infrastructure.mcp.mcts_calculation_mcp_tools import (
                MCTSCalculationMCPTools,
            )

            # Import CLRS Tree MCP Tools
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import (
                CLRSMCPTools,
                TreeMCPTools,
                GleanAnalysisMCPTools,
                GLEAN_MCP_TOOLS,
            )

            # Initialize all new MCP tools
            feature_store = FeatureStore()
            data_ingestion = DataIngestion()

            # Initialize CLRS and Tree MCP tools
            glean_client = GleanClient()
            clrs_mcp_tools = CLRSMCPTools(glean_client)
            tree_mcp_tools = TreeMCPTools(glean_client)
            enhanced_glean_tools = GleanAnalysisMCPTools(glean_client)

            # Create all MCP tool instances
            new_mcp_tools = [
                ("feature_store", FeatureStoreMCPTools(feature_store)),
                ("data_analysis", DataAnalysisMCPTools(data_ingestion)),
                ("clrs_algorithms", CLRSAlgorithmsMCPTools()),
                ("technical_analysis_skills", TechnicalAnalysisSkillsMCPTools()),
                ("ml_models", MLModelsMCPTools()),
                ("code_quality", CodeQualityMCPTools()),
                ("mcts_calculation", MCTSCalculationMCPTools()),
                ("clrs_analysis", clrs_mcp_tools),
                ("tree_analysis", tree_mcp_tools),
                ("enhanced_glean", enhanced_glean_tools),
            ]

            # Register all new MCP tools
            for tool_name, tool_instance in new_mcp_tools:
                if hasattr(tool_instance, "register_tools"):
                    tool_instance.register_tools(self.server)

            # Register CLRS Tree MCP Tools directly
            await self._register_clrs_mcp_tools(
                clrs_mcp_tools, tree_mcp_tools, enhanced_glean_tools
            )

            # Register Glean MCP tools from the imported registry
            for tool_name, tool_config in GLEAN_MCP_TOOLS.items():
                self.server.add_tool(
                    name=f"glean_{tool_name}",
                    description=tool_config["metadata"]["description"],
                    handler=tool_config["function"],
                    input_schema=tool_config["metadata"].get("inputSchema", {}),
                )

            # Initialize A2A agent registry integration
            from src.cryptotrading.core.protocols.a2a.agent_registry_integration import (
                initialize_mcp_a2a_integration,
                start_mcp_agents,
            )

            # Initialize and register MCP agents with A2A protocol
            await initialize_mcp_a2a_integration()
            await start_mcp_agents()

            # Register all agent MCP tools
            agent_tools = [
                ("technical_analysis", technical_analysis_mcp_tools),
                ("mcts_calculation", mcts_calculation_mcp_tools),
                ("historical_data", historical_data_mcp_tools),
                ("database", database_mcp_tools),
                ("agent_manager", agent_manager_mcp_tools),
                ("glean_agent", glean_agent_mcp_tools),
                ("ml_agent", ml_agent_mcp_tools),
                ("feature_store", new_mcp_tools[0][1]),
                ("data_analysis", new_mcp_tools[1][1]),
                ("clrs_algorithms", new_mcp_tools[2][1]),
                ("technical_analysis_skills", new_mcp_tools[3][1]),
                ("ml_models", new_mcp_tools[4][1]),
                ("code_quality", new_mcp_tools[5][1]),
                ("mcts_calculation", new_mcp_tools[6][1]),
            ]

            for agent_name, tool_instance in agent_tools:
                if hasattr(tool_instance, "tools") and hasattr(tool_instance, "handle_tool_call"):
                    for tool_def in tool_instance.tools:
                        tool_name = f"{agent_name}_{tool_def['name']}"

                        # Create wrapper function for the tool
                        async def tool_wrapper(
                            *args, tool_instance=tool_instance, tool_name=tool_def["name"], **kwargs
                        ):
                            # Convert args/kwargs to arguments dict
                            if args and isinstance(args[0], dict):
                                arguments = args[0]
                            else:
                                arguments = kwargs

                            return await tool_instance.handle_tool_call(tool_name, arguments)

                        # Register tool with MCP server
                        self.server.add_tool(
                            name=tool_name,
                            description=f"[{agent_name.upper()}] {tool_def['description']}",
                            handler=tool_wrapper,
                            input_schema=tool_def.get("inputSchema", {}),
                        )

            print(f"Successfully registered MCP tools from {len(agent_tools)} specialized agents")

        except ImportError as e:
            print(f"Warning: Could not import some agent MCP tools: {e}")

        # Initialize MCP Agent Lifecycle Manager
        await self._initialize_agent_lifecycle_manager()

    async def _register_clrs_mcp_tools(self, clrs_tools, tree_tools, enhanced_tools):
        """Register CLRS tree MCP tools with the server"""

        # CLRS Analysis Tools
        self.server.add_tool(
            name="clrs_dependency_analysis",
            description="Analyze code dependencies using CLRS graph algorithms",
            handler=clrs_tools.analyze_dependency_graph,
            input_schema={
                "type": "object",
                "properties": {
                    "modules": {"type": "object", "description": "Module dependency mapping"}
                },
                "required": ["modules"],
            },
        )

        self.server.add_tool(
            name="clrs_code_similarity",
            description="Analyze code similarity using CLRS string algorithms",
            handler=clrs_tools.analyze_code_similarity,
            input_schema={
                "type": "object",
                "properties": {
                    "code1": {"type": "string", "description": "First code snippet"},
                    "code2": {"type": "string", "description": "Second code snippet"},
                },
                "required": ["code1", "code2"],
            },
        )

        self.server.add_tool(
            name="clrs_sort_symbols",
            description="Sort code symbols using CLRS sorting algorithms",
            handler=clrs_tools.sort_code_symbols,
            input_schema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Code symbols to sort",
                    },
                    "sort_by": {"type": "string", "default": "usage"},
                    "algorithm": {"type": "string", "default": "quicksort"},
                },
                "required": ["symbols"],
            },
        )

        self.server.add_tool(
            name="clrs_search_symbols",
            description="Search code symbols using CLRS binary search",
            handler=clrs_tools.search_code_symbols,
            input_schema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Code symbols to search",
                    },
                    "target_name": {"type": "string", "description": "Target symbol name"},
                },
                "required": ["symbols", "target_name"],
            },
        )

        self.server.add_tool(
            name="clrs_find_call_path",
            description="Find shortest call path using Dijkstra's algorithm",
            handler=clrs_tools.find_shortest_call_path,
            input_schema={
                "type": "object",
                "properties": {
                    "call_graph": {"type": "object", "description": "Function call graph"},
                    "start_function": {"type": "string", "description": "Start function"},
                    "end_function": {"type": "string", "description": "End function"},
                },
                "required": ["call_graph", "start_function", "end_function"],
            },
        )

        self.server.add_tool(
            name="clrs_pattern_matching",
            description="Find code patterns using KMP string matching",
            handler=clrs_tools.find_code_patterns,
            input_schema={
                "type": "object",
                "properties": {
                    "source_code": {"type": "string", "description": "Source code to search"},
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patterns to find",
                    },
                },
                "required": ["source_code", "patterns"],
            },
        )

        # Tree Analysis Tools
        self.server.add_tool(
            name="tree_ast_analysis",
            description="Process AST structure using tree operations",
            handler=tree_tools.process_ast_structure,
            input_schema={
                "type": "object",
                "properties": {
                    "ast_data": {"type": "object", "description": "AST data structure"},
                    "operation": {"type": "string", "default": "get_depth"},
                },
                "required": ["ast_data"],
            },
        )

        self.server.add_tool(
            name="tree_hierarchy_analysis",
            description="Analyze hierarchical code structure",
            handler=tree_tools.analyze_code_hierarchy,
            input_schema={
                "type": "object",
                "properties": {"codebase": {"type": "object", "description": "Codebase structure"}},
                "required": ["codebase"],
            },
        )

        self.server.add_tool(
            name="tree_structure_diff",
            description="Compare two code structures and show differences",
            handler=tree_tools.compare_code_structures,
            input_schema={
                "type": "object",
                "properties": {
                    "old_structure": {"type": "object", "description": "Old structure"},
                    "new_structure": {"type": "object", "description": "New structure"},
                },
                "required": ["old_structure", "new_structure"],
            },
        )

        self.server.add_tool(
            name="tree_config_merge",
            description="Merge configuration structures",
            handler=tree_tools.merge_configurations,
            input_schema={
                "type": "object",
                "properties": {
                    "base_config": {"type": "object", "description": "Base configuration"},
                    "override_config": {"type": "object", "description": "Override configuration"},
                },
                "required": ["base_config", "override_config"],
            },
        )

        # Enhanced Analysis Tools
        self.server.add_tool(
            name="clrs_comprehensive_analysis",
            description="Comprehensive analysis combining CLRS algorithms and Tree operations",
            handler=enhanced_tools.comprehensive_code_analysis,
            input_schema={
                "type": "object",
                "properties": {
                    "codebase_data": {"type": "object", "description": "Complete codebase data"}
                },
                "required": ["codebase_data"],
            },
        )

        self.server.add_tool(
            name="clrs_optimization_recommendations",
            description="Get optimization recommendations using CLRS+Tree analysis",
            handler=enhanced_tools.optimize_code_structure,
            input_schema={
                "type": "object",
                "properties": {
                    "current_structure": {
                        "type": "object",
                        "description": "Current code structure",
                    },
                    "optimization_goals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["reduce_complexity", "improve_modularity"],
                    },
                },
                "required": ["current_structure"],
            },
        )

    async def _initialize_agent_lifecycle_manager(self):
        """Initialize the MCP Agent Lifecycle Manager"""
        try:
            from cryptotrading.core.agents.mcp_agent_lifecycle_manager import (
                get_mcp_agent_lifecycle_manager,
            )

            # Get the global lifecycle manager
            self.lifecycle_manager = get_mcp_agent_lifecycle_manager()

            # Add lifecycle management tools
            await self._add_lifecycle_management_tools()

            print("MCP Agent Lifecycle Manager initialized successfully")

        except Exception as e:
            print(f"Warning: Could not initialize MCP Agent Lifecycle Manager: {e}")

    async def _add_lifecycle_management_tools(self):
        """Add lifecycle management tools to MCP server"""

        # Start agent tool
        async def start_agent(agent_id: str) -> dict:
            try:
                success = await self.lifecycle_manager.start_agent(agent_id)
                return {"success": success, "agent_id": agent_id, "action": "start"}
            except Exception as e:
                return {"success": False, "error": str(e), "agent_id": agent_id}

        # Stop agent tool
        async def stop_agent(agent_id: str) -> dict:
            try:
                success = await self.lifecycle_manager.stop_agent(agent_id)
                return {"success": success, "agent_id": agent_id, "action": "stop"}
            except Exception as e:
                return {"success": False, "error": str(e), "agent_id": agent_id}

        # Restart agent tool
        async def restart_agent(agent_id: str) -> dict:
            try:
                success = await self.lifecycle_manager.restart_agent(agent_id)
                return {"success": success, "agent_id": agent_id, "action": "restart"}
            except Exception as e:
                return {"success": False, "error": str(e), "agent_id": agent_id}

        # Get agent health tool
        async def get_agent_health(agent_id: str) -> dict:
            try:
                health = await self.lifecycle_manager.get_agent_health(agent_id)
                if health:
                    return {
                        "agent_id": agent_id,
                        "state": health.state.value,
                        "health_score": health.health_score,
                        "uptime_seconds": health.uptime_seconds,
                        "memory_entries": health.memory_entries,
                        "error_count": health.error_count,
                        "last_activity": health.last_activity.isoformat(),
                    }
                else:
                    return {"error": f"Agent {agent_id} not found"}
            except Exception as e:
                return {"error": str(e)}

        # Get system health tool
        async def get_system_health() -> dict:
            try:
                return await self.lifecycle_manager.get_system_health()
            except Exception as e:
                return {"error": str(e)}

        # Start lifecycle manager tool
        async def start_lifecycle_manager() -> dict:
            try:
                await self.lifecycle_manager.start()
                return {"success": True, "action": "start_lifecycle_manager"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Add tools to MCP server
        self.server.add_tool(
            name="lifecycle_start_agent",
            description="Start a specific MCP agent",
            handler=start_agent,
            input_schema={
                "agent_id": {"type": "string", "description": "ID of the agent to start"}
            },
        )

        self.server.add_tool(
            name="lifecycle_stop_agent",
            description="Stop a specific MCP agent",
            handler=stop_agent,
            input_schema={"agent_id": {"type": "string", "description": "ID of the agent to stop"}},
        )

        self.server.add_tool(
            name="lifecycle_restart_agent",
            description="Restart a specific MCP agent",
            handler=restart_agent,
            input_schema={
                "agent_id": {"type": "string", "description": "ID of the agent to restart"}
            },
        )

        self.server.add_tool(
            name="lifecycle_get_agent_health",
            description="Get health metrics for a specific agent",
            handler=get_agent_health,
            input_schema={
                "agent_id": {"type": "string", "description": "ID of the agent to check"}
            },
        )

        self.server.add_tool(
            name="lifecycle_get_system_health",
            description="Get overall system health and agent status",
            handler=get_system_health,
            input_schema={},
        )

        self.server.add_tool(
            name="lifecycle_start_manager",
            description="Start the MCP Agent Lifecycle Manager",
            handler=start_lifecycle_manager,
            input_schema={},
        )

    async def _add_basic_crypto_tools(self):
        """Add basic crypto tools - REMOVED MOCK DATA"""

        async def get_portfolio_summary() -> dict:
            """Real portfolio summary from database"""
            try:
                db = get_database()
                # Query actual portfolio data
                portfolio_data = await db.execute_query(
                    "SELECT symbol, amount, current_value FROM portfolios WHERE user_id = ?",
                    ["default_user"],
                )

                total_value = sum(row.get("current_value", 0) for row in portfolio_data)
                positions = [
                    {
                        "symbol": row["symbol"],
                        "amount": row["amount"],
                        "value": row["current_value"],
                    }
                    for row in portfolio_data
                ]

                return {
                    "total_value": total_value,
                    "positions": positions,
                    "last_updated": datetime.utcnow().isoformat(),
                    "source": "database",
                }
            except Exception as e:
                return {
                    "error": f"Portfolio data unavailable: {str(e)}",
                    "total_value": 0.0,
                    "positions": [],
                    "last_updated": datetime.utcnow().isoformat(),
                    "source": "error",
                }

        async def get_market_data(symbol: str) -> dict:
            """Real market data from unified provider"""
            try:
                from src.cryptotrading.data.providers.unified_provider import UnifiedDataProvider

                provider = UnifiedDataProvider()

                # Get real market data
                market_data = await provider.get_real_time_price(symbol)

                return {
                    "symbol": symbol.upper(),
                    "price": market_data.get("price", 0.0),
                    "change_24h": market_data.get("change_24h", 0.0),
                    "volume": market_data.get("volume_24h", 0.0),
                    "timestamp": market_data.get("timestamp"),
                    "source": market_data.get("source", "unknown"),
                }
            except Exception as e:
                # Don't return fake data - let the error propagate
                raise Exception(f"Market data unavailable for {symbol}: {str(e)}")

        # Add basic crypto tools
        self.server.add_simple_tool("get_portfolio", "Get portfolio summary", get_portfolio_summary)
        self.server.add_simple_tool(
            "get_market_data",
            "Get market data for symbol",
            get_market_data,
            {"symbol": {"type": "string", "description": "Trading symbol (BTC, ETH, etc.)"}},
        )


# Global server instance (reused across requests)
vercel_server = VercelMCPServer()


async def handle_mcp_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP JSON-RPC request"""
    try:
        # Initialize server if needed
        server = await vercel_server.initialize()

        # Handle the request
        response = await server.server.handle_request(request_data)
        return response

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_data.get("id"),
            "error": {"code": -32603, "message": "Internal error", "data": str(e)},
        }


def handler(request):
    """Vercel serverless function handler"""
    from urllib.parse import parse_qs

    # Handle different HTTP methods
    method = request.method

    if method == "OPTIONS":
        # CORS preflight
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
            },
            "body": "",
        }

    elif method == "GET":
        # Health check / status endpoint
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps(
                {
                    "status": "healthy",
                    "service": "MCP Server",
                    "version": "1.0.0",
                    "timestamp": datetime.utcnow().isoformat(),
                    "endpoints": {"mcp": "/api/mcp", "tools": "/api/mcp-tools"},
                }
            ),
        }

    elif method == "POST":
        try:
            # Parse request body
            if hasattr(request, "body"):
                body = request.body
            else:
                body = request.get("body", "{}")

            if isinstance(body, bytes):
                body = body.decode("utf-8")

            request_data = json.loads(body)

            # Handle MCP request
            response = asyncio.run(handle_mcp_request(request_data))

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps(response),
            }

        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    }
                ),
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32603, "message": "Internal error", "data": str(e)},
                    }
                ),
            }

    else:
        return {
            "statusCode": 405,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Method not allowed"}),
        }


# Export for Vercel
def app(request):
    """Main Vercel handler"""
    return handler(request)
