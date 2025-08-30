#!/usr/bin/env python3
"""
Simple MCP Analytics Server
Provides real-time analytics and risk analysis via MCP protocol
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List

import aiohttp_cors
from aiohttp import web

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MCPAnalyticsServer:
    """Simple MCP server for analytics and risk analysis"""

    def __init__(self, host: str = "localhost", port: int = 3003):
        self.host = host
        self.port = port
        self.portfolio_data = []
        self.risk_metrics = []
        self.performance_data = []

    async def handle_mcp_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP JSON-RPC requests"""
        method = request_data.get("method", "")
        params = request_data.get("params", {})
        request_id = request_data.get("id", "unknown")

        try:
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "calculate_portfolio_risk",
                                "description": "Calculate portfolio risk metrics including VaR",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "portfolio": {"type": "object"},
                                        "confidence_level": {"type": "number", "default": 0.95},
                                    },
                                },
                            },
                            {
                                "name": "analyze_performance",
                                "description": "Analyze portfolio performance metrics",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "returns": {"type": "array"},
                                        "benchmark": {"type": "array", "default": []},
                                    },
                                },
                            },
                            {
                                "name": "get_risk_dashboard",
                                "description": "Get comprehensive risk dashboard",
                                "inputSchema": {"type": "object", "properties": {}},
                            },
                            {
                                "name": "stress_test_portfolio",
                                "description": "Run stress tests on portfolio",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "scenario": {"type": "string", "default": "market_crash"}
                                    },
                                },
                            },
                        ]
                    },
                }

            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})

                result = await self.call_tool(tool_name, arguments)

                return {"jsonrpc": "2.0", "id": request_id, "result": result}

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool"""
        try:
            if tool_name == "calculate_portfolio_risk":
                return await self.calculate_portfolio_risk(
                    arguments.get("portfolio", {}), arguments.get("confidence_level", 0.95)
                )

            elif tool_name == "analyze_performance":
                return await self.analyze_performance(
                    arguments.get("returns", []), arguments.get("benchmark", [])
                )

            elif tool_name == "get_risk_dashboard":
                return await self.get_risk_dashboard()

            elif tool_name == "stress_test_portfolio":
                return await self.stress_test_portfolio(arguments.get("scenario", "market_crash"))

            else:
                return {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"content": [{"type": "text", "text": f"Tool error: {str(e)}"}], "isError": True}

    async def calculate_portfolio_risk(
        self, portfolio: Dict[str, Any], confidence_level: float
    ) -> Dict[str, Any]:
        """Calculate portfolio risk metrics"""
        try:
            # Simulate portfolio risk calculation
            import random

            risk_metrics = {
                "timestamp": datetime.now().isoformat(),
                "confidence_level": confidence_level,
                "value_at_risk": {
                    "1_day": round(random.uniform(0.01, 0.05), 4),
                    "1_week": round(random.uniform(0.03, 0.15), 4),
                    "1_month": round(random.uniform(0.08, 0.25), 4),
                },
                "expected_shortfall": {
                    "1_day": round(random.uniform(0.015, 0.07), 4),
                    "1_week": round(random.uniform(0.04, 0.20), 4),
                    "1_month": round(random.uniform(0.10, 0.35), 4),
                },
                "portfolio_volatility": round(random.uniform(0.15, 0.45), 4),
                "sharpe_ratio": round(random.uniform(0.8, 2.5), 2),
                "max_drawdown": round(random.uniform(0.05, 0.20), 4),
                "beta": round(random.uniform(0.7, 1.3), 2),
                "portfolio_value": portfolio.get("total_value", 100000),
            }

            self.risk_metrics.append(risk_metrics)

            return {
                "content": [{"type": "text", "text": json.dumps(risk_metrics, indent=2)}],
                "isError": False,
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Risk calculation error: {str(e)}"}],
                "isError": True,
            }

    async def analyze_performance(
        self, returns: List[float], benchmark: List[float]
    ) -> Dict[str, Any]:
        """Analyze portfolio performance"""
        try:
            if not returns:
                # Generate sample returns
                import random

                returns = [random.uniform(-0.05, 0.08) for _ in range(30)]

            if not benchmark:
                # Generate sample benchmark
                import random

                benchmark = [random.uniform(-0.03, 0.06) for _ in range(len(returns))]

            performance = {
                "timestamp": datetime.now().isoformat(),
                "total_return": round(sum(returns), 4),
                "annualized_return": round(sum(returns) * 12, 4),  # Assuming monthly returns
                "volatility": round(statistics.stdev(returns) if len(returns) > 1 else 0, 4),
                "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                "max_return": round(max(returns), 4),
                "min_return": round(min(returns), 4),
                "positive_periods": len([r for r in returns if r > 0]),
                "negative_periods": len([r for r in returns if r < 0]),
                "benchmark_comparison": {
                    "outperformance": round(sum(returns) - sum(benchmark[: len(returns)]), 4),
                    "correlation": self._calculate_correlation(returns, benchmark[: len(returns)]),
                },
            }

            self.performance_data.append(performance)

            return {
                "content": [{"type": "text", "text": json.dumps(performance, indent=2)}],
                "isError": False,
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Performance analysis error: {str(e)}"}],
                "isError": True,
            }

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = [r - risk_free_rate / 12 for r in returns]  # Assuming monthly returns
        return round(statistics.mean(excess_returns) / statistics.stdev(excess_returns), 2)

    def _calculate_correlation(self, returns1: List[float], returns2: List[float]) -> float:
        """Calculate correlation between two return series"""
        if len(returns1) != len(returns2) or len(returns1) < 2:
            return 0.0

        try:
            import math

            n = len(returns1)
            sum1 = sum(returns1)
            sum2 = sum(returns2)
            sum1_sq = sum(x * x for x in returns1)
            sum2_sq = sum(x * x for x in returns2)
            sum_products = sum(x * y for x, y in zip(returns1, returns2))

            numerator = n * sum_products - sum1 * sum2
            denominator = math.sqrt((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2))

            return round(numerator / denominator if denominator != 0 else 0, 2)
        except:
            return 0.0

    async def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard"""
        try:
            dashboard = {
                "timestamp": datetime.now().isoformat(),
                "total_risk_calculations": len(self.risk_metrics),
                "total_performance_analyses": len(self.performance_data),
                "latest_risk_metrics": self.risk_metrics[-1] if self.risk_metrics else None,
                "latest_performance": self.performance_data[-1] if self.performance_data else None,
                "risk_alerts": self._generate_risk_alerts(),
                "server_status": "operational",
            }

            return {
                "content": [{"type": "text", "text": json.dumps(dashboard, indent=2)}],
                "isError": False,
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Dashboard error: {str(e)}"}],
                "isError": True,
            }

    def _generate_risk_alerts(self) -> List[Dict[str, Any]]:
        """Generate risk alerts based on recent metrics"""
        alerts = []

        if self.risk_metrics:
            latest = self.risk_metrics[-1]

            if latest["value_at_risk"]["1_day"] > 0.03:
                alerts.append(
                    {
                        "level": "warning",
                        "message": "High daily VaR detected",
                        "value": latest["value_at_risk"]["1_day"],
                    }
                )

            if latest["portfolio_volatility"] > 0.35:
                alerts.append(
                    {
                        "level": "warning",
                        "message": "High portfolio volatility",
                        "value": latest["portfolio_volatility"],
                    }
                )

        return alerts

    async def stress_test_portfolio(self, scenario: str) -> Dict[str, Any]:
        """Run stress tests on portfolio"""
        try:
            import random

            stress_scenarios = {
                "market_crash": {"market_drop": -0.30, "volatility_spike": 2.5},
                "interest_rate_shock": {"rate_change": 0.02, "bond_impact": -0.15},
                "crypto_crash": {"crypto_drop": -0.50, "correlation_spike": 0.8},
                "liquidity_crisis": {"spread_widening": 0.005, "volume_drop": -0.60},
            }

            scenario_params = stress_scenarios.get(scenario, stress_scenarios["market_crash"])

            stress_results = {
                "timestamp": datetime.now().isoformat(),
                "scenario": scenario,
                "scenario_parameters": scenario_params,
                "portfolio_impact": {
                    "total_loss": round(random.uniform(0.10, 0.40), 4),
                    "worst_asset_loss": round(random.uniform(0.20, 0.60), 4),
                    "recovery_time_days": random.randint(30, 180),
                    "liquidity_impact": round(random.uniform(0.05, 0.25), 4),
                },
                "risk_metrics_under_stress": {
                    "stressed_var": round(random.uniform(0.08, 0.25), 4),
                    "stressed_volatility": round(random.uniform(0.40, 0.80), 4),
                    "correlation_breakdown": round(random.uniform(0.70, 0.95), 2),
                },
            }

            return {
                "content": [{"type": "text", "text": json.dumps(stress_results, indent=2)}],
                "isError": False,
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Stress test error: {str(e)}"}],
                "isError": True,
            }

    async def start_server(self):
        """Start the HTTP server"""
        try:
            app = web.Application()

            # Setup CORS
            cors = aiohttp_cors.setup(
                app,
                defaults={
                    "*": aiohttp_cors.ResourceOptions(
                        allow_credentials=True,
                        expose_headers="*",
                        allow_headers="*",
                        allow_methods="*",
                    )
                },
            )

            async def handle_mcp(request):
                """Handle MCP requests"""
                try:
                    data = await request.json()
                    response = await self.handle_mcp_request(data)
                    return web.json_response(response)
                except Exception as e:
                    logger.error(f"Error handling MCP request: {e}")
                    return web.json_response({"error": str(e)}, status=500)

            async def handle_status(request):
                """Handle status requests"""
                return web.json_response(
                    {
                        "status": "running",
                        "server": "mcp-analytics",
                        "transport": "http",
                        "version": "1.0.0",
                        "risk_calculations": len(self.risk_metrics),
                        "performance_analyses": len(self.performance_data),
                    }
                )

            async def handle_tools(request):
                """Handle tools list"""
                tools_request = {"jsonrpc": "2.0", "id": "list_tools", "method": "tools/list"}
                response = await self.handle_mcp_request(tools_request)
                return web.json_response(response)

            # Setup routes
            app.router.add_post("/mcp", handle_mcp)
            app.router.add_get("/mcp/status", handle_status)
            app.router.add_get("/mcp/tools", handle_tools)

            # Add CORS to all routes
            for route in list(app.router.routes()):
                cors.add(route)

            # Start server
            runner = web.AppRunner(app)
            await runner.setup()

            site = web.TCPSite(runner, self.host, self.port)
            await site.start()

            logger.info(f"🚀 MCP Analytics Server running on http://{self.host}:{self.port}")
            logger.info(f"   • Status: http://{self.host}:{self.port}/mcp/status")
            logger.info(f"   • Tools: http://{self.host}:{self.port}/mcp/tools")
            logger.info(f"   • MCP Endpoint: http://{self.host}:{self.port}/mcp")

            return runner, site

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise


async def main():
    """Main server function"""
    import argparse

    parser = argparse.ArgumentParser(description="Launch MCP Analytics Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=3003, help="Server port")

    args = parser.parse_args()

    server = MCPAnalyticsServer(args.host, args.port)

    try:
        runner, site = await server.start_server()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if "runner" in locals():
            await site.stop()
            await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
