#!/usr/bin/env python3
"""
Agent Manager CLI - A2A Strands-based CLI for agent management.

Uses MCP tools to execute all agent management operations including:
- A2A registration
- MCP tool segregation
- Blockchain registration
- Lifecycle management
"""

import asyncio
import click
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from ..specialized.agent_manager import AgentManagerAgent
from ..strands import StrandsAgent
from ...protocols.a2a.a2a_protocol import A2AMessage, MessageType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentManagerCLI:
    """CLI for Agent Manager using MCP tools."""
    
    def __init__(self):
        self.agent_manager = None
        self.strands_agent = None
        self.mcp_tools = {}
        
    async def initialize(self):
        """Initialize Agent Manager with Strands and MCP tools."""
        # Create Agent Manager instance
        self.agent_manager = AgentManagerAgent(agent_id="agent-manager-001")
        
        # Initialize as Strands agent for MCP capabilities
        self.strands_agent = StrandsAgent(
            agent_id="agent-manager-strands",
            config={
                "mcp_tools_path": "src/cryptotrading/core/agents/mcp_tools/agent_manager_tools.json",
                "enable_blockchain": True,
                "enable_compliance": True
            }
        )
        
        # Load MCP tools
        await self._load_mcp_tools()
        
        # Initialize agent manager
        await self.agent_manager.initialize()
        
        logger.info("Agent Manager CLI initialized with MCP tools")
    
    async def _load_mcp_tools(self):
        """Load MCP tools configuration."""
        tools_path = Path("src/cryptotrading/core/agents/mcp_tools/agent_manager_tools.json")
        
        if tools_path.exists():
            with open(tools_path, 'r') as f:
                tools_config = json.load(f)
                self.mcp_tools = tools_config.get("tools", {})
                logger.info(f"Loaded {len(self.mcp_tools)} MCP tools")
        else:
            logger.warning("MCP tools configuration not found")
    
    async def execute_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool through the Strands agent."""
        if tool_name not in self.mcp_tools:
            return {"error": f"Unknown MCP tool: {tool_name}"}
        
        # Create MCP tool invocation message
        message = A2AMessage(
            type=MessageType.MCP_TOOL_INVOCATION,
            sender="agent-manager-cli",
            receiver="agent-manager-001",
            payload={
                "tool": tool_name,
                "parameters": params
            }
        )
        
        # Execute through agent manager
        result = await self.agent_manager.process_message(message)
        
        if result:
            return result.payload
        else:
            # Direct execution fallback
            method_name = f"_mcp_{tool_name}"
            if hasattr(self.agent_manager, method_name):
                method = getattr(self.agent_manager, method_name)
                return await method(**params)
            else:
                return {"error": f"Tool {tool_name} not implemented"}
    
    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        mcp_tools: List[str],
        blockchain: bool = True
    ) -> Dict[str, Any]:
        """Register an agent with A2A network and blockchain."""
        logger.info(f"Registering agent {agent_id}...")
        
        # Step 1: Validate compliance
        compliance_result = await self.execute_mcp_tool(
            "validate_compliance",
            {
                "agent_id": agent_id,
                "capabilities": capabilities,
                "mcp_tools": mcp_tools
            }
        )
        
        if not compliance_result.get("compliant", False):
            logger.error(f"Agent {agent_id} failed compliance: {compliance_result.get('violations')}")
            return compliance_result
        
        # Step 2: Enforce MCP segregation
        segregation_result = await self.execute_mcp_tool(
            "enforce_mcp_segregation",
            {
                "agent_id": agent_id,
                "requested_tools": mcp_tools,
                "agent_capabilities": capabilities
            }
        )
        
        approved_tools = segregation_result.get("approved_tools", [])
        logger.info(f"Approved MCP tools: {approved_tools}")
        
        # Step 3: Generate skill card
        skill_card_result = await self.execute_mcp_tool(
            "generate_skill_card",
            {
                "agent_id": agent_id,
                "capabilities": capabilities
            }
        )
        
        skill_card = skill_card_result.get("skill_card", {})
        
        # Step 4: Register with A2A
        registration_result = await self.execute_mcp_tool(
            "register_agent",
            {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "capabilities": capabilities,
                "mcp_tools": approved_tools,
                "skill_card": skill_card,
                "blockchain_register": blockchain
            }
        )
        
        # Step 5: Blockchain registration if requested
        if blockchain and registration_result.get("status") == "success":
            blockchain_result = await self.execute_mcp_tool(
                "blockchain_register",
                {
                    "agent_id": agent_id,
                    "skill_card": skill_card
                }
            )
            registration_result["blockchain_tx"] = blockchain_result.get("transaction_hash")
        
        logger.info(f"Agent {agent_id} registration complete: {registration_result}")
        return registration_result
    
    async def manage_lifecycle(self, agent_id: str, action: str) -> Dict[str, Any]:
        """Manage agent lifecycle (start, stop, restart, health_check)."""
        result = await self.execute_mcp_tool(
            "manage_lifecycle",
            {
                "agent_id": agent_id,
                "action": action
            }
        )
        
        logger.info(f"Lifecycle action {action} for {agent_id}: {result.get('status')}")
        return result
    
    async def query_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query registered agents."""
        result = await self.execute_mcp_tool(
            "query_registry",
            {"filter": filters or {}}
        )
        
        return result
    
    async def audit_compliance(self, detailed: bool = False) -> Dict[str, Any]:
        """Audit all agents for compliance."""
        result = await self.execute_mcp_tool(
            "audit_compliance",
            {
                "include_blockchain": True,
                "detailed_report": detailed
            }
        )
        
        return result
    
    async def monitor_health(self) -> Dict[str, Any]:
        """Monitor health of all agents."""
        result = await self.execute_mcp_tool(
            "monitor_health",
            {"include_metrics": True}
        )
        
        return result


# CLI Commands
@click.group()
@click.pass_context
def cli(ctx):
    """Agent Manager CLI - Manage A2A agents with MCP tools."""
    ctx.ensure_object(dict)
    ctx.obj['manager'] = AgentManagerCLI()


@cli.command()
@click.argument('agent_id')
@click.argument('agent_type')
@click.option('--capabilities', '-c', multiple=True, help='Agent capabilities')
@click.option('--mcp-tools', '-t', multiple=True, help='Required MCP tools')
@click.option('--no-blockchain', is_flag=True, help='Skip blockchain registration')
@click.pass_context
def register(ctx, agent_id, agent_type, capabilities, mcp_tools, no_blockchain):
    """Register a new agent with A2A network."""
    async def _register():
        manager = ctx.obj['manager']
        await manager.initialize()
        
        result = await manager.register_agent(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=list(capabilities),
            mcp_tools=list(mcp_tools),
            blockchain=not no_blockchain
        )
        
        if result.get("status") == "success":
            click.echo(f"✅ Agent {agent_id} registered successfully")
            click.echo(f"   Compliance Score: {result.get('compliance_score')}")
            if result.get("blockchain_tx"):
                click.echo(f"   Blockchain TX: {result.get('blockchain_tx')}")
        else:
            click.echo(f"❌ Registration failed: {result.get('error')}")
    
    asyncio.run(_register())


@cli.command()
@click.argument('agent_id')
@click.argument('action', type=click.Choice(['start', 'stop', 'restart', 'health']))
@click.pass_context
def lifecycle(ctx, agent_id, action):
    """Manage agent lifecycle."""
    async def _lifecycle():
        manager = ctx.obj['manager']
        await manager.initialize()
        
        result = await manager.manage_lifecycle(agent_id, action)
        
        click.echo(f"Agent {agent_id} - Action: {action}")
        click.echo(f"Status: {result.get('status')}")
        if result.get('health'):
            click.echo(f"Health: {json.dumps(result['health'], indent=2)}")
    
    asyncio.run(_lifecycle())


@cli.command()
@click.option('--type', '-t', help='Filter by agent type')
@click.option('--capability', '-c', help='Filter by capability')
@click.option('--status', '-s', help='Filter by status')
@click.option('--blockchain', is_flag=True, help='Only show blockchain registered')
@click.pass_context
def list(ctx, type, capability, status, blockchain):
    """List registered agents."""
    async def _list():
        manager = ctx.obj['manager']
        await manager.initialize()
        
        filters = {}
        if type:
            filters['agent_type'] = type
        if capability:
            filters['capabilities'] = [capability]
        if status:
            filters['status'] = status
        if blockchain:
            filters['blockchain_only'] = True
        
        agents = await manager.query_agents(filters)
        
        click.echo("Registered Agents:")
        click.echo("-" * 60)
        for agent in agents:
            click.echo(f"  {agent['agent_id']}")
            click.echo(f"    Status: {agent['status']}")
            click.echo(f"    Capabilities: {', '.join(agent['capabilities'][:3])}...")
            if agent.get('blockchain_address'):
                click.echo(f"    Blockchain: {agent['blockchain_address']}")
    
    asyncio.run(_list())


@cli.command()
@click.option('--detailed', '-d', is_flag=True, help='Generate detailed report')
@click.pass_context
def audit(ctx, detailed):
    """Audit all agents for A2A compliance."""
    async def _audit():
        manager = ctx.obj['manager']
        await manager.initialize()
        
        result = await manager.audit_compliance(detailed=detailed)
        
        click.echo("A2A Compliance Audit")
        click.echo("=" * 60)
        click.echo(f"Compliant Agents: {len(result.get('compliant_agents', []))}")
        click.echo(f"Non-Compliant Agents: {len(result.get('non_compliant_agents', []))}")
        click.echo(f"Total Compliance Score: {result.get('total_score', 0):.2f}")
        
        if result.get('non_compliant_agents'):
            click.echo("\nNon-Compliant Agents:")
            for agent in result['non_compliant_agents']:
                click.echo(f"  - {agent}")
        
        if detailed and result.get('report'):
            click.echo("\nDetailed Report:")
            click.echo(json.dumps(result['report'], indent=2))
    
    asyncio.run(_audit())


@cli.command()
@click.pass_context
def health(ctx):
    """Monitor health of all agents."""
    async def _health():
        manager = ctx.obj['manager']
        await manager.initialize()
        
        result = await manager.monitor_health()
        
        click.echo("Agent Health Monitor")
        click.echo("=" * 60)
        
        healthy = result.get('healthy_agents', [])
        unhealthy = result.get('unhealthy_agents', [])
        
        click.echo(f"✅ Healthy: {len(healthy)}")
        click.echo(f"❌ Unhealthy: {len(unhealthy)}")
        
        if unhealthy:
            click.echo("\nUnhealthy Agents:")
            for agent in unhealthy:
                click.echo(f"  - {agent}")
        
        if result.get('recommendations'):
            click.echo("\nRecommendations:")
            for rec in result['recommendations']:
                click.echo(f"  • {rec}")
    
    asyncio.run(_health())


@cli.command()
@click.pass_context
def register_trading(ctx):
    """Register the Trading Algorithm Agent (convenience command)."""
    async def _register_trading():
        manager = ctx.obj['manager']
        await manager.initialize()
        
        # Trading Algorithm Agent registration
        result = await manager.register_agent(
            agent_id="trading_algorithm_agent",
            agent_type="trading_algorithm",
            capabilities=[
                "grid_trading", "dollar_cost_averaging", "arbitrage_detection",
                "momentum_trading", "mean_reversion", "scalping",
                "market_making", "breakout_trading", "ml_predictions",
                "multi_strategy_management", "risk_management",
                "portfolio_optimization", "signal_generation",
                "strategy_analysis", "backtesting"
            ],
            mcp_tools=[
                "grid_create", "grid_rebalance", "grid_monitor",
                "dca_execute", "dca_smart_adjust", "dca_schedule",
                "arbitrage_scan", "arbitrage_execute", "momentum_scan",
                "risk_calculate", "position_size", "portfolio_optimize"
            ],
            blockchain=True
        )
        
        if result.get("status") == "success":
            click.echo("✅ Trading Algorithm Agent registered successfully!")
            click.echo(f"   Registration ID: {result.get('registration_id')}")
            click.echo(f"   Compliance Score: {result.get('compliance_score')}")
            click.echo(f"   Blockchain TX: {result.get('blockchain_tx')}")
        else:
            click.echo(f"❌ Registration failed: {result}")
    
    asyncio.run(_register_trading())


@cli.command()
@click.pass_context
def register_all(ctx):
    """Register all specialized agents."""
    async def _register_all():
        manager = ctx.obj['manager']
        await manager.initialize()
        
        agents_to_register = [
            {
                "agent_id": "trading_algorithm_agent",
                "agent_type": "trading_algorithm",
                "capabilities": ["signal_generation", "strategy_analysis", "backtesting"],
                "mcp_tools": ["grid_create", "dca_execute", "risk_calculate"]
            },
            {
                "agent_id": "data_analysis_agent",
                "agent_type": "data_analysis",
                "capabilities": ["data_processing", "statistical_analysis", "pattern_recognition"],
                "mcp_tools": ["analyze_data", "generate_report"]
            },
            {
                "agent_id": "feature_store_agent",
                "agent_type": "feature_store",
                "capabilities": ["feature_storage", "feature_retrieval", "feature_versioning"],
                "mcp_tools": ["store_feature", "retrieve_feature", "validate_feature"]
            }
        ]
        
        for agent_config in agents_to_register:
            click.echo(f"\nRegistering {agent_config['agent_id']}...")
            result = await manager.register_agent(**agent_config, blockchain=True)
            
            if result.get("status") == "success":
                click.echo(f"  ✅ Success - Score: {result.get('compliance_score')}")
            else:
                click.echo(f"  ❌ Failed: {result.get('error')}")
    
    asyncio.run(_register_all())


if __name__ == '__main__':
    cli()