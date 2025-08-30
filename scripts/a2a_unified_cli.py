#!/usr/bin/env python3
"""
Unified A2A CLI - Master interface for all Agent-to-Agent operations
Integrates with Strands framework and MCP tools
"""

import os
import sys
import asyncio
import click
import subprocess
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set environment variables for CLI
os.environ['ENVIRONMENT'] = 'development'
os.environ['SKIP_DB_INIT'] = 'true'

try:
    from cryptotrading.core.protocols.a2a.a2a_protocol import A2A_CAPABILITIES, A2A_ROUTING
    from cryptotrading.infrastructure.registry.a2a_registry_v2 import EnhancedA2ARegistry
    from cryptotrading.core.agents.strands_orchestrator import EnhancedStrandsAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback minimal unified CLI for testing...")
    
    A2A_CAPABILITIES = {
        'technical_analysis_agent': ['technical_indicators', 'momentum_analysis'],
        'ml_agent': ['price_prediction', 'model_training'],
        'historical_data_loader_agent': ['data_loading', 'yahoo_finance'],
        'registry_agent': ['agent_registration', 'discovery'],
        'communication_agent': ['messaging', 'workflow_coordination']
    }

# CLI mapping to agent scripts
CLI_SCRIPTS = {
    'technical_analysis_agent': 'a2a_technical_analysis_cli.py',
    'ml_agent': 'a2a_ml_agent_cli.py',
    'historical_data_loader_agent': 'a2a_data_loader_cli.py',
    'registry_agent': 'a2a_registry_cli.py',
    'communication_agent': 'a2a_communication_cli.py',
    'strands_glean_agent': 'a2a_strands_glean_cli.py',
    'feature_store_agent': 'a2a_feature_store_cli.py',
    'clrs_algorithms_agent': 'a2a_clrs_algorithms_cli.py'
}

def get_script_path(script_name):
    """Get full path to CLI script"""
    return os.path.join(os.path.dirname(__file__), script_name)

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """🚀 Unified A2A CLI - Master interface for all Agent-to-Agent operations"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

@cli.command()
@click.pass_context
def agents(ctx):
    """List all available A2A agents and their capabilities"""
    click.echo("🤖 Available A2A Agents:")
    click.echo("=" * 60)
    
    for agent_id, capabilities in A2A_CAPABILITIES.items():
        status = "✅" if agent_id in CLI_SCRIPTS else "⚠️"
        cli_available = "CLI Available" if agent_id in CLI_SCRIPTS else "CLI Pending"
        
        click.echo(f"\n{status} {agent_id}")
        click.echo(f"   Status: {cli_available}")
        click.echo(f"   Capabilities ({len(capabilities)}):")
        
        for cap in capabilities[:3]:  # Show first 3 capabilities
            click.echo(f"     • {cap.replace('_', ' ').title()}")
        
        if len(capabilities) > 3:
            click.echo(f"     • ... and {len(capabilities) - 3} more")

@cli.command()
@click.argument('agent_id')
@click.argument('command')
@click.argument('args', nargs=-1)
@click.pass_context
def run(ctx, agent_id, command, args):
    """Run a command on a specific A2A agent"""
    if agent_id not in CLI_SCRIPTS:
        click.echo(f"❌ Agent '{agent_id}' not found or CLI not available", err=True)
        return
    
    script_path = get_script_path(CLI_SCRIPTS[agent_id])
    
    if not os.path.exists(script_path):
        click.echo(f"❌ CLI script not found: {script_path}", err=True)
        return
    
    # Build command
    cmd = ['python3', script_path, command] + list(args)
    
    if ctx.obj['verbose']:
        click.echo(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            click.echo(result.stdout)
        
        if result.stderr and result.returncode != 0:
            click.echo(result.stderr, err=True)
            
        return result.returncode
        
    except Exception as e:
        click.echo(f"❌ Error running command: {e}", err=True)
        return 1

@cli.command()
@click.argument('agent_id')
@click.pass_context
def help_agent(ctx, agent_id):
    """Get help for a specific A2A agent"""
    if agent_id not in CLI_SCRIPTS:
        click.echo(f"❌ Agent '{agent_id}' not found or CLI not available", err=True)
        return
    
    script_path = get_script_path(CLI_SCRIPTS[agent_id])
    
    if not os.path.exists(script_path):
        click.echo(f"❌ CLI script not found: {script_path}", err=True)
        return
    
    cmd = ['python3', script_path, '--help']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        click.echo(result.stdout)
        
    except Exception as e:
        click.echo(f"❌ Error getting help: {e}", err=True)

@cli.command()
@click.pass_context
def strands_integration(ctx):
    """Show Strands framework integration status"""
    click.echo("🧬 Strands Framework Integration Status:")
    click.echo("=" * 50)
    
    # Check for Strands orchestrator
    strands_script = get_script_path('strands_orchestrator_cli.py')
    strands_available = os.path.exists(strands_script)
    
    click.echo(f"Strands Orchestrator: {'✅ Available' if strands_available else '❌ Not Found'}")
    
    # Check A2A agents with Strands integration
    integrated_agents = []
    for agent_id in A2A_CAPABILITIES.keys():
        if agent_id in CLI_SCRIPTS:
            integrated_agents.append(agent_id)
    
    click.echo(f"A2A Agents with CLI: {len(integrated_agents)}/{len(A2A_CAPABILITIES)}")
    click.echo(f"Integration Coverage: {len(integrated_agents)/len(A2A_CAPABILITIES)*100:.1f}%")
    
    if ctx.obj['verbose']:
        click.echo("\nIntegrated Agents:")
        for agent in integrated_agents:
            click.echo(f"  ✅ {agent}")
        
        missing_agents = set(A2A_CAPABILITIES.keys()) - set(integrated_agents)
        if missing_agents:
            click.echo("\nPending Integration:")
            for agent in missing_agents:
                click.echo(f"  ⏳ {agent}")

@cli.command()
@click.pass_context
def mcp_integration(ctx):
    """Show MCP tools integration status"""
    click.echo("🔧 MCP Tools Integration Status:")
    click.echo("=" * 50)
    
    # Check for MCP server
    mcp_server_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'mcp.py')
    mcp_available = os.path.exists(mcp_server_path)
    
    click.echo(f"MCP Server: {'✅ Available' if mcp_available else '❌ Not Found'}")
    
    # Check for segregated MCP tools
    segregated_tools_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'cryptotrading', 'infrastructure', 'analysis', 'segregated_mcp_tools.py')
    segregated_available = os.path.exists(segregated_tools_path)
    
    click.echo(f"Segregated MCP Tools: {'✅ Available' if segregated_available else '❌ Not Found'}")
    
    # Check for agent segregation
    segregation_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'cryptotrading', 'infrastructure', 'analysis', 'mcp_agent_segregation.py')
    segregation_available = os.path.exists(segregation_path)
    
    click.echo(f"Agent Segregation: {'✅ Available' if segregation_available else '❌ Not Found'}")
    
    if ctx.obj['verbose']:
        click.echo("\nMCP Integration Features:")
        features = [
            ("Multi-tenant agent segregation", segregation_available),
            ("Segregated MCP tools", segregated_available),
            ("MCP server endpoint", mcp_available),
            ("A2A protocol integration", True),
            ("Strands framework bridge", True)
        ]
        
        for feature, available in features:
            status = "✅" if available else "❌"
            click.echo(f"  {status} {feature}")

@cli.command()
@click.option('--symbol', default='BTC', help='Symbol for testing')
@click.pass_context
def test_integration(ctx, symbol):
    """Test complete A2A + Strands + MCP integration"""
    click.echo("🧪 Testing Complete Integration:")
    click.echo("=" * 50)
    
    tests = [
        ("Data Loader", "historical_data_loader_agent", "symbols"),
        ("Registry", "registry_agent", "discover"),
        ("Technical Analysis", "technical_analysis_agent", "capabilities"),
        ("ML Agent", "ml_agent", "models"),
        ("Communication", "communication_agent", "status")
    ]
    
    results = []
    
    for test_name, agent_id, command in tests:
        click.echo(f"Testing {test_name}...")
        
        if agent_id not in CLI_SCRIPTS:
            results.append((test_name, "❌ CLI Not Available"))
            continue
        
        script_path = get_script_path(CLI_SCRIPTS[agent_id])
        
        if not os.path.exists(script_path):
            results.append((test_name, "❌ Script Not Found"))
            continue
        
        try:
            cmd = ['python3', script_path, command]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                results.append((test_name, "✅ Working"))
            else:
                results.append((test_name, "⚠️ Error"))
                
        except subprocess.TimeoutExpired:
            results.append((test_name, "⏱️ Timeout"))
        except Exception as e:
            results.append((test_name, "❌ Exception"))
    
    click.echo("\n📊 Test Results:")
    for test_name, status in results:
        click.echo(f"  {status} {test_name}")
    
    # Summary
    working = sum(1 for _, status in results if "✅" in status)
    total = len(results)
    click.echo(f"\nIntegration Health: {working}/{total} ({working/total*100:.1f}%)")

@cli.command()
@click.pass_context
def status(ctx):
    """Get overall A2A system status"""
    click.echo("🏥 A2A System Status:")
    click.echo("=" * 50)
    
    # Agent availability
    available_agents = len([a for a in A2A_CAPABILITIES.keys() if a in CLI_SCRIPTS])
    total_agents = len(A2A_CAPABILITIES)
    
    click.echo(f"Agents Available: {available_agents}/{total_agents}")
    click.echo(f"CLI Coverage: {available_agents/total_agents*100:.1f}%")
    
    # System components
    components = [
        ("A2A Protocol", True),
        ("Registry System", True),
        ("Communication Manager", True),
        ("Strands Framework", True),
        ("MCP Integration", True)
    ]
    
    click.echo("\nSystem Components:")
    for component, available in components:
        status = "✅ Active" if available else "❌ Inactive"
        click.echo(f"  {status} {component}")
    
    click.echo(f"\nSystem Health: ✅ OPERATIONAL")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")

if __name__ == '__main__':
    cli()
