#!/usr/bin/env python3
"""
A2A Strands Glean Agent CLI
Provides command-line interface for code analysis and Glean operations
"""

import asyncio
import os
import sys
from datetime import datetime

import click

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set environment variables for CLI
os.environ["ENVIRONMENT"] = "development"
os.environ["SKIP_DB_INIT"] = "true"

try:
    from cryptotrading.core.agents.specialized.strands_glean_agent import StrandsGleanAgent
    from cryptotrading.core.protocols.a2a.a2a_protocol import A2A_CAPABILITIES
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback minimal Strands Glean agent for CLI testing...")

    class FallbackStrandsGleanAgent:
        """Minimal Strands Glean agent for CLI testing when imports fail"""

        def __init__(self):
            self.agent_id = "strands_glean_agent"
            self.capabilities = [
                "code_analysis",
                "dependency_mapping",
                "symbol_search",
                "code_navigation",
                "insight_generation",
                "coverage_validation",
                "change_monitoring",
                "code_quality",
            ]

        async def analyze_code_structure(self, path, language="python"):
            """Mock code structure analysis"""
            return {
                "path": path,
                "language": language,
                "files_analyzed": 45,
                "classes_found": 12,
                "functions_found": 89,
                "complexity_score": 7.2,
                "maintainability_index": 68.5,
                "test_coverage": 0.82,
                "timestamp": datetime.now().isoformat(),
            }

        async def search_symbols(self, query, scope="global"):
            """Mock symbol search"""
            symbols = [
                {"name": f"{query}Handler", "type": "class", "file": "handlers.py", "line": 45},
                {
                    "name": f"process_{query.lower()}",
                    "type": "function",
                    "file": "processors.py",
                    "line": 123,
                },
                {
                    "name": f"{query.upper()}_CONFIG",
                    "type": "constant",
                    "file": "config.py",
                    "line": 12,
                },
            ]
            return {
                "query": query,
                "scope": scope,
                "results": symbols,
                "total_matches": len(symbols),
                "timestamp": datetime.now().isoformat(),
            }

        async def generate_dependency_map(self, root_path):
            """Mock dependency mapping"""
            return {
                "root_path": root_path,
                "dependencies": {
                    "internal": ["core.agents", "infrastructure.data", "protocols.a2a"],
                    "external": ["pandas", "numpy", "asyncio", "click"],
                    "circular": [],
                },
                "dependency_graph": {"nodes": 25, "edges": 48, "clusters": 5},
                "timestamp": datetime.now().isoformat(),
            }

        async def validate_coverage(self, target_path):
            """Mock coverage validation"""
            return {
                "target_path": target_path,
                "coverage_metrics": {
                    "line_coverage": 0.847,
                    "branch_coverage": 0.723,
                    "function_coverage": 0.891,
                },
                "blind_spots": [
                    {"file": "error_handlers.py", "lines": [45, 67, 89]},
                    {"file": "edge_cases.py", "lines": [23, 34]},
                ],
                "recommendations": [
                    "Add tests for error handling paths",
                    "Improve edge case coverage",
                ],
                "timestamp": datetime.now().isoformat(),
            }

    StrandsGleanAgent = FallbackStrandsGleanAgent

# Global agent instance
agent = None


def get_agent():
    """Get or create agent instance"""
    global agent
    if agent is None:
        agent = StrandsGleanAgent()
    return agent


def async_command(f):
    """Decorator to run async commands"""

    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """A2A Strands Glean Agent CLI - Code analysis and insight generation"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("path")
@click.option("--language", default="python", help="Programming language")
@click.pass_context
@async_command
async def analyze(ctx, path, language):
    """Analyze code structure and complexity"""
    agent = get_agent()

    try:
        result = await agent.analyze_code_structure(path, language)

        click.echo(f"🔍 Code Analysis: {path}")
        click.echo("=" * 50)
        click.echo(f"Language: {result['language'].title()}")
        click.echo(f"Files Analyzed: {result['files_analyzed']}")
        click.echo(f"Classes Found: {result['classes_found']}")
        click.echo(f"Functions Found: {result['functions_found']}")
        click.echo()
        click.echo("📊 Quality Metrics:")
        click.echo(f"  Complexity Score: {result['complexity_score']:.1f}/10")
        click.echo(f"  Maintainability: {result['maintainability_index']:.1f}/100")
        click.echo(f"  Test Coverage: {result['test_coverage']:.1%}")

        if ctx.obj["verbose"]:
            click.echo(f"\nTimestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error analyzing code: {e}", err=True)


@cli.command()
@click.argument("query")
@click.option("--scope", default="global", help="Search scope (global, local, module)")
@click.pass_context
@async_command
async def search(ctx, query, scope):
    """Search for symbols in codebase"""
    agent = get_agent()

    try:
        result = await agent.search_symbols(query, scope)

        click.echo(f"🔎 Symbol Search: '{query}'")
        click.echo(f"Scope: {scope} | Matches: {result['total_matches']}")
        click.echo()

        for symbol in result["results"]:
            click.echo(f"📍 {symbol['name']} ({symbol['type']})")
            click.echo(f"   File: {symbol['file']}:{symbol['line']}")
            click.echo()

        if ctx.obj["verbose"]:
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error searching symbols: {e}", err=True)


@cli.command()
@click.argument("root_path")
@click.pass_context
@async_command
async def dependencies(ctx, root_path):
    """Generate dependency map"""
    agent = get_agent()

    try:
        result = await agent.generate_dependency_map(root_path)

        click.echo(f"🕸️  Dependency Map: {root_path}")
        click.echo("=" * 50)

        deps = result["dependencies"]
        click.echo(f"Internal Dependencies ({len(deps['internal'])}):")
        for dep in deps["internal"]:
            click.echo(f"  • {dep}")
        click.echo()

        click.echo(f"External Dependencies ({len(deps['external'])}):")
        for dep in deps["external"]:
            click.echo(f"  • {dep}")
        click.echo()

        if deps["circular"]:
            click.echo("⚠️  Circular Dependencies:")
            for dep in deps["circular"]:
                click.echo(f"  • {dep}")
        else:
            click.echo("✅ No circular dependencies found")

        click.echo()
        graph = result["dependency_graph"]
        click.echo(
            f"Graph: {graph['nodes']} nodes, {graph['edges']} edges, {graph['clusters']} clusters"
        )

        if ctx.obj["verbose"]:
            click.echo(f"\nTimestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error generating dependency map: {e}", err=True)


@cli.command()
@click.argument("target_path")
@click.pass_context
@async_command
async def coverage(ctx, target_path):
    """Validate test coverage and identify blind spots"""
    agent = get_agent()

    try:
        result = await agent.validate_coverage(target_path)

        click.echo(f"🎯 Coverage Validation: {target_path}")
        click.echo("=" * 50)

        metrics = result["coverage_metrics"]
        click.echo("📊 Coverage Metrics:")
        click.echo(f"  Line Coverage: {metrics['line_coverage']:.1%}")
        click.echo(f"  Branch Coverage: {metrics['branch_coverage']:.1%}")
        click.echo(f"  Function Coverage: {metrics['function_coverage']:.1%}")
        click.echo()

        if result["blind_spots"]:
            click.echo("🔴 Blind Spots Found:")
            for spot in result["blind_spots"]:
                lines_str = ", ".join(map(str, spot["lines"]))
                click.echo(f"  • {spot['file']}: lines {lines_str}")
        else:
            click.echo("✅ No blind spots detected")

        click.echo()
        click.echo("💡 Recommendations:")
        for rec in result["recommendations"]:
            click.echo(f"  • {rec}")

        if ctx.obj["verbose"]:
            click.echo(f"\nTimestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error validating coverage: {e}", err=True)


@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    agent = get_agent()

    click.echo("🔧 Strands Glean Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")


@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    agent = get_agent()

    click.echo("🏥 Strands Glean Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    cli()
