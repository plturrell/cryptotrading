#!/usr/bin/env python3
"""
Vercel Glean CLI - Real Glean implementation for Vercel deployment
Uses SCIP indexing and Angle queries without Docker dependencies
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from cryptotrading.infrastructure.analysis.angle_parser import PYTHON_QUERIES, create_query
from cryptotrading.infrastructure.analysis.vercel_glean_client import VercelGleanClient

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--project-root", "-p", default=str(project_root), help="Project root directory")
@click.pass_context
def cli(ctx, verbose, project_root):
    """Vercel Glean CLI - Real Facebook Glean concepts for Vercel deployment"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ctx.ensure_object(dict)
    ctx.obj["project_root"] = project_root

    click.echo("🔍 Vercel Glean CLI")
    click.echo("Real Glean implementation using SCIP indexing")


@cli.command("index")
@click.option("--unit", "-u", default="default", help="Unit name for indexing")
@click.option("--force", "-f", is_flag=True, help="Force reindexing")
@click.pass_context
def index_cmd(ctx, unit, force):
    """Index the project using SCIP"""

    async def index_project():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            click.echo(f"📦 Indexing project for unit '{unit}'...")

            result = await client.index_project(unit, force_reindex=force)

            if result["status"] == "success":
                click.echo("✅ Indexing completed successfully!")
                click.echo(f"📊 Files indexed: {result['stats']['files_indexed']}")
                click.echo(f"🔗 Symbols found: {result['stats']['total_symbols']}")
                click.echo(f"📍 Occurrences: {result['stats']['total_occurrences']}")
                click.echo(f"💾 Facts stored: {result['facts_stored']}")
            elif result["status"] == "already_indexed":
                click.echo(f"ℹ️  Unit '{unit}' already indexed (use --force to reindex)")
            else:
                click.echo(f"❌ Indexing failed: {result.get('error', 'Unknown error')}")

    asyncio.run(index_project())


@cli.command("query")
@click.argument("angle_query")
@click.option("--format", "-f", default="json", type=click.Choice(["json", "table", "compact"]))
@click.pass_context
def query_cmd(ctx, angle_query, format):
    """Execute an Angle query

    Examples:
        vercel_glean_cli.py query 'python.Declaration { kind = "function", name = ?name }'
        vercel_glean_cli.py query 'python.Reference { target = "MyClass" }'
    """

    async def execute_query():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            click.echo(f"🔍 Executing query: {angle_query}")

            results = await client.query(angle_query)

            if format == "json":
                click.echo(json.dumps(results, indent=2))
            elif format == "table":
                _display_table_format(results)
            else:  # compact
                _display_compact_format(results)

    asyncio.run(execute_query())


@cli.command("functions")
@click.option("--name", "-n", help="Specific function name to find")
@click.option("--file", "-f", help="File to search in")
@click.pass_context
def functions_cmd(ctx, name, file):
    """Find function definitions"""

    async def find_functions():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            if name:
                results = await client.find_function_definitions(name)
                click.echo(f"🔍 Functions named '{name}':")
            else:
                results = await client.find_function_definitions()
                click.echo("🔍 All functions:")

            for result in results:
                key = result.get("key", {})
                value = result.get("value", {})
                click.echo(f"  📁 {key.get('file', 'unknown')}")
                click.echo(f"  ⚡ {key.get('name', 'unnamed')} ({value.get('kind', 'unknown')})")
                if value.get("signature"):
                    click.echo(f"     {value['signature']}")
                click.echo()

    asyncio.run(find_functions())


@cli.command("classes")
@click.option("--name", "-n", help="Specific class name to find")
@click.pass_context
def classes_cmd(ctx, name):
    """Find class definitions"""

    async def find_classes():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            if name:
                results = await client.find_class_definitions(name)
                click.echo(f"🔍 Classes named '{name}':")
            else:
                results = await client.find_class_definitions()
                click.echo("🔍 All classes:")

            for result in results:
                key = result.get("key", {})
                value = result.get("value", {})
                click.echo(f"  📁 {key.get('file', 'unknown')}")
                click.echo(f"  🏗️  {key.get('name', 'unnamed')} ({value.get('kind', 'unknown')})")
                if value.get("documentation"):
                    click.echo(f"     {value['documentation'][:100]}...")
                click.echo()

    asyncio.run(find_classes())


@cli.command("references")
@click.argument("symbol")
@click.pass_context
def references_cmd(ctx, symbol):
    """Find all references to a symbol"""

    async def find_refs():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            click.echo(f"🔍 Finding references to '{symbol}'...")

            results = await client.find_references(symbol)

            if not results:
                click.echo("No references found")
                return

            click.echo(f"Found {len(results)} references:")
            for result in results:
                key = result.get("key", {})
                click.echo(f"  📁 {key.get('file', 'unknown')}")
                if key.get("span"):
                    span = key["span"]
                    click.echo(f"     Line {span[0] + 1}, Column {span[1] + 1}")
                click.echo()

    asyncio.run(find_refs())


@cli.command("deps")
@click.argument("module")
@click.option("--depth", "-d", default=3, help="Maximum dependency depth")
@click.pass_context
def deps_cmd(ctx, module, depth):
    """Analyze module dependencies"""

    async def analyze_deps():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            click.echo(f"🔍 Analyzing dependencies for '{module}' (depth {depth})...")

            result = await client.analyze_dependencies(module, depth)

            if "error" in result:
                click.echo(f"❌ Error: {result['error']}")
                return

            deps = result["dependencies"]
            click.echo(f"📦 Module: {result['module']}")
            click.echo(f"🔗 Total dependencies: {result['total_dependencies']}")
            click.echo()

            if deps["direct"]:
                click.echo("📋 Direct dependencies:")
                for dep in deps["direct"]:
                    click.echo(f"  • {dep}")
                click.echo()

            if deps["transitive"]:
                click.echo("🔄 Transitive dependencies:")
                for dep in deps["transitive"]:
                    depth_level = deps["depth_map"].get(dep, 0)
                    click.echo(f"  {'  ' * depth_level}• {dep} (depth {depth_level})")

    asyncio.run(analyze_deps())


@cli.command("arch")
@click.option("--rules", "-r", help="Custom rules JSON file")
@click.pass_context
def arch_cmd(ctx, rules):
    """Validate architectural constraints"""

    async def validate_arch():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            click.echo("🏗️  Validating architecture...")

            rules_dict = None
            if rules:
                with open(rules, "r") as f:
                    rules_dict = json.load(f)

            result = await client.validate_architecture(rules_dict)

            if result["status"] == "error":
                click.echo(f"❌ Validation failed: {result['error']}")
                return

            violations = result["violations"]

            if not violations:
                click.echo("✅ No architectural violations found!")
            else:
                click.echo(f"⚠️  Found {len(violations)} violations:")

                for violation in violations:
                    severity = violation["severity"]
                    emoji = "🔴" if severity == "high" else "🟡" if severity == "medium" else "⚪"

                    click.echo(f"\n{emoji} {violation['type'].upper()} ({severity})")
                    click.echo(f"  Source: {violation['source']}")
                    click.echo(f"  Target: {violation['target']}")
                    if violation.get("rule", {}).get("message"):
                        click.echo(f"  Reason: {violation['rule']['message']}")

    asyncio.run(validate_arch())


@cli.command("stats")
@click.pass_context
def stats_cmd(ctx):
    """Show storage statistics"""

    async def show_stats():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            stats = await client.get_storage_stats()

            click.echo("📊 Glean Storage Statistics")
            click.echo("=" * 30)
            click.echo(f"Total facts: {stats['total_facts']}")
            click.echo(f"Predicates: {stats['total_predicates']}")
            click.echo(f"Storage size: {stats['storage_size_mb']} MB")
            click.echo()

            if stats["predicate_counts"]:
                click.echo("📋 Facts by predicate:")
                for predicate, count in list(stats["predicate_counts"].items())[:10]:
                    click.echo(f"  {predicate}: {count}")

                if len(stats["predicate_counts"]) > 10:
                    click.echo(f"  ... and {len(stats['predicate_counts']) - 10} more")

    asyncio.run(show_stats())


@cli.command("export")
@click.argument("unit")
@click.argument("output_file")
@click.pass_context
def export_cmd(ctx, unit, output_file):
    """Export unit facts to JSON"""

    async def export_unit():
        async with VercelGleanClient(ctx.obj["project_root"]) as client:
            click.echo(f"📤 Exporting unit '{unit}' to {output_file}...")

            await client.export_unit(unit, output_file)
            click.echo("✅ Export completed!")

    asyncio.run(export_unit())


def _display_table_format(results):
    """Display results in table format"""
    if not results:
        click.echo("No results found")
        return

    click.echo("Results:")
    click.echo("-" * 50)

    for i, result in enumerate(results, 1):
        click.echo(f"{i}. Predicate: {result.get('predicate', 'unknown')}")

        if result.get("key"):
            click.echo(f"   Key: {json.dumps(result['key'], indent=6)}")

        if result.get("value"):
            click.echo(f"   Value: {json.dumps(result['value'], indent=8)}")

        click.echo()


def _display_compact_format(results):
    """Display results in compact format"""
    click.echo(f"Found {len(results)} results")

    for result in results:
        predicate = result.get("predicate", "unknown")
        key = result.get("key", {})

        if "name" in key and "file" in key:
            click.echo(f"{predicate}: {key['name']} in {key['file']}")
        elif "file" in key:
            click.echo(f"{predicate}: {key['file']}")
        else:
            click.echo(f"{predicate}: {json.dumps(key)}")


if __name__ == "__main__":
    cli()
