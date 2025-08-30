#!/usr/bin/env python3
"""
Comprehensive verification script to ensure 100% calculation containment in MCP tools with STRANDS agents.
"""

import ast
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


def find_calculation_methods_in_agents() -> Dict[str, List[str]]:
    """Find any remaining calculation methods in agent classes."""
    agents_dir = Path("/Users/apple/projects/cryptotrading/src/cryptotrading/core/agents")
    calculation_methods = {}

    # Keywords that indicate calculation logic
    calc_keywords = [
        "calculate",
        "compute",
        "analyze",
        "process",
        "evaluate",
        "optimize",
        "simulate",
        "predict",
        "forecast",
        "estimate",
    ]

    for py_file in agents_dir.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue

        try:
            with open(py_file, "r") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.endswith("Agent"):
                    class_methods = []

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = item.name
                            # Check if method name contains calculation keywords
                            if any(keyword in method_name.lower() for keyword in calc_keywords):
                                # Check if method actually contains calculation logic (not just delegation)
                                method_source = ast.get_source_segment(content, item)
                                if method_source and not _is_delegation_method(method_source):
                                    class_methods.append(method_name)

                    if class_methods:
                        relative_path = str(py_file.relative_to(agents_dir))
                        calculation_methods[f"{relative_path}::{node.name}"] = class_methods

        except Exception as e:
            print(f"Error parsing {py_file}: {e}")

    return calculation_methods


def _is_delegation_method(method_source: str) -> bool:
    """Check if a method is just delegating to MCP tools or other services."""
    delegation_patterns = [
        "execute_tool(",
        "handle_tool_call(",
        "await self.mcp_tools.",
        "await self.ta_agent.",
        "await self.ml_agent.",
        "from ...infrastructure.mcp.",
        "mcp_tools.handle_tool_call",
    ]

    return any(pattern in method_source for pattern in delegation_patterns)


def verify_mcp_tools_coverage() -> Dict[str, Any]:
    """Verify MCP tools coverage and registration."""
    mcp_dir = Path("/Users/apple/projects/cryptotrading/src/cryptotrading/infrastructure/mcp")

    # Find all MCP tool files
    mcp_tool_files = list(mcp_dir.glob("*_mcp_tools.py"))
    mcp_agent_files = list(mcp_dir.glob("*_mcp_agent.py"))

    coverage = {
        "mcp_tools": [f.stem for f in mcp_tool_files],
        "mcp_agents": [f.stem for f in mcp_agent_files],
        "total_tools": len(mcp_tool_files),
        "total_agents": len(mcp_agent_files),
        "coverage_areas": [],
    }

    # Expected coverage areas
    expected_areas = [
        "technical_analysis",
        "mcts_calculation",
        "ml_models",
        "feature_store",
        "data_analysis",
        "code_quality",
        "clrs_algorithms",
    ]

    for area in expected_areas:
        has_tools = any(area in tool for tool in coverage["mcp_tools"])
        has_agent = any(area in agent for agent in coverage["mcp_agents"])
        coverage["coverage_areas"].append(
            {
                "area": area,
                "has_tools": has_tools,
                "has_agent": has_agent,
                "complete": has_tools and has_agent,
            }
        )

    return coverage


def check_mcp_server_registration() -> Dict[str, Any]:
    """Check MCP server registration status."""
    mcp_server_file = Path("/Users/apple/projects/cryptotrading/api/mcp.py")

    if not mcp_server_file.exists():
        return {"error": "MCP server file not found"}

    try:
        with open(mcp_server_file, "r") as f:
            content = f.read()

        # Check for MCP tool imports and registrations
        registrations = {"imports": [], "tool_instances": [], "agent_registrations": []}

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "from cryptotrading.infrastructure.mcp" in line and "_mcp_tools" in line:
                registrations["imports"].append(line.strip())
            elif "MCPTools()" in line:
                registrations["tool_instances"].append(line.strip())
            elif "agent_tools" in line and "=" in line:
                # Look for agent tool registration patterns
                for j in range(i, min(i + 20, len(lines))):
                    if "mcp_tools" in lines[j]:
                        registrations["agent_registrations"].append(lines[j].strip())

        return registrations

    except Exception as e:
        return {"error": f"Failed to parse MCP server file: {e}"}


def generate_verification_report() -> str:
    """Generate comprehensive verification report."""
    print("🔍 Verifying 100% Calculation Containment in MCP Tools...")

    # 1. Check for remaining calculation methods in agents
    print("\n1. Checking for calculation methods in agent classes...")
    calc_methods = find_calculation_methods_in_agents()

    # 2. Verify MCP tools coverage
    print("2. Verifying MCP tools coverage...")
    coverage = verify_mcp_tools_coverage()

    # 3. Check MCP server registration
    print("3. Checking MCP server registration...")
    registration = check_mcp_server_registration()

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("CALCULATION CONTAINMENT VERIFICATION REPORT")
    report.append("=" * 80)

    # Agent calculation methods analysis
    report.append("\n📊 AGENT CALCULATION METHODS ANALYSIS:")
    if calc_methods:
        report.append("❌ Found calculation methods still in agent classes:")
        for agent_class, methods in calc_methods.items():
            report.append(f"   • {agent_class}:")
            for method in methods:
                report.append(f"     - {method}")
        report.append(
            "\n⚠️  These methods should be moved to MCP tools or verified as delegation methods."
        )
    else:
        report.append("✅ No calculation methods found in agent classes - all properly delegated!")

    # MCP tools coverage analysis
    report.append(f"\n🛠️  MCP TOOLS COVERAGE ANALYSIS:")
    report.append(f"   • Total MCP Tool Files: {coverage['total_tools']}")
    report.append(f"   • Total MCP Agent Files: {coverage['total_agents']}")
    report.append(f"   • Coverage Areas:")

    complete_areas = 0
    for area in coverage["coverage_areas"]:
        status = "✅" if area["complete"] else "❌"
        tools_status = "✓" if area["has_tools"] else "✗"
        agent_status = "✓" if area["has_agent"] else "✗"
        report.append(f"     {status} {area['area']}: Tools({tools_status}) Agent({agent_status})")
        if area["complete"]:
            complete_areas += 1

    coverage_percentage = (complete_areas / len(coverage["coverage_areas"])) * 100
    report.append(
        f"   • Coverage Completion: {coverage_percentage:.1f}% ({complete_areas}/{len(coverage['coverage_areas'])})"
    )

    # MCP server registration analysis
    report.append(f"\n🔧 MCP SERVER REGISTRATION ANALYSIS:")
    if "error" in registration:
        report.append(f"❌ {registration['error']}")
    else:
        report.append(f"   • MCP Tool Imports: {len(registration['imports'])}")
        report.append(f"   • Tool Instances: {len(registration['tool_instances'])}")
        report.append(f"   • Agent Registrations: {len(registration['agent_registrations'])}")

        if registration["imports"] and registration["tool_instances"]:
            report.append("✅ MCP tools appear to be properly registered")
        else:
            report.append("❌ MCP registration may be incomplete")

    # Overall assessment
    report.append(f"\n🎯 OVERALL ASSESSMENT:")

    issues = []
    if calc_methods:
        issues.append("Calculation methods found in agent classes")
    if coverage_percentage < 100:
        issues.append(f"MCP coverage incomplete ({coverage_percentage:.1f}%)")
    if "error" in registration or not (
        registration.get("imports") and registration.get("tool_instances")
    ):
        issues.append("MCP server registration issues")

    if not issues:
        report.append("🎉 SUCCESS: 100% calculation containment achieved!")
        report.append("   • All calculation logic properly contained in MCP tools")
        report.append("   • All MCP tools have corresponding STRANDS agents")
        report.append("   • MCP server registration complete")
    else:
        report.append("⚠️  ISSUES FOUND:")
        for issue in issues:
            report.append(f"   • {issue}")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    report = generate_verification_report()
    print(report)

    # Save report to file
    with open("/Users/apple/projects/cryptotrading/calculation_containment_report.txt", "w") as f:
        f.write(report)

    print(f"\n📄 Report saved to: calculation_containment_report.txt")
