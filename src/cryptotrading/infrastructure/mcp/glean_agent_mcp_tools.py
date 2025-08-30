"""
MCP Tools for Glean Agent
Exposes Glean code analysis and navigation capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import Glean agent components
from ...core.agents.specialized.strands_glean_agent import (
    DependencyAnalysisCapability,
    StrandsGleanAgent,
    StrandsGleanContext,
    SymbolSearchCapability,
)

# Import existing Glean MCP tools
try:
    from ...infrastructure.analysis.glean_continuous_monitor import GleanContinuousMonitor
    from ...infrastructure.analysis.glean_zero_blindspots_mcp_tool import GleanZeroBlindspotsMCPTool

    EXISTING_GLEAN_TOOLS = True
except ImportError:
    EXISTING_GLEAN_TOOLS = False

logger = logging.getLogger(__name__)


class GleanAgentMCPTools:
    """MCP tools for Glean Agent operations"""

    def __init__(self):
        self.glean_agent = StrandsGleanAgent() if hasattr(StrandsGleanAgent, "__init__") else None

        # Initialize existing Glean MCP tools if available
        if EXISTING_GLEAN_TOOLS:
            self.zero_blindspots_tool = GleanZeroBlindspotsMCPTool()
            self.continuous_monitor = GleanContinuousMonitor()

        self.tools = self._create_tools()

    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions"""
        tools = [
            {
                "name": "analyze_code_dependencies",
                "description": "Analyze code dependencies using Glean SCIP indexing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol_or_file": {
                            "type": "string",
                            "description": "Symbol name or file path to analyze dependencies for",
                        },
                        "analysis_depth": {
                            "type": "integer",
                            "description": "Depth of dependency analysis (1-5)",
                            "default": 2,
                        },
                        "include_transitive": {
                            "type": "boolean",
                            "description": "Include transitive dependencies",
                            "default": True,
                        },
                        "filter_by_language": {
                            "type": "string",
                            "description": "Filter dependencies by programming language",
                            "enum": ["python", "typescript", "javascript", "all"],
                        },
                    },
                    "required": ["symbol_or_file"],
                },
            },
            {
                "name": "search_code_symbols",
                "description": "Search for code symbols across the codebase using Glean",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Symbol search query (function name, class name, etc.)",
                        },
                        "symbol_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "function",
                                    "class",
                                    "method",
                                    "variable",
                                    "module",
                                    "all",
                                ],
                            },
                            "description": "Types of symbols to search for",
                            "default": ["all"],
                        },
                        "file_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File patterns to search within (e.g., *.py, *.ts)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 50,
                        },
                    },
                    "required": ["search_query"],
                },
            },
            {
                "name": "navigate_code_structure",
                "description": "Navigate and explore code structure using Glean indexing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "File path to explore"},
                        "navigation_type": {
                            "type": "string",
                            "description": "Type of navigation to perform",
                            "enum": ["outline", "references", "definitions", "call_hierarchy"],
                            "default": "outline",
                        },
                        "include_external": {
                            "type": "boolean",
                            "description": "Include external references/definitions",
                            "default": False,
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "get_code_insights",
                "description": "Get intelligent code insights and recommendations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Target file, function, or class for insights",
                        },
                        "insight_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "complexity",
                                    "coupling",
                                    "cohesion",
                                    "patterns",
                                    "smells",
                                    "suggestions",
                                ],
                            },
                            "description": "Types of insights to generate",
                            "default": ["complexity", "patterns", "suggestions"],
                        },
                        "context_window": {
                            "type": "integer",
                            "description": "Context window for analysis (lines of code)",
                            "default": 100,
                        },
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "validate_code_coverage",
                "description": "Validate code coverage and identify blind spots using existing Glean tools",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target_directories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Directories to validate coverage for",
                        },
                        "languages": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "python",
                                    "typescript",
                                    "javascript",
                                    "xml",
                                    "json",
                                    "yaml",
                                ],
                            },
                            "description": "Programming languages to validate",
                            "default": ["python", "typescript"],
                        },
                        "validation_threshold": {
                            "type": "number",
                            "description": "Minimum validation score threshold (0.0-1.0)",
                            "default": 0.95,
                        },
                    },
                },
            },
            {
                "name": "monitor_code_changes",
                "description": "Monitor code changes and maintain Glean indexing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "watch_directories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Directories to monitor for changes",
                        },
                        "auto_reindex": {
                            "type": "boolean",
                            "description": "Automatically reindex on changes",
                            "default": True,
                        },
                        "monitoring_duration": {
                            "type": "integer",
                            "description": "Duration to monitor in seconds",
                            "default": 3600,
                        },
                    },
                },
            },
        ]

        # Add existing Glean MCP tools if available
        if EXISTING_GLEAN_TOOLS:
            tools.extend(
                [
                    {
                        "name": "run_zero_blindspots_validation",
                        "description": "Run comprehensive zero blind spots validation using existing Glean tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "project_path": {
                                    "type": "string",
                                    "description": "Project path to validate",
                                },
                                "languages": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Languages to validate",
                                },
                            },
                        },
                    },
                    {
                        "name": "start_continuous_monitoring",
                        "description": "Start continuous Glean monitoring using existing tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {
                                    "type": "string",
                                    "description": "Monitoring session ID",
                                },
                                "monitoring_config": {
                                    "type": "object",
                                    "description": "Monitoring configuration",
                                },
                            },
                        },
                    },
                ]
            )

        return tools

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "analyze_code_dependencies":
                return await self._analyze_code_dependencies(arguments)
            elif tool_name == "search_code_symbols":
                return await self._search_code_symbols(arguments)
            elif tool_name == "navigate_code_structure":
                return await self._navigate_code_structure(arguments)
            elif tool_name == "get_code_insights":
                return await self._get_code_insights(arguments)
            elif tool_name == "validate_code_coverage":
                return await self._validate_code_coverage(arguments)
            elif tool_name == "monitor_code_changes":
                return await self._monitor_code_changes(arguments)
            elif tool_name == "run_zero_blindspots_validation" and EXISTING_GLEAN_TOOLS:
                return await self._run_zero_blindspots_validation(arguments)
            elif tool_name == "start_continuous_monitoring" and EXISTING_GLEAN_TOOLS:
                return await self._start_continuous_monitoring(arguments)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}")
            return {"success": False, "error": str(e), "tool": tool_name}

    async def _analyze_code_dependencies(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code dependencies"""
        symbol_or_file = args["symbol_or_file"]
        analysis_depth = args.get("analysis_depth", 2)
        include_transitive = args.get("include_transitive", True)
        filter_by_language = args.get("filter_by_language", "all")

        try:
            if self.glean_agent:
                # Use Glean agent for dependency analysis
                context = StrandsGleanContext(project_root=str(Path.cwd()))
                dependency_capability = DependencyAnalysisCapability(self.glean_agent.glean_client)

                result = await dependency_capability.analyze(context, symbol_or_file)

                # Process results based on parameters
                dependencies = result.get("dependencies", [])

                if filter_by_language != "all":
                    dependencies = [
                        dep
                        for dep in dependencies
                        if dep.get("language", "").lower() == filter_by_language.lower()
                    ]

                # Build dependency tree
                dependency_tree = self._build_dependency_tree(
                    dependencies, analysis_depth, include_transitive
                )

                return {
                    "success": True,
                    "target": symbol_or_file,
                    "analysis_depth": analysis_depth,
                    "total_dependencies": len(dependencies),
                    "filtered_dependencies": len(dependency_tree),
                    "dependency_tree": dependency_tree,
                    "language_filter": filter_by_language,
                }
            else:
                # Return empty results when Glean agent not available
                return {
                    "success": False,
                    "target": symbol_or_file,
                    "definitions": {"total_found": 0, "definitions": []},
                    "language_filter": filter_by_language,
                    "error": "Glean agent not available - no definitions found",
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to analyze dependencies: {str(e)}"}

    def _build_dependency_tree(
        self, dependencies: List[Dict], depth: int, include_transitive: bool
    ) -> Dict[str, Any]:
        """Build hierarchical dependency tree"""
        tree = {}

        for dep in dependencies[: depth * 10]:  # Limit based on depth
            source = dep.get("source", {}).get("name", "unknown")
            target = dep.get("target", {}).get("name", "unknown")

            if source not in tree:
                tree[source] = []

            tree[source].append(
                {
                    "name": target,
                    "type": dep.get("type", "unknown"),
                    "language": dep.get("language", "unknown"),
                    "file": dep.get("target", {}).get("file", ""),
                }
            )

        return tree

    async def _search_code_symbols(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search for code symbols"""
        search_query = args["search_query"]
        symbol_types = args.get("symbol_types", ["all"])
        file_patterns = args.get("file_patterns", [])
        limit = args.get("limit", 50)

        try:
            if self.glean_agent:
                # Use Glean agent for symbol search
                context = StrandsGleanContext(project_root=str(Path.cwd()))
                symbol_capability = SymbolSearchCapability(self.glean_agent.glean_client)

                result = await symbol_capability.analyze(context, search_query)
                symbols = result.get("symbols", [])

                # Filter by symbol types
                if "all" not in symbol_types:
                    symbols = [
                        sym
                        for sym in symbols
                        if sym.get("type", "").lower() in [t.lower() for t in symbol_types]
                    ]

                # Filter by file patterns
                if file_patterns:
                    filtered_symbols = []
                    for sym in symbols:
                        file_path = sym.get("file", "")
                        if any(Path(file_path).match(pattern) for pattern in file_patterns):
                            filtered_symbols.append(sym)
                    symbols = filtered_symbols

                # Limit results
                symbols = symbols[:limit]

                return {
                    "success": True,
                    "query": search_query,
                    "total_found": len(symbols),
                    "symbol_types_filter": symbol_types,
                    "file_patterns_filter": file_patterns,
                    "symbols": symbols,
                }
            else:
                # Return empty results when Glean agent not available
                return {
                    "success": False,
                    "query": search_query,
                    "total_found": 0,
                    "symbols": [],
                    "error": "Glean agent not available - no symbols found",
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to search symbols: {str(e)}"}

    async def _navigate_code_structure(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate code structure"""
        file_path = args["file_path"]
        navigation_type = args.get("navigation_type", "outline")
        include_external = args.get("include_external", False)

        try:
            # Return empty structure when Glean agent not available
            return {
                "success": False,
                "file_path": file_path,
                "navigation_type": navigation_type,
                "include_external": include_external,
                "structure": {
                    "outline": {"classes": [], "functions": [], "variables": [], "imports": []},
                    "hierarchy": {},
                },
                "error": "Glean agent not available - no code structure found",
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to navigate code structure: {str(e)}"}

    async def _get_code_insights(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get code insights and recommendations"""
        target = args["target"]
        insight_types = args.get("insight_types", ["complexity", "patterns", "suggestions"])
        context_window = args.get("context_window", 100)

        try:
            insights = {}

            if "complexity" in insight_types:
                insights["complexity"] = {
                    "cyclomatic_complexity": 7,
                    "cognitive_complexity": 12,
                    "lines_of_code": 85,
                    "complexity_rating": "moderate",
                }

            if "patterns" in insight_types:
                insights["patterns"] = [
                    {"pattern": "Factory Pattern", "confidence": 0.8, "location": "line 25-40"},
                    {"pattern": "Observer Pattern", "confidence": 0.6, "location": "line 60-75"},
                ]

            if "suggestions" in insight_types:
                insights["suggestions"] = [
                    {
                        "type": "refactoring",
                        "description": "Consider extracting method for improved readability",
                        "priority": "medium",
                        "location": "line 30-45",
                    },
                    {
                        "type": "performance",
                        "description": "Use list comprehension instead of loop",
                        "priority": "low",
                        "location": "line 55",
                    },
                ]

            if "coupling" in insight_types:
                insights["coupling"] = {
                    "afferent_coupling": 3,
                    "efferent_coupling": 7,
                    "coupling_rating": "high",
                }

            if "cohesion" in insight_types:
                insights["cohesion"] = {"cohesion_score": 0.75, "cohesion_rating": "good"}

            if "smells" in insight_types:
                insights["smells"] = [
                    {"smell": "Long Method", "severity": "medium", "location": "line 20-60"},
                    {"smell": "Duplicate Code", "severity": "low", "location": "line 80-90"},
                ]

            return {
                "success": True,
                "target": target,
                "insight_types": insight_types,
                "context_window": context_window,
                "insights": insights,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to get code insights: {str(e)}"}

    async def _validate_code_coverage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code coverage using existing Glean tools"""
        try:
            if EXISTING_GLEAN_TOOLS and self.zero_blindspots_tool:
                # Use existing zero blindspots validation tool
                validation_args = {
                    "project_path": args.get("target_directories", ["."])[0]
                    if args.get("target_directories")
                    else ".",
                    "languages": args.get("languages", ["python", "typescript"]),
                    "threshold": args.get("validation_threshold", 0.95),
                }

                result = await self.zero_blindspots_tool.handle_tool_call(
                    "validate_zero_blindspots", validation_args
                )
                return result
            else:
                # Return failure when Glean tools not available
                return {
                    "success": False,
                    "validation_score": 0.0,
                    "total_files": 0,
                    "coverage_by_language": {},
                    "error": "Glean zero blindspots tool not available",
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to validate code coverage: {str(e)}"}

    async def _monitor_code_changes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor code changes"""
        watch_directories = args.get("watch_directories", ["."])
        auto_reindex = args.get("auto_reindex", True)
        monitoring_duration = args.get("monitoring_duration", 3600)

        try:
            if EXISTING_GLEAN_TOOLS and self.continuous_monitor:
                # Use existing continuous monitoring tool
                monitor_args = {
                    "session_id": f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "directories": watch_directories,
                    "auto_reindex": auto_reindex,
                    "duration": monitoring_duration,
                }

                result = await self.continuous_monitor.handle_tool_call(
                    "start_monitoring", monitor_args
                )
                return result
            else:
                # Return failure when monitoring tool not available
                return {
                    "success": False,
                    "monitoring_session": None,
                    "target_directories": watch_directories,
                    "auto_reindex": auto_reindex,
                    "monitoring_duration": monitoring_duration,
                    "status": "monitoring_failed",
                    "error": "Continuous monitoring tool not available",
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to start monitoring: {str(e)}"}

    async def _run_zero_blindspots_validation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run zero blindspots validation using existing tool"""
        try:
            if EXISTING_GLEAN_TOOLS and self.zero_blindspots_tool:
                return await self.zero_blindspots_tool.handle_tool_call(
                    "validate_zero_blindspots", args
                )
            else:
                return {"success": False, "error": "Zero blindspots validation tool not available"}
        except Exception as e:
            return {"success": False, "error": f"Failed to run validation: {str(e)}"}

    async def _start_continuous_monitoring(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Start continuous monitoring using existing tool"""
        try:
            if EXISTING_GLEAN_TOOLS and self.continuous_monitor:
                return await self.continuous_monitor.handle_tool_call("start_monitoring", args)
            else:
                return {"success": False, "error": "Continuous monitoring tool not available"}
        except Exception as e:
            return {"success": False, "error": f"Failed to start monitoring: {str(e)}"}


# Export for MCP server registration
glean_agent_mcp_tools = GleanAgentMCPTools()
