"""
Code Quality MCP Tools - All code analysis and quality calculations
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...infrastructure.analysis.code_analyzer import CodeAnalyzer
from ...infrastructure.analysis.code_quality_intelligence import CodeQualityIntelligence
from ...infrastructure.analysis.impact_analyzer import ImpactAnalyzer

logger = logging.getLogger(__name__)


class CodeQualityMCPTools:
    """MCP tools for code quality analysis and calculations"""

    def __init__(self):
        self.quality_intelligence = CodeQualityIntelligence()
        self.code_analyzer = CodeAnalyzer()
        self.impact_analyzer = ImpactAnalyzer()
        self.tools = self._create_tools()

    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions for code quality tools"""
        return [
            {
                "name": "analyze_code_quality",
                "description": "Perform comprehensive code quality analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to file or directory to analyze",
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["full", "quick", "security"],
                            "default": "full",
                        },
                        "include_metrics": {"type": "boolean", "default": True},
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "calculate_complexity_metrics",
                "description": "Calculate code complexity metrics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_code": {"type": "string", "description": "Source code to analyze"},
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript", "typescript"],
                            "default": "python",
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["cyclomatic", "halstead", "maintainability"],
                        },
                    },
                    "required": ["source_code"],
                },
            },
            {
                "name": "detect_code_smells",
                "description": "Detect code smells and anti-patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file to analyze"},
                        "smell_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of smells to detect",
                        },
                        "severity_threshold": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "default": "medium",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "analyze_dependencies",
                "description": "Analyze code dependencies and coupling",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string", "description": "Path to project root"},
                        "dependency_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["imports", "calls", "inheritance"],
                        },
                        "depth_limit": {"type": "integer", "default": 5},
                    },
                    "required": ["project_path"],
                },
            },
            {
                "name": "calculate_impact_analysis",
                "description": "Calculate impact of code changes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "changed_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of changed file paths",
                        },
                        "project_path": {"type": "string", "description": "Path to project root"},
                        "analysis_depth": {
                            "type": "string",
                            "enum": ["shallow", "medium", "deep"],
                            "default": "medium",
                        },
                    },
                    "required": ["changed_files", "project_path"],
                },
            },
            {
                "name": "generate_quality_report",
                "description": "Generate comprehensive quality report",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string", "description": "Path to project root"},
                        "report_format": {
                            "type": "string",
                            "enum": ["json", "html", "markdown"],
                            "default": "json",
                        },
                        "include_recommendations": {"type": "boolean", "default": True},
                    },
                    "required": ["project_path"],
                },
            },
        ]

    def register_tools(self, server):
        """Register all code quality tools with MCP server"""
        for tool_def in self.tools:
            tool_name = tool_def["name"]

            @server.call_tool()
            async def handle_tool(name: str, arguments: dict) -> dict:
                if name == tool_name:
                    return await self.handle_tool_call(tool_name, arguments)
                return {"error": f"Unknown tool: {name}"}

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls for code quality"""
        try:
            if tool_name == "analyze_code_quality":
                return await self._handle_analyze_code_quality(arguments)
            elif tool_name == "calculate_complexity_metrics":
                return await self._handle_calculate_complexity_metrics(arguments)
            elif tool_name == "detect_code_smells":
                return await self._handle_detect_code_smells(arguments)
            elif tool_name == "analyze_dependencies":
                return await self._handle_analyze_dependencies(arguments)
            elif tool_name == "calculate_impact_analysis":
                return await self._handle_calculate_impact_analysis(arguments)
            elif tool_name == "generate_quality_report":
                return await self._handle_generate_quality_report(arguments)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error in code quality tool {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_analyze_code_quality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code quality analysis requests"""
        try:
            file_path = args["file_path"]
            analysis_type = args.get("analysis_type", "full")
            include_metrics = args.get("include_metrics", True)

            analysis_results = await self.quality_intelligence.analyze_quality(
                file_path, analysis_type, include_metrics
            )

            return {
                "success": True,
                "quality_analysis": analysis_results,
                "file_path": file_path,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_calculate_complexity_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complexity metrics calculation"""
        try:
            source_code = args["source_code"]
            language = args.get("language", "python")
            metrics = args.get("metrics", ["cyclomatic", "halstead", "maintainability"])

            complexity_results = await self.code_analyzer.calculate_complexity(
                source_code, language, metrics
            )

            return {
                "success": True,
                "complexity_metrics": complexity_results,
                "language": language,
                "metrics_calculated": metrics,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_detect_code_smells(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code smell detection"""
        try:
            file_path = args["file_path"]
            smell_types = args.get("smell_types", [])
            severity_threshold = args.get("severity_threshold", "medium")

            smell_results = await self.code_analyzer.detect_smells(
                file_path, smell_types, severity_threshold
            )

            return {
                "success": True,
                "code_smells": smell_results,
                "file_path": file_path,
                "severity_threshold": severity_threshold,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_analyze_dependencies(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dependency analysis"""
        try:
            project_path = args["project_path"]
            dependency_types = args.get("dependency_types", ["imports", "calls", "inheritance"])
            depth_limit = args.get("depth_limit", 5)

            dependency_results = await self.code_analyzer.analyze_dependencies(
                project_path, dependency_types, depth_limit
            )

            return {
                "success": True,
                "dependency_analysis": dependency_results,
                "project_path": project_path,
                "dependency_types": dependency_types,
                "depth_limit": depth_limit,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_calculate_impact_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle impact analysis calculation"""
        try:
            changed_files = args["changed_files"]
            project_path = args["project_path"]
            analysis_depth = args.get("analysis_depth", "medium")

            impact_results = await self.impact_analyzer.calculate_impact(
                changed_files, project_path, analysis_depth
            )

            return {
                "success": True,
                "impact_analysis": impact_results,
                "changed_files": changed_files,
                "project_path": project_path,
                "analysis_depth": analysis_depth,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_generate_quality_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quality report generation"""
        try:
            project_path = args["project_path"]
            report_format = args.get("report_format", "json")
            include_recommendations = args.get("include_recommendations", True)

            report_results = await self.quality_intelligence.generate_report(
                project_path, report_format, include_recommendations
            )

            return {
                "success": True,
                "quality_report": report_results,
                "project_path": project_path,
                "report_format": report_format,
                "include_recommendations": include_recommendations,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
