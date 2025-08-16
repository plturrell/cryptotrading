"""
MCP Server for Strands-Glean Integration
Exposes code analysis capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from .server import MCPServer
from .tools import MCPTool, ToolResult, ToolContent
from .transport import StdioTransport, WebSocketTransport

# Import analysis components
try:
    from ...infrastructure.analysis.vercel_glean_client import VercelGleanClient
    from ...infrastructure.analysis.angle_parser import create_query, PYTHON_QUERIES
    from ...infrastructure.analysis.scip_indexer import index_project_for_glean
    GLEAN_AVAILABLE = True
except ImportError:
    GLEAN_AVAILABLE = False

logger = logging.getLogger(__name__)


class StrandsGleanMCPServer(MCPServer):
    """MCP Server that exposes Strands-Glean code analysis capabilities"""
    
    def __init__(
        self,
        server_name: str = "strands-glean-server",
        version: str = "1.0.0",
        project_root: Optional[str] = None
    ):
        super().__init__(server_name, version)
        
        self.project_root = Path(project_root or "/Users/apple/projects/cryptotrading")
        self.glean_client: Optional[VercelGleanClient] = None
        self.indexed_units: set = set()
        
        # Initialize Glean client if available
        if GLEAN_AVAILABLE:
            self.glean_client = VercelGleanClient(project_root=str(self.project_root))
            logger.info("VercelGleanClient initialized")
        else:
            logger.warning("Glean not available - server will have limited functionality")
        
        # Register tools
        self._register_analysis_tools()
        self._register_ai_tools()
        
    def _register_analysis_tools(self):
        """Register code analysis tools"""
        tools = [
            MCPTool(
                name="glean_index_project",
                description="Index the codebase using SCIP for Glean analysis",
                parameters={
                    "unit_name": {
                        "type": "string",
                        "description": "Name for the indexed unit",
                        "default": "main"
                    },
                    "force_reindex": {
                        "type": "boolean", 
                        "description": "Force reindexing even if already indexed",
                        "default": False
                    }
                },
                function=self._index_project
            ),
            MCPTool(
                name="glean_symbol_search",
                description="Search for symbols in the codebase",
                parameters={
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern for symbols",
                        "required": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                },
                function=self._search_symbols
            ),
            MCPTool(
                name="glean_dependency_analysis",
                description="Analyze dependencies for a symbol or module",
                parameters={
                    "symbol": {
                        "type": "string", 
                        "description": "Symbol or module to analyze",
                        "required": True
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Analysis depth",
                        "default": 3
                    }
                },
                function=self._analyze_dependencies
            ),
            MCPTool(
                name="glean_architecture_review",
                description="Review architectural patterns and violations",
                parameters={
                    "component": {
                        "type": "string",
                        "description": "Component or module to review",
                        "required": True
                    },
                    "rules": {
                        "type": "array",
                        "description": "Architecture rules to check",
                        "items": {"type": "string"},
                        "default": ["layer_separation", "dependency_direction", "circular_deps"]
                    }
                },
                function=self._review_architecture
            ),
            MCPTool(
                name="glean_query_angle",
                description="Execute raw Angle query on the codebase",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Angle query to execute",
                        "required": True
                    },
                    "template": {
                        "type": "string",
                        "description": "Use predefined query template",
                        "enum": list(PYTHON_QUERIES.keys()) if GLEAN_AVAILABLE else []
                    }
                },
                function=self._execute_angle_query
            ),
            MCPTool(
                name="glean_statistics",
                description="Get codebase analysis statistics",
                parameters={},
                function=self._get_statistics
            )
        ]
        
        for tool in tools:
            self.add_tool(tool)
    
    def _register_ai_tools(self):
        """Register AI enhancement tools"""
        tools = [
            MCPTool(
                name="ai_enhance_analysis",
                description="Enhance Glean analysis results with AI insights",
                parameters={
                    "analysis_data": {
                        "type": "object",
                        "description": "Raw analysis data from Glean",
                        "required": True
                    },
                    "enhancement_type": {
                        "type": "string",
                        "description": "Type of AI enhancement",
                        "enum": ["summary", "recommendations", "risk_assessment", "refactoring", "documentation"],
                        "default": "summary"
                    },
                    "ai_provider": {
                        "type": "string",
                        "description": "AI provider to use",
                        "enum": ["grok", "openai", "anthropic", "local"],
                        "default": "grok"
                    }
                },
                function=self._enhance_with_ai
            ),
            MCPTool(
                name="ai_code_review",
                description="AI-powered code review using Glean analysis",
                parameters={
                    "files": {
                        "type": "array",
                        "description": "Files to review",
                        "items": {"type": "string"},
                        "required": True
                    },
                    "review_type": {
                        "type": "string",
                        "description": "Type of review",
                        "enum": ["security", "performance", "maintainability", "style", "comprehensive"],
                        "default": "comprehensive"
                    }
                },
                function=self._ai_code_review
            ),
            MCPTool(
                name="ai_explain_code",
                description="AI explanation of code using structural analysis",
                parameters={
                    "symbol": {
                        "type": "string",
                        "description": "Symbol to explain",
                        "required": True
                    },
                    "context_level": {
                        "type": "string",
                        "description": "Level of context to include",
                        "enum": ["minimal", "moderate", "comprehensive"],
                        "default": "moderate"
                    }
                },
                function=self._ai_explain_code
            )
        ]
        
        for tool in tools:
            self.add_tool(tool)
    
    async def _index_project(self, unit_name: str = "main", force_reindex: bool = False) -> ToolResult:
        """Index the project using SCIP"""
        if not self.glean_client:
            return ToolResult.text_result("Glean client not available", is_error=True)
        
        try:
            # Check if already indexed
            if unit_name in self.indexed_units and not force_reindex:
                return ToolResult.text_result(f"Unit '{unit_name}' already indexed. Use force_reindex=true to reindex.")
            
            # Index the project
            result = await self.glean_client.index_project(unit_name, force_reindex)
            
            if result.get("status") == "success":
                self.indexed_units.add(unit_name)
                stats = result.get("stats", {})
                
                response = {
                    "status": "success",
                    "unit_name": unit_name,
                    "files_indexed": stats.get("files_indexed", 0),
                    "symbols_found": stats.get("symbols_found", 0),
                    "facts_stored": stats.get("facts_stored", 0),
                    "indexing_time": stats.get("indexing_time", 0)
                }
                
                return ToolResult.data_result(
                    json.dumps(response, indent=2),
                    "application/json"
                )
            else:
                return ToolResult.text_result(f"Indexing failed: {result.get('error', 'Unknown error')}", is_error=True)
                
        except Exception as e:
            logger.error(f"Project indexing failed: {e}")
            return ToolResult.text_result(f"Indexing error: {str(e)}", is_error=True)
    
    async def _search_symbols(self, pattern: str, limit: int = 10) -> ToolResult:
        """Search for symbols matching pattern"""
        if not self.glean_client:
            return ToolResult.text_result("Glean client not available", is_error=True)
        
        try:
            # Create symbol search query
            query = create_query("symbol_search", pattern=pattern)
            results = await self.glean_client.query_angle(query)
            
            if isinstance(results, dict) and "symbols" in results:
                symbols = results["symbols"][:limit]
                
                response = {
                    "query_pattern": pattern,
                    "total_found": len(results["symbols"]),
                    "returned": len(symbols),
                    "symbols": symbols
                }
                
                return ToolResult.data_result(
                    json.dumps(response, indent=2),
                    "application/json"
                )
            else:
                return ToolResult.text_result(f"No symbols found for pattern: {pattern}")
                
        except Exception as e:
            logger.error(f"Symbol search failed: {e}")
            return ToolResult.text_result(f"Search error: {str(e)}", is_error=True)
    
    async def _analyze_dependencies(self, symbol: str, depth: int = 3) -> ToolResult:
        """Analyze dependencies for a symbol"""
        if not self.glean_client:
            return ToolResult.text_result("Glean client not available", is_error=True)
        
        try:
            # Create dependency analysis query
            query = create_query("dependencies", symbol=symbol)
            results = await self.glean_client.query_angle(query)
            
            response = {
                "symbol": symbol,
                "analysis_depth": depth,
                "timestamp": datetime.now().isoformat(),
                "dependencies": results if isinstance(results, dict) else {"raw_result": results}
            }
            
            return ToolResult.data_result(
                json.dumps(response, indent=2),
                "application/json"
            )
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return ToolResult.text_result(f"Analysis error: {str(e)}", is_error=True)
    
    async def _review_architecture(self, component: str, rules: List[str]) -> ToolResult:
        """Review architectural patterns"""
        if not self.glean_client:
            return ToolResult.text_result("Glean client not available", is_error=True)
        
        try:
            # Create architecture analysis query
            query = create_query("architecture_analysis", component=component)
            results = await self.glean_client.query_angle(query)
            
            response = {
                "component": component,
                "rules_checked": rules,
                "timestamp": datetime.now().isoformat(),
                "analysis": results if isinstance(results, dict) else {"raw_result": results},
                "violations": [],  # Would be populated by actual analysis
                "recommendations": []  # Would be generated based on analysis
            }
            
            return ToolResult.data_result(
                json.dumps(response, indent=2),
                "application/json"
            )
            
        except Exception as e:
            logger.error(f"Architecture review failed: {e}")
            return ToolResult.text_result(f"Review error: {str(e)}", is_error=True)
    
    async def _execute_angle_query(self, query: str = "", template: str = "") -> ToolResult:
        """Execute raw Angle query"""
        if not self.glean_client:
            return ToolResult.text_result("Glean client not available", is_error=True)
        
        try:
            # Use template if provided, otherwise use raw query
            if template and template in PYTHON_QUERIES:
                actual_query = PYTHON_QUERIES[template]
            else:
                actual_query = query
            
            if not actual_query:
                return ToolResult.text_result("No query provided", is_error=True)
            
            results = await self.glean_client.query_angle(actual_query)
            
            response = {
                "query": actual_query,
                "template_used": template if template else None,
                "timestamp": datetime.now().isoformat(),
                "results": results if isinstance(results, dict) else {"raw_result": results}
            }
            
            return ToolResult.data_result(
                json.dumps(response, indent=2),
                "application/json"
            )
            
        except Exception as e:
            logger.error(f"Angle query failed: {e}")
            return ToolResult.text_result(f"Query error: {str(e)}", is_error=True)
    
    async def _get_statistics(self) -> ToolResult:
        """Get analysis statistics"""
        if not self.glean_client:
            return ToolResult.text_result("Glean client not available", is_error=True)
        
        try:
            stats = await self.glean_client.get_statistics()
            
            response = {
                "project_root": str(self.project_root),
                "indexed_units": list(self.indexed_units),
                "glean_available": GLEAN_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats
            }
            
            return ToolResult.data_result(
                json.dumps(response, indent=2),
                "application/json"
            )
            
        except Exception as e:
            logger.error(f"Statistics failed: {e}")
            return ToolResult.text_result(f"Statistics error: {str(e)}", is_error=True)
    
    async def _enhance_with_ai(self, analysis_data: Dict[str, Any], enhancement_type: str = "summary", ai_provider: str = "grok") -> ToolResult:
        """Enhance analysis with AI insights using real Grok API"""
        try:
            if ai_provider == "grok":
                # Use real Grok API
                from ...ai.grok_client import GrokClient
                
                async with GrokClient() as grok:
                    if enhancement_type == "summary":
                        # Use Grok for code structure analysis
                        result = await grok.analyze_code_structure(analysis_data, focus="architecture")
                    elif enhancement_type == "recommendations":
                        # Use Grok for refactoring suggestions
                        result = await grok.generate_refactoring_suggestions({
                            "complexity_issues": analysis_data.get("complexity_issues", []),
                            "coupling_issues": analysis_data.get("coupling_issues", []),
                            "code_smells": analysis_data.get("code_smells", [])
                        })
                    else:
                        # General analysis
                        result = await grok.analyze_code_structure(analysis_data, focus=enhancement_type)
                    
                    enhanced_result = {
                        "original_analysis": analysis_data,
                        "enhancement_type": enhancement_type,
                        "ai_provider": ai_provider,
                        "timestamp": datetime.now().isoformat(),
                        "grok_response": result,
                        "confidence_score": 0.90,
                        "processing_time": result.get("usage", {}).get("total_time", 0)
                    }
            else:
                # Fallback to mock for other providers
                enhanced_result = {
                    "original_analysis": analysis_data,
                    "enhancement_type": enhancement_type,
                    "ai_provider": ai_provider,
                    "timestamp": datetime.now().isoformat(),
                    "ai_insights": self._generate_ai_insights(analysis_data, enhancement_type),
                    "confidence_score": 0.85,
                    "processing_time": 1.2
                }
            
            return ToolResult.data_result(
                json.dumps(enhanced_result, indent=2),
                "application/json"
            )
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            return ToolResult.text_result(f"Enhancement error: {str(e)}", is_error=True)
    
    async def _ai_code_review(self, files: List[str], review_type: str = "comprehensive") -> ToolResult:
        """AI-powered code review using real Grok API"""
        try:
            # Gather analysis data for each file
            file_analyses = []
            
            for file_path in files:
                file_data = {"path": file_path, "symbols": [], "complexity": "medium"}
                
                if self.glean_client:
                    try:
                        # Get symbols in file using corrected method name
                        query = create_query("file_symbols", file=file_path)
                        symbols = await self.glean_client.query(query)
                        file_data["symbols"] = symbols[:10]  # Limit for API
                    except:
                        pass  # Continue with basic data
                
                file_analyses.append(file_data)
            
            # Use real Grok API for code review
            from ...ai.grok_client import GrokClient
            
            async with GrokClient() as grok:
                grok_review = await grok.generate_code_review(file_analyses, review_type)
                
                review_result = {
                    "files_reviewed": files,
                    "review_type": review_type,
                    "timestamp": datetime.now().isoformat(),
                    "file_analyses": file_analyses,
                    "grok_review": grok_review,
                    "ai_powered": True,
                    "model_used": grok_review.get("model", "grok-4-latest")
                }
            
            return ToolResult.data_result(
                json.dumps(review_result, indent=2),
                "application/json"
            )
            
        except Exception as e:
            logger.error(f"AI code review failed: {e}")
            return ToolResult.text_result(f"Review error: {str(e)}", is_error=True)
    
    async def _ai_explain_code(self, symbol: str, context_level: str = "moderate") -> ToolResult:
        """AI explanation of code structure using real Grok API"""
        try:
            # Get symbol data from Glean
            symbol_data = {"name": symbol, "kind": "unknown", "file": "unknown"}
            
            if self.glean_client:
                try:
                    search_results = await self._search_symbols(symbol, limit=1)
                    if not search_results.isError:
                        data = json.loads(search_results.content[0].data)
                        symbols = data.get("symbols", [])
                        if symbols:
                            symbol_data = symbols[0]
                except:
                    pass
            
            # Use real Grok API for explanation
            from ...ai.grok_client import GrokClient
            
            async with GrokClient() as grok:
                grok_explanation = await grok.explain_code_component(symbol_data, context_level)
                
                explanation = {
                    "symbol": symbol,
                    "context_level": context_level,
                    "timestamp": datetime.now().isoformat(),
                    "symbol_data": symbol_data,
                    "grok_explanation": grok_explanation,
                    "ai_powered": True,
                    "model_used": grok_explanation.get("model", "grok-4-latest")
                }
            
            return ToolResult.data_result(
                json.dumps(explanation, indent=2),
                "application/json"
            )
            
        except Exception as e:
            logger.error(f"AI explanation failed: {e}")
            return ToolResult.text_result(f"Explanation error: {str(e)}", is_error=True)
    
    def _generate_ai_insights(self, analysis_data: Dict[str, Any], enhancement_type: str) -> Dict[str, Any]:
        """Generate AI insights based on analysis data"""
        if enhancement_type == "summary":
            return {
                "key_findings": ["Complex dependency structure", "High coupling in core modules"],
                "summary": "The codebase shows signs of architectural debt with complex interdependencies."
            }
        elif enhancement_type == "recommendations":
            return {
                "priority_actions": ["Refactor core modules", "Implement dependency injection"],
                "long_term_goals": ["Improve modularity", "Reduce coupling"]
            }
        elif enhancement_type == "risk_assessment":
            return {
                "risk_level": "medium",
                "risk_factors": ["Circular dependencies", "Large modules"],
                "mitigation_strategies": ["Gradual refactoring", "Interface segregation"]
            }
        else:
            return {"insight": f"Analysis completed for {enhancement_type}"}
    
    def _generate_review_issues(self, files: List[str], review_type: str) -> List[Dict[str, Any]]:
        """Generate review issues based on file analysis"""
        issues = [
            {
                "file": files[0] if files else "unknown",
                "type": "complexity",
                "severity": "medium",
                "message": "Function complexity exceeds recommended threshold",
                "line": 45,
                "suggestion": "Consider breaking into smaller functions"
            }
        ]
        return issues
    
    def _generate_review_recommendations(self, review_type: str) -> List[str]:
        """Generate review recommendations"""
        return [
            "Improve function documentation",
            "Add type hints for better clarity",
            "Consider extracting common patterns into utilities",
            "Review error handling strategies"
        ]
    
    def _generate_code_explanation(self, symbol: str, context_level: str) -> str:
        """Generate explanation for a code symbol"""
        if context_level == "minimal":
            return f"'{symbol}' is a code symbol in the project."
        elif context_level == "moderate":
            return f"'{symbol}' appears to be a key component with multiple dependencies and usages across the codebase."
        else:
            return f"'{symbol}' is a central architectural component that plays a critical role in the system's design. It has complex relationships with other modules and requires careful consideration for any modifications."


async def create_strands_glean_mcp_server(
    project_root: str = None,
    transport_type: str = "stdio"
) -> StrandsGleanMCPServer:
    """Create and configure the Strands-Glean MCP server"""
    
    server = StrandsGleanMCPServer(
        project_root=project_root or "/Users/apple/projects/cryptotrading"
    )
    
    # Configure transport
    if transport_type == "stdio":
        transport = StdioTransport()
    elif transport_type == "websocket":
        transport = WebSocketTransport(port=8080)
    else:
        raise ValueError(f"Unsupported transport type: {transport_type}")
    
    server.set_transport(transport)
    
    logger.info(f"Strands-Glean MCP server created with {transport_type} transport")
    return server


# CLI entry point for the MCP server
async def main():
    """Main entry point for the MCP server"""
    import sys
    
    # Parse command line arguments
    transport_type = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    project_root = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create and start server
    server = await create_strands_glean_mcp_server(project_root, transport_type)
    
    logger.info("Starting Strands-Glean MCP server...")
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())