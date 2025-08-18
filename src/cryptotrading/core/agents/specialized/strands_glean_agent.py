"""
Strands-Glean Integration Agent
Combines Strands framework with Glean code analysis for intelligent codebase navigation
"""
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from dataclasses import dataclass, field
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

# Conditional imports to handle missing dependencies
try:
    from ..strands import StrandsAgent
    STRANDS_AVAILABLE = True
except ImportError:
    from ..base import BaseAgent as StrandsAgent
    STRANDS_AVAILABLE = False

try:
    from ...infrastructure.analysis.vercel_glean_client import VercelGleanClient
    from ...infrastructure.analysis.angle_parser import create_query, PYTHON_QUERIES
    from ...infrastructure.analysis.scip_indexer import SCIPSymbol, SCIPDocument
    from ...infrastructure.analysis.scip_data_flow_indexer import DataFlowSCIPIndexer
    from ...infrastructure.analysis.glean_data_schemas import ALL_DATA_TRACKING_SCHEMAS
    GLEAN_AVAILABLE = True
except ImportError:
    GLEAN_AVAILABLE = False
    # Mock classes
    class VercelGleanClient:
        def __init__(self, **kwargs): pass
    class SCIPSymbol:
        def __init__(self, **kwargs): pass
    class SCIPDocument:
        def __init__(self, **kwargs): pass

logger = logging.getLogger(__name__)


@dataclass
class StrandsGleanContext:
    """Context for Strands-Glean integration"""
    project_root: str
    indexed_files: Set[str] = field(default_factory=set)
    active_analysis: Optional[Dict[str, Any]] = None
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    code_symbols: Dict[str, List[SCIPSymbol]] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)


class CodeAnalysisCapability(ABC):
    """Abstract base for code analysis capabilities"""

    @abstractmethod
    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        pass


class DependencyAnalysisCapability(CodeAnalysisCapability):
    """Analyze code dependencies using Glean"""

    def __init__(self, glean_client: VercelGleanClient):
        self.glean_client = glean_client

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze dependencies for a given symbol or file"""
        try:
            # Create Angle query for dependencies
            angle_query = create_query("dependencies", {"symbol": query})

            # Execute query using Glean client
            results = await self.glean_client.query_angle(angle_query)

            # Update context with dependency information
            if results and 'dependencies' in results:
                for dep in results['dependencies']:
                    source = dep.get('source', {}).get('file', '')
                    target = dep.get('target', {}).get('file', '')
                    if source and target:
                        if source not in context.dependency_graph:
                            context.dependency_graph[source] = set()
                        context.dependency_graph[source].add(target)

            return {
                "status": "success",
                "dependencies": results.get('dependencies', []),
                "graph": dict(context.dependency_graph),
                "query": query
            }
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class SymbolSearchCapability(CodeAnalysisCapability):
    """Search for code symbols using Glean SCIP indexing"""

    def __init__(self, glean_client: VercelGleanClient):
        self.glean_client = glean_client

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Search for symbols matching the query"""
        try:
            # Create Angle query for symbol search
            angle_query = create_query("symbol_search", {"pattern": query})

            # Execute query
            results = await self.glean_client.query_angle(angle_query)

            # Update context with found symbols
            if results and 'symbols' in results:
                for symbol_data in results['symbols']:
                    file_path = symbol_data.get('file', '')
                    if file_path not in context.code_symbols:
                        context.code_symbols[file_path] = []

                    # Create SCIPSymbol object
                    symbol = SCIPSymbol(
                        name=symbol_data.get('name', ''),
                        kind=symbol_data.get('kind', ''),
                        position=symbol_data.get('position', {}),
                        file_path=file_path
                    )
                    context.code_symbols[file_path].append(symbol)

            return {
                "status": "success",
                "symbols": results.get('symbols', []),
                "matches": len(results.get('symbols', [])),
                "query": query
            }
        except Exception as e:
            logger.error(f"Symbol search failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class ArchitectureAnalysisCapability(CodeAnalysisCapability):
    """Analyze architectural patterns and violations"""

    def __init__(self, glean_client: VercelGleanClient):
        self.glean_client = glean_client

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze architectural patterns"""
        try:
            # Create Angle query for architecture analysis
            angle_query = create_query("architecture_analysis", {"component": query})

            # Execute query
            results = await self.glean_client.query_angle(angle_query)

            return {
                "status": "success",
                "architecture": results.get('architecture', {}),
                "violations": results.get('violations', []),
                "patterns": results.get('patterns', []),
                "query": query
            }
        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class StrandsGleanAgent(StrandsAgent):
    """Strands agent with Glean integration for intelligent code analysis"""

    def __init__(
        self,
        agent_id: str = "strands-glean-agent",
        project_root: str = None,
        glean_storage_path: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(agent_id=agent_id, **kwargs)

        self.project_root = project_root or "/Users/apple/projects/cryptotrading"
        self.context = StrandsGleanContext(project_root=self.project_root)

        # Initialize Glean client if available
        if GLEAN_AVAILABLE:
            self.glean_client = VercelGleanClient(
                project_root=self.project_root,
                storage_path=glean_storage_path
            )
        else:
            self.glean_client = None
            logger.warning("Glean not available, using mock client")

        # Initialize capabilities
        self.capabilities = {}
        if self.glean_client:
            self.capabilities.update({
                "dependency_analysis": DependencyAnalysisCapability(self.glean_client),
                "symbol_search": SymbolSearchCapability(self.glean_client),
                "architecture_analysis": ArchitectureAnalysisCapability(self.glean_client),
                "data_flow_analysis": DataFlowAnalysisCapability(self.glean_client),
                "parameter_analysis": ParameterAnalysisCapability(self.glean_client),
                "factor_analysis": FactorAnalysisCapability(self.glean_client),
                "data_quality_analysis": DataQualityAnalysisCapability(self.glean_client)
            })

        # Add CLRS algorithmic analysis capabilities
        try:
            from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import (
                CLRSMCPTools, TreeMCPTools, GleanAnalysisMCPTools
            )
            from cryptotrading.infrastructure.analysis.glean_client import GleanClient

            # Initialize CLRS tools
            self.clrs_tools = CLRSMCPTools(self.glean_client or GleanClient())
            self.tree_tools = TreeMCPTools(self.glean_client or GleanClient())
            self.enhanced_tools = GleanAnalysisMCPTools(self.glean_client or GleanClient())

            # Add CLRS capabilities
            self.capabilities.update({
                "clrs_dependency_analysis": CLRSAnalysisCapability(self.clrs_tools),
                "clrs_code_similarity": CLRSCodeSimilarityCapability(self.clrs_tools),
                "clrs_pattern_matching": CLRSPatternCapability(self.clrs_tools),
                "tree_structure_analysis": TreeStructureCapability(self.tree_tools),
                "comprehensive_analysis": ComprehensiveAnalysisCapability(self.enhanced_tools)
            })

            logger.info("CLRS algorithmic analysis capabilities enabled")
        except ImportError as e:
            logger.warning(f"CLRS tools not available: {e}")

        # Initialize memory system for code analysis caching
        asyncio.create_task(self._initialize_memory_system())

        logger.info(f"StrandsGleanAgent initialized with {len(self.capabilities)} capabilities")

    async def _initialize_memory_system(self):
        """Initialize memory system for code analysis caching and learning"""
        try:
            # Store Glean agent configuration
            await self.store_memory(
                "glean_agent_config",
                {
                    "agent_id": self.agent_id,
                    "project_root": self.project_root,
                    "capabilities": list(self.capabilities.keys()),
                    "glean_available": GLEAN_AVAILABLE,
                    "initialized_at": datetime.now().isoformat()
                },
                {"type": "configuration", "persistent": True}
            )

            # Initialize analysis cache
            await self.store_memory(
                "analysis_cache",
                {},
                {"type": "analysis_cache", "persistent": True}
            )

            # Initialize dependency graph cache
            await self.store_memory(
                "dependency_graph_cache",
                {},
                {"type": "dependency_cache", "persistent": True}
            )

            # Initialize symbol search cache
            await self.store_memory(
                "symbol_search_cache",
                {},
                {"type": "symbol_cache", "persistent": True}
            )

            # Initialize query performance tracking
            await self.store_memory(
                "query_performance",
                {"total_queries": 0, "avg_response_time": 0, "success_rate": 0},
                {"type": "performance_tracking", "persistent": True}
            )

            logger.info(f"Memory system initialized for Glean Agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Glean agent memory system: {e}")

    async def initialize(self) -> bool:
        """Initialize the agent and index the project"""
        try:
            if not self.glean_client:
                logger.warning("No Glean client available")
                return True

            # Index the project
            logger.info("Indexing project for Glean analysis...")
            index_result = await self.glean_client.index_project(
                unit_name=f"strands-{self.agent_id}",
                force_reindex=False
            )

            if index_result.get("status") in ["success", "already_indexed"]:
                logger.info(f"Project indexing completed: {index_result}")
                return True
            else:
                logger.error(f"Project indexing failed: {index_result}")
                return False

        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False

    async def analyze_code(
        self,
        analysis_type: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform code analysis using specified capability with memory caching"""
        if analysis_type not in self.capabilities:
            return {
                "status": "error",
                "error": f"Unknown analysis type: {analysis_type}",
                "available_types": list(self.capabilities.keys())
            }

        try:
            # Check cache for recent analysis
            cache_key = f"{analysis_type}_{hash(query)}_{hash(str(context) if context else '')}"
            cached_result = await self.retrieve_memory(f"analysis_cache_{cache_key}")
            if cached_result:
                logger.info(f"Using cached result for {analysis_type} query: {query[:50]}...")
                return cached_result

            # Update context if provided
            if context:
                for key, value in context.items():
                    if hasattr(self.context, key):
                        setattr(self.context, key, value)

            # Perform analysis
            start_time = datetime.now()
            capability = self.capabilities[analysis_type]
            result = await capability.analyze(self.context, query)
            end_time = datetime.now()

            # Cache successful results for 30 minutes
            if result.get("status") == "success":
                expiration = datetime.now().timestamp() + 1800  # 30 minutes
                await self.store_memory(
                    f"analysis_cache_{cache_key}",
                    result,
                    {"type": "analysis_cache", "expires_at": expiration}
                )

            # Update performance tracking
            await self._track_query_performance(analysis_type, start_time, end_time, result.get("status") == "success")

            # Record query in history
            self.context.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": analysis_type,
                "query": query,
                "result_status": result.get("status", "unknown"),
                "response_time_ms": (end_time - start_time).total_seconds() * 1000
            })

            return result
        except Exception as e:
            # Store error for learning
            error_data = {
                "error": str(e),
                "analysis_type": analysis_type,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            await self.store_memory(
                f"analysis_error_{datetime.now().timestamp()}",
                error_data,
                {"type": "error_log"}
            )
            logger.error(f"Code analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _track_query_performance(
        self, analysis_type: str, start_time: datetime,
        end_time: datetime, success: bool
    ):
        """Track query performance metrics in memory"""
        try:
            default_data = {
                "total_queries": 0, "avg_response_time": 0,
                "success_rate": 0, "by_type": {}
            }
            performance_data = await self.retrieve_memory("query_performance") or default_data

            response_time = (end_time - start_time).total_seconds() * 1000

            # Update overall metrics
            performance_data["total_queries"] += 1
            current_avg = performance_data["avg_response_time"]
            total_queries = performance_data["total_queries"]
            new_avg = (current_avg * (total_queries - 1) + response_time) / total_queries
            performance_data["avg_response_time"] = new_avg

            # Update success rate
            if "successful_queries" not in performance_data:
                performance_data["successful_queries"] = 0
            if success:
                performance_data["successful_queries"] += 1
            successful = performance_data["successful_queries"]
            total = performance_data["total_queries"]
            performance_data["success_rate"] = successful / total

            # Update by type metrics
            if analysis_type not in performance_data["by_type"]:
                performance_data["by_type"][analysis_type] = {
                    "count": 0, "avg_time": 0, "success_count": 0
                }

            type_data = performance_data["by_type"][analysis_type]
            type_data["count"] += 1
            count = type_data["count"]
            old_avg = type_data["avg_time"]
            type_data["avg_time"] = (old_avg * (count - 1) + response_time) / count
            if success:
                type_data["success_count"] += 1

            await self.store_memory("query_performance", performance_data, {"type": "performance_tracking"})

        except Exception as e:
            logger.error(f"Failed to track query performance: {e}")

    async def find_dependencies(self, symbol: str) -> Dict[str, Any]:
        """Find dependencies for a symbol"""
        return await self.analyze_code("dependency_analysis", symbol)

    async def search_symbols(self, pattern: str) -> Dict[str, Any]:
        """Search for symbols matching pattern"""
        return await self.analyze_code("symbol_search", pattern)

    async def analyze_architecture(self, component: str) -> Dict[str, Any]:
        """Analyze architectural patterns for a component"""
        return await self.analyze_code("architecture_analysis", component)

    async def analyze_data_flow(self, symbol_or_component: str) -> Dict[str, Any]:
        """Analyze data flow for a specific symbol or component"""
        return await self.analyze_code("data_flow_analysis", symbol_or_component)

    async def analyze_parameters(self, category: str = "") -> Dict[str, Any]:
        """Analyze configuration parameters by category"""
        return await self.analyze_code("parameter_analysis", category)

    async def analyze_factors(self, symbol: str = "") -> Dict[str, Any]:
        """Analyze crypto factors for a specific symbol or all symbols"""
        return await self.analyze_code("factor_analysis", symbol)

    async def analyze_data_quality(self, data_id: str = "") -> Dict[str, Any]:
        """Analyze data quality metrics"""
        return await self.analyze_code("data_quality_analysis", data_id)

    async def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current analysis context"""
        return {
            "project_root": self.context.project_root,
            "indexed_files": len(self.context.indexed_files),
            "total_symbols": sum(len(symbols) for symbols in self.context.code_symbols.values()),
            "dependency_nodes": len(self.context.dependency_graph),
            "query_history": len(self.context.query_history),
            "capabilities": list(self.capabilities.keys()),
            "glean_available": GLEAN_AVAILABLE,
            "strands_available": STRANDS_AVAILABLE
        }

    async def process_strand(self, strand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a strand with Glean-powered analysis"""
        try:
            strand_type = strand_data.get("type", "unknown")
            strand_query = strand_data.get("query", "")

            if strand_type == "code_analysis":
                analysis_type = strand_data.get("analysis_type", "symbol_search")
                return await self.analyze_code(analysis_type, strand_query, strand_data.get("context"))

            elif strand_type == "dependency_trace":
                return await self.find_dependencies(strand_query)

            elif strand_type == "symbol_lookup":
                return await self.search_symbols(strand_query)

            elif strand_type == "architecture_review":
                return await self.analyze_architecture(strand_query)

            elif strand_type == "data_flow_analysis":
                return await self.analyze_data_flow(strand_query)

            elif strand_type == "parameter_analysis":
                return await self.analyze_parameters(strand_query)

            elif strand_type == "factor_analysis":
                return await self.analyze_factors(strand_query)

            elif strand_type == "data_quality_analysis":
                return await self.analyze_data_quality(strand_query)

            elif strand_type == "clrs_dependency_analysis":
                return await self.analyze_code("clrs_dependency_analysis", strand_query, strand_data.get("context"))

            elif strand_type == "clrs_code_similarity":
                return await self.analyze_code("clrs_code_similarity", strand_query, strand_data.get("context"))

            elif strand_type == "clrs_pattern_matching":
                return await self.analyze_code("clrs_pattern_matching", strand_query, strand_data.get("context"))

            elif strand_type == "tree_structure_analysis":
                return await self.analyze_code("tree_structure_analysis", strand_query, strand_data.get("context"))

            elif strand_type == "comprehensive_analysis":
                return await self.analyze_code("comprehensive_analysis", strand_query, strand_data.get("context"))

            else:
                return {
                    "status": "error",
                    "error": f"Unknown strand type: {strand_type}",
                    "supported_types": [
                        "code_analysis", "dependency_trace", "symbol_lookup",
                        "architecture_review", "data_flow_analysis", "parameter_analysis",
                        "factor_analysis", "data_quality_analysis", "clrs_dependency_analysis",
                        "clrs_code_similarity", "clrs_pattern_matching", "tree_structure_analysis",
                        "comprehensive_analysis"
                    ]
                }
        except Exception as e:
            logger.error(f"Strand processing failed: {e}")
            return {"status": "error", "error": str(e)}


# Factory function for easy instantiation
async def create_strands_glean_agent(
    project_root: str = None,
    agent_id: str = None,
    **kwargs
) -> StrandsGleanAgent:
    """Create and initialize a StrandsGleanAgent"""
    agent = StrandsGleanAgent(
        agent_id=agent_id or f"strands-glean-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        project_root=project_root,
        **kwargs
    )

    # Initialize the agent
    success = await agent.initialize()
    if not success:
        logger.warning("Agent initialization had issues, but continuing...")

    return agent


# CLI integration function for testing
async def test_strands_glean_agent():
    """Test function for the StrandsGleanAgent"""
    agent = await create_strands_glean_agent()

    print("=== StrandsGleanAgent Test ===")

    # Get context summary
    summary = await agent.get_context_summary()
    print(f"Context: {json.dumps(summary, indent=2)}")

    # Test symbol search
    symbol_results = await agent.search_symbols("Agent")
    print(f"Symbol search results: {json.dumps(symbol_results, indent=2)}")

    # Test dependency analysis
    if symbol_results.get("symbols"):
        first_symbol = symbol_results["symbols"][0]
        dep_results = await agent.find_dependencies(first_symbol.get("name", ""))
        print(f"Dependency results: {json.dumps(dep_results, indent=2)}")

    return agent


class DataFlowAnalysisCapability(CodeAnalysisCapability):
    """Analyze data flow through the system"""

    def __init__(self, glean_client: VercelGleanClient):
        self.glean_client = glean_client
        self.data_flow_indexer = DataFlowSCIPIndexer("/Users/apple/projects/cryptotrading")

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze data flow for a specific component or symbol"""
        try:
            # Create Angle query for data flow
            angle_query = create_query("data_flow", {
                "predicate": "crypto.DataInput",
                "filter": {"symbol": query} if query else {}
            })

            # Execute query
            results = await self.glean_client.query_angle(angle_query)

            # Also get data outputs and lineage
            output_query = create_query("data_flow", {
                "predicate": "crypto.DataOutput",
                "filter": {"symbol": query} if query else {}
            })
            output_results = await self.glean_client.query_angle(output_query)

            lineage_query = create_query("data_flow", {
                "predicate": "crypto.DataLineage",
                "filter": {}
            })
            lineage_results = await self.glean_client.query_angle(lineage_query)

            return {
                "status": "success",
                "inputs": results.get('results', []),
                "outputs": output_results.get('results', []),
                "lineage": lineage_results.get('results', []),
                "query": query
            }
        except Exception as e:
            logger.error(f"Data flow analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class ParameterAnalysisCapability(CodeAnalysisCapability):
    """Analyze configuration parameters across the system"""

    def __init__(self, glean_client: VercelGleanClient):
        self.glean_client = glean_client

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze parameters for a specific component or category"""
        try:
            # Create Angle query for parameters
            angle_query = create_query("parameter_analysis", {
                "predicate": "crypto.Parameter",
                "filter": {"category": query} if query else {}
            })

            # Execute query
            results = await self.glean_client.query_angle(angle_query)

            # Analyze parameter patterns
            parameters = results.get('results', [])
            analysis = {
                "total_parameters": len(parameters),
                "by_type": {},
                "by_category": {},
                "value_ranges": {},
                "most_common": []
            }

            for param in parameters:
                value = param.get('value', {})
                param_type = value.get('param_type', 'unknown')
                category = value.get('category', 'unknown')

                analysis["by_type"][param_type] = analysis["by_type"].get(param_type, 0) + 1
                analysis["by_category"][category] = analysis["by_category"].get(category, 0) + 1

            return {
                "status": "success",
                "parameters": parameters,
                "analysis": analysis,
                "query": query
            }
        except Exception as e:
            logger.error(f"Parameter analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class FactorAnalysisCapability(CodeAnalysisCapability):
    """Analyze crypto factors and their calculations"""

    def __init__(self, glean_client: VercelGleanClient):
        self.glean_client = glean_client

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze factors for a specific symbol or category"""
        try:
            # Create Angle query for factors
            angle_query = create_query("factor_analysis", {
                "predicate": "crypto.Factor",
                "filter": {"symbol": query} if query else {}
            })

            # Execute query
            results = await self.glean_client.query_angle(angle_query)

            # Get factor calculations
            calc_query = create_query("factor_analysis", {
                "predicate": "crypto.FactorCalculation",
                "filter": {"symbol": query} if query else {}
            })
            calc_results = await self.glean_client.query_angle(calc_query)

            # Analyze factor patterns
            factors = results.get('results', [])
            calculations = calc_results.get('results', [])

            analysis = {
                "total_factors": len(factors),
                "by_category": {},
                "by_symbol": {},
                "calculation_stats": {
                    "total_calculations": len(calculations),
                    "avg_execution_time": 0,
                    "error_rate": 0
                }
            }

            # Analyze factors
            for factor in factors:
                value = factor.get('value', {})
                category = value.get('category', 'unknown')
                symbol = factor.get('key', {}).get('symbol', 'unknown')

                analysis["by_category"][category] = analysis["by_category"].get(category, 0) + 1
                analysis["by_symbol"][symbol] = analysis["by_symbol"].get(symbol, 0) + 1

            # Analyze calculations
            if calculations:
                total_time = 0
                errors = 0

                for calc in calculations:
                    value = calc.get('value', {})
                    exec_time = value.get('execution_time_ms', 0)
                    total_time += exec_time

                    if value.get('error'):
                        errors += 1

                analysis["calculation_stats"]["avg_execution_time"] = total_time / len(calculations)
                analysis["calculation_stats"]["error_rate"] = (errors / len(calculations)) * 100

            return {
                "status": "success",
                "factors": factors,
                "calculations": calculations,
                "analysis": analysis,
                "query": query
            }
        except Exception as e:
            logger.error(f"Factor analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class DataQualityAnalysisCapability(CodeAnalysisCapability):
    """Analyze data quality metrics and issues"""

    def __init__(self, glean_client: VercelGleanClient):
        self.glean_client = glean_client

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze data quality for a specific data source or type"""
        try:
            # Create Angle query for data quality
            angle_query = create_query("data_quality", {
                "predicate": "crypto.DataQuality",
                "filter": {"data_id": query} if query else {}
            })

            # Execute query
            results = await self.glean_client.query_angle(angle_query)

            # Analyze quality metrics
            quality_data = results.get('results', [])
            analysis = {
                "total_assessments": len(quality_data),
                "avg_score": 0,
                "by_metric": {},
                "issues": [],
                "trends": {}
            }

            if quality_data:
                total_score = 0
                for assessment in quality_data:
                    value = assessment.get('value', {})
                    score = value.get('score', 0)
                    metric = assessment.get('key', {}).get('metric', 'unknown')
                    issues = value.get('issues', '[]')

                    total_score += score

                    if metric not in analysis["by_metric"]:
                        analysis["by_metric"][metric] = {"scores": [], "avg": 0}
                    analysis["by_metric"][metric]["scores"].append(score)

                    # Parse issues
                    try:
                        issue_list = json.loads(issues)
                        analysis["issues"].extend(issue_list)
                    except:
                        pass

                analysis["avg_score"] = total_score / len(quality_data)

                # Calculate averages by metric
                for metric, data in analysis["by_metric"].items():
                    data["avg"] = sum(data["scores"]) / len(data["scores"])

            return {
                "status": "success",
                "quality_data": quality_data,
                "analysis": analysis,
                "query": query
            }
        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


# CLRS Analysis Capabilities
class CLRSAnalysisCapability(CodeAnalysisCapability):
    """CLRS algorithmic dependency analysis"""

    def __init__(self, clrs_tools):
        self.clrs_tools = clrs_tools

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze dependencies using CLRS graph algorithms"""
        try:
            # Convert context dependency graph to modules format
            modules = dict(context.dependency_graph)
            if query and query not in modules:
                modules[query] = []

            return await self.clrs_tools.analyze_dependency_graph(modules)
        except Exception as e:
            logger.error(f"CLRS dependency analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class CLRSCodeSimilarityCapability(CodeAnalysisCapability):
    """CLRS code similarity analysis"""

    def __init__(self, clrs_tools):
        self.clrs_tools = clrs_tools

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze code similarity using CLRS string algorithms"""
        try:
            # Query should be in format "code1|code2"
            if "|" not in query:
                return {"status": "error", "error": "Query must be in format 'code1|code2'"}

            code1, code2 = query.split("|", 1)
            return await self.clrs_tools.analyze_code_similarity(code1, code2)
        except Exception as e:
            logger.error(f"CLRS code similarity failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class CLRSPatternCapability(CodeAnalysisCapability):
    """CLRS pattern matching analysis"""

    def __init__(self, clrs_tools):
        self.clrs_tools = clrs_tools

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Find patterns using CLRS string matching"""
        try:
            # Use indexed files as source code base
            patterns = [query] if query else ["def ", "class ", "import "]

            # Get source code from first indexed file
            if context.indexed_files:
                source_code = str(context.indexed_files[0])  # Simplified
            else:
                source_code = query

            return await self.clrs_tools.find_code_patterns(source_code, patterns)
        except Exception as e:
            logger.error(f"CLRS pattern matching failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class TreeStructureCapability(CodeAnalysisCapability):
    """Tree structure analysis capability"""

    def __init__(self, tree_tools):
        self.tree_tools = tree_tools

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Analyze hierarchical code structure"""
        try:
            # Convert context to codebase structure
            codebase = {
                "files": list(context.indexed_files),
                "symbols": context.code_symbols,
                "dependencies": dict(context.dependency_graph)
            }

            return await self.tree_tools.analyze_code_hierarchy(codebase)
        except Exception as e:
            logger.error(f"Tree structure analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


class ComprehensiveAnalysisCapability(CodeAnalysisCapability):
    """Comprehensive analysis using CLRS + Tree operations"""

    def __init__(self, enhanced_tools):
        self.enhanced_tools = enhanced_tools

    async def analyze(self, context: StrandsGleanContext, query: str) -> Dict[str, Any]:
        """Comprehensive analysis combining all techniques"""
        try:
            # Build comprehensive codebase data
            codebase_data = {
                "modules": dict(context.dependency_graph),
                "file_structure": {
                    "files": list(context.indexed_files),
                    "symbols": context.code_symbols
                },
                "code_samples": [query] if query else []
            }

            return await self.enhanced_tools.comprehensive_code_analysis(codebase_data)
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"status": "error", "error": str(e), "query": query}


# Update the StrandsGleanAgent to include new capabilities
def _update_agent_capabilities(agent: 'StrandsGleanAgent'):
    """Update agent with new data analysis capabilities"""
    if agent.glean_client and GLEAN_AVAILABLE:
        agent.capabilities.update({
            "data_flow_analysis": DataFlowAnalysisCapability(agent.glean_client),
            "parameter_analysis": ParameterAnalysisCapability(agent.glean_client),
            "factor_analysis": FactorAnalysisCapability(agent.glean_client),
            "data_quality_analysis": DataQualityAnalysisCapability(agent.glean_client)
        })


if __name__ == "__main__":
    asyncio.run(test_strands_glean_agent())