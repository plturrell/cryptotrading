"""
MCP Tools for CLRS Algorithms and Tree Library Integration

Provides MCP-compatible tools that combine CLRS algorithmic reasoning
with Tree library operations for advanced code analysis.
Implements strict agent segregation and multi-tenancy.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass

from .clrs_algorithms import (
    CLRSSortingAlgorithms,
    CLRSSearchAlgorithms, 
    CLRSGraphAlgorithms,
    CLRSDynamicProgramming,
    CLRSStringAlgorithms,
    Graph,
    WeightedGraph
)
from .tree_library import (
    TreeOperations,
    PathOperations,
    StructuralAnalysis,
    TreeDiffMerge,
    ASTProcessor,
    HierarchicalCodeIndex,
    ConfigurationManager
)
from .mcp_agent_segregation import (
    AgentContext,
    ResourceType,
    SecureToolWrapper,
    get_segregation_manager,
    require_agent_auth
)
from .glean_client import GleanClient

logger = logging.getLogger(__name__)

class CLRSMCPTools:
    """MCP tools for CLRS algorithmic reasoning"""
    
    def __init__(self, glean_client: GleanClient):
        self.glean_client = glean_client
        self.sorting = CLRSSortingAlgorithms()
        self.search = CLRSSearchAlgorithms()
        self.graph = CLRSGraphAlgorithms()
        self.dp = CLRSDynamicProgramming()
        self.string = CLRSStringAlgorithms()
        self.dependency_analyzer = CodeDependencyAnalyzer()
        self.similarity_analyzer = CodeSimilarityAnalyzer()
    
    async def sort_code_symbols(self, symbols: List[Dict[str, Any]], sort_by: str = "usage", algorithm: str = "quicksort") -> Dict[str, Any]:
        """Sort code symbols using CLRS algorithms"""
        try:
            # Define comparison function based on sort_by parameter
            def compare_symbols(a: Dict[str, Any], b: Dict[str, Any]) -> int:
                val_a = a.get(sort_by, 0)
                val_b = b.get(sort_by, 0)
                return -1 if val_a < val_b else (1 if val_a > val_b else 0)
            
            # Apply selected sorting algorithm
            if algorithm == "insertion":
                sorted_symbols = self.sorting.insertion_sort(symbols, compare_symbols)
            elif algorithm == "bubble":
                sorted_symbols = self.sorting.bubble_sort(symbols, compare_symbols)
            elif algorithm == "heap":
                sorted_symbols = self.sorting.heapsort(symbols, compare_symbols)
            else:  # default to quicksort
                sorted_symbols = self.sorting.quicksort(symbols, compare_symbols)
            
            return {
                "success": True,
                "sorted_symbols": sorted_symbols,
                "algorithm_used": algorithm,
                "sort_criteria": sort_by,
                "total_symbols": len(sorted_symbols)
            }
            
        except Exception as e:
            logger.error("Symbol sorting failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def search_code_symbols(self, symbols: List[Dict[str, Any]], target_name: str) -> Dict[str, Any]:
        """Search for code symbols using binary search"""
        try:
            # Sort symbols by name first for binary search
            sorted_symbols = self.sorting.quicksort(
                symbols, 
                lambda a, b: -1 if a.get("name", "") < b.get("name", "") else (1 if a.get("name", "") > b.get("name", "") else 0)
            )
            
            # Create list of just names for binary search
            symbol_names = [s.get("name", "") for s in sorted_symbols]
            
            # Perform binary search
            index = self.search.binary_search(
                symbol_names, 
                target_name,
                lambda a, b: -1 if a < b else (1 if a > b else 0)
            )
            
            result = {
                "success": True,
                "found": index != -1,
                "index": index if index != -1 else None,
                "symbol": sorted_symbols[index] if index != -1 else None,
                "search_algorithm": "binary_search",
                "total_searched": len(symbols)
            }
            
            return result
            
        except Exception as e:
            logger.error("Symbol search failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def analyze_dependency_graph(self, modules: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze code dependencies using graph algorithms"""
        try:
            analysis = self.dependency_analyzer.analyze_dependencies(modules)
            
            return {
                "success": True,
                "analysis": analysis,
                "algorithms_used": ["dfs", "topological_sort", "cycle_detection"],
                "recommendations": self._generate_dependency_recommendations(analysis)
            }
            
        except Exception as e:
            logger.error("Dependency analysis failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def find_shortest_call_path(self, call_graph: Dict[str, List[str]], start_function: str, end_function: str) -> Dict[str, Any]:
        """Find shortest path between functions using Dijkstra's algorithm"""
        try:
            # Build weighted graph from call graph
            graph = WeightedGraph()
            
            for func, calls in call_graph.items():
                graph.add_node(func)
                for called_func in calls:
                    graph.add_weighted_edge(func, called_func, 1.0)  # Unit weight
            
            # Run Dijkstra's algorithm
            shortest_paths = self.graph.dijkstra(graph, start_function)
            
            # Reconstruct path to end function
            path = []
            current = end_function
            
            while current is not None:
                path.append(current)
                current = shortest_paths.predecessors.get(current)
            
            path.reverse()
            
            return {
                "success": True,
                "path": path if path[0] == start_function else [],
                "distance": shortest_paths.distances.get(end_function, float('inf')),
                "algorithm_used": "dijkstra",
                "path_exists": shortest_paths.distances.get(end_function, float('inf')) != float('inf')
            }
            
        except Exception as e:
            logger.error("Call path analysis failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def analyze_code_similarity(self, code1: str, code2: str) -> Dict[str, Any]:
        """Analyze code similarity using dynamic programming"""
        try:
            similarity = self.similarity_analyzer.analyze_similarity(code1, code2)
            
            return {
                "success": True,
                "similarity": similarity,
                "algorithm_used": "longest_common_subsequence",
                "recommendation": self._get_similarity_recommendation(similarity["similarity_ratio"])
            }
            
        except Exception as e:
            logger.error("Code similarity analysis failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def find_code_patterns(self, source_code: str, patterns: List[str]) -> Dict[str, Any]:
        """Find code patterns using string matching algorithms"""
        try:
            results = {}
            
            for pattern in patterns:
                # Use KMP for efficient pattern matching
                matches = self.string.kmp_string_match(source_code, pattern)
                results[pattern] = {
                    "matches": len(matches),
                    "positions": matches,
                    "algorithm": "kmp"
                }
            
            return {
                "success": True,
                "pattern_matches": results,
                "total_patterns": len(patterns),
                "source_length": len(source_code)
            }
            
        except Exception as e:
            logger.error("Pattern matching failed: %s", e)
            return {"success": False, "error": str(e)}
    
    def _generate_dependency_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on dependency analysis"""
        recommendations = []
        
        if analysis.get("circular_dependencies"):
            recommendations.append("Break circular dependencies to improve modularity")
        
        if len(analysis.get("critical_modules", [])) > 5:
            recommendations.append("Consider splitting highly coupled modules")
        
        if analysis.get("total_dependencies", 0) > analysis.get("total_modules", 1) * 3:
            recommendations.append("High dependency ratio - consider architectural refactoring")
        
        return recommendations
    
    def _get_similarity_recommendation(self, ratio: float) -> str:
        """Get recommendation based on similarity ratio"""
        if ratio > 0.8:
            return "High similarity - consider extracting common functionality"
        elif ratio > 0.5:
            return "Moderate similarity - review for potential refactoring"
        else:
            return "Low similarity - code appears to be sufficiently different"

class TreeMCPTools:
    """MCP tools for Tree library operations"""
    
    def __init__(self, glean_client: GleanClient):
        self.glean_client = glean_client
        self.tree_ops = TreeOperations()
        self.path_ops = PathOperations()
        self.structural = StructuralAnalysis()
        self.diff_merge = TreeDiffMerge()
        self.hierarchical_index = HierarchicalCodeIndex()
        self.config_manager = ConfigurationManager()
    
    async def process_ast_structure(self, ast_data: Dict[str, Any], operation: str, **kwargs) -> Dict[str, Any]:
        """Process AST using tree operations"""
        try:
            if operation == "flatten":
                result = self.tree_ops.flatten(ast_data)
                return {
                    "success": True,
                    "operation": "flatten",
                    "result": result,
                    "leaf_count": len(result)
                }
            
            elif operation == "get_depth":
                depth = self.structural.get_depth(ast_data)
                return {
                    "success": True,
                    "operation": "get_depth",
                    "depth": depth,
                    "complexity_assessment": self._assess_complexity(depth)
                }
            
            elif operation == "find_nodes":
                predicate_type = kwargs.get("node_type", "function")
                matches = self.structural.find_substructures(
                    ast_data,
                    lambda node: isinstance(node, dict) and node.get("type") == predicate_type
                )
                return {
                    "success": True,
                    "operation": "find_nodes",
                    "matches": [{"path": match.keys, "node": match.value} for match in matches],
                    "count": len(matches)
                }
            
            elif operation == "get_paths":
                paths = self.path_ops.get_all_paths(ast_data)
                return {
                    "success": True,
                    "operation": "get_paths",
                    "paths": [{"path": path.keys, "value_type": type(path.value).__name__} for path in paths],
                    "total_paths": len(paths)
                }
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error("AST processing failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def analyze_code_hierarchy(self, codebase: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hierarchical structure of codebase"""
        try:
            # Index the codebase
            self.hierarchical_index.index_codebase(codebase)
            
            # Analyze structure
            depth = self.structural.get_depth(self.hierarchical_index.code_structure)
            node_count = self.structural.get_node_count(self.hierarchical_index.code_structure)
            leaf_count = self.structural.get_leaf_count(self.hierarchical_index.code_structure)
            
            return {
                "success": True,
                "hierarchy_analysis": {
                    "max_depth": depth,
                    "total_nodes": node_count,
                    "leaf_nodes": leaf_count,
                    "branching_factor": node_count / max(depth, 1),
                    "complexity_score": self._calculate_hierarchy_complexity(depth, node_count, leaf_count)
                },
                "structure_type": "hierarchical_index"
            }
            
        except Exception as e:
            logger.error("Hierarchy analysis failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def compare_code_structures(self, old_structure: Dict[str, Any], new_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two code structures and show differences"""
        try:
            diff = self.diff_merge.diff(old_structure, new_structure)
            
            return {
                "success": True,
                "diff_analysis": {
                    "added_count": len(diff.added),
                    "removed_count": len(diff.removed),
                    "modified_count": len(diff.modified),
                    "added_paths": [path.keys for path in diff.added],
                    "removed_paths": [path.keys for path in diff.removed],
                    "modified_paths": [change["path"].keys for change in diff.modified],
                    "change_impact": self._assess_change_impact(diff)
                },
                "algorithm_used": "tree_diff"
            }
            
        except Exception as e:
            logger.error("Structure comparison failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def navigate_code_path(self, structure: Dict[str, Any], path: List[Union[str, int]], operation: str = "get") -> Dict[str, Any]:
        """Navigate to specific path in code structure"""
        try:
            if operation == "get":
                value = self.path_ops.get_path(structure, path)
                return {
                    "success": True,
                    "operation": "get_path",
                    "path": path,
                    "value": value,
                    "value_type": type(value).__name__ if value is not None else "None",
                    "exists": value is not None
                }
            
            elif operation == "analyze":
                value = self.path_ops.get_path(structure, path)
                if value is not None:
                    analysis = {
                        "depth_from_root": len(path),
                        "is_leaf": self.structural.is_leaf(value),
                        "value_type": type(value).__name__
                    }
                    
                    if not self.structural.is_leaf(value):
                        analysis.update({
                            "child_count": len(value) if isinstance(value, (list, dict)) else 0,
                            "subtree_depth": self.structural.get_depth(value),
                            "subtree_nodes": self.structural.get_node_count(value)
                        })
                    
                    return {
                        "success": True,
                        "operation": "analyze_path",
                        "path": path,
                        "analysis": analysis
                    }
                else:
                    return {
                        "success": False,
                        "error": "Path does not exist",
                        "path": path
                    }
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error("Path navigation failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def merge_configurations(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration structures"""
        try:
            merged = self.config_manager.merge_configs(base_config, override_config)
            
            # Analyze merge results
            base_paths = len(self.path_ops.get_all_paths(base_config))
            override_paths = len(self.path_ops.get_all_paths(override_config))
            merged_paths = len(self.path_ops.get_all_paths(merged))
            
            return {
                "success": True,
                "merged_config": merged,
                "merge_analysis": {
                    "base_paths": base_paths,
                    "override_paths": override_paths,
                    "merged_paths": merged_paths,
                    "paths_added": merged_paths - base_paths,
                    "merge_complexity": self._assess_merge_complexity(base_config, override_config, merged)
                }
            }
            
        except Exception as e:
            logger.error("Configuration merge failed: %s", e)
            return {"success": False, "error": str(e)}
    
    def _assess_complexity(self, depth: int) -> str:
        """Assess complexity based on depth"""
        if depth <= 3:
            return "low"
        elif depth <= 6:
            return "moderate"
        else:
            return "high"
    
    def _calculate_hierarchy_complexity(self, depth: int, nodes: int, leaves: int) -> float:
        """Calculate complexity score for hierarchy"""
        # Normalized complexity score (0-1)
        depth_factor = min(depth / 10, 1.0)
        density_factor = min((nodes - leaves) / max(nodes, 1), 1.0)
        return (depth_factor + density_factor) / 2
    
    def _assess_change_impact(self, diff) -> str:
        """Assess impact of changes"""
        total_changes = len(diff.added) + len(diff.removed) + len(diff.modified)
        
        if total_changes <= 5:
            return "low"
        elif total_changes <= 20:
            return "moderate"
        else:
            return "high"
    
    def _assess_merge_complexity(self, base: Dict, override: Dict, merged: Dict) -> str:
        """Assess complexity of merge operation"""
        base_size = len(str(base))
        override_size = len(str(override))
        merged_size = len(str(merged))
        
        complexity_ratio = merged_size / max(base_size + override_size, 1)
        
        if complexity_ratio <= 0.8:
            return "simple"
        elif complexity_ratio <= 1.2:
            return "moderate"
        else:
            return "complex"

class EnhancedGleanMCPTools:
    """Enhanced Glean MCP tools with CLRS and Tree capabilities"""
    
    def __init__(self, glean_client: GleanClient):
        self.glean_client = glean_client
        self.clrs_tools = CLRSMCPTools(glean_client)
        self.tree_tools = TreeMCPTools(glean_client)
    
    async def comprehensive_code_analysis(self, codebase_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis combining CLRS algorithms and Tree operations"""
        try:
            results = {
                "success": True,
                "analysis_components": [],
                "overall_metrics": {}
            }
            
            # Extract modules and dependencies for CLRS analysis
            if "modules" in codebase_data:
                dep_analysis = await self.clrs_tools.analyze_dependency_graph(codebase_data["modules"])
                results["dependency_analysis"] = dep_analysis
                results["analysis_components"].append("dependency_graph")
            
            # Hierarchical structure analysis using Tree operations
            if "file_structure" in codebase_data:
                hierarchy_analysis = await self.tree_tools.analyze_code_hierarchy(codebase_data["file_structure"])
                results["hierarchy_analysis"] = hierarchy_analysis
                results["analysis_components"].append("hierarchy_structure")
            
            # Code similarity analysis if multiple code samples provided
            if "code_samples" in codebase_data and len(codebase_data["code_samples"]) >= 2:
                samples = codebase_data["code_samples"]
                similarity_results = []
                
                for i in range(len(samples) - 1):
                    similarity = await self.clrs_tools.analyze_code_similarity(samples[i], samples[i + 1])
                    similarity_results.append(similarity)
                
                results["similarity_analysis"] = similarity_results
                results["analysis_components"].append("code_similarity")
            
            # Calculate overall metrics
            results["overall_metrics"] = self._calculate_overall_metrics(results)
            
            return results
            
        except Exception as e:
            logger.error("Comprehensive analysis failed: %s", e)
            return {"success": False, "error": str(e)}
    
    async def optimize_code_structure(self, current_structure: Dict[str, Any], optimization_goals: List[str]) -> Dict[str, Any]:
        """Optimize code structure using algorithmic insights"""
        try:
            recommendations = []
            optimizations = {}
            
            # Analyze current structure
            hierarchy_analysis = await self.tree_tools.analyze_code_hierarchy(current_structure)
            
            if "reduce_complexity" in optimization_goals:
                complexity_opts = self._generate_complexity_optimizations(hierarchy_analysis)
                optimizations["complexity"] = complexity_opts
                recommendations.extend(complexity_opts.get("recommendations", []))
            
            if "improve_modularity" in optimization_goals and "modules" in current_structure:
                dep_analysis = await self.clrs_tools.analyze_dependency_graph(current_structure["modules"])
                modularity_opts = self._generate_modularity_optimizations(dep_analysis)
                optimizations["modularity"] = modularity_opts
                recommendations.extend(modularity_opts.get("recommendations", []))
            
            if "optimize_performance" in optimization_goals:
                perf_opts = self._generate_performance_optimizations(current_structure)
                optimizations["performance"] = perf_opts
                recommendations.extend(perf_opts.get("recommendations", []))
            
            return {
                "success": True,
                "optimization_analysis": optimizations,
                "recommendations": recommendations,
                "priority_actions": self._prioritize_recommendations(recommendations),
                "estimated_impact": self._estimate_optimization_impact(optimizations)
            }
            
        except Exception as e:
            logger.error("Structure optimization failed: %s", e)
            return {"success": False, "error": str(e)}
    
    def _calculate_overall_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall codebase metrics"""
        metrics = {
            "complexity_score": 0.0,
            "modularity_score": 0.0,
            "maintainability_score": 0.0
        }
        
        # Factor in dependency analysis
        if "dependency_analysis" in analysis_results:
            dep_data = analysis_results["dependency_analysis"].get("analysis", {})
            if dep_data.get("circular_dependencies"):
                metrics["modularity_score"] -= 0.3
            
            critical_modules = len(dep_data.get("critical_modules", []))
            if critical_modules > 5:
                metrics["complexity_score"] += 0.2
        
        # Factor in hierarchy analysis
        if "hierarchy_analysis" in analysis_results:
            hier_data = analysis_results["hierarchy_analysis"].get("hierarchy_analysis", {})
            complexity_score = hier_data.get("complexity_score", 0.0)
            metrics["complexity_score"] += complexity_score
        
        # Normalize scores to 0-1 range
        for key in metrics:
            metrics[key] = max(0.0, min(1.0, metrics[key]))
        
        # Calculate overall maintainability
        metrics["maintainability_score"] = (
            (1.0 - metrics["complexity_score"]) * 0.4 +
            metrics["modularity_score"] * 0.6
        )
        
        return metrics
    
    def _generate_complexity_optimizations(self, hierarchy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complexity reduction recommendations"""
        recommendations = []
        analysis = hierarchy_analysis.get("hierarchy_analysis", {})
        
        if analysis.get("max_depth", 0) > 6:
            recommendations.append("Consider flattening deep hierarchies to reduce complexity")
        
        if analysis.get("complexity_score", 0) > 0.7:
            recommendations.append("High complexity detected - consider breaking down large components")
        
        return {
            "current_complexity": analysis.get("complexity_score", 0),
            "target_complexity": max(0.3, analysis.get("complexity_score", 0) - 0.2),
            "recommendations": recommendations
        }
    
    def _generate_modularity_optimizations(self, dep_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate modularity improvement recommendations"""
        recommendations = []
        analysis = dep_analysis.get("analysis", {})
        
        if analysis.get("circular_dependencies"):
            recommendations.append("Break circular dependencies to improve modularity")
        
        critical_count = len(analysis.get("critical_modules", []))
        if critical_count > 3:
            recommendations.append(f"Reduce coupling in {critical_count} critical modules")
        
        return {
            "circular_dependencies": len(analysis.get("circular_dependencies", [])),
            "critical_modules": critical_count,
            "recommendations": recommendations
        }
    
    def _generate_performance_optimizations(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze structure size and depth for performance implications
        if isinstance(structure, dict):
            total_keys = len(str(structure))
            if total_keys > 10000:
                recommendations.append("Large structure detected - consider lazy loading or pagination")
        
        return {
            "structure_size": len(str(structure)),
            "recommendations": recommendations
        }
    
    def _prioritize_recommendations(self, recommendations: List[str]) -> List[str]:
        """Prioritize recommendations by impact"""
        priority_keywords = {
            "circular": 1,  # Highest priority
            "complexity": 2,
            "coupling": 3,
            "performance": 4
        }
        
        def get_priority(rec: str) -> int:
            for keyword, priority in priority_keywords.items():
                if keyword in rec.lower():
                    return priority
            return 5  # Default priority
        
        return sorted(recommendations, key=get_priority)
    
    def _estimate_optimization_impact(self, optimizations: Dict[str, Any]) -> str:
        """Estimate overall impact of optimizations"""
        total_issues = 0
        
        for opt_type, opt_data in optimizations.items():
            if isinstance(opt_data, dict):
                recommendations = opt_data.get("recommendations", [])
                total_issues += len(recommendations)
        
        if total_issues <= 2:
            return "low"
        elif total_issues <= 5:
            return "moderate"
        else:
            return "high"

# Import and expose segregated MCP analysis tools for compatibility
from .segregated_mcp_tools import (
    DependencyGraphTool,
    CodeSimilarityTool,
    HierarchicalIndexingTool,
    ConfigurationMergeTool,
    OptimizationRecommendationTool
)

# Import Glean MCP tools
from .glean_zero_blindspots_mcp_tool import (
    glean_zero_blindspots_validator_tool,
    GLEAN_ZERO_BLINDSPOTS_TOOL_METADATA
)
from .glean_continuous_monitor import (
    glean_continuous_monitor_tool,
    GLEAN_CONTINUOUS_MONITOR_TOOL_METADATA
)

# Register all Glean MCP tools
GLEAN_MCP_TOOLS = {
    "glean_zero_blindspots_validator": {
        "function": glean_zero_blindspots_validator_tool,
        "metadata": GLEAN_ZERO_BLINDSPOTS_TOOL_METADATA
    },
    "glean_continuous_monitor": {
        "function": glean_continuous_monitor_tool,
        "metadata": GLEAN_CONTINUOUS_MONITOR_TOOL_METADATA
    }
}

# Expose CLRSAnalysisTool from all_segregated_mcp_tools for compatibility
from .all_segregated_mcp_tools import CLRSAnalysisTool
