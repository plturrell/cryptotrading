"""
Code Analyzer - Production implementation for crypto trading platform
Provides comprehensive code analysis using Glean integration
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    import networkx as nx
except ImportError:
    # Fallback for environments without networkx
    nx = None

from .glean_client import CodeReference, CodeSymbol, GleanClient

logger = logging.getLogger(__name__)


@dataclass
class DependencyNode:
    """Represents a node in the dependency graph"""

    name: str
    type: str  # module, class, function
    file_path: str
    dependencies: Set[str]
    dependents: Set[str]
    complexity_score: float = 0.0


@dataclass
class ComplexityMetrics:
    """Code complexity metrics"""

    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    function_count: int
    class_count: int
    coupling_score: float
    cohesion_score: float


class DependencyAnalyzer:
    """Analyzes code dependencies using Glean"""

    def __init__(self, glean_client: GleanClient):
        self.glean = glean_client
        self.dependency_graph = nx.DiGraph() if nx else None
        self.module_cache: Dict[str, Dict] = {}

    async def build_dependency_graph(self, root_module: str = "cryptotrading") -> nx.DiGraph:
        """Build complete dependency graph for the codebase"""
        logger.info("Building dependency graph for %s", root_module)

        # Start with root module
        await self._add_module_to_graph(root_module)

        # Process all modules recursively
        modules_to_process = [root_module]
        processed_modules = set()

        while modules_to_process:
            current_module = modules_to_process.pop(0)
            if current_module in processed_modules:
                continue

            processed_modules.add(current_module)

            # Get module dependencies
            deps = await self.glean.get_dependencies(current_module, depth=1)

            for dep in deps["direct"]:
                if not self.dependency_graph.has_node(dep):
                    await self._add_module_to_graph(dep)
                    modules_to_process.append(dep)

                # Add edge
                self.dependency_graph.add_edge(current_module, dep)

        logger.info(
            "Dependency graph built: %d nodes, %d edges",
            len(self.dependency_graph.nodes),
            len(self.dependency_graph.edges),
        )
        return self.dependency_graph

    async def _add_module_to_graph(self, module: str):
        """Add a module to the dependency graph"""
        if self.dependency_graph.has_node(module):
            return

        # Get module facts
        facts = await self.glean.get_module_facts(module)
        self.module_cache[module] = facts

        # Add node with metadata
        self.dependency_graph.add_node(
            module,
            **{
                "symbol_count": facts["symbol_count"],
                "dependency_count": facts["dependency_count"],
                "dependent_count": facts["dependent_count"],
                "type": "module",
            },
        )

    async def analyze_module_dependencies(self, module: str, max_depth: int = 3) -> Dict[str, Any]:
        """Analyze dependencies for a specific module"""
        deps = await self.glean.get_dependencies(module, depth=max_depth)

        # Calculate dependency metrics
        direct_count = len(deps["direct"])
        transitive_count = len(deps["transitive"])

        # Analyze dependency types
        internal_deps = [d for d in deps["direct"] if d.startswith("cryptotrading")]
        external_deps = [d for d in deps["direct"] if not d.startswith("cryptotrading")]

        # Calculate coupling score
        coupling_score = self._calculate_coupling_score(direct_count, transitive_count)

        return {
            "module": module,
            "direct_dependencies": deps["direct"],
            "transitive_dependencies": deps["transitive"],
            "dependency_counts": {
                "direct": direct_count,
                "transitive": transitive_count,
                "internal": len(internal_deps),
                "external": len(external_deps),
            },
            "coupling_score": coupling_score,
            "risk_level": self._assess_dependency_risk(coupling_score, direct_count),
        }

    def _calculate_coupling_score(self, direct: int, transitive: int) -> float:
        """Calculate coupling score based on dependencies"""
        if direct == 0:
            return 0.0

        # Formula: (direct + transitive/2) / ideal_max_deps
        ideal_max_deps = 10  # Ideal maximum dependencies
        score = (direct + transitive / 2) / ideal_max_deps
        return min(score, 1.0)  # Cap at 1.0

    def _assess_dependency_risk(self, coupling_score: float, direct_deps: int) -> str:
        """Assess risk level based on coupling"""
        if coupling_score > 0.8 or direct_deps > 15:
            return "high"
        elif coupling_score > 0.5 or direct_deps > 8:
            return "medium"
        else:
            return "low"

    async def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the codebase"""
        if not self.dependency_graph:
            await self.build_dependency_graph()

        cycles = []
        try:
            # Find strongly connected components
            sccs = list(nx.strongly_connected_components(self.dependency_graph))

            for scc in sccs:
                if len(scc) > 1:  # Circular dependency
                    cycle = list(scc)
                    cycles.append(cycle)

        except (nx.NetworkXError, AttributeError) as e:
            logger.error("Error finding circular dependencies: %s", e)

        return cycles

    async def get_critical_modules(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify the most critical modules in the codebase"""
        if not self.dependency_graph:
            await self.build_dependency_graph()

        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(self.dependency_graph)
        in_degree = dict(self.dependency_graph.in_degree())
        out_degree = dict(self.dependency_graph.out_degree())

        critical_modules = []

        for module in self.dependency_graph.nodes():
            criticality_score = (
                betweenness.get(module, 0) * 0.4
                + (in_degree.get(module, 0) / max(in_degree.values(), default=1)) * 0.3
                + (out_degree.get(module, 0) / max(out_degree.values(), default=1)) * 0.3
            )

            critical_modules.append(
                {
                    "module": module,
                    "criticality_score": criticality_score,
                    "betweenness_centrality": betweenness.get(module, 0),
                    "incoming_dependencies": in_degree.get(module, 0),
                    "outgoing_dependencies": out_degree.get(module, 0),
                }
            )

        # Sort by criticality score
        critical_modules.sort(key=lambda x: x["criticality_score"], reverse=True)
        return critical_modules[:top_n]


class CodeAnalyzer:
    """Main code analyzer orchestrating all analysis functions"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or "/Users/apple/projects/cryptotrading")
        self.glean_client = None
        self.dependency_analyzer = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the analyzer with Glean client"""
        try:
            self.glean_client = GleanClient(project_root=str(self.project_root))

            async with self.glean_client:
                # Ensure codebase is indexed
                if not await self.glean_client.index_codebase():
                    logger.error("Failed to index codebase")
                    return False

                self.dependency_analyzer = DependencyAnalyzer(self.glean_client)
                self._initialized = True
                return True

        except (ImportError, ConnectionError, ValueError) as e:
            logger.error("Failed to initialize CodeAnalyzer: %s", e)
            return False

    async def analyze_symbol_usage(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive analysis of symbol usage"""
        if not self._initialized:
            await self.initialize()

        async with self.glean_client:
            # Find all references
            references = await self.glean_client.find_references(symbol)

            # Categorize references
            categorized = self._categorize_references(references)

            # Analyze usage patterns
            usage_patterns = self._analyze_usage_patterns(references)

            return {
                "symbol": symbol,
                "total_references": len(references),
                "categorized_references": categorized,
                "usage_patterns": usage_patterns,
                "files_affected": len(set(ref.file_path for ref in references)),
                "modules_affected": len(
                    set(self._get_module_from_path(ref.file_path) for ref in references)
                ),
            }

    def _categorize_references(
        self, references: List[CodeReference]
    ) -> Dict[str, List[CodeReference]]:
        """Categorize references by type"""
        categorized = defaultdict(list)

        for ref in references:
            categorized[ref.reference_type].append(ref)

        return dict(categorized)

    def _analyze_usage_patterns(self, references: List[CodeReference]) -> Dict[str, Any]:
        """Analyze patterns in symbol usage"""
        if not references:
            return {}

        # File distribution
        file_counts = defaultdict(int)
        for ref in references:
            file_counts[ref.file_path] += 1

        # Most used files
        most_used_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Module distribution
        module_counts = defaultdict(int)
        for ref in references:
            module = self._get_module_from_path(ref.file_path)
            module_counts[module] += 1

        return {
            "most_used_files": most_used_files,
            "module_distribution": dict(module_counts),
            "average_usage_per_file": len(references) / len(file_counts) if file_counts else 0,
        }

    def _get_module_from_path(self, file_path: str) -> str:
        """Extract module name from file path"""
        path = Path(file_path)
        if "src/cryptotrading" in str(path):
            # Extract module path
            parts = path.parts
            try:
                crypto_index = parts.index("cryptotrading")
                module_parts = parts[crypto_index:]
                # Remove .py extension
                if module_parts[-1].endswith(".py"):
                    module_parts = module_parts[:-1] + (module_parts[-1][:-3],)
                return ".".join(module_parts)
            except (ValueError, IndexError):
                return "unknown"
        return "external"

    async def analyze_code_complexity(self, module: str) -> ComplexityMetrics:
        """Analyze code complexity for a module"""
        if not self._initialized:
            await self.initialize()

        async with self.glean_client:
            # Get module facts
            facts = await self.glean_client.get_module_facts(module)

            # Calculate complexity metrics
            symbols = facts["symbols"]

            function_count = len([s for s in symbols if s.type == "function"])
            class_count = len([s for s in symbols if s.type == "class"])

            # Estimate complexity (would need AST analysis for precise values)
            cyclomatic_complexity = self._estimate_cyclomatic_complexity(symbols)
            cognitive_complexity = self._estimate_cognitive_complexity(symbols)

            # Calculate coupling and cohesion
            coupling_score = len(facts["imports"]) / 10.0  # Normalize to 0-1
            cohesion_score = self._calculate_cohesion_score(symbols)

            return ComplexityMetrics(
                cyclomatic_complexity=cyclomatic_complexity,
                cognitive_complexity=cognitive_complexity,
                lines_of_code=self._estimate_loc(module),
                function_count=function_count,
                class_count=class_count,
                coupling_score=min(coupling_score, 1.0),
                cohesion_score=cohesion_score,
            )

    def _estimate_cyclomatic_complexity(self, symbols: List[CodeSymbol]) -> int:
        """Estimate cyclomatic complexity"""
        # Simple estimation based on function count
        function_count = len([s for s in symbols if s.type == "function"])
        return max(1, function_count * 2)  # Rough estimate

    def _estimate_cognitive_complexity(self, symbols: List[CodeSymbol]) -> int:
        """Estimate cognitive complexity"""
        # Simple estimation
        function_count = len([s for s in symbols if s.type == "function"])
        class_count = len([s for s in symbols if s.type == "class"])
        return function_count + class_count * 2

    def _calculate_cohesion_score(self, symbols: List[CodeSymbol]) -> float:
        """Calculate module cohesion score"""
        if not symbols:
            return 0.0

        # Simple heuristic: ratio of classes to total symbols
        class_count = len([s for s in symbols if s.type == "class"])
        total_symbols = len(symbols)

        if total_symbols == 0:
            return 0.0

        # Higher class ratio indicates better cohesion
        return min(class_count / total_symbols * 2, 1.0)

    def _estimate_loc(self, module: str) -> int:
        """Estimate lines of code for a module"""
        try:
            module_path = (
                self.project_root / "src" / "cryptotrading" / f"{module.replace('.', '/')}.py"
            )
            if module_path.exists():
                with open(module_path, "r", encoding="utf-8") as f:
                    return len(
                        [line for line in f if line.strip() and not line.strip().startswith("#")]
                    )
        except (IOError, OSError, UnicodeDecodeError):
            pass
        return 0

    async def detect_dead_code(self) -> List[Dict[str, Any]]:
        """Detect potentially dead code"""
        if not self._initialized:
            await self.initialize()

        async with self.glean_client:
            dead_code_candidates = []

            # Get all function definitions in the codebase
            src_path = self.project_root / "src" / "cryptotrading"

            for py_file in src_path.rglob("*.py"):
                symbols = await self.glean_client.get_file_symbols(str(py_file))

                for symbol in symbols:
                    if symbol.type == "function":
                        # Check if function has references
                        refs = await self.glean_client.find_references(symbol.name)

                        # Filter out self-references (definition)
                        external_refs = [
                            ref
                            for ref in refs
                            if ref.file_path != symbol.file_path or ref.line != symbol.line
                        ]

                        if len(external_refs) == 0:
                            # Potential dead code
                            dead_code_candidates.append(
                                {
                                    "symbol": symbol.name,
                                    "type": symbol.type,
                                    "file": symbol.file_path,
                                    "line": symbol.line,
                                    "reason": "No external references found",
                                }
                            )

            return dead_code_candidates

    async def analyze_dependencies(self, module: str = None, max_depth: int = 3) -> Dict[str, Any]:
        """Analyze dependencies for a module or entire codebase"""
        if not self._initialized:
            await self.initialize()

        if module:
            # Analyze specific module
            return await self.dependency_analyzer.analyze_module_dependencies(module, max_depth)
        else:
            # Analyze entire codebase
            async with self.glean_client:
                # Build dependency graph
                graph = await self.dependency_analyzer.build_dependency_graph()

                # Get critical modules
                critical_modules = await self.dependency_analyzer.get_critical_modules()

                # Find circular dependencies
                circular_deps = await self.dependency_analyzer.find_circular_dependencies()

                # Calculate overall metrics
                total_modules = len(graph.nodes())
                total_edges = len(graph.edges())
                avg_dependencies = total_edges / total_modules if total_modules > 0 else 0

                return {
                    "codebase_overview": {
                        "total_modules": total_modules,
                        "total_dependencies": total_edges,
                        "average_dependencies_per_module": avg_dependencies,
                    },
                    "critical_modules": critical_modules,
                    "circular_dependencies": circular_deps,
                    "dependency_graph_stats": {
                        "nodes": total_modules,
                        "edges": total_edges,
                        "density": nx.density(graph) if total_modules > 1 else 0,
                    },
                }

    async def cleanup(self):
        """Cleanup resources"""
        if self.glean_client:
            await self.glean_client.cleanup()
