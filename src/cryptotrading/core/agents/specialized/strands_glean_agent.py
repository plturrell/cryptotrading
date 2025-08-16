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
                "architecture_analysis": ArchitectureAnalysisCapability(self.glean_client)
            })
        
        logger.info(f"StrandsGleanAgent initialized with {len(self.capabilities)} capabilities")
    
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
        """Perform code analysis using specified capability"""
        if analysis_type not in self.capabilities:
            return {
                "status": "error",
                "error": f"Unknown analysis type: {analysis_type}",
                "available_types": list(self.capabilities.keys())
            }
        
        try:
            # Update context if provided
            if context:
                for key, value in context.items():
                    if hasattr(self.context, key):
                        setattr(self.context, key, value)
            
            # Perform analysis
            capability = self.capabilities[analysis_type]
            result = await capability.analyze(self.context, query)
            
            # Record query in history
            self.context.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": analysis_type,
                "query": query,
                "result_status": result.get("status", "unknown")
            })
            
            return result
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def find_dependencies(self, symbol: str) -> Dict[str, Any]:
        """Find dependencies for a symbol"""
        return await self.analyze_code("dependency_analysis", symbol)
    
    async def search_symbols(self, pattern: str) -> Dict[str, Any]:
        """Search for symbols matching pattern"""
        return await self.analyze_code("symbol_search", pattern)
    
    async def analyze_architecture(self, component: str) -> Dict[str, Any]:
        """Analyze architectural patterns for a component"""
        return await self.analyze_code("architecture_analysis", component)
    
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
            
            else:
                return {
                    "status": "error",
                    "error": f"Unknown strand type: {strand_type}",
                    "supported_types": ["code_analysis", "dependency_trace", "symbol_lookup", "architecture_review"]
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


if __name__ == "__main__":
    asyncio.run(test_strands_glean_agent())