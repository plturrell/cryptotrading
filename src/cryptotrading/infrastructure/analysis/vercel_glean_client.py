"""
Vercel-Compatible Glean Client
Real Glean implementation using SCIP indexing without Docker dependencies
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime

from .scip_indexer import PythonSCIPIndexer, index_project_for_glean
from .angle_parser import AngleQueryEngine, create_query, PYTHON_QUERIES
from .glean_storage import GleanStorage, GleanFact, initialize_python_schemas

logger = logging.getLogger(__name__)


class VercelGleanClient:
    """Vercel-compatible Glean client using serverless storage and SCIP indexing"""
    
    def __init__(self, project_root: str = None, storage_path: Optional[Path] = None):
        self.project_root = Path(project_root or "/Users/apple/projects/cryptotrading")
        self.storage = GleanStorage(storage_path)
        self.indexer = PythonSCIPIndexer(self.project_root)
        self.query_engine: Optional[AngleQueryEngine] = None
        self.indexed_units: Set[str] = set()
        
        # Initialize Python schemas
        initialize_python_schemas(self.storage)
        logger.info("Initialized Vercel Glean client")
    
    async def index_project(self, unit_name: str = "default", force_reindex: bool = False) -> Dict[str, Any]:
        """Index the entire project using SCIP"""
        try:
            # Check if already indexed
            if unit_name in self.indexed_units and not force_reindex:
                logger.info(f"Unit '{unit_name}' already indexed")
                return {"status": "already_indexed", "unit": unit_name}
            
            # Delete existing unit if force reindexing
            if force_reindex:
                self.storage.delete_unit(unit_name)
            
            # Generate SCIP index and Glean facts
            logger.info(f"Indexing project for unit '{unit_name}'...")
            result = index_project_for_glean(str(self.project_root))
            
            # Store facts in Glean storage
            facts_stored = self.storage.store_facts(result["glean_facts"], unit_name)
            
            # Update query engine with new facts
            await self._refresh_query_engine()
            
            self.indexed_units.add(unit_name)
            
            return {
                "status": "success",
                "unit": unit_name,
                "stats": result["stats"],
                "facts_stored": facts_stored
            }
            
        except Exception as e:
            logger.error(f"Failed to index project: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _refresh_query_engine(self) -> None:
        """Refresh the query engine with current facts"""
        # Get all facts from storage
        predicates = self.storage.get_predicates()
        all_facts = []
        
        for predicate in predicates:
            facts = self.storage.query_facts(predicate)
            for fact in facts:
                all_facts.append(fact.to_dict())
        
        self.query_engine = AngleQueryEngine(all_facts)
        logger.info(f"Query engine refreshed with {len(all_facts)} facts")
    
    async def query(self, angle_query: str) -> List[Dict[str, Any]]:
        """Execute an Angle query"""
        if not self.query_engine:
            await self._refresh_query_engine()
        
        try:
            results = self.query_engine.query(angle_query)
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    async def find_function_definitions(self, function_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find function definitions"""
        if function_name:
            query = create_query("find_definitions", name=function_name, kind="function")
        else:
            query = PYTHON_QUERIES["find_function"]
        
        return await self.query(query)
    
    async def find_class_definitions(self, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find class definitions"""
        if class_name:
            query = create_query("find_definitions", name=class_name, kind="class")
        else:
            query = PYTHON_QUERIES["find_class"]
        
        return await self.query(query)
    
    async def find_references(self, symbol: str) -> List[Dict[str, Any]]:
        """Find all references to a symbol"""
        query = create_query("find_references", symbol=symbol)
        return await self.query(query)
    
    async def find_imports(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find import statements"""
        if file_path:
            # Custom query for imports in specific file
            query = f"""
                python.Reference {{
                    symbol_roles = 8,
                    file = "{file_path}",
                    target = ?module
                }}
            """
        else:
            query = PYTHON_QUERIES["find_imports"]
        
        return await self.query(query)
    
    async def get_file_symbols(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all symbols defined in a file"""
        query = create_query("file_symbols", file=file_path)
        return await self.query(query)
    
    async def analyze_dependencies(self, module: str, max_depth: int = 3) -> Dict[str, Any]:
        """Analyze module dependencies using Glean facts"""
        try:
            # Find all imports for the module
            imports_query = f"""
                python.Reference {{
                    file = "{module}",
                    symbol_roles = 8,
                    target = ?imported_module
                }}
            """
            
            imports = await self.query(imports_query)
            
            # Build dependency graph
            dependencies = {
                "direct": [],
                "transitive": [],
                "depth_map": {}
            }
            
            for imp in imports:
                target = imp.get("key", {}).get("target", "")
                if target:
                    dependencies["direct"].append(target)
                    dependencies["depth_map"][target] = 1
            
            # Recursively find transitive dependencies
            visited = set(dependencies["direct"])
            current_depth = 1
            
            while current_depth < max_depth:
                current_level = [mod for mod, depth in dependencies["depth_map"].items() 
                               if depth == current_depth]
                
                for mod in current_level:
                    mod_imports_query = f"""
                        python.Reference {{
                            file = "{mod}",
                            symbol_roles = 8,
                            target = ?imported_module
                        }}
                    """
                    
                    mod_imports = await self.query(mod_imports_query)
                    
                    for imp in mod_imports:
                        target = imp.get("key", {}).get("target", "")
                        if target and target not in visited:
                            dependencies["transitive"].append(target)
                            dependencies["depth_map"][target] = current_depth + 1
                            visited.add(target)
                
                current_depth += 1
            
            return {
                "module": module,
                "dependencies": dependencies,
                "total_dependencies": len(visited),
                "max_depth_analyzed": max_depth
            }
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {"module": module, "error": str(e)}
    
    async def validate_architecture(self, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate architectural constraints"""
        if not rules:
            rules = self._get_default_architecture_rules()
        
        violations = []
        
        try:
            # Check layer dependencies
            for forbidden in rules.get("forbidden_dependencies", []):
                source_pattern = forbidden["source"]
                target_pattern = forbidden["target"]
                
                # Query for violations
                violation_query = f"""
                    python.Reference {{
                        file = ?source_file,
                        target = ?target_module
                    }}
                """
                
                refs = await self.query(violation_query)
                
                for ref in refs:
                    source_file = ref.get("key", {}).get("file", "")
                    target_module = ref.get("key", {}).get("target", "")
                    
                    if (source_pattern in source_file and 
                        target_pattern in target_module):
                        violations.append({
                            "type": "forbidden_dependency",
                            "source": source_file,
                            "target": target_module,
                            "rule": forbidden,
                            "severity": forbidden.get("severity", "medium")
                        })
            
            return {
                "status": "completed",
                "violations": violations,
                "total_violations": len(violations),
                "rules_checked": len(rules.get("forbidden_dependencies", []))
            }
            
        except Exception as e:
            logger.error(f"Architecture validation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_default_architecture_rules(self) -> Dict[str, Any]:
        """Get default architecture rules for crypto trading platform"""
        return {
            "forbidden_dependencies": [
                {
                    "source": "cryptotrading/core",
                    "target": "cryptotrading/infrastructure",
                    "severity": "high",
                    "message": "Core should not depend on infrastructure"
                },
                {
                    "source": "cryptotrading/data",
                    "target": "cryptotrading/core/agents",
                    "severity": "high",
                    "message": "Data layer should not depend on core business logic"
                },
                {
                    "source": "cryptotrading/utils",
                    "target": "cryptotrading/core",
                    "severity": "medium",
                    "message": "Utils should not depend on core components"
                }
            ],
            "size_limits": {
                "max_functions_per_file": 20,
                "max_classes_per_file": 10,
                "max_lines_per_file": 500
            }
        }
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self.storage.get_stats()
    
    async def export_unit(self, unit: str, output_path: str) -> None:
        """Export unit facts to JSON"""
        self.storage.export_unit(unit, Path(output_path))
    
    async def import_facts(self, input_path: str, unit: Optional[str] = None) -> int:
        """Import facts from JSON file"""
        return self.storage.import_facts(Path(input_path), unit)
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        # No async resources to cleanup in this implementation
        pass
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


# Convenience aliases for backward compatibility
GleanClient = VercelGleanClient