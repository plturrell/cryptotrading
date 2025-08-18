"""
Enhanced Angle Queries for Multi-Language Support
Extends the Angle query system to support SAP CAP, JavaScript/UI5, and configuration files
"""

from typing import Dict, List, Optional, Any
from .angle_parser import AngleQueryEngine, PYTHON_QUERIES

# SAP CAP Angle queries
CAP_QUERIES = {
    "find_entity": """
        cap.Entity {
            name = ?name,
            namespace = ?namespace,
            file = ?file
        }
    """,
    
    "find_service": """
        cap.Service {
            name = ?name,
            namespace = ?namespace,
            file = ?file
        }
    """,
    
    "find_associations": """
        cap.Element {
            entity = "{entity}",
            type = "Association",
            target = ?target
        }
    """,
    
    "find_compositions": """
        cap.Element {
            entity = "{entity}",
            type = "Composition", 
            target = ?target
        }
    """,
    
    "find_entity_elements": """
        cap.Element {
            entity = "{entity}",
            name = ?element_name,
            type = ?element_type
        }
    """,
    
    "find_service_entities": """
        cap.Service {
            name = "{service}",
            entities = ?exposed_entities
        }
    """,
    
    "find_cap_imports": """
        cap.Using {
            file = ?file,
            import = ?imported_module
        }
    """
}

# TypeScript Angle queries
TYPESCRIPT_QUERIES = {
    "find_ts_interface": """
        typescript.Declaration {
            name = ?name,
            kind = "interface",
            file = ?file
        }
    """,
    
    "find_ts_class": """
        typescript.Declaration {
            name = ?name,
            kind = "class",
            file = ?file,
            extends = ?parent_class
        }
    """,
    
    "find_ts_function": """
        typescript.Declaration {
            name = ?name,
            kind = "function",
            file = ?file,
            type_annotation = ?return_type
        }
    """,
    
    "find_ts_type": """
        typescript.Declaration {
            name = ?name,
            kind = "type",
            file = ?file
        }
    """,
    
    "find_ts_enum": """
        typescript.Declaration {
            name = ?name,
            kind = "enum",
            file = ?file
        }
    """,
    
    "find_ts_imports": """
        typescript.Import {
            file = ?file,
            module = ?imported_module,
            import_type = ?import_type
        }
    """
}

# JavaScript/UI5 Angle queries
JAVASCRIPT_QUERIES = {
    "find_js_function": """
        javascript.Function {
            name = ?name,
            file = ?file,
            is_async = ?is_async
        }
    """,
    
    "find_js_class": """
        javascript.Class {
            name = ?name,
            file = ?file,
            extends = ?parent_class
        }
    """,
    
    "find_js_imports": """
        javascript.Import {
            file = ?file,
            module = ?imported_module
        }
    """,
    
    "find_ui5_controller": """
        ui5.Controller {
            name = ?name,
            file = ?file,
            extends = ?base_controller
        }
    """,
    
    "find_ui5_view": """
        ui5.View {
            name = ?name,
            file = ?file,
            controller = ?controller_name
        }
    """,
    
    "find_event_handlers": """
        ui5.Controller {
            name = "{controller}",
            event_handlers = ?handlers
        }
    """,
    
    "find_ui5_dependencies": """
        ui5.Controller {
            name = "{controller}",
            dependencies = ?deps
        }
    """
}

# Configuration file queries
CONFIG_QUERIES = {
    "find_config_files": """
        config.File {
            path = ?path,
            type = ?config_type
        }
    """,
    
    "find_json_configs": """
        config.File {
            type = "json",
            path = ?path,
            keys = ?configuration_keys
        }
    """
}

# Cross-language relationship queries
CROSS_LANGUAGE_QUERIES = {
    "find_ui5_python_integration": """
        ui5.Controller {
            name = ?controller,
            file = ?ui5_file
        }
        python.Declaration {
            name = ?python_class,
            file = ?python_file
        }
    """,
    
    "find_cap_service_usage": """
        cap.Service {
            name = ?service,
            file = ?cap_file
        }
        javascript.Import {
            module = ?service_import,
            file = ?js_file
        }
    """,
    
    "find_all_languages_in_file": """
        src.File {
            path = ?file_path,
            language = ?language
        }
    """
}

# Combined query dictionary
ALL_QUERIES = {
    **PYTHON_QUERIES,
    **CAP_QUERIES, 
    **TYPESCRIPT_QUERIES,
    **JAVASCRIPT_QUERIES,
    **CONFIG_QUERIES,
    **CROSS_LANGUAGE_QUERIES
}

class EnhancedAngleQueryEngine(AngleQueryEngine):
    """Enhanced query engine with multi-language support"""
    
    def __init__(self, facts: List[Dict[str, Any]]):
        super().__init__(facts)
        self.language_stats = self._calculate_language_stats()
    
    def _calculate_language_stats(self) -> Dict[str, int]:
        """Calculate statistics by language"""
        stats = {}
        for fact in self.facts:
            if fact.get("predicate") == "src.File":
                language = fact.get("value", {}).get("language", "unknown")
                stats[language] = stats.get(language, 0) + 1
        return stats
    
    def query_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Query facts for a specific language"""
        results = []
        for fact in self.facts:
            if fact.get("predicate") == "src.File":
                if fact.get("value", {}).get("language") == language:
                    results.append(fact)
        return results
    
    def find_cross_language_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find relationships between different languages"""
        relationships = {
            "ui5_controllers_with_python": [],
            "cap_services_with_js_usage": [],
            "config_files_with_code_refs": []
        }
        
        # Find UI5 controllers that might relate to Python classes
        ui5_controllers = [f for f in self.facts if f.get("predicate") == "ui5.Controller"]
        python_classes = [f for f in self.facts if f.get("predicate") == "python.Declaration" 
                         and f.get("value", {}).get("kind") == "class"]
        
        for controller in ui5_controllers:
            controller_name = controller.get("key", {}).get("name", "")
            for py_class in python_classes:
                class_name = py_class.get("key", {}).get("name", "")
                if controller_name.lower() in class_name.lower() or class_name.lower() in controller_name.lower():
                    relationships["ui5_controllers_with_python"].append({
                        "controller": controller,
                        "python_class": py_class
                    })
        
        return relationships
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all languages"""
        predicate_counts = {}
        for fact in self.facts:
            predicate = fact.get("predicate", "unknown")
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
        
        return {
            "total_facts": len(self.facts),
            "language_distribution": self.language_stats,
            "predicate_distribution": predicate_counts,
            "supported_languages": list(self.language_stats.keys()),
            "cross_language_relationships": len(self.find_cross_language_relationships())
        }

def create_enhanced_query(template: str, **kwargs) -> str:
    """Create an enhanced Angle query from template with parameters"""
    query = ALL_QUERIES.get(template, template)
    
    for key, value in kwargs.items():
        query = query.replace(f"{{{key}}}", str(value))
    
    return query

# Specialized query functions for each language
def find_cap_entities(engine: EnhancedAngleQueryEngine, namespace: str = None) -> List[Dict[str, Any]]:
    """Find all CAP entities, optionally filtered by namespace"""
    if namespace:
        query = create_enhanced_query("find_entity", namespace=namespace)
    else:
        query = ALL_QUERIES["find_entity"]
    return engine.query(query)

def find_ui5_controllers(engine: EnhancedAngleQueryEngine) -> List[Dict[str, Any]]:
    """Find all UI5 controllers"""
    return engine.query(ALL_QUERIES["find_ui5_controller"])

def find_javascript_functions(engine: EnhancedAngleQueryEngine, is_async: bool = None) -> List[Dict[str, Any]]:
    """Find JavaScript functions, optionally filtered by async status"""
    if is_async is not None:
        query = create_enhanced_query("find_js_function", is_async=str(is_async).lower())
    else:
        query = ALL_QUERIES["find_js_function"]
    return engine.query(query)

def find_typescript_interfaces(engine: EnhancedAngleQueryEngine) -> List[Dict[str, Any]]:
    """Find all TypeScript interfaces"""
    return engine.query(ALL_QUERIES["find_ts_interface"])

def find_typescript_classes(engine: EnhancedAngleQueryEngine) -> List[Dict[str, Any]]:
    """Find all TypeScript classes"""
    return engine.query(ALL_QUERIES["find_ts_class"])

def find_typescript_functions(engine: EnhancedAngleQueryEngine) -> List[Dict[str, Any]]:
    """Find all TypeScript functions"""
    return engine.query(ALL_QUERIES["find_ts_function"])

def analyze_project_architecture(engine: EnhancedAngleQueryEngine) -> Dict[str, Any]:
    """Analyze the overall project architecture across all languages"""
    return {
        "python_components": {
            "classes": len([f for f in engine.facts if f.get("predicate") == "python.Declaration" 
                           and f.get("value", {}).get("kind") == "class"]),
            "functions": len([f for f in engine.facts if f.get("predicate") == "python.Declaration" 
                            and f.get("value", {}).get("kind") == "function"])
        },
        "cap_components": {
            "entities": len([f for f in engine.facts if f.get("predicate") == "cap.Entity"]),
            "services": len([f for f in engine.facts if f.get("predicate") == "cap.Service"])
        },
        "ui5_components": {
            "controllers": len([f for f in engine.facts if f.get("predicate") == "ui5.Controller"]),
            "views": len([f for f in engine.facts if f.get("predicate") == "ui5.View"])
        },
        "javascript_components": {
            "functions": len([f for f in engine.facts if f.get("predicate") == "javascript.Function"]),
            "classes": len([f for f in engine.facts if f.get("predicate") == "javascript.Class"])
        }
    }
