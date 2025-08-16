"""
Tree Library Implementation for Nested Data Structure Processing
Generalizes map function to apply operations to nested structures while preserving structure
"""

from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple, TypeVar
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import copy

T = TypeVar('T')
U = TypeVar('U')

# Type definitions
NestedStructure = Union[
    T,
    List['NestedStructure[T]'],
    Dict[str, 'NestedStructure[T]'],
    Tuple['NestedStructure[T]', ...],
    Set['NestedStructure[T]']
]

@dataclass
class TreePath:
    """Represents a path to a value in a nested structure"""
    keys: List[Union[str, int]]
    value: Any

@dataclass
class TreeDiff:
    """Represents differences between two tree structures"""
    added: List[TreePath]
    removed: List[TreePath]
    modified: List[Dict[str, Any]]

class TreeOperations:
    """Core tree operations for nested data structures"""
    
    @staticmethod
    def flatten(structure: NestedStructure[T]) -> List[T]:
        """Flatten nested structure into a list of leaf values"""
        result = []
        
        def _flatten_recursive(obj: Any):
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _flatten_recursive(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    _flatten_recursive(value)
            elif isinstance(obj, set):
                for item in obj:
                    _flatten_recursive(item)
            else:
                # Leaf node
                result.append(obj)
        
        _flatten_recursive(structure)
        return result
    
    @staticmethod
    def map_structure(fn: Callable[[T], U], structure: NestedStructure[T]) -> NestedStructure[U]:
        """Apply function to each leaf while preserving structure"""
        def _map_recursive(obj: Any) -> Any:
            if isinstance(obj, list):
                return [_map_recursive(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(_map_recursive(item) for item in obj)
            elif isinstance(obj, dict):
                return {key: _map_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, set):
                return {_map_recursive(item) for item in obj}
            else:
                # Leaf node - apply function
                return fn(obj)
        
        return _map_recursive(structure)
    
    @staticmethod
    def filter_structure(predicate: Callable[[T], bool], structure: NestedStructure[T]) -> NestedStructure[T]:
        """Filter structure keeping only elements that satisfy predicate"""
        def _filter_recursive(obj: Any) -> Any:
            if isinstance(obj, list):
                filtered = [_filter_recursive(item) for item in obj]
                return [item for item in filtered if item is not None]
            elif isinstance(obj, tuple):
                filtered = tuple(_filter_recursive(item) for item in obj)
                return tuple(item for item in filtered if item is not None)
            elif isinstance(obj, dict):
                filtered = {}
                for key, value in obj.items():
                    filtered_value = _filter_recursive(value)
                    if filtered_value is not None:
                        filtered[key] = filtered_value
                return filtered
            elif isinstance(obj, set):
                filtered = {_filter_recursive(item) for item in obj}
                return {item for item in filtered if item is not None}
            else:
                # Leaf node - apply predicate
                return obj if predicate(obj) else None
        
        return _filter_recursive(structure)
    
    @staticmethod
    def reduce_structure(fn: Callable[[U, T], U], initial: U, structure: NestedStructure[T]) -> U:
        """Reduce structure to a single value"""
        flattened = TreeOperations.flatten(structure)
        result = initial
        for item in flattened:
            result = fn(result, item)
        return result

class PathOperations:
    """Operations for navigating and manipulating paths in nested structures"""
    
    @staticmethod
    def get_path(structure: NestedStructure[T], path: List[Union[str, int]]) -> Optional[T]:
        """Get value at specified path"""
        current = structure
        
        try:
            for key in path:
                if isinstance(current, dict):
                    current = current[key]
                elif isinstance(current, (list, tuple)):
                    current = current[int(key)]
                else:
                    return None
            return current
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    
    @staticmethod
    def set_path(structure: NestedStructure[T], path: List[Union[str, int]], value: T) -> NestedStructure[T]:
        """Set value at specified path, creating intermediate structures as needed"""
        if not path:
            return value
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(structure)
        current = result
        
        # Navigate to parent of target
        for key in path[:-1]:
            if isinstance(current, dict):
                if key not in current:
                    # Determine if next key suggests list or dict
                    next_key = path[path.index(key) + 1]
                    current[key] = [] if isinstance(next_key, int) else {}
                current = current[key]
            elif isinstance(current, list):
                idx = int(key)
                # Extend list if necessary
                while len(current) <= idx:
                    current.append(None)
                if current[idx] is None:
                    # Determine structure type for next level
                    next_key = path[path.index(key) + 1]
                    current[idx] = [] if isinstance(next_key, int) else {}
                current = current[idx]
        
        # Set the final value
        final_key = path[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            idx = int(final_key)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        
        return result
    
    @staticmethod
    def delete_path(structure: NestedStructure[T], path: List[Union[str, int]]) -> NestedStructure[T]:
        """Delete value at specified path"""
        if not path:
            return None
        
        result = copy.deepcopy(structure)
        current = result
        
        # Navigate to parent
        try:
            for key in path[:-1]:
                if isinstance(current, dict):
                    current = current[key]
                elif isinstance(current, (list, tuple)):
                    current = current[int(key)]
            
            # Delete final key
            final_key = path[-1]
            if isinstance(current, dict):
                del current[final_key]
            elif isinstance(current, list):
                current.pop(int(final_key))
        
        except (KeyError, IndexError, ValueError, TypeError):
            pass  # Path doesn't exist
        
        return result
    
    @staticmethod
    def get_all_paths(structure: NestedStructure[T]) -> List[TreePath]:
        """Get all paths and their values in the structure"""
        paths = []
        
        def _collect_paths(obj: Any, current_path: List[Union[str, int]]):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = current_path + [key]
                    if StructuralAnalysis.is_leaf(value):
                        paths.append(TreePath(keys=new_path, value=value))
                    else:
                        _collect_paths(value, new_path)
            elif isinstance(obj, (list, tuple)):
                for i, value in enumerate(obj):
                    new_path = current_path + [i]
                    if StructuralAnalysis.is_leaf(value):
                        paths.append(TreePath(keys=new_path, value=value))
                    else:
                        _collect_paths(value, new_path)
            elif isinstance(obj, set):
                # Sets don't have stable ordering, so we can't provide meaningful paths
                for i, value in enumerate(sorted(obj, key=str)):
                    new_path = current_path + [f"set_item_{i}"]
                    if StructuralAnalysis.is_leaf(value):
                        paths.append(TreePath(keys=new_path, value=value))
                    else:
                        _collect_paths(value, new_path)
            else:
                # Leaf at root level
                paths.append(TreePath(keys=current_path, value=obj))
        
        _collect_paths(structure, [])
        return paths
    

class StructuralAnalysis:
    """Analysis operations for nested structures"""
    
    @staticmethod
    def get_depth(structure: NestedStructure[T]) -> int:
        """Get maximum depth of nested structure"""
        def _depth_recursive(obj: Any, current_depth: int = 0) -> int:
            if isinstance(obj, (dict, list, tuple, set)):
                if not obj:  # Empty container
                    return current_depth
                
                max_child_depth = current_depth
                if isinstance(obj, dict):
                    for value in obj.values():
                        max_child_depth = max(max_child_depth, _depth_recursive(value, current_depth + 1))
                else:  # list, tuple, set
                    for item in obj:
                        max_child_depth = max(max_child_depth, _depth_recursive(item, current_depth + 1))
                return max_child_depth
            else:
                return current_depth
        
        return _depth_recursive(structure)
    
    @staticmethod
    def get_leaf_count(structure: NestedStructure[T]) -> int:
        """Count number of leaf nodes"""
        return len(TreeOperations.flatten(structure))
    
    @staticmethod
    def get_node_count(structure: NestedStructure[T]) -> int:
        """Count total number of nodes (including containers)"""
        def _count_recursive(obj: Any) -> int:
            count = 1  # Count current node
            
            if isinstance(obj, dict):
                for value in obj.values():
                    count += _count_recursive(value)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    count += _count_recursive(item)
            
            return count
        
        return _count_recursive(structure)
    
    @staticmethod
    def is_leaf(structure: NestedStructure[T]) -> bool:
        """Check if structure is a leaf node"""
        return not isinstance(structure, (dict, list, tuple, set))
    
    @staticmethod
    def find_substructures(structure: NestedStructure[T], predicate: Callable[[Any], bool]) -> List[TreePath]:
        """Find all substructures matching predicate"""
        matches = []
        
        def _search_recursive(obj: Any, current_path: List[Union[str, int]]):
            if predicate(obj):
                matches.append(TreePath(keys=current_path.copy(), value=obj))
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    _search_recursive(value, current_path + [key])
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _search_recursive(item, current_path + [i])
            elif isinstance(obj, set):
                for i, item in enumerate(sorted(obj, key=str)):
                    _search_recursive(item, current_path + [f"set_item_{i}"])
        
        _search_recursive(structure, [])
        return matches

class TreeDiffMerge:
    """Operations for comparing and merging tree structures"""
    
    @staticmethod
    def diff(old_tree: NestedStructure[T], new_tree: NestedStructure[T]) -> TreeDiff:
        """Compare two tree structures and return differences"""
        old_paths = {tuple(p.keys): p.value for p in PathOperations.get_all_paths(old_tree)}
        new_paths = {tuple(p.keys): p.value for p in PathOperations.get_all_paths(new_tree)}
        
        added = []
        removed = []
        modified = []
        
        # Find added paths
        for path_tuple, value in new_paths.items():
            if path_tuple not in old_paths:
                added.append(TreePath(keys=list(path_tuple), value=value))
        
        # Find removed paths
        for path_tuple, value in old_paths.items():
            if path_tuple not in new_paths:
                removed.append(TreePath(keys=list(path_tuple), value=value))
        
        # Find modified paths
        for path_tuple in old_paths:
            if path_tuple in new_paths and old_paths[path_tuple] != new_paths[path_tuple]:
                modified.append({
                    'path': TreePath(keys=list(path_tuple), value=new_paths[path_tuple]),
                    'old_value': old_paths[path_tuple],
                    'new_value': new_paths[path_tuple]
                })
        
        return TreeDiff(added=added, removed=removed, modified=modified)
    
    @staticmethod
    def merge(base: NestedStructure[T], changes: TreeDiff) -> NestedStructure[T]:
        """Apply changes to base structure"""
        result = copy.deepcopy(base)
        
        # Apply additions and modifications
        for path_obj in changes.added:
            result = PathOperations.set_path(result, path_obj.keys, path_obj.value)
        
        for change in changes.modified:
            result = PathOperations.set_path(result, change['path'].keys, change['new_value'])
        
        # Apply removals
        for path_obj in changes.removed:
            result = PathOperations.delete_path(result, path_obj.keys)
        
        return result
    
    @staticmethod
    def patch(structure: NestedStructure[T], patches: TreeDiff) -> NestedStructure[T]:
        """Apply patches to structure (alias for merge)"""
        return TreeDiffMerge.merge(structure, patches)

# AST Processing Classes
class ASTNode:
    """Base class for AST nodes"""
    def __init__(self, node_type: str, **kwargs):
        self.type = node_type
        self.attributes = kwargs

class FunctionNode(ASTNode):
    """Function definition node"""
    def __init__(self, name: str, parameters: List[str], body: Any):
        super().__init__("function")
        self.name = name
        self.parameters = parameters
        self.body = body

class VariableNode(ASTNode):
    """Variable declaration node"""
    def __init__(self, name: str, value: Any):
        super().__init__("variable")
        self.name = name
        self.value = value

class ClassNode(ASTNode):
    """Class definition node"""
    def __init__(self, name: str, methods: List[FunctionNode]):
        super().__init__("class")
        self.name = name
        self.methods = methods

class ASTVisitor:
    """Visitor pattern for AST traversal"""
    
    def visit_node(self, node: ASTNode, path: TreePath) -> None:
        """Visit generic node"""
        pass
    
    def visit_function(self, node: FunctionNode, path: TreePath) -> None:
        """Visit function node"""
        self.visit_node(node, path)
    
    def visit_variable(self, node: VariableNode, path: TreePath) -> None:
        """Visit variable node"""
        self.visit_node(node, path)
    
    def visit_class(self, node: ClassNode, path: TreePath) -> None:
        """Visit class node"""
        self.visit_node(node, path)

class ASTProcessor:
    """AST processing using tree operations"""
    
    @staticmethod
    def traverse_ast(ast: NestedStructure[ASTNode], visitor: ASTVisitor) -> None:
        """Traverse AST with visitor pattern"""
        paths = PathOperations.get_all_paths(ast)
        
        for path in paths:
            node = path.value
            if isinstance(node, ASTNode):
                if isinstance(node, FunctionNode):
                    visitor.visit_function(node, path)
                elif isinstance(node, VariableNode):
                    visitor.visit_variable(node, path)
                elif isinstance(node, ClassNode):
                    visitor.visit_class(node, path)
                else:
                    visitor.visit_node(node, path)
    
    @staticmethod
    def transform_ast(ast: NestedStructure[ASTNode], transformer: Callable[[ASTNode], ASTNode]) -> NestedStructure[ASTNode]:
        """Transform AST using tree operations"""
        return TreeOperations.map_structure(
            lambda node: transformer(node) if isinstance(node, ASTNode) else node,
            ast
        )
    
    @staticmethod
    def find_nodes(ast: NestedStructure[ASTNode], predicate: Callable[[ASTNode], bool]) -> List[ASTNode]:
        """Find nodes matching predicate"""
        matches = StructuralAnalysis.find_substructures(ast, predicate)
        return [match.value for match in matches if isinstance(match.value, ASTNode)]
    
    @staticmethod
    def replace_nodes(ast: NestedStructure[ASTNode], replacements: Dict[ASTNode, ASTNode]) -> NestedStructure[ASTNode]:
        """Replace nodes in AST"""
        return TreeOperations.map_structure(
            lambda node: replacements.get(node, node) if isinstance(node, ASTNode) else node,
            ast
        )

# Glean Integration Classes
class HierarchicalCodeIndex:
    """Hierarchical code indexing using tree operations"""
    
    def __init__(self):
        self.code_structure: NestedStructure[Any] = {}
        self.tree_ops = TreeOperations()
        self.path_ops = PathOperations()
    
    def index_codebase(self, files: Dict[str, Any]) -> None:
        """Build hierarchical index of codebase"""
        self.code_structure = self._build_code_hierarchy(files)
    
    def _build_code_hierarchy(self, files: Dict[str, Any]) -> NestedStructure[Any]:
        """Build hierarchical structure from files"""
        hierarchy = {}
        
        for file_path, content in files.items():
            # Split path into components
            path_parts = file_path.split('/')
            
            # Build nested structure
            current = hierarchy
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add file content
            current[path_parts[-1]] = content
        
        return hierarchy
    
    def find_symbols(self, query: str) -> List[Any]:
        """Find symbols matching query"""
        def matches_query(element: Any) -> bool:
            if hasattr(element, 'name'):
                return query.lower() in element.name.lower()
            elif isinstance(element, str):
                return query.lower() in element.lower()
            return False
        
        filtered = TreeOperations.filter_structure(matches_query, self.code_structure)
        return TreeOperations.flatten(filtered)
    
    def get_code_path(self, symbol: str) -> Optional[TreePath]:
        """Get path to symbol in code hierarchy"""
        paths = PathOperations.get_all_paths(self.code_structure)
        
        for path in paths:
            if hasattr(path.value, 'name') and path.value.name == symbol:
                return path
            elif isinstance(path.value, str) and symbol in path.value:
                return path
        
        return None
    
    def get_module_structure(self, module_path: List[str]) -> Optional[Any]:
        """Get structure of specific module"""
        return PathOperations.get_path(self.code_structure, module_path)

class ConfigurationManager:
    """Configuration management using tree operations"""
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration structures"""
        def merge_values(base_val: Any, override_val: Any) -> Any:
            if isinstance(base_val, dict) and isinstance(override_val, dict):
                merged = base_val.copy()
                for key, value in override_val.items():
                    if key in merged:
                        merged[key] = merge_values(merged[key], value)
                    else:
                        merged[key] = value
                return merged
            else:
                return override_val
        
        return merge_values(base, override)
    
    def validate_config_structure(self, config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema"""
        errors = []
        
        def validate_recursive(config_part: Any, schema_part: Any, path: List[str]):
            if isinstance(schema_part, dict):
                if not isinstance(config_part, dict):
                    errors.append(f"Expected dict at {'.'.join(path)}, got {type(config_part)}")
                    return
                
                for key, schema_value in schema_part.items():
                    if key not in config_part:
                        errors.append(f"Missing required key: {'.'.join(path + [key])}")
                    else:
                        validate_recursive(config_part[key], schema_value, path + [key])
            elif isinstance(schema_part, type):
                if not isinstance(config_part, schema_part):
                    errors.append(f"Type mismatch at {'.'.join(path)}: expected {schema_part}, got {type(config_part)}")
        
        validate_recursive(config, schema, [])
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }

# Performance optimizations would be implemented here
class PerformanceOptimizations:
    """Performance optimizations for tree operations"""
    
    @staticmethod
    def lazy_map_structure(fn: Callable[[T], U], structure: NestedStructure[T]):
        """Lazy evaluation version of map_structure"""
        # This would implement lazy evaluation using generators
        # For now, return regular map_structure
        return TreeOperations.map_structure(fn, structure)
    
    @staticmethod
    def memoized_map_structure(fn: Callable[[T], U]):
        """Memoized version of map_structure"""
        cache = {}
        
        def memoized_fn(structure: NestedStructure[T]) -> NestedStructure[U]:
            # Simple memoization based on structure hash
            structure_key = str(structure)  # Simplified - would need better hashing
            if structure_key not in cache:
                cache[structure_key] = TreeOperations.map_structure(fn, structure)
            return cache[structure_key]
        
        return memoized_fn
