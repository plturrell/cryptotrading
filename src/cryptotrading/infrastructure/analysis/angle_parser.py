"""
Angle Query Language Parser for Glean
Implements a subset of Glean's Angle query language for Python analysis
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of Angle queries"""
    PATTERN = "pattern"
    RECURSIVE = "recursive"
    NEGATION = "negation"
    FILTER = "filter"


@dataclass
class AngleQuery:
    """Parsed Angle query representation"""
    predicate: str
    patterns: Dict[str, Any] = field(default_factory=dict)
    filters: List[str] = field(default_factory=list)
    recursive: bool = False
    limit: Optional[int] = None
    
    def matches(self, fact: Dict[str, Any]) -> bool:
        """Check if a fact matches this query"""
        # Check predicate match
        if fact.get("predicate") != self.predicate:
            return False
        
        # Check pattern matches
        for key, pattern in self.patterns.items():
            fact_value = fact.get("key", {}).get(key) or fact.get("value", {}).get(key)
            
            if isinstance(pattern, str) and pattern.startswith("?"):
                # Variable binding - always matches
                continue
            elif fact_value != pattern:
                return False
        
        return True


class AngleParser:
    """Parser for Angle query language"""
    
    # Angle query patterns
    PREDICATE_PATTERN = re.compile(r'(\w+(?:\.\w+)*)\s*{')
    FIELD_PATTERN = re.compile(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\??\w+|\d+)')
    RECURSIVE_PATTERN = re.compile(r'(\w+)\*')
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
    
    def parse(self, query: str) -> AngleQuery:
        """Parse an Angle query string"""
        query = query.strip()
        
        # Extract predicate
        predicate_match = self.PREDICATE_PATTERN.search(query)
        if not predicate_match:
            raise ValueError(f"Invalid Angle query: no predicate found in '{query}'")
        
        predicate = predicate_match.group(1)
        
        # Extract query body
        start = predicate_match.end() - 1
        body = self._extract_body(query[start:])
        
        # Parse patterns
        patterns = self._parse_patterns(body)
        
        # Check for recursive query
        recursive = self.RECURSIVE_PATTERN.search(query) is not None
        
        return AngleQuery(
            predicate=predicate,
            patterns=patterns,
            recursive=recursive
        )
    
    def _extract_body(self, query: str) -> str:
        """Extract the body of a query between { }"""
        depth = 0
        start = -1
        end = -1
        
        for i, char in enumerate(query):
            if char == '{':
                if depth == 0:
                    start = i + 1
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if start == -1 or end == -1:
            raise ValueError("Unmatched braces in query")
        
        return query[start:end]
    
    def _parse_patterns(self, body: str) -> Dict[str, Any]:
        """Parse field patterns from query body"""
        patterns = {}
        
        for match in self.FIELD_PATTERN.finditer(body):
            field = match.group(1)
            value = match.group(2)
            
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.isdigit():
                value = int(value)
            
            patterns[field] = value
        
        return patterns


class AngleQueryEngine:
    """Execute Angle queries against Glean facts"""
    
    def __init__(self, facts: List[Dict[str, Any]]):
        self.facts = facts
        self.parser = AngleParser()
        
        # Index facts by predicate for performance
        self.fact_index: Dict[str, List[Dict[str, Any]]] = {}
        for fact in facts:
            predicate = fact.get("predicate", "")
            if predicate not in self.fact_index:
                self.fact_index[predicate] = []
            self.fact_index[predicate].append(fact)
    
    def query(self, angle_query: str) -> List[Dict[str, Any]]:
        """Execute an Angle query and return matching facts"""
        try:
            parsed_query = self.parser.parse(angle_query)
            
            # Get candidate facts
            candidates = self.fact_index.get(parsed_query.predicate, [])
            
            # Filter by patterns
            results = []
            for fact in candidates:
                if parsed_query.matches(fact):
                    results.append(fact)
            
            # Apply limit if specified
            if parsed_query.limit:
                results = results[:parsed_query.limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute Angle query: {e}")
            return []
    
    def query_with_bindings(self, angle_query: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]:
        """Execute query and return results with variable bindings"""
        parsed_query = self.parser.parse(angle_query)
        results = []
        bindings: Dict[str, List[Any]] = {}
        
        candidates = self.fact_index.get(parsed_query.predicate, [])
        
        for fact in candidates:
            if self._matches_with_bindings(parsed_query, fact, bindings):
                results.append(fact)
        
        return results, bindings
    
    def _matches_with_bindings(self, query: AngleQuery, fact: Dict[str, Any], 
                               bindings: Dict[str, List[Any]]) -> bool:
        """Check if fact matches query and capture variable bindings"""
        if fact.get("predicate") != query.predicate:
            return False
        
        for key, pattern in query.patterns.items():
            fact_value = fact.get("key", {}).get(key) or fact.get("value", {}).get(key)
            
            if isinstance(pattern, str) and pattern.startswith("?"):
                # Variable binding
                var_name = pattern[1:]
                if var_name not in bindings:
                    bindings[var_name] = []
                bindings[var_name].append(fact_value)
            elif fact_value != pattern:
                return False
        
        return True


# Common Angle queries for Python code analysis
PYTHON_QUERIES = {
    "find_function": """
        python.Declaration {
            name = ?name,
            kind = "function",
            file = ?file
        }
    """,
    
    "find_class": """
        python.Declaration {
            name = ?name,
            kind = "class",
            file = ?file
        }
    """,
    
    "find_imports": """
        python.Reference {
            symbol_roles = 8,
            file = ?file,
            target = ?module
        }
    """,
    
    "find_references": """
        python.Reference {
            target = "{symbol}",
            file = ?file,
            span = ?location
        }
    """,
    
    "find_definitions": """
        python.Declaration {
            name = "{name}",
            file = ?file,
            kind = ?kind
        }
    """,
    
    "file_symbols": """
        python.Declaration {
            file = "{file}",
            name = ?name,
            kind = ?kind
        }
    """
}


def create_query(template: str, **kwargs) -> str:
    """Create an Angle query from a template with parameters"""
    query = PYTHON_QUERIES.get(template, template)
    
    for key, value in kwargs.items():
        query = query.replace(f"{{{key}}}", str(value))
    
    return query


# Example usage functions
def find_all_functions(engine: AngleQueryEngine) -> List[Dict[str, Any]]:
    """Find all function declarations"""
    return engine.query(PYTHON_QUERIES["find_function"])


def find_references_to_symbol(engine: AngleQueryEngine, symbol: str) -> List[Dict[str, Any]]:
    """Find all references to a specific symbol"""
    query = create_query("find_references", symbol=symbol)
    return engine.query(query)


def find_symbols_in_file(engine: AngleQueryEngine, file_path: str) -> List[Dict[str, Any]]:
    """Find all symbols defined in a specific file"""
    query = create_query("file_symbols", file=file_path)
    return engine.query(query)