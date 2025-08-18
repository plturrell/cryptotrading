"""
Advanced Angle Query Templates for Glean Code Analysis
Sophisticated query patterns for comprehensive codebase understanding
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"


class AnalysisScope(Enum):
    """Scope of analysis"""
    FILE = "file"
    MODULE = "module"
    PACKAGE = "package"
    PROJECT = "project"
    CROSS_PROJECT = "cross_project"


@dataclass
class QueryTemplate:
    """Advanced query template with metadata"""
    name: str
    description: str
    complexity: QueryComplexity
    scope: AnalysisScope
    query_pattern: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "json"
    use_cases: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def render(self, **kwargs) -> str:
        """Render query with parameters"""
        formatted_query = self.query_pattern
        
        # Apply template parameters
        for param, value in {**self.parameters, **kwargs}.items():
            placeholder = f"{{{param}}}"
            if placeholder in formatted_query:
                formatted_query = formatted_query.replace(placeholder, str(value))
        
        return formatted_query


class AdvancedAngleQueryBuilder:
    """Builder for sophisticated Angle queries"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.custom_templates: Dict[str, QueryTemplate] = {}
    
    def _initialize_templates(self) -> Dict[str, QueryTemplate]:
        """Initialize predefined advanced query templates"""
        templates = {}
        
        # Architecture Analysis Templates
        templates["dependency_graph"] = QueryTemplate(
            name="dependency_graph",
            description="Generate comprehensive dependency graph with metrics",
            complexity=QueryComplexity.ADVANCED,
            scope=AnalysisScope.PROJECT,
            query_pattern="""
            query DependencyGraph($root: String) {
              codebase.file.imports where src.file =~ $root
              | map(.dst.file, .src.file, .import_type, .line_number)
              | group_by(.dst.file)
              | map({
                  file: .key,
                  dependencies: .values | map({
                    source: .src.file,
                    type: .import_type,
                    line: .line_number
                  }),
                  dependency_count: .values | length,
                  fan_in: codebase.file.imports where dst.file == .key | length
                })
              | sort_by(.dependency_count) desc
            }
            """,
            parameters={"root": "src/"},
            use_cases=[
                "Architecture visualization",
                "Dependency analysis", 
                "Circular dependency detection",
                "Module coupling analysis"
            ]
        )
        
        templates["code_complexity_analysis"] = QueryTemplate(
            name="code_complexity_analysis",
            description="Analyze code complexity metrics across the codebase",
            complexity=QueryComplexity.EXPERT,
            scope=AnalysisScope.PROJECT,
            query_pattern="""
            query CodeComplexityAnalysis($min_complexity: Int) {
              codebase.function
              | map({
                  name: .name.name,
                  file: .file.name,
                  line_count: .span.end.line - .span.start.line + 1,
                  cyclomatic_complexity: .cyclomatic_complexity // 1,
                  parameters: .parameters | length,
                  return_statements: codebase.stmt.return where function == . | length,
                  nested_depth: .max_nesting_depth // 1,
                  calls_made: codebase.expr.call where caller == . | length
                })
              | where .cyclomatic_complexity >= $min_complexity
              | sort_by(.cyclomatic_complexity) desc
              | map(. + {
                  complexity_category: (
                    if .cyclomatic_complexity <= 5 then "simple"
                    elif .cyclomatic_complexity <= 10 then "moderate"  
                    elif .cyclomatic_complexity <= 20 then "complex"
                    else "very_complex"
                  ),
                  maintainability_score: (
                    100 - (.cyclomatic_complexity * 2) - (.nested_depth * 5) - (.line_count / 10)
                  )
                })
            }
            """,
            parameters={"min_complexity": 5},
            use_cases=[
                "Code quality assessment",
                "Refactoring prioritization",
                "Technical debt analysis",
                "Performance optimization targets"
            ]
        )
        
        templates["api_surface_analysis"] = QueryTemplate(
            name="api_surface_analysis", 
            description="Analyze public API surface and usage patterns",
            complexity=QueryComplexity.ADVANCED,
            scope=AnalysisScope.MODULE,
            query_pattern="""
            query APIAnalysis($module_pattern: String) {
              codebase.declaration
              | where .visibility == "public" and .file.name =~ $module_pattern
              | map({
                  name: .name.name,
                  type: .kind,
                  file: .file.name,
                  line: .span.start.line,
                  signature: .signature // "",
                  docstring: .docstring.text // "",
                  usages: codebase.expr.ref where target == . | length,
                  external_usages: codebase.expr.ref 
                    where target == . and caller.file.name !~ $module_pattern | length,
                  deprecated: .annotations | any(.name.name == "deprecated"),
                  parameters: (if .kind == "function" then .parameters | length else 0),
                  return_type: .return_type.name // "unknown"
                })
              | sort_by(.external_usages) desc
              | map(. + {
                  api_stability: (
                    if .usages == 0 then "unused"
                    elif .external_usages == 0 then "internal"
                    elif .external_usages < 5 then "limited"
                    elif .external_usages < 20 then "stable"
                    else "core"
                  ),
                  documentation_quality: (
                    if .docstring | length > 100 then "good"
                    elif .docstring | length > 20 then "minimal"
                    else "missing"
                  )
                })
            }
            """,
            parameters={"module_pattern": "src/api/"},
            use_cases=[
                "API design review",
                "Breaking change impact analysis", 
                "Documentation coverage",
                "Public interface optimization"
            ]
        )
        
        templates["security_pattern_analysis"] = QueryTemplate(
            name="security_pattern_analysis",
            description="Identify potential security vulnerabilities and patterns",
            complexity=QueryComplexity.EXPERT,
            scope=AnalysisScope.PROJECT,
            query_pattern="""
            query SecurityAnalysis($risk_patterns: [String]) {
              // Find potential SQL injection points
              sql_risks: codebase.expr.call 
                where .name.name in ["execute", "query", "raw"] and
                      .arguments | any(.kind == "string_concat" or .kind == "f_string")
                | map({
                    type: "sql_injection_risk",
                    function: .name.name,
                    file: .file.name,
                    line: .span.start.line,
                    severity: "high"
                  }),
              
              // Find hardcoded secrets
              secret_risks: codebase.expr.literal
                where .kind == "string" and 
                      (.value | test("(?i)(password|secret|key|token).*=.*['\"][a-zA-Z0-9]{8,}"))
                | map({
                    type: "hardcoded_secret",
                    value: .value | truncate(20),
                    file: .file.name,
                    line: .span.start.line,
                    severity: "critical"
                  }),
              
              // Find unsafe eval/exec usage
              eval_risks: codebase.expr.call
                where .name.name in ["eval", "exec", "compile"] and
                      .arguments | any(.kind != "literal")
                | map({
                    type: "code_injection_risk", 
                    function: .name.name,
                    file: .file.name,
                    line: .span.start.line,
                    severity: "high"
                  }),
              
              // Find missing input validation
              validation_risks: codebase.function
                where .parameters | length > 0 and
                      not (.body | any(.kind == "assert" or .kind == "if_stmt"))
                | map({
                    type: "missing_input_validation",
                    function: .name.name,
                    file: .file.name,
                    line: .span.start.line,
                    severity: "medium"
                  })
            }
            """,
            parameters={"risk_patterns": ["password", "secret", "key", "token"]},
            use_cases=[
                "Security code review",
                "Vulnerability assessment",
                "Compliance auditing",
                "Security pattern enforcement"
            ]
        )
        
        templates["performance_hotspot_analysis"] = QueryTemplate(
            name="performance_hotspot_analysis",
            description="Identify performance bottlenecks and optimization opportunities",
            complexity=QueryComplexity.ADVANCED,
            scope=AnalysisScope.PROJECT,
            query_pattern="""
            query PerformanceAnalysis($complexity_threshold: Int) {
              // Functions with high complexity or long execution paths
              hotspot_functions: codebase.function
                where (.cyclomatic_complexity // 1) >= $complexity_threshold or
                      (.span.end.line - .span.start.line) > 50
                | map({
                    name: .name.name,
                    file: .file.name,
                    complexity: .cyclomatic_complexity // 1,
                    lines: .span.end.line - .span.start.line + 1,
                    loops: codebase.stmt.for where function == . | length,
                    nested_calls: codebase.expr.call where caller == . | length,
                    recursive: codebase.expr.call 
                      where caller == . and .name.name == .caller.name.name | length > 0
                  })
                | sort_by(.complexity) desc,
              
              // Expensive operations patterns
              expensive_operations: codebase.expr.call
                where .name.name in ["sort", "reverse", "copy", "deepcopy", "pickle", "json.loads"]
                | group_by(.file.name)
                | map({
                    file: .key,
                    expensive_calls: .values | length,
                    operations: .values | map(.name.name) | unique
                  })
                | sort_by(.expensive_calls) desc,
              
              // Large data structure operations
              large_data_patterns: codebase.expr.list_comp
                | where .iter.kind == "call" or .condition.kind == "call"
                | map({
                    file: .file.name,
                    line: .span.start.line,
                    type: "complex_list_comprehension",
                    nested_depth: .nesting_level // 1
                  })
            }
            """,
            parameters={"complexity_threshold": 10},
            use_cases=[
                "Performance optimization",
                "Code profiling preparation",
                "Scalability analysis",
                "Resource usage optimization"
            ]
        )
        
        templates["test_coverage_analysis"] = QueryTemplate(
            name="test_coverage_analysis",
            description="Analyze test coverage and testing patterns",
            complexity=QueryComplexity.INTERMEDIATE,
            scope=AnalysisScope.PROJECT,
            query_pattern="""
            query TestCoverageAnalysis($test_pattern: String) {
              // Find all testable functions (public, non-test)
              testable_functions: codebase.function
                where .visibility == "public" and 
                      not (.file.name =~ $test_pattern) and
                      not (.name.name | startswith("_"))
                | map({
                    name: .name.name,
                    file: .file.name,
                    line: .span.start.line,
                    complexity: .cyclomatic_complexity // 1,
                    parameters: .parameters | length
                  }),
              
              // Find test functions  
              test_functions: codebase.function
                where .file.name =~ $test_pattern or .name.name | startswith("test_")
                | map({
                    name: .name.name,
                    file: .file.name,
                    line: .span.start.line,
                    tested_function: .name.name | gsub("test_"; ""),
                    assertions: codebase.expr.call where caller == . and 
                                .name.name | startswith("assert") | length,
                    mocks: codebase.expr.call where caller == . and
                           .name.name | contains("mock") | length
                  }),
              
              // Coverage analysis
              coverage_report: testable_functions 
                | map(. + {
                    has_test: (test_functions | any(.tested_function == .name)),
                    test_count: (test_functions | map(select(.tested_function == .name)) | length),
                    coverage_score: (
                      if test_functions | any(.tested_function == .name) then 
                        min(100, (test_functions | map(select(.tested_function == .name)) | length) * 25)
                      else 0
                    )
                  })
                | sort_by(.coverage_score) asc
            }
            """,
            parameters={"test_pattern": "test.*\\.py$"},
            use_cases=[
                "Test coverage assessment",
                "Testing strategy planning",
                "Quality assurance review",
                "CI/CD optimization"
            ]
        )
        
        templates["refactoring_opportunities"] = QueryTemplate(
            name="refactoring_opportunities",
            description="Identify code duplication and refactoring opportunities",
            complexity=QueryComplexity.EXPERT,
            scope=AnalysisScope.PROJECT,
            query_pattern="""
            query RefactoringOpportunities($similarity_threshold: Float) {
              // Find duplicate code patterns
              similar_functions: codebase.function
                | combinations(2)
                | where .[0].name.name != .[1].name.name and
                        .[0].file.name != .[1].file.name and
                        similarity(.[0].body, .[1].body) > $similarity_threshold
                | map({
                    function1: {
                      name: .[0].name.name,
                      file: .[0].file.name,
                      lines: .[0].span.end.line - .[0].span.start.line + 1
                    },
                    function2: {
                      name: .[1].name.name, 
                      file: .[1].file.name,
                      lines: .[1].span.end.line - .[1].span.start.line + 1
                    },
                    similarity_score: similarity(.[0].body, .[1].body),
                    refactoring_potential: "high"
                  }),
              
              // Find large classes that could be decomposed
              large_classes: codebase.class
                | where (.methods | length) > 10 or 
                        (.span.end.line - .span.start.line) > 200
                | map({
                    name: .name.name,
                    file: .file.name,
                    methods: .methods | length,
                    lines: .span.end.line - .span.start.line + 1,
                    responsibilities: .methods | group_by(.name.name | split("_")[0]) | length,
                    decomposition_score: (.methods | length) + (.lines / 50)
                  })
                | sort_by(.decomposition_score) desc,
              
              // Find god functions (too many responsibilities)
              god_functions: codebase.function
                | where (.span.end.line - .span.start.line) > 50 and
                        (.cyclomatic_complexity // 1) > 15
                | map({
                    name: .name.name,
                    file: .file.name,
                    lines: .span.end.line - .span.start.line + 1,
                    complexity: .cyclomatic_complexity // 1,
                    parameters: .parameters | length,
                    local_vars: codebase.var where function == . | length,
                    refactoring_urgency: (
                      (.lines / 25) + (.cyclomatic_complexity // 1 / 5) + (.parameters / 3)
                    )
                  })
                | sort_by(.refactoring_urgency) desc
            }
            """,
            parameters={"similarity_threshold": 0.8},
            use_cases=[
                "Code refactoring planning",
                "Technical debt reduction",
                "Code quality improvement",
                "Maintenance cost reduction"
            ]
        )
        
        templates["architecture_metrics"] = QueryTemplate(
            name="architecture_metrics",
            description="Calculate comprehensive architecture and design metrics",
            complexity=QueryComplexity.EXPERT,
            scope=AnalysisScope.PROJECT,
            query_pattern="""
            query ArchitectureMetrics($package_pattern: String) {
              // Calculate coupling metrics
              coupling_metrics: codebase.file
                | where .name =~ $package_pattern
                | map({
                    file: .name,
                    efferent_coupling: codebase.file.imports where src.file == .name | length,
                    afferent_coupling: codebase.file.imports where dst.file == .name | length,
                    instability: (
                      let ec = codebase.file.imports where src.file == .name | length;
                      let ac = codebase.file.imports where dst.file == .name | length;
                      if (ec + ac) > 0 then ec / (ec + ac) else 0
                    ),
                    abstractness: (
                      let total_classes = codebase.class where file == .name | length;
                      let abstract_classes = codebase.class 
                        where file == .name and .is_abstract | length;
                      if total_classes > 0 then abstract_classes / total_classes else 0
                    )
                  })
                | map(. + {
                    distance_from_main: abs(.instability + .abstractness - 1),
                    architectural_health: (
                      if .distance_from_main < 0.2 then "excellent"
                      elif .distance_from_main < 0.4 then "good"
                      elif .distance_from_main < 0.6 then "moderate"
                      else "poor"
                    )
                  }),
              
              // Calculate cohesion metrics
              cohesion_metrics: codebase.class
                | where .file.name =~ $package_pattern
                | map({
                    class: .name.name,
                    file: .file.name,
                    methods: .methods | length,
                    attributes: .attributes | length,
                    method_attribute_interactions: (
                      .methods | map(
                        codebase.expr.attr where method == . and
                        .attr in .class.attributes | length
                      ) | add
                    ),
                    lcom: (
                      let method_count = .methods | length;
                      let attr_count = .attributes | length;
                      let interactions = .method_attribute_interactions;
                      if attr_count > 0 and method_count > 1 then
                        (method_count - (interactions / attr_count)) / (method_count - 1)
                      else 0
                    )
                  })
                | map(. + {
                    cohesion_level: (
                      if .lcom < 0.2 then "high"
                      elif .lcom < 0.5 then "moderate"
                      else "low"
                    )
                  })
            }
            """,
            parameters={"package_pattern": "src/"},
            use_cases=[
                "Architecture assessment",
                "Design quality evaluation",
                "Technical debt measurement",
                "Refactoring prioritization"
            ]
        )
        
        return templates
    
    def get_template(self, name: str) -> Optional[QueryTemplate]:
        """Get a query template by name"""
        return self.templates.get(name) or self.custom_templates.get(name)
    
    def list_templates(self, complexity: Optional[QueryComplexity] = None,
                      scope: Optional[AnalysisScope] = None) -> List[QueryTemplate]:
        """List available templates with optional filtering"""
        all_templates = list(self.templates.values()) + list(self.custom_templates.values())
        
        filtered = all_templates
        if complexity:
            filtered = [t for t in filtered if t.complexity == complexity]
        if scope:
            filtered = [t for t in filtered if t.scope == scope]
        
        return filtered
    
    def add_custom_template(self, template: QueryTemplate):
        """Add a custom query template"""
        self.custom_templates[template.name] = template
    
    def generate_query(self, template_name: str, **parameters) -> str:
        """Generate a query from a template with parameters"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.render(**parameters)
    
    def get_query_suggestions(self, keywords: List[str]) -> List[QueryTemplate]:
        """Get query template suggestions based on keywords"""
        suggestions = []
        all_templates = list(self.templates.values()) + list(self.custom_templates.values())
        
        for template in all_templates:
            score = 0
            searchable_text = f"{template.name} {template.description} {' '.join(template.use_cases)}"
            
            for keyword in keywords:
                if keyword.lower() in searchable_text.lower():
                    score += 1
            
            if score > 0:
                suggestions.append((template, score))
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [template for template, _ in suggestions]
    
    def create_composite_query(self, template_names: List[str], **parameters) -> str:
        """Create a composite query from multiple templates"""
        queries = []
        
        for template_name in template_names:
            template = self.get_template(template_name)
            if template:
                query = template.render(**parameters)
                queries.append(f"// Query: {template.name}\n{query}")
        
        return "\n\n".join(queries)
    
    def validate_query_syntax(self, query: str) -> Dict[str, Any]:
        """Basic validation of Angle query syntax"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check for basic syntax elements
        if not re.search(r'query\s+\w+', query):
            validation_result["errors"].append("Query must start with 'query QueryName'")
            validation_result["valid"] = False
        
        # Check for balanced braces
        open_braces = query.count('{')
        close_braces = query.count('}')
        if open_braces != close_braces:
            validation_result["errors"].append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
            validation_result["valid"] = False
        
        # Check for common patterns
        if 'codebase.' not in query:
            validation_result["warnings"].append("Query doesn't reference 'codebase' - may not return results")
        
        return validation_result


# Predefined query collections for common use cases
QUERY_COLLECTIONS = {
    "code_quality": [
        "code_complexity_analysis",
        "test_coverage_analysis", 
        "refactoring_opportunities"
    ],
    "architecture": [
        "dependency_graph",
        "architecture_metrics",
        "api_surface_analysis"
    ],
    "security": [
        "security_pattern_analysis"
    ],
    "performance": [
        "performance_hotspot_analysis",
        "code_complexity_analysis"
    ],
    "maintenance": [
        "refactoring_opportunities",
        "test_coverage_analysis",
        "dependency_graph"
    ]
}


def create_advanced_query_builder() -> AdvancedAngleQueryBuilder:
    """Factory function to create a configured query builder"""
    return AdvancedAngleQueryBuilder()


def get_query_collection(collection_name: str) -> List[str]:
    """Get a predefined collection of related queries"""
    return QUERY_COLLECTIONS.get(collection_name, [])


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    builder = create_advanced_query_builder()
    
    # List all templates
    print("Available Query Templates:")
    for template in builder.list_templates():
        print(f"- {template.name} ({template.complexity.value}) - {template.description}")
    
    # Generate a specific query
    print("\nGenerated Dependency Graph Query:")
    dependency_query = builder.generate_query("dependency_graph", root="src/cryptotrading/")
    print(dependency_query)
    
    # Get suggestions based on keywords
    print("\nSuggestions for 'security performance':")
    suggestions = builder.get_query_suggestions(["security", "performance"])
    for suggestion in suggestions:
        print(f"- {suggestion.name}: {suggestion.description}")