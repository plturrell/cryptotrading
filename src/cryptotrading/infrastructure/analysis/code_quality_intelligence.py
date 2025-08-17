"""
Advanced Code Quality Intelligence Layer
Provides comprehensive code quality analysis beyond basic indexing
"""

import ast
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import difflib

logger = logging.getLogger(__name__)

@dataclass
class ComplexityMetrics:
    """Cyclomatic complexity metrics for a function"""
    function_name: str
    file_path: str
    line_number: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    parameter_count: int
    lines_of_code: int
    nesting_depth: int
    return_points: int

@dataclass
class DuplicationBlock:
    """Represents a block of duplicated code"""
    hash_signature: str
    files: List[str]
    line_ranges: List[Tuple[int, int]]
    similarity_score: float
    lines_count: int
    code_snippet: str

@dataclass
class CouplingMetrics:
    """Class/module coupling metrics"""
    entity_name: str
    entity_type: str  # class, module, function
    file_path: str
    afferent_coupling: int  # incoming dependencies
    efferent_coupling: int  # outgoing dependencies
    instability: float  # Ce / (Ca + Ce)
    abstractness: float
    distance_from_main: float

@dataclass
class DocumentationMetrics:
    """Documentation coverage metrics"""
    entity_name: str
    entity_type: str
    file_path: str
    has_docstring: bool
    docstring_length: int
    has_type_hints: bool
    parameter_documentation: float  # percentage of params documented
    return_documentation: bool

@dataclass
class QualityReport:
    """Comprehensive code quality report"""
    timestamp: str
    project_path: str
    total_files: int
    total_functions: int
    total_classes: int
    
    # Complexity Analysis
    complexity_metrics: List[ComplexityMetrics]
    avg_cyclomatic_complexity: float
    high_complexity_functions: int  # >10 complexity
    
    # Duplication Analysis
    duplication_blocks: List[DuplicationBlock]
    duplication_percentage: float
    total_duplicated_lines: int
    
    # Coupling Analysis
    coupling_metrics: List[CouplingMetrics]
    avg_instability: float
    highly_coupled_entities: int
    
    # Documentation Analysis
    documentation_metrics: List[DocumentationMetrics]
    documentation_coverage: float
    type_hint_coverage: float
    
    # Overall Scores
    maintainability_score: float  # 0-100
    technical_debt_score: float   # 0-100 (lower is better)
    code_quality_grade: str       # A, B, C, D, F

class CodeQualityIntelligence:
    """Advanced code quality analysis engine"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.complexity_analyzer = ComplexityAnalyzer()
        self.duplication_detector = SemanticDuplicationDetector()
        self.coupling_analyzer = CouplingAnalyzer()
        self.documentation_analyzer = DocumentationAnalyzer()
    
    def analyze_project_quality(self) -> QualityReport:
        """Perform comprehensive code quality analysis"""
        logger.info("🧠 Starting comprehensive code quality analysis...")
        
        # Collect all Python files
        python_files = [f for f in self.project_root.rglob("*.py") 
                       if "node_modules" not in str(f) and "__pycache__" not in str(f)]
        
        # Initialize collectors
        all_complexity = []
        all_duplications = []
        all_coupling = []
        all_documentation = []
        total_functions = 0
        total_classes = 0
        
        # Analyze each file
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                tree = ast.parse(content)
                
                # Complexity analysis
                complexity_results = self.complexity_analyzer.analyze_file(py_file, tree, content)
                all_complexity.extend(complexity_results)
                
                # Count entities
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                
                # Coupling analysis
                coupling_results = self.coupling_analyzer.analyze_file(py_file, tree)
                all_coupling.extend(coupling_results)
                
                # Documentation analysis
                doc_results = self.documentation_analyzer.analyze_file(py_file, tree)
                all_documentation.extend(doc_results)
                
            except Exception as e:
                logger.warning("Failed to analyze %s: %s", py_file, e)
                continue
        
        # Cross-file duplication analysis
        logger.info("🔍 Analyzing code duplication across files...")
        all_duplications = self.duplication_detector.detect_duplications(python_files)
        
        # Calculate aggregate metrics
        report = self._generate_quality_report(
            python_files, all_complexity, all_duplications, 
            all_coupling, all_documentation, total_functions, total_classes
        )
        
        logger.info("✅ Code quality analysis complete")
        return report
    
    def _generate_quality_report(self, files: List[Path], complexity: List[ComplexityMetrics],
                               duplications: List[DuplicationBlock], coupling: List[CouplingMetrics],
                               documentation: List[DocumentationMetrics], 
                               total_functions: int, total_classes: int) -> QualityReport:
        """Generate comprehensive quality report"""
        
        # Complexity metrics
        avg_complexity = sum(c.cyclomatic_complexity for c in complexity) / len(complexity) if complexity else 0
        high_complexity = len([c for c in complexity if c.cyclomatic_complexity > 10])
        
        # Duplication metrics
        total_duplicated_lines = sum(d.lines_count * (len(d.files) - 1) for d in duplications)
        total_lines = sum(len(open(f, 'r').readlines()) for f in files if f.exists())
        duplication_percentage = (total_duplicated_lines / total_lines * 100) if total_lines > 0 else 0
        
        # Coupling metrics
        avg_instability = sum(c.instability for c in coupling) / len(coupling) if coupling else 0
        highly_coupled = len([c for c in coupling if c.efferent_coupling > 10])
        
        # Documentation metrics
        documented_entities = len([d for d in documentation if d.has_docstring])
        doc_coverage = (documented_entities / len(documentation) * 100) if documentation else 0
        
        type_hinted_entities = len([d for d in documentation if d.has_type_hints])
        type_hint_coverage = (type_hinted_entities / len(documentation) * 100) if documentation else 0
        
        # Calculate overall scores
        maintainability_score = self._calculate_maintainability_score(
            avg_complexity, duplication_percentage, avg_instability, doc_coverage
        )
        
        technical_debt_score = self._calculate_technical_debt_score(
            high_complexity, total_duplicated_lines, highly_coupled, len(documentation) - documented_entities
        )
        
        quality_grade = self._calculate_quality_grade(maintainability_score)
        
        return QualityReport(
            timestamp=datetime.now().isoformat(),
            project_path=str(self.project_root),
            total_files=len(files),
            total_functions=total_functions,
            total_classes=total_classes,
            complexity_metrics=complexity,
            avg_cyclomatic_complexity=avg_complexity,
            high_complexity_functions=high_complexity,
            duplication_blocks=duplications,
            duplication_percentage=duplication_percentage,
            total_duplicated_lines=total_duplicated_lines,
            coupling_metrics=coupling,
            avg_instability=avg_instability,
            highly_coupled_entities=highly_coupled,
            documentation_metrics=documentation,
            documentation_coverage=doc_coverage,
            type_hint_coverage=type_hint_coverage,
            maintainability_score=maintainability_score,
            technical_debt_score=technical_debt_score,
            code_quality_grade=quality_grade
        )
    
    def _calculate_maintainability_score(self, complexity: float, duplication: float, 
                                       instability: float, documentation: float) -> float:
        """Calculate maintainability score (0-100, higher is better)"""
        # Normalize and weight factors
        complexity_score = max(0, 100 - (complexity - 5) * 10)  # Penalty after complexity 5
        duplication_score = max(0, 100 - duplication * 2)       # 2 points per % duplication
        coupling_score = max(0, 100 - instability * 100)        # Instability 0-1 scale
        doc_score = documentation                                # Already 0-100
        
        # Weighted average
        return (complexity_score * 0.3 + duplication_score * 0.3 + 
                coupling_score * 0.2 + doc_score * 0.2)
    
    def _calculate_technical_debt_score(self, high_complexity: int, duplicated_lines: int,
                                      highly_coupled: int, undocumented: int) -> float:
        """Calculate technical debt score (0-100, lower is better)"""
        # Each factor contributes to debt
        complexity_debt = min(100, high_complexity * 5)
        duplication_debt = min(100, duplicated_lines / 100)
        coupling_debt = min(100, highly_coupled * 10)
        documentation_debt = min(100, undocumented / 10)
        
        return (complexity_debt + duplication_debt + coupling_debt + documentation_debt) / 4
    
    def _calculate_quality_grade(self, maintainability_score: float) -> str:
        """Calculate letter grade based on maintainability score"""
        if maintainability_score >= 90:
            return "A"
        elif maintainability_score >= 80:
            return "B"
        elif maintainability_score >= 70:
            return "C"
        elif maintainability_score >= 60:
            return "D"
        else:
            return "F"

class ComplexityAnalyzer:
    """Analyzes cyclomatic and cognitive complexity"""
    
    def analyze_file(self, file_path: Path, tree: ast.AST, content: str) -> List[ComplexityMetrics]:
        """Analyze complexity metrics for all functions in a file"""
        results = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics = self._analyze_function_complexity(node, file_path, content)
                results.append(metrics)
        
        return results
    
    def _analyze_function_complexity(self, node: ast.FunctionDef, file_path: Path, content: str) -> ComplexityMetrics:
        """Analyze complexity for a single function"""
        # Cyclomatic complexity
        cyclomatic = self._calculate_cyclomatic_complexity(node)
        
        # Cognitive complexity (more sophisticated)
        cognitive = self._calculate_cognitive_complexity(node)
        
        # Other metrics
        param_count = len(node.args.args)
        lines_of_code = (node.end_lineno - node.lineno) if hasattr(node, 'end_lineno') else 0
        nesting_depth = self._calculate_nesting_depth(node)
        return_points = self._count_return_statements(node)
        
        return ComplexityMetrics(
            function_name=node.name,
            file_path=str(file_path),
            line_number=node.lineno,
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            parameter_count=param_count,
            lines_of_code=lines_of_code,
            nesting_depth=nesting_depth,
            return_points=return_points
        )
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (decision points + 1)"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # And/Or operators add complexity
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cognitive complexity (more nuanced than cyclomatic)"""
        cognitive = 0
        nesting_level = 0
        
        def visit_node(n, level=0):
            nonlocal cognitive
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                cognitive += 1 + level  # Base + nesting penalty
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level + 1)
            elif isinstance(n, ast.ExceptHandler):
                cognitive += 1 + level
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level + 1)
            elif isinstance(n, (ast.BoolOp, ast.Compare)):
                cognitive += 1
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level)
            else:
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level)
        
        visit_node(node)
        return cognitive
    
    def _calculate_nesting_depth(self, node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        def visit_node(n, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                for child in ast.iter_child_nodes(n):
                    visit_node(child, depth + 1)
            else:
                for child in ast.iter_child_nodes(n):
                    visit_node(child, depth)
        
        visit_node(node)
        return max_depth
    
    def _count_return_statements(self, node: ast.FunctionDef) -> int:
        """Count return statements in function"""
        return len([n for n in ast.walk(node) if isinstance(n, ast.Return)])

class SemanticDuplicationDetector:
    """Detects semantic code duplication beyond simple text matching"""
    
    def detect_duplications(self, files: List[Path]) -> List[DuplicationBlock]:
        """Detect code duplications across files"""
        duplications = []
        
        # Extract code blocks from all files
        code_blocks = self._extract_code_blocks(files)
        
        # Group by semantic similarity
        similarity_groups = self._group_by_similarity(code_blocks)
        
        # Convert to duplication blocks
        for group in similarity_groups:
            if len(group) > 1:  # Only duplications
                duplication = self._create_duplication_block(group)
                duplications.append(duplication)
        
        return duplications
    
    def _extract_code_blocks(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Extract meaningful code blocks from files"""
        blocks = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                lines = content.split('\n')
                
                # Extract functions and classes
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        start_line = node.lineno - 1
                        end_line = getattr(node, 'end_lineno', start_line + 10) - 1
                        
                        # Get code block
                        code_lines = lines[start_line:end_line + 1]
                        code_text = '\n'.join(code_lines)
                        
                        # Create normalized version for comparison
                        normalized = self._normalize_code(code_text)
                        
                        blocks.append({
                            'file': str(file_path),
                            'name': node.name,
                            'type': type(node).__name__,
                            'start_line': start_line + 1,
                            'end_line': end_line + 1,
                            'code': code_text,
                            'normalized': normalized,
                            'hash': hashlib.md5(normalized.encode()).hexdigest()
                        })
            
            except Exception as e:
                logger.warning("Failed to extract blocks from %s: %s", file_path, e)
                continue
        
        return blocks
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison (remove whitespace, comments, etc.)"""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Remove variable names (basic)
        code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', code)
        
        return code.strip()
    
    def _group_by_similarity(self, blocks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group code blocks by similarity"""
        groups = []
        processed = set()
        
        for i, block1 in enumerate(blocks):
            if i in processed:
                continue
            
            group = [block1]
            processed.add(i)
            
            for j, block2 in enumerate(blocks[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_similarity(block1, block2)
                if similarity > 0.8:  # 80% similarity threshold
                    group.append(block2)
                    processed.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> float:
        """Calculate similarity between two code blocks"""
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, block1['normalized'], block2['normalized'])
        return matcher.ratio()
    
    def _create_duplication_block(self, group: List[Dict[str, Any]]) -> DuplicationBlock:
        """Create duplication block from similar code blocks"""
        files = [block['file'] for block in group]
        line_ranges = [(block['start_line'], block['end_line']) for block in group]
        
        # Calculate average similarity
        similarities = []
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                sim = self._calculate_similarity(group[i], group[j])
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Use first block as representative
        representative = group[0]
        
        return DuplicationBlock(
            hash_signature=representative['hash'],
            files=files,
            line_ranges=line_ranges,
            similarity_score=avg_similarity,
            lines_count=representative['end_line'] - representative['start_line'] + 1,
            code_snippet=representative['code'][:200] + "..." if len(representative['code']) > 200 else representative['code']
        )

class CouplingAnalyzer:
    """Analyzes coupling and cohesion metrics"""
    
    def analyze_file(self, file_path: Path, tree: ast.AST) -> List[CouplingMetrics]:
        """Analyze coupling metrics for classes and modules"""
        results = []
        
        # Analyze imports (efferent coupling)
        imports = self._extract_imports(tree)
        
        # Analyze classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics = self._analyze_class_coupling(node, file_path, imports)
                results.append(metrics)
        
        return results
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imported modules/names"""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                for alias in node.names:
                    imports.add(alias.name)
        
        return imports
    
    def _analyze_class_coupling(self, node: ast.ClassDef, file_path: Path, imports: Set[str]) -> CouplingMetrics:
        """Analyze coupling for a single class"""
        # Efferent coupling (outgoing dependencies)
        efferent = len(imports)
        
        # Afferent coupling would require cross-project analysis
        # For now, use a simple heuristic based on inheritance and method calls
        afferent = len(node.bases)  # Inheritance relationships
        
        # Calculate instability (Ce / (Ca + Ce))
        total_coupling = afferent + efferent
        instability = efferent / total_coupling if total_coupling > 0 else 0
        
        # Abstractness (abstract methods / total methods)
        total_methods = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
        abstract_methods = len([n for n in node.body 
                              if isinstance(n, ast.FunctionDef) and self._is_abstract_method(n)])
        abstractness = abstract_methods / total_methods if total_methods > 0 else 0
        
        # Distance from main sequence
        distance = abs(abstractness + instability - 1)
        
        return CouplingMetrics(
            entity_name=node.name,
            entity_type="class",
            file_path=str(file_path),
            afferent_coupling=afferent,
            efferent_coupling=efferent,
            instability=instability,
            abstractness=abstractness,
            distance_from_main=distance
        )
    
    def _is_abstract_method(self, node: ast.FunctionDef) -> bool:
        """Check if method is abstract (raises NotImplementedError or has @abstractmethod)"""
        # Check for NotImplementedError
        for stmt in node.body:
            if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Name):
                if stmt.exc.id == "NotImplementedError":
                    return True
        
        # Check for @abstractmethod decorator
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                return True
        
        return False

class DocumentationAnalyzer:
    """Analyzes documentation coverage and quality"""
    
    def analyze_file(self, file_path: Path, tree: ast.AST) -> List[DocumentationMetrics]:
        """Analyze documentation metrics for all entities in a file"""
        results = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                metrics = self._analyze_entity_documentation(node, file_path)
                results.append(metrics)
        
        return results
    
    def _analyze_entity_documentation(self, node: ast.AST, file_path: Path) -> DocumentationMetrics:
        """Analyze documentation for a single entity"""
        entity_name = node.name
        entity_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"
        
        # Check for docstring
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None
        docstring_length = len(docstring) if docstring else 0
        
        # Check for type hints
        has_type_hints = False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check return type annotation
            has_return_type = node.returns is not None
            
            # Check parameter type annotations
            param_annotations = sum(1 for arg in node.args.args if arg.annotation is not None)
            total_params = len(node.args.args)
            
            has_type_hints = has_return_type and (param_annotations == total_params)
            
            # Parameter documentation coverage
            param_doc_coverage = self._calculate_param_documentation_coverage(docstring, node.args.args)
            
            # Return documentation
            return_documented = self._has_return_documentation(docstring)
        else:
            param_doc_coverage = 1.0  # Classes don't have parameters
            return_documented = True
        
        return DocumentationMetrics(
            entity_name=entity_name,
            entity_type=entity_type,
            file_path=str(file_path),
            has_docstring=has_docstring,
            docstring_length=docstring_length,
            has_type_hints=has_type_hints,
            parameter_documentation=param_doc_coverage,
            return_documentation=return_documented
        )
    
    def _calculate_param_documentation_coverage(self, docstring: Optional[str], params: List[ast.arg]) -> float:
        """Calculate what percentage of parameters are documented"""
        if not docstring or not params:
            return 1.0
        
        # Simple heuristic: check if parameter names appear in docstring
        documented_params = 0
        for param in params:
            if param.arg in docstring:
                documented_params += 1
        
        return documented_params / len(params)
    
    def _has_return_documentation(self, docstring: Optional[str]) -> bool:
        """Check if return value is documented"""
        if not docstring:
            return False
        
        # Look for common return documentation patterns
        return_patterns = [r'returns?:', r'return\s', r'-> ']
        for pattern in return_patterns:
            if re.search(pattern, docstring, re.IGNORECASE):
                return True
        
        return False
