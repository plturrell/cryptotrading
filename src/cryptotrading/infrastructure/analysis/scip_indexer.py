"""
SCIP Indexer for Python - Generates Glean-compatible facts
Uses SCIP (Source Code Index Protocol) to index Python code for Glean
"""

import ast
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SCIPSymbol:
    """SCIP symbol representation"""

    scheme: str = "scip-python"
    package: str = ""
    descriptors: List[str] = field(default_factory=list)

    def to_string(self) -> str:
        """Convert to SCIP symbol string format"""
        desc_str = "#".join(self.descriptors) if self.descriptors else ""
        return f"{self.scheme} {self.package} {desc_str}"


@dataclass
class SCIPDocument:
    """SCIP document with occurrences and symbols"""

    relative_path: str
    language: str = "python"
    occurrences: List[Dict[str, Any]] = field(default_factory=list)
    symbols: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SCIPIndex:
    """SCIP index containing all documents"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    documents: List[SCIPDocument] = field(default_factory=list)

    def to_json(self) -> str:
        """Convert to SCIP JSON format"""
        return json.dumps(
            {
                "metadata": self.metadata,
                "documents": [
                    {
                        "relative_path": doc.relative_path,
                        "language": doc.language,
                        "occurrences": doc.occurrences,
                        "symbols": doc.symbols,
                    }
                    for doc in self.documents
                ],
            },
            indent=2,
        )


class PythonSCIPIndexer:
    """Indexes Python code into SCIP format for Glean consumption"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.index = SCIPIndex(
            metadata={
                "version": "0.1",
                "project_root": str(self.project_root),
                "tool": "scip-python-glean",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.symbol_table: Dict[str, SCIPSymbol] = {}

    def index_file(self, file_path: Path) -> Optional[SCIPDocument]:
        """Index a single Python file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            relative_path = file_path.relative_to(self.project_root)

            document = SCIPDocument(relative_path=str(relative_path), language="python")

            # Visit all nodes in the AST
            visitor = SCIPVisitor(
                file_path=relative_path, content=content, document=document, indexer=self
            )
            visitor.visit(tree)

            return document

        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            return None

    def index_directory(self, directory: Path) -> None:
        """Index all Python files in a directory"""
        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            doc = self.index_file(py_file)
            if doc:
                self.index.documents.append(doc)
                logger.info(f"Indexed {py_file}")

    def generate_glean_facts(self) -> List[Dict[str, Any]]:
        """Generate Glean facts from SCIP index"""
        facts = []

        for doc in self.index.documents:
            # Generate file facts
            facts.append(
                {
                    "predicate": "src.File",
                    "key": {"path": doc.relative_path},
                    "value": {"language": "Python"},
                }
            )

            # Generate symbol facts
            for symbol in doc.symbols:
                facts.append(
                    {
                        "predicate": "python.Declaration",
                        "key": {"name": symbol["symbol"], "file": doc.relative_path},
                        "value": {
                            "kind": symbol.get("kind", "unknown"),
                            "signature": symbol.get("signature", ""),
                        },
                    }
                )

            # Generate reference facts
            for occurrence in doc.occurrences:
                if occurrence.get("symbol_roles") & 1:  # Definition
                    continue

                facts.append(
                    {
                        "predicate": "python.Reference",
                        "key": {
                            "target": occurrence["symbol"],
                            "file": doc.relative_path,
                            "span": occurrence["range"],
                        },
                        "value": {},
                    }
                )

        return facts


class SCIPVisitor(ast.NodeVisitor):
    """AST visitor that generates SCIP occurrences and symbols"""

    def __init__(
        self, file_path: Path, content: str, document: SCIPDocument, indexer: "PythonSCIPIndexer"
    ):
        self.file_path = file_path
        self.lines = content.splitlines()
        self.document = document
        self.indexer = indexer
        self.scope_stack: List[str] = []

    def _get_position(self, node: ast.AST) -> List[int]:
        """Get SCIP position array [line, col, line, col]"""
        return [
            node.lineno - 1,
            node.col_offset,
            node.end_lineno - 1 if node.end_lineno else node.lineno - 1,
            node.end_col_offset if node.end_col_offset else node.col_offset,
        ]

    def _make_symbol(self, name: str, kind: str) -> str:
        """Create a SCIP symbol string"""
        package = str(self.file_path).replace("/", ".").replace(".py", "")
        descriptors = self.scope_stack + [f"{kind}/{name}"]
        symbol = SCIPSymbol(package=package, descriptors=descriptors)
        return symbol.to_string()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition"""
        symbol = self._make_symbol(node.name, "def")

        # Add symbol definition
        self.document.symbols.append(
            {
                "symbol": symbol,
                "kind": "function",
                "display_name": node.name,
                "signature": self._get_function_signature(node),
                "documentation": ast.get_docstring(node) or "",
            }
        )

        # Add occurrence
        self.document.occurrences.append(
            {"range": self._get_position(node), "symbol": symbol, "symbol_roles": 1}  # Definition
        )

        # Visit function body
        self.scope_stack.append(f"def/{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition"""
        symbol = self._make_symbol(node.name, "class")

        # Add symbol definition
        self.document.symbols.append(
            {
                "symbol": symbol,
                "kind": "class",
                "display_name": node.name,
                "documentation": ast.get_docstring(node) or "",
            }
        )

        # Add occurrence
        self.document.occurrences.append(
            {"range": self._get_position(node), "symbol": symbol, "symbol_roles": 1}  # Definition
        )

        # Visit class body
        self.scope_stack.append(f"class/{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement"""
        for alias in node.names:
            module_name = alias.name

            # Create reference to imported module
            self.document.occurrences.append(
                {
                    "range": self._get_position(node),
                    "symbol": f"scip-python module {module_name}",
                    "symbol_roles": 8,  # Import
                }
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statement"""
        if node.module:
            module_name = node.module

            for alias in node.names:
                symbol_name = alias.name

                # Create reference to imported symbol
                self.document.occurrences.append(
                    {
                        "range": self._get_position(node),
                        "symbol": f"scip-python module {module_name} {symbol_name}",
                        "symbol_roles": 8,  # Import
                    }
                )

    def visit_Name(self, node: ast.Name) -> None:
        """Visit name reference"""
        # Skip definitions (handled elsewhere)
        if isinstance(node.ctx, ast.Store):
            return

        # Try to resolve the symbol
        # In a real implementation, this would do proper name resolution
        # For now, we'll create a reference occurrence
        self.document.occurrences.append(
            {
                "range": self._get_position(node),
                "symbol": f"scip-python local {node.id}",
                "symbol_roles": 2,  # Reference
            }
        )

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        params = []
        for arg in node.args.args:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            params.append(param)

        signature = f"def {node.name}({', '.join(params)})"
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        return signature


def index_project_for_glean(project_path: str) -> Dict[str, Any]:
    """Main entry point to index a Python project for Glean"""
    project_root = Path(project_path)
    indexer = PythonSCIPIndexer(project_root)

    # Index the source directory
    src_dir = project_root / "src"
    if src_dir.exists():
        indexer.index_directory(src_dir)
    else:
        indexer.index_directory(project_root)

    # Generate outputs
    return {
        "scip_index": json.loads(indexer.index.to_json()),
        "glean_facts": indexer.generate_glean_facts(),
        "stats": {
            "files_indexed": len(indexer.index.documents),
            "total_symbols": sum(len(doc.symbols) for doc in indexer.index.documents),
            "total_occurrences": sum(len(doc.occurrences) for doc in indexer.index.documents),
        },
    }
