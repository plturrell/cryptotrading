"""
TypeScript SCIP Indexer for Glean Integration
Generates SCIP facts from TypeScript files for comprehensive code analysis
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TypeScriptSymbol:
    """TypeScript symbol representation"""

    name: str
    kind: str  # interface, class, function, variable, type, enum, etc.
    line: int
    column: int
    file_path: str
    visibility: str = "public"
    type_annotation: Optional[str] = None
    generic_params: List[str] = None
    extends: Optional[str] = None
    implements: List[str] = None
    decorators: List[str] = None
    is_exported: bool = False


class TypeScriptSCIPIndexer:
    """SCIP indexer for TypeScript files"""

    def __init__(self):
        # TypeScript patterns
        self.interface_pattern = re.compile(
            r"(?:export\s+)?interface\s+(\w+)(?:\s*<([^>]+)>)?\s*(?:extends\s+([^{]+))?\s*{"
        )
        self.class_pattern = re.compile(
            r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s*<([^>]+)>)?\s*(?:extends\s+(\w+))?\s*(?:implements\s+([^{]+))?\s*{"
        )
        self.function_pattern = re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)(?:\s*<([^>]+)>)?\s*\([^)]*\)(?:\s*:\s*([^{]+))?\s*{"
        )
        self.method_pattern = re.compile(
            r"(?:(public|private|protected|readonly)\s+)?(?:(static|async)\s+)?(\w+)(?:\s*<([^>]+)>)?\s*\([^)]*\)(?:\s*:\s*([^{]+))?\s*{"
        )
        self.variable_pattern = re.compile(
            r"(?:export\s+)?(?:const|let|var)\s+(\w+)(?:\s*:\s*([^=]+))?\s*="
        )
        self.type_pattern = re.compile(r"(?:export\s+)?type\s+(\w+)(?:\s*<([^>]+)>)?\s*=")
        self.enum_pattern = re.compile(r"(?:export\s+)?enum\s+(\w+)\s*{")
        self.import_pattern = re.compile(
            r'import\s+(?:{([^}]+)}|(\w+)|\*\s+as\s+(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]'
        )
        self.decorator_pattern = re.compile(r"@(\w+)(?:\([^)]*\))?")

    def index_typescript_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Index a single TypeScript file and generate SCIP facts"""
        facts = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return facts

        lines = content.split("\n")
        symbols = self._extract_symbols(content, file_path)

        # Generate file fact
        facts.append(
            {
                "predicate": "src.File",
                "key": {"path": file_path},
                "value": {"language": "typescript", "lines": len(lines), "symbols": len(symbols)},
            }
        )

        # Generate symbol facts
        for symbol in symbols:
            facts.extend(self._generate_symbol_facts(symbol))

        # Generate import facts
        imports = self._extract_imports(content, file_path)
        for imp in imports:
            facts.append(
                {
                    "predicate": "typescript.Import",
                    "key": {"file": file_path, "module": imp["module"]},
                    "value": {"imported_names": imp["names"], "import_type": imp["type"]},
                }
            )

        return facts

    def _extract_symbols(self, content: str, file_path: str) -> List[TypeScriptSymbol]:
        """Extract TypeScript symbols from file content"""
        symbols = []
        lines = content.split("\n")

        # Find decorators first
        decorators_by_line = {}
        for i, line in enumerate(lines):
            decorator_matches = self.decorator_pattern.findall(line)
            if decorator_matches:
                decorators_by_line[i + 1] = decorator_matches

        # Extract interfaces
        for match in self.interface_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            symbols.append(
                TypeScriptSymbol(
                    name=match.group(1),
                    kind="interface",
                    line=line_num,
                    column=match.start() - content.rfind("\n", 0, match.start()),
                    file_path=file_path,
                    generic_params=match.group(2).split(",") if match.group(2) else [],
                    extends=match.group(3).strip() if match.group(3) else None,
                    is_exported="export" in match.group(0),
                    decorators=decorators_by_line.get(line_num - 1, []),
                )
            )

        # Extract classes
        for match in self.class_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            symbols.append(
                TypeScriptSymbol(
                    name=match.group(1),
                    kind="class",
                    line=line_num,
                    column=match.start() - content.rfind("\n", 0, match.start()),
                    file_path=file_path,
                    generic_params=match.group(2).split(",") if match.group(2) else [],
                    extends=match.group(3).strip() if match.group(3) else None,
                    implements=match.group(4).split(",") if match.group(4) else [],
                    is_exported="export" in match.group(0),
                    decorators=decorators_by_line.get(line_num - 1, []),
                )
            )

        # Extract functions
        for match in self.function_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            symbols.append(
                TypeScriptSymbol(
                    name=match.group(1),
                    kind="function",
                    line=line_num,
                    column=match.start() - content.rfind("\n", 0, match.start()),
                    file_path=file_path,
                    generic_params=match.group(2).split(",") if match.group(2) else [],
                    type_annotation=match.group(3).strip() if match.group(3) else None,
                    is_exported="export" in match.group(0),
                    decorators=decorators_by_line.get(line_num - 1, []),
                )
            )

        # Extract methods (inside classes)
        for match in self.method_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            visibility = match.group(1) or "public"
            modifiers = [match.group(2)] if match.group(2) else []

            symbols.append(
                TypeScriptSymbol(
                    name=match.group(3),
                    kind="method",
                    line=line_num,
                    column=match.start() - content.rfind("\n", 0, match.start()),
                    file_path=file_path,
                    visibility=visibility,
                    generic_params=match.group(4).split(",") if match.group(4) else [],
                    type_annotation=match.group(5).strip() if match.group(5) else None,
                    decorators=decorators_by_line.get(line_num - 1, []),
                )
            )

        # Extract variables/constants
        for match in self.variable_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            symbols.append(
                TypeScriptSymbol(
                    name=match.group(1),
                    kind="variable",
                    line=line_num,
                    column=match.start() - content.rfind("\n", 0, match.start()),
                    file_path=file_path,
                    type_annotation=match.group(2).strip() if match.group(2) else None,
                    is_exported="export" in match.group(0),
                )
            )

        # Extract type aliases
        for match in self.type_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            symbols.append(
                TypeScriptSymbol(
                    name=match.group(1),
                    kind="type",
                    line=line_num,
                    column=match.start() - content.rfind("\n", 0, match.start()),
                    file_path=file_path,
                    generic_params=match.group(2).split(",") if match.group(2) else [],
                    is_exported="export" in match.group(0),
                )
            )

        # Extract enums
        for match in self.enum_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            symbols.append(
                TypeScriptSymbol(
                    name=match.group(1),
                    kind="enum",
                    line=line_num,
                    column=match.start() - content.rfind("\n", 0, match.start()),
                    file_path=file_path,
                    is_exported="export" in match.group(0),
                )
            )

        return symbols

    def _extract_imports(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract import statements"""
        imports = []

        for match in self.import_pattern.finditer(content):
            if match.group(1):  # Named imports
                names = [name.strip() for name in match.group(1).split(",")]
                import_type = "named"
            elif match.group(2):  # Default import
                names = [match.group(2)]
                import_type = "default"
            elif match.group(3):  # Namespace import
                names = [match.group(3)]
                import_type = "namespace"
            else:
                names = []
                import_type = "side_effect"

            imports.append({"module": match.group(4), "names": names, "type": import_type})

        return imports

    def _generate_symbol_facts(self, symbol: TypeScriptSymbol) -> List[Dict[str, Any]]:
        """Generate SCIP facts for a TypeScript symbol"""
        facts = []

        # Declaration fact
        facts.append(
            {
                "predicate": "typescript.Declaration",
                "key": {"name": symbol.name, "file": symbol.file_path, "line": symbol.line},
                "value": {
                    "kind": symbol.kind,
                    "visibility": symbol.visibility,
                    "type_annotation": symbol.type_annotation,
                    "generic_params": symbol.generic_params or [],
                    "extends": symbol.extends,
                    "implements": symbol.implements or [],
                    "decorators": symbol.decorators or [],
                    "is_exported": symbol.is_exported,
                    "column": symbol.column,
                },
            }
        )

        # Reference facts for extends/implements
        if symbol.extends:
            facts.append(
                {
                    "predicate": "typescript.Reference",
                    "key": {
                        "file": symbol.file_path,
                        "line": symbol.line,
                        "target": symbol.extends,
                    },
                    "value": {"reference_type": "extends", "symbol": symbol.name},
                }
            )

        for impl in symbol.implements or []:
            facts.append(
                {
                    "predicate": "typescript.Reference",
                    "key": {"file": symbol.file_path, "line": symbol.line, "target": impl.strip()},
                    "value": {"reference_type": "implements", "symbol": symbol.name},
                }
            )

        return facts


def index_typescript_files(root_path: str) -> List[Dict[str, Any]]:
    """Index all TypeScript files in a directory tree"""
    indexer = TypeScriptSCIPIndexer()
    all_facts = []

    root = Path(root_path)
    ts_files = list(root.rglob("*.ts")) + list(root.rglob("*.tsx"))

    for ts_file in ts_files:
        if ts_file.is_file():
            facts = indexer.index_typescript_file(str(ts_file))
            all_facts.extend(facts)

    return all_facts
