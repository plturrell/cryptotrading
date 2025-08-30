"""
JavaScript/UI5 SCIP Indexer - Generates Glean-compatible facts for JavaScript and SAP UI5
Indexes .js files, controllers, views, and UI5 components
"""

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JSFunction:
    """JavaScript function representation"""

    name: str
    parameters: List[str] = field(default_factory=list)
    is_async: bool = False
    is_arrow: bool = False
    line_start: int = 0
    line_end: int = 0


@dataclass
class JSClass:
    """JavaScript class representation"""

    name: str
    extends: Optional[str] = None
    methods: List[JSFunction] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


@dataclass
class UI5Controller:
    """UI5 controller representation"""

    name: str
    extends: str = ""
    methods: List[JSFunction] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    event_handlers: List[str] = field(default_factory=list)


@dataclass
class UI5View:
    """UI5 XML view representation"""

    name: str
    controller: Optional[str] = None
    controls: List[Dict[str, Any]] = field(default_factory=list)
    bindings: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)


@dataclass
class JSDocument:
    """JavaScript document with parsed content"""

    relative_path: str
    language: str = "javascript"
    functions: List[JSFunction] = field(default_factory=list)
    classes: List[JSClass] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    ui5_controller: Optional[UI5Controller] = None
    ui5_view: Optional[UI5View] = None


class JavaScriptUI5Indexer:
    """Indexes JavaScript and UI5 files into SCIP format for Glean"""

    # JavaScript parsing patterns
    FUNCTION_PATTERN = re.compile(r"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*{", re.MULTILINE)
    ARROW_FUNCTION_PATTERN = re.compile(
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>", re.MULTILINE
    )
    CLASS_PATTERN = re.compile(r"class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{", re.MULTILINE)
    METHOD_PATTERN = re.compile(r"(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*{", re.MULTILINE)
    IMPORT_PATTERN = re.compile(
        r'import\s+(?:{([^}]+)}|\*\s+as\s+(\w+)|(\w+))\s+from\s+["\']([^"\']+)["\']', re.MULTILINE
    )
    REQUIRE_PATTERN = re.compile(
        r'(?:const|let|var)\s+(?:{([^}]+)}|(\w+))\s*=\s*require\s*\(\s*["\']([^"\']+)["\']\s*\)',
        re.MULTILINE,
    )
    EXPORT_PATTERN = re.compile(
        r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)", re.MULTILINE
    )

    # UI5 specific patterns
    UI5_CONTROLLER_PATTERN = re.compile(
        r"sap\.ui\.define\s*\(\s*\[([^\]]+)\]\s*,\s*function\s*\(([^)]*)\)", re.MULTILINE
    )
    UI5_EXTEND_PATTERN = re.compile(r'(\w+)\.extend\s*\(\s*["\']([^"\']+)["\']', re.MULTILINE)
    EVENT_HANDLER_PATTERN = re.compile(r"on\w+\s*:\s*function\s*\(([^)]*)\)", re.MULTILINE)

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.documents: List[JSDocument] = []

    def index_file(self, file_path: Path) -> Optional[JSDocument]:
        """Index a single JavaScript file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            relative_path = file_path.relative_to(self.project_root)
            document = JSDocument(relative_path=str(relative_path))

            # Parse functions
            self._parse_functions(content, document)

            # Parse classes
            self._parse_classes(content, document)

            # Parse imports/requires
            self._parse_imports(content, document)

            # Parse exports
            self._parse_exports(content, document)

            # Check if it's a UI5 controller
            if self._is_ui5_controller(content):
                document.ui5_controller = self._parse_ui5_controller(content, file_path.stem)

            return document

        except Exception as e:
            logger.error("Failed to index JS file %s: %s", file_path, str(e))
            return None

    def index_xml_view(self, file_path: Path) -> Optional[JSDocument]:
        """Index a UI5 XML view file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            relative_path = file_path.relative_to(self.project_root)
            document = JSDocument(relative_path=str(relative_path), language="xml")

            # Parse XML view
            document.ui5_view = self._parse_ui5_view(content, file_path.stem)

            return document

        except Exception as e:
            logger.error("Failed to index XML view %s: %s", file_path, str(e))
            return None

    def _parse_functions(self, content: str, document: JSDocument):
        """Parse JavaScript functions"""

        # Regular functions
        for match in self.FUNCTION_PATTERN.finditer(content):
            func_name = match.group(1)
            params = [p.strip() for p in match.group(2).split(",") if p.strip()]
            line_num = content[: match.start()].count("\n")

            is_async = "async" in match.group(0)

            function = JSFunction(
                name=func_name,
                parameters=params,
                is_async=is_async,
                is_arrow=False,
                line_start=line_num,
            )
            document.functions.append(function)

        # Arrow functions
        for match in self.ARROW_FUNCTION_PATTERN.finditer(content):
            func_name = match.group(1)
            params = [p.strip() for p in match.group(2).split(",") if p.strip()]
            line_num = content[: match.start()].count("\n")

            is_async = "async" in match.group(0)

            function = JSFunction(
                name=func_name,
                parameters=params,
                is_async=is_async,
                is_arrow=True,
                line_start=line_num,
            )
            document.functions.append(function)

    def _parse_classes(self, content: str, document: JSDocument):
        """Parse JavaScript classes"""
        for match in self.CLASS_PATTERN.finditer(content):
            class_name = match.group(1)
            extends = match.group(2)
            line_num = content[: match.start()].count("\n")

            js_class = JSClass(name=class_name, extends=extends, line_start=line_num)

            # Find class methods
            class_start = match.end()
            brace_count = 1
            class_end = class_start

            for i, char in enumerate(content[class_start:], class_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        class_end = i
                        break

            class_body = content[class_start:class_end]

            # Parse methods within class
            for method_match in self.METHOD_PATTERN.finditer(class_body):
                method_name = method_match.group(1)
                params = [p.strip() for p in method_match.group(2).split(",") if p.strip()]

                method = JSFunction(
                    name=method_name, parameters=params, is_async="async" in method_match.group(0)
                )
                js_class.methods.append(method)

            document.classes.append(js_class)

    def _parse_imports(self, content: str, document: JSDocument):
        """Parse import and require statements"""
        # ES6 imports
        for match in self.IMPORT_PATTERN.finditer(content):
            module_path = match.group(4)
            document.imports.append(module_path)

        # CommonJS requires
        for match in self.REQUIRE_PATTERN.finditer(content):
            module_path = match.group(3)
            document.imports.append(module_path)

    def _parse_exports(self, content: str, document: JSDocument):
        """Parse export statements"""
        for match in self.EXPORT_PATTERN.finditer(content):
            export_name = match.group(1)
            document.exports.append(export_name)

    def _is_ui5_controller(self, content: str) -> bool:
        """Check if file is a UI5 controller"""
        return bool(self.UI5_CONTROLLER_PATTERN.search(content))

    def _parse_ui5_controller(self, content: str, filename: str) -> UI5Controller:
        """Parse UI5 controller specific content"""
        controller = UI5Controller(name=filename)

        # Parse sap.ui.define dependencies
        define_match = self.UI5_CONTROLLER_PATTERN.search(content)
        if define_match:
            deps_str = define_match.group(1)
            deps = [dep.strip().strip("\"'") for dep in deps_str.split(",")]
            controller.dependencies = deps

        # Parse extend call
        extend_match = self.UI5_EXTEND_PATTERN.search(content)
        if extend_match:
            controller.extends = extend_match.group(2)

        # Parse event handlers and methods in UI5 controller
        for match in self.EVENT_HANDLER_PATTERN.finditer(content):
            handler_name = match.group(0).split(":")[0].strip()
            controller.event_handlers.append(handler_name)

        # Parse UI5 controller methods (including onInit, onPress, etc.)
        ui5_method_pattern = re.compile(r"(\w+)\s*:\s*function\s*\([^)]*\)\s*{", re.MULTILINE)
        for match in ui5_method_pattern.finditer(content):
            method_name = match.group(1)
            line_num = content[: match.start()].count("\n")

            method = JSFunction(
                name=method_name,
                parameters=[],  # Could parse parameters if needed
                is_async=False,
                is_arrow=False,
                line_start=line_num,
            )
            controller.methods.append(method)

        # Also parse async methods in UI5 controllers
        ui5_async_method_pattern = re.compile(r"async\s+(\w+)\s*\([^)]*\)\s*{", re.MULTILINE)
        for match in ui5_async_method_pattern.finditer(content):
            method_name = match.group(1)
            line_num = content[: match.start()].count("\n")

            method = JSFunction(
                name=method_name, parameters=[], is_async=True, is_arrow=False, line_start=line_num
            )
            controller.methods.append(method)

        return controller

    def _parse_ui5_view(self, content: str, filename: str) -> UI5View:
        """Parse UI5 XML view"""
        view = UI5View(name=filename)

        try:
            # Parse XML
            root = ET.fromstring(content)

            # Extract controller name
            controller_attr = root.get("controllerName")
            if controller_attr:
                view.controller = controller_attr

            # Extract controls and bindings
            self._extract_xml_controls(root, view)

        except ET.ParseError as e:
            logger.warning("Failed to parse XML view %s: %s", filename, str(e))

        return view

    def _extract_xml_controls(self, element: ET.Element, view: UI5View):
        """Recursively extract controls from XML"""
        # Extract control info
        control_info = {"tag": element.tag, "id": element.get("id"), "class": element.get("class")}
        view.controls.append(control_info)

        # Extract bindings
        for attr_name, attr_value in element.attrib.items():
            if "{" in attr_value and "}" in attr_value:
                view.bindings.append(f"{attr_name}: {attr_value}")

            # Extract event handlers
            if attr_name.startswith("on") or "press" in attr_name.lower():
                view.events.append(f"{attr_name}: {attr_value}")

        # Recurse into children
        for child in element:
            self._extract_xml_controls(child, view)

    def index_directory(self, directory: Path) -> None:
        """Index all JavaScript and XML files in a directory"""
        # Index JavaScript files
        for js_file in directory.rglob("*.js"):
            if "node_modules" in str(js_file) or "__pycache__" in str(js_file):
                continue

            doc = self.index_file(js_file)
            if doc:
                self.documents.append(doc)
                logger.info("Indexed JS file %s", js_file)

        # Index XML view files
        for xml_file in directory.rglob("*.xml"):
            if "node_modules" in str(xml_file):
                continue

            doc = self.index_xml_view(xml_file)
            if doc:
                self.documents.append(doc)
                logger.info("Indexed XML view %s", xml_file)

    def generate_glean_facts(self) -> List[Dict[str, Any]]:
        """Generate Glean facts from JavaScript documents"""
        facts = []

        for doc in self.documents:
            # Generate file facts
            facts.append(
                {
                    "predicate": "src.File",
                    "key": {"path": doc.relative_path},
                    "value": {"language": doc.language.title()},
                }
            )

            # Generate function facts
            for func in doc.functions:
                facts.append(
                    {
                        "predicate": "javascript.Function",
                        "key": {"name": func.name, "file": doc.relative_path},
                        "value": {
                            "parameters": func.parameters,
                            "is_async": func.is_async,
                            "is_arrow": func.is_arrow,
                            "line": func.line_start,
                        },
                    }
                )

            # Generate class facts
            for cls in doc.classes:
                facts.append(
                    {
                        "predicate": "javascript.Class",
                        "key": {"name": cls.name, "file": doc.relative_path},
                        "value": {
                            "extends": cls.extends,
                            "methods": [m.name for m in cls.methods],
                            "line": cls.line_start,
                        },
                    }
                )

            # Generate import facts
            for imp in doc.imports:
                facts.append(
                    {
                        "predicate": "javascript.Import",
                        "key": {"file": doc.relative_path, "module": imp},
                        "value": {},
                    }
                )

            # Generate UI5 controller facts
            if doc.ui5_controller:
                controller = doc.ui5_controller
                facts.append(
                    {
                        "predicate": "ui5.Controller",
                        "key": {"name": controller.name, "file": doc.relative_path},
                        "value": {
                            "extends": controller.extends,
                            "dependencies": controller.dependencies,
                            "event_handlers": controller.event_handlers,
                        },
                    }
                )

                # Generate function facts for UI5 controller methods
                for method in controller.methods:
                    facts.append(
                        {
                            "predicate": "javascript.Function",
                            "key": {"name": method.name, "file": doc.relative_path},
                            "value": {
                                "parameters": method.parameters,
                                "is_async": method.is_async,
                                "is_arrow": method.is_arrow,
                                "line": method.line_start,
                                "context": "ui5_controller",
                            },
                        }
                    )

            # Generate UI5 view facts
            if doc.ui5_view:
                view = doc.ui5_view
                facts.append(
                    {
                        "predicate": "ui5.View",
                        "key": {"name": view.name, "file": doc.relative_path},
                        "value": {
                            "controller": view.controller,
                            "controls": len(view.controls),
                            "bindings": view.bindings,
                            "events": view.events,
                        },
                    }
                )

        return facts


def index_javascript_ui5_for_glean(project_path: str) -> Dict[str, Any]:
    """Main entry point to index JavaScript/UI5 project for Glean"""
    project_root = Path(project_path)
    indexer = JavaScriptUI5Indexer(project_root)

    # Index the entire project
    indexer.index_directory(project_root)

    # Generate outputs
    return {
        "js_documents": len(indexer.documents),
        "glean_facts": indexer.generate_glean_facts(),
        "stats": {
            "files_indexed": len(indexer.documents),
            "total_functions": sum(len(doc.functions) for doc in indexer.documents),
            "total_classes": sum(len(doc.classes) for doc in indexer.documents),
            "ui5_controllers": sum(1 for doc in indexer.documents if doc.ui5_controller),
            "ui5_views": sum(1 for doc in indexer.documents if doc.ui5_view),
        },
    }
