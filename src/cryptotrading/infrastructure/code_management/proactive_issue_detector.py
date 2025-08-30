"""
Proactive Issue Detection and Auto-Fixing System
AI-powered system that detects issues before they become problems and fixes them automatically
"""

import ast
import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .intelligent_code_manager import CodeIssue, FixStatus, IssueType

logger = logging.getLogger(__name__)


@dataclass
class ProactiveRule:
    """Rule for proactive issue detection"""

    name: str
    pattern: str
    issue_type: IssueType
    severity: int
    auto_fix_template: Optional[str]
    description: str


class ProactiveIssueDetector:
    """
    Proactive issue detection and auto-fixing system
    Scans code for potential issues before they become problems
    """

    def __init__(self, project_root: str = ".", database_adapter=None):
        self.project_root = Path(project_root)
        self.detected_issues = []
        self.database_adapter = database_adapter
        self.detection_rules = self._setup_detection_rules()

    def _setup_detection_rules(self) -> List[ProactiveRule]:
        """Setup proactive detection rules"""
        return [
            # Security Issues
            ProactiveRule(
                name="hardcoded_secrets",
                pattern=r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']',
                issue_type=IssueType.SECURITY,
                severity=9,
                auto_fix_template="Move to environment variable",
                description="Hardcoded secrets detected",
            ),
            ProactiveRule(
                name="sql_injection_risk",
                pattern=r'execute\s*\(\s*["\'].*%.*["\']',
                issue_type=IssueType.SECURITY,
                severity=8,
                auto_fix_template="Use parameterized queries",
                description="Potential SQL injection vulnerability",
            ),
            # Performance Issues
            ProactiveRule(
                name="inefficient_loops",
                pattern=r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(",
                issue_type=IssueType.PERFORMANCE,
                severity=5,
                auto_fix_template="Use enumerate() or direct iteration",
                description="Inefficient loop pattern detected",
            ),
            ProactiveRule(
                name="repeated_calculations",
                pattern=r"(\w+\([^)]*\))\s*[+\-*/]\s*\1",
                issue_type=IssueType.PERFORMANCE,
                severity=4,
                auto_fix_template="Cache calculation result",
                description="Repeated calculation detected",
            ),
            # Code Quality Issues
            ProactiveRule(
                name="long_functions",
                pattern=r"def\s+\w+.*?(?=\ndef|\nclass|\Z)",
                issue_type=IssueType.MAINTAINABILITY,
                severity=6,
                auto_fix_template="Break into smaller functions",
                description="Function too long (>50 lines)",
            ),
            ProactiveRule(
                name="magic_numbers",
                pattern=r"\b(?<![\w.])\d{2,}\b(?![\w.])",
                issue_type=IssueType.CODE_SMELL,
                severity=3,
                auto_fix_template="Extract to named constant",
                description="Magic number detected",
            ),
            # Technical Debt
            ProactiveRule(
                name="todo_comments",
                pattern=r"#\s*(TODO|FIXME|HACK|XXX).*",
                issue_type=IssueType.TECHNICAL_DEBT,
                severity=3,
                auto_fix_template=None,
                description="Technical debt comment found",
            ),
            ProactiveRule(
                name="empty_except",
                pattern=r"except[^:]*:\s*pass",
                issue_type=IssueType.CRITICAL,
                severity=7,
                auto_fix_template="Add proper exception handling",
                description="Empty except block detected",
            ),
            # Import Issues
            ProactiveRule(
                name="unused_imports",
                pattern=r"^import\s+(\w+)(?!.*\1)",
                issue_type=IssueType.CODE_SMELL,
                severity=2,
                auto_fix_template="Remove unused import",
                description="Unused import detected",
            ),
            ProactiveRule(
                name="wildcard_imports",
                pattern=r"from\s+\w+\s+import\s+\*",
                issue_type=IssueType.CODE_SMELL,
                severity=4,
                auto_fix_template="Use specific imports",
                description="Wildcard import detected",
            ),
        ]

    async def scan_project(self) -> List[CodeIssue]:
        """Scan entire project for proactive issues"""
        logger.info("🔍 Starting proactive issue scan...")

        issues = []

        # Scan Python files
        python_files = list(self.project_path.rglob("*.py"))
        python_files = [f for f in python_files if self._should_scan_file(f)]

        for py_file in python_files:
            file_issues = await self._scan_file(py_file)
            issues.extend(file_issues)

        # Scan configuration files
        config_issues = await self._scan_config_files()
        issues.extend(config_issues)

        # Analyze code complexity
        complexity_issues = await self._analyze_complexity()
        issues.extend(complexity_issues)

        # Check dependencies
        dependency_issues = await self._check_dependencies()
        issues.extend(dependency_issues)

        self.detected_issues = issues
        logger.info("Found %d proactive issues", len(issues))

        return issues

    def _should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned"""
        exclude_patterns = [
            "__pycache__",
            "node_modules",
            ".git",
            "venv",
            "env",
            ".pytest_cache",
            "dist",
            "build",
            ".vercel",
        ]

        return not any(pattern in str(file_path) for pattern in exclude_patterns)

    async def _scan_file(self, file_path: Path) -> List[CodeIssue]:
        """Scan a single file for issues"""
        issues = []

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Apply regex rules
            for rule in self.rules:
                if rule.name == "long_functions":
                    # Special handling for function length
                    function_issues = self._check_function_length(content, file_path, rule)
                    issues.extend(function_issues)
                else:
                    # Regular regex matching
                    matches = re.finditer(rule.pattern, content, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1

                        issue = CodeIssue(
                            id=f"proactive_{rule.name}_{file_path.name}_{line_num}_{datetime.now().timestamp()}",
                            type=rule.issue_type,
                            severity=rule.severity,
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=line_num,
                            description=f"{rule.description}: {match.group(0)[:100]}",
                            suggested_fix=rule.auto_fix_template,
                            auto_fixable=rule.auto_fix_template is not None,
                            detected_at=datetime.now().isoformat(),
                        )
                        issues.append(issue)

            # AST-based analysis
            ast_issues = self._analyze_ast(content, file_path)
            issues.extend(ast_issues)

        except Exception as e:
            logger.warning("Error scanning file %s: %s", file_path, e)

        return issues

    def _check_function_length(
        self, content: str, file_path: Path, rule: ProactiveRule
    ) -> List[CodeIssue]:
        """Check for overly long functions"""
        issues = []
        lines = content.split("\n")

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_start = node.lineno
                    func_end = node.end_lineno or func_start
                    func_length = func_end - func_start + 1

                    if func_length > 50:  # Configurable threshold
                        issue = CodeIssue(
                            id=f"long_function_{file_path.name}_{node.name}_{datetime.now().timestamp()}",
                            type=rule.issue_type,
                            severity=rule.severity
                            + (func_length // 25),  # Increase severity for longer functions
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=func_start,
                            description=f"Function '{node.name}' is {func_length} lines long (>50 lines)",
                            suggested_fix=f"Break '{node.name}' into smaller functions",
                            auto_fixable=False,  # Requires human judgment
                            detected_at=datetime.now().isoformat(),
                        )
                        issues.append(issue)

        except SyntaxError:
            pass  # Skip files with syntax errors

        return issues

    def _analyze_ast(self, content: str, file_path: Path) -> List[CodeIssue]:
        """Analyze code using AST for deeper insights"""
        issues = []

        try:
            tree = ast.parse(content)

            # Check for complex expressions
            for node in ast.walk(tree):
                if isinstance(node, ast.BoolOp) and len(node.values) > 5:
                    issue = CodeIssue(
                        id=f"complex_boolean_{file_path.name}_{node.lineno}_{datetime.now().timestamp()}",
                        type=IssueType.MAINTAINABILITY,
                        severity=5,
                        file_path=str(file_path.relative_to(self.project_path)),
                        line_number=node.lineno,
                        description=f"Complex boolean expression with {len(node.values)} conditions",
                        suggested_fix="Break into smaller logical units",
                        auto_fixable=False,
                        detected_at=datetime.now().isoformat(),
                    )
                    issues.append(issue)

                # Check for deeply nested code
                if (
                    isinstance(node, (ast.If, ast.For, ast.While))
                    and self._get_nesting_depth(node) > 4
                ):
                    issue = CodeIssue(
                        id=f"deep_nesting_{file_path.name}_{node.lineno}_{datetime.now().timestamp()}",
                        type=IssueType.MAINTAINABILITY,
                        severity=6,
                        file_path=str(file_path.relative_to(self.project_path)),
                        line_number=node.lineno,
                        description="Deeply nested code structure detected",
                        suggested_fix="Extract nested logic into separate functions",
                        auto_fixable=False,
                        detected_at=datetime.now().isoformat(),
                    )
                    issues.append(issue)

        except SyntaxError:
            pass

        return issues

    def _get_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate nesting depth of AST node"""
        max_depth = depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._get_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)

        return max_depth

    async def _scan_config_files(self) -> List[CodeIssue]:
        """Scan configuration files for issues"""
        issues = []

        # Check requirements.txt for outdated packages
        req_files = list(self.project_path.glob("*requirements*.txt"))
        for req_file in req_files:
            try:
                content = req_file.read_text()
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    if line.strip() and not line.startswith("#"):
                        # Check for unpinned versions
                        if "==" not in line and ">=" not in line and "~=" not in line:
                            issue = CodeIssue(
                                id=f"unpinned_dependency_{req_file.name}_{i}_{datetime.now().timestamp()}",
                                type=IssueType.TECHNICAL_DEBT,
                                severity=4,
                                file_path=str(req_file.relative_to(self.project_path)),
                                line_number=i,
                                description=f"Unpinned dependency: {line.strip()}",
                                suggested_fix="Pin to specific version",
                                auto_fixable=False,
                                detected_at=datetime.now().isoformat(),
                            )
                            issues.append(issue)

            except Exception as e:
                logger.warning("Error scanning %s: %s", req_file, e)

        return issues

    async def _analyze_complexity(self) -> List[CodeIssue]:
        """Analyze code complexity metrics"""
        issues = []

        # This would integrate with complexity analysis tools
        # For now, we'll do basic analysis

        python_files = list(self.project_path.rglob("*.py"))
        python_files = [f for f in python_files if self._should_scan_file(f)]

        for py_file in python_files:
            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)

                        if complexity > 10:  # High complexity threshold
                            issue = CodeIssue(
                                id=f"high_complexity_{py_file.name}_{node.name}_{datetime.now().timestamp()}",
                                type=IssueType.MAINTAINABILITY,
                                severity=7,
                                file_path=str(py_file.relative_to(self.project_path)),
                                line_number=node.lineno,
                                description=f"Function '{node.name}' has high cyclomatic complexity: {complexity}",
                                suggested_fix="Reduce complexity by extracting methods or simplifying logic",
                                auto_fixable=False,
                                detected_at=datetime.now().isoformat(),
                            )
                            issues.append(issue)

            except Exception as e:
                logger.warning("Error analyzing complexity for %s: %s", py_file, e)

        return issues

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    async def _check_dependencies(self) -> List[CodeIssue]:
        """Check for dependency-related issues"""
        issues = []

        # Check for security vulnerabilities in dependencies
        # This would integrate with safety, pip-audit, etc.

        # For now, check for common problematic patterns
        python_files = list(self.project_path.rglob("*.py"))
        python_files = [f for f in python_files if self._should_scan_file(f)]

        for py_file in python_files:
            try:
                content = py_file.read_text()

                # Check for deprecated imports
                deprecated_imports = ["imp", "optparse", "distutils", "platform.dist"]

                for dep in deprecated_imports:
                    if f"import {dep}" in content or f"from {dep}" in content:
                        line_num = (
                            content.split("\n").index(
                                next(line for line in content.split("\n") if dep in line)
                            )
                            + 1
                        )

                        issue = CodeIssue(
                            id=f"deprecated_import_{py_file.name}_{dep}_{datetime.now().timestamp()}",
                            type=IssueType.TECHNICAL_DEBT,
                            severity=5,
                            file_path=str(py_file.relative_to(self.project_path)),
                            line_number=line_num,
                            description=f"Deprecated import: {dep}",
                            suggested_fix=f"Replace {dep} with modern alternative",
                            auto_fixable=False,
                            detected_at=datetime.now().isoformat(),
                        )
                        issues.append(issue)

            except Exception as e:
                logger.warning("Error checking dependencies for %s: %s", py_file, e)

        return issues

    async def auto_fix_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Automatically fix issues where possible"""
        fixed_issues = []

        for issue in issues:
            if issue.auto_fixable:
                try:
                    success = await self._apply_proactive_fix(issue)
                    if success:
                        issue.fix_status = FixStatus.COMPLETED
                        fixed_issues.append(issue)
                        logger.info("✅ Auto-fixed proactive issue: %s", issue.description)
                    else:
                        issue.fix_status = FixStatus.FAILED
                except Exception as e:
                    issue.fix_status = FixStatus.FAILED
                    logger.error("Failed to auto-fix issue %s: %s", issue.id, e)

        return fixed_issues

    async def _apply_proactive_fix(self, issue: CodeIssue) -> bool:
        """Apply automatic fix for proactive issues"""
        file_path = self.project_path / issue.file_path

        try:
            content = file_path.read_text()
            lines = content.split("\n")

            if "unused_imports" in issue.id:
                # Remove unused import line
                if issue.line_number and issue.line_number <= len(lines):
                    lines.pop(issue.line_number - 1)
                    file_path.write_text("\n".join(lines))
                    return True

            elif "magic_numbers" in issue.id:
                # This would require more sophisticated analysis
                # For now, just flag for manual review
                return False

            # Add more auto-fix implementations here

        except Exception as e:
            logger.error("Error applying fix for %s: %s", issue.id, e)
            return False

        return False

    def get_issue_summary(self) -> Dict[str, Any]:
        """Get summary of detected issues"""
        if not self.detected_issues:
            return {"total": 0, "by_type": {}, "by_severity": {}}

        by_type = {}
        by_severity = {}

        for issue in self.detected_issues:
            # Count by type
            type_name = issue.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Count by severity
            severity_range = (
                "low" if issue.severity <= 3 else "medium" if issue.severity <= 6 else "high"
            )
            by_severity[severity_range] = by_severity.get(severity_range, 0) + 1

        return {
            "total": len(self.detected_issues),
            "by_type": by_type,
            "by_severity": by_severity,
            "auto_fixable": len([i for i in self.detected_issues if i.auto_fixable]),
            "critical": len([i for i in self.detected_issues if i.severity >= 8]),
        }
