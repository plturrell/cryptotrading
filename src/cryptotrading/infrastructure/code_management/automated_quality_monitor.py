"""
Automated Code Quality Monitor
Real-time code quality monitoring with automated issue detection and fixing
"""

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database_adapter import CodeManagementDatabaseAdapter
from .intelligent_code_manager import CodeIssue, FixStatus, IssueType

logger = logging.getLogger(__name__)


@dataclass
class QualityCheck:
    """Represents a quality check configuration"""

    name: str
    command: str
    parser: str  # Function name to parse results
    auto_fix: bool
    severity_mapping: Dict[str, int]


class AutomatedQualityMonitor:
    """Automated code quality monitoring with real-time analysis"""

    def __init__(
        self, project_path: str, database_adapter: Optional[CodeManagementDatabaseAdapter] = None
    ):
        self.project_path = Path(project_path)
        self.quality_checks = self._setup_quality_checks()
        self.results_history = []
        self.database_adapter = database_adapter

    def _setup_quality_checks(self) -> List[QualityCheck]:
        """Setup quality check configurations"""
        return [
            QualityCheck(
                name="pylint",
                command="pylint --output-format=json",
                parser="parse_pylint_results",
                auto_fix=False,
                severity_mapping={"error": 9, "warning": 5, "refactor": 3, "convention": 2},
            ),
            QualityCheck(
                name="mypy",
                command="mypy --show-error-codes --json-report",
                parser="parse_mypy_results",
                auto_fix=False,
                severity_mapping={"error": 8, "note": 3},
            ),
            QualityCheck(
                name="bandit",
                command="bandit -r -f json",
                parser="parse_bandit_results",
                auto_fix=False,
                severity_mapping={"HIGH": 9, "MEDIUM": 6, "LOW": 3},
            ),
            QualityCheck(
                name="black",
                command="black --check --diff",
                parser="parse_black_results",
                auto_fix=True,
                severity_mapping={"format": 4},
            ),
            QualityCheck(
                name="isort",
                command="isort --check-only --diff",
                parser="parse_isort_results",
                auto_fix=True,
                severity_mapping={"import": 3},
            ),
            QualityCheck(
                name="flake8",
                command="flake8 --format=json",
                parser="parse_flake8_results",
                auto_fix=False,
                severity_mapping={"E": 6, "W": 4, "F": 7, "C": 3},
            ),
        ]

    async def run_continuous_monitoring(self, interval_seconds: int = 300) -> None:
        """Run continuous quality monitoring"""
        logger.info(
            "🔄 Starting continuous quality monitoring (interval: %d seconds)", interval_seconds
        )

        while True:
            try:
                # Run all quality checks
                results = await self.run_all_checks()

                # Process results and detect issues
                issues = self.process_results(results)

                # Auto-fix issues where possible
                fixed_issues = await self.auto_fix_issues(issues)

                # Log summary
                logger.info(
                    "Quality check complete: %d issues found, %d auto-fixed",
                    len(issues),
                    len(fixed_issues),
                )

                # Store results
                await self.store_results(results, issues, fixed_issues)

                # Wait for next cycle
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error("Error in quality monitoring: %s", e)
                await asyncio.sleep(60)

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all configured quality checks"""
        logger.info("🔍 Running all quality checks...")

        results = {}

        for check in self.quality_checks:
            try:
                result = await self.run_single_check(check)
                results[check.name] = result
                logger.info("✅ %s check completed", check.name)
            except Exception as e:
                logger.error("❌ %s check failed: %s", check.name, e)
                results[check.name] = {"error": str(e)}

        return results

    async def run_single_check(self, check: QualityCheck) -> Dict[str, Any]:
        """Run a single quality check"""
        # Build command for Python files
        python_files = list(self.project_path.rglob("*.py"))
        python_files = [
            f for f in python_files if "node_modules" not in str(f) and "__pycache__" not in str(f)
        ]

        if not python_files:
            return {"files": [], "issues": []}

        # Run the command
        cmd = (
            f"{check.command} {' '.join(str(f) for f in python_files[:10])}"  # Limit files for demo
        )

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Parse results using the specified parser
            parser_method = getattr(self, check.parser, None)
            if parser_method:
                return parser_method(stdout.decode(), stderr.decode(), check)
            else:
                return {"raw_stdout": stdout.decode(), "raw_stderr": stderr.decode()}

        except Exception as e:
            return {"error": str(e)}

    def parse_pylint_results(self, stdout: str, stderr: str, check: QualityCheck) -> Dict[str, Any]:
        """Parse pylint JSON output"""
        try:
            if stdout.strip():
                data = json.loads(stdout)
                issues = []
                for item in data:
                    issues.append(
                        {
                            "file": item.get("path", ""),
                            "line": item.get("line", 0),
                            "column": item.get("column", 0),
                            "type": item.get("type", ""),
                            "symbol": item.get("symbol", ""),
                            "message": item.get("message", ""),
                            "severity": check.severity_mapping.get(item.get("type", ""), 5),
                        }
                    )
                return {"issues": issues, "total": len(issues)}
            return {"issues": [], "total": 0}
        except json.JSONDecodeError:
            return {"error": "Failed to parse pylint output", "raw": stdout}

    def parse_mypy_results(self, stdout: str, stderr: str, check: QualityCheck) -> Dict[str, Any]:
        """Parse mypy output"""
        issues = []
        for line in stdout.split("\n"):
            if ":" in line and ("error:" in line or "note:" in line):
                parts = line.split(":")
                if len(parts) >= 4:
                    file_path = parts[0]
                    line_num = parts[1] if parts[1].isdigit() else 0
                    message = ":".join(parts[3:]).strip()
                    issue_type = "error" if "error:" in line else "note"

                    issues.append(
                        {
                            "file": file_path,
                            "line": int(line_num) if str(line_num).isdigit() else 0,
                            "message": message,
                            "type": issue_type,
                            "severity": check.severity_mapping.get(issue_type, 5),
                        }
                    )

        return {"issues": issues, "total": len(issues)}

    def parse_bandit_results(self, stdout: str, stderr: str, check: QualityCheck) -> Dict[str, Any]:
        """Parse bandit JSON output"""
        try:
            if stdout.strip():
                data = json.loads(stdout)
                issues = []
                for result in data.get("results", []):
                    issues.append(
                        {
                            "file": result.get("filename", ""),
                            "line": result.get("line_number", 0),
                            "test_id": result.get("test_id", ""),
                            "test_name": result.get("test_name", ""),
                            "issue_severity": result.get("issue_severity", "LOW"),
                            "issue_confidence": result.get("issue_confidence", "LOW"),
                            "issue_text": result.get("issue_text", ""),
                            "severity": check.severity_mapping.get(
                                result.get("issue_severity", "LOW"), 3
                            ),
                        }
                    )
                return {"issues": issues, "total": len(issues)}
            return {"issues": [], "total": 0}
        except json.JSONDecodeError:
            return {"error": "Failed to parse bandit output", "raw": stdout}

    def parse_black_results(self, stdout: str, stderr: str, check: QualityCheck) -> Dict[str, Any]:
        """Parse black formatting check output"""
        issues = []
        if "would reformat" in stdout or "reformatted" in stdout:
            for line in stdout.split("\n"):
                if "would reformat" in line:
                    file_path = line.split("would reformat")[0].strip()
                    issues.append(
                        {
                            "file": file_path,
                            "line": 0,
                            "message": "File needs formatting",
                            "type": "format",
                            "severity": check.severity_mapping.get("format", 4),
                            "auto_fixable": True,
                        }
                    )

        return {"issues": issues, "total": len(issues)}

    def parse_isort_results(self, stdout: str, stderr: str, check: QualityCheck) -> Dict[str, Any]:
        """Parse isort import sorting check output"""
        issues = []
        if "Skipped" not in stdout and stdout.strip():
            for line in stdout.split("\n"):
                if line.strip() and not line.startswith("---") and not line.startswith("+++"):
                    issues.append(
                        {
                            "file": "imports",
                            "line": 0,
                            "message": "Import order needs fixing",
                            "type": "import",
                            "severity": check.severity_mapping.get("import", 3),
                            "auto_fixable": True,
                        }
                    )
                    break  # One issue per file

        return {"issues": issues, "total": len(issues)}

    def parse_flake8_results(self, stdout: str, stderr: str, check: QualityCheck) -> Dict[str, Any]:
        """Parse flake8 output"""
        issues = []
        for line in stdout.split("\n"):
            if ":" in line:
                parts = line.split(":")
                if len(parts) >= 4:
                    file_path = parts[0]
                    line_num = parts[1] if parts[1].isdigit() else 0
                    col_num = parts[2] if parts[2].isdigit() else 0
                    message = ":".join(parts[3:]).strip()

                    # Extract error code
                    error_code = ""
                    if message and len(message) > 4:
                        error_code = message.split()[0]

                    error_type = error_code[0] if error_code else "E"

                    issues.append(
                        {
                            "file": file_path,
                            "line": int(line_num) if str(line_num).isdigit() else 0,
                            "column": int(col_num) if str(col_num).isdigit() else 0,
                            "code": error_code,
                            "message": message,
                            "type": error_type,
                            "severity": check.severity_mapping.get(error_type, 5),
                        }
                    )

        return {"issues": issues, "total": len(issues)}

    def process_results(self, results: Dict[str, Any]) -> List[CodeIssue]:
        """Process quality check results into CodeIssue objects"""
        issues = []

        for tool_name, result in results.items():
            if "error" in result:
                continue

            tool_issues = result.get("issues", [])
            for issue_data in tool_issues:
                issue = CodeIssue(
                    id=f"{tool_name}_{len(issues)}_{datetime.now().timestamp()}",
                    type=self._map_issue_type(issue_data.get("type", ""), tool_name),
                    severity=issue_data.get("severity", 5),
                    file_path=issue_data.get("file", ""),
                    line_number=issue_data.get("line", 0)
                    if issue_data.get("line", 0) > 0
                    else None,
                    description=f"[{tool_name}] {issue_data.get('message', 'Unknown issue')}",
                    suggested_fix=self._generate_fix_suggestion(issue_data, tool_name),
                    auto_fixable=issue_data.get("auto_fixable", tool_name in ["black", "isort"]),
                    detected_at=datetime.now().isoformat(),
                )
                issues.append(issue)

        return issues

    def _map_issue_type(self, issue_type: str, tool_name: str) -> IssueType:
        """Map tool-specific issue types to our IssueType enum"""
        if tool_name == "bandit":
            return IssueType.SECURITY
        elif tool_name in ["black", "isort"]:
            return IssueType.CODE_SMELL
        elif issue_type in ["error", "E"]:
            return IssueType.CRITICAL
        elif issue_type in ["warning", "W"]:
            return IssueType.MAINTAINABILITY
        elif issue_type in ["refactor", "convention", "C"]:
            return IssueType.TECHNICAL_DEBT
        else:
            return IssueType.CODE_SMELL

    def _generate_fix_suggestion(self, issue_data: Dict[str, Any], tool_name: str) -> Optional[str]:
        """Generate fix suggestions for issues"""
        if tool_name == "black":
            return "Run 'black' to auto-format this file"
        elif tool_name == "isort":
            return "Run 'isort' to fix import ordering"
        elif tool_name == "pylint":
            symbol = issue_data.get("symbol", "")
            if symbol:
                return f"Address pylint {symbol} issue"
        elif tool_name == "mypy":
            return "Add type annotations or fix type errors"
        elif tool_name == "bandit":
            return "Review security issue and implement secure alternative"

        return None

    async def auto_fix_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Auto-fix issues where possible"""
        fixed_issues = []

        for issue in issues:
            if issue.auto_fixable:
                try:
                    success = await self._apply_auto_fix(issue)
                    if success:
                        issue.fix_status = FixStatus.COMPLETED
                        fixed_issues.append(issue)
                        logger.info("✅ Auto-fixed: %s", issue.description)
                    else:
                        issue.fix_status = FixStatus.FAILED
                except Exception as e:
                    issue.fix_status = FixStatus.FAILED
                    logger.error("Failed to auto-fix issue: %s", e)

        return fixed_issues

    async def _apply_auto_fix(self, issue: CodeIssue) -> bool:
        """Apply automatic fix for specific issue types"""
        if "black" in issue.description.lower():
            # Run black on the file
            cmd = f"black {issue.file_path}"
            process = await asyncio.create_subprocess_shell(cmd, cwd=self.project_path)
            await process.communicate()
            return process.returncode == 0

        elif "isort" in issue.description.lower():
            # Run isort on the file
            cmd = f"isort {issue.file_path}"
            process = await asyncio.create_subprocess_shell(cmd, cwd=self.project_path)
            await process.communicate()
            return process.returncode == 0

        return False

    async def store_results(
        self, results: Dict[str, Any], issues: List[CodeIssue], fixed_issues: List[CodeIssue]
    ) -> None:
        """Store monitoring results"""
        timestamp = datetime.now().isoformat()

        summary = {
            "timestamp": timestamp,
            "total_issues": len(issues),
            "fixed_issues": len(fixed_issues),
            "issues_by_tool": {},
            "issues_by_severity": {},
        }

        # Count by tool and severity
        for issue in issues:
            tool = (
                issue.description.split("]")[0].replace("[", "")
                if "]" in issue.description
                else "unknown"
            )
            summary["issues_by_tool"][tool] = summary["issues_by_tool"].get(tool, 0) + 1
            summary["issues_by_severity"][str(issue.severity)] = (
                summary["issues_by_severity"].get(str(issue.severity), 0) + 1
            )

        # Store to database if available
        if self.database_adapter:
            try:
                # Save all issues to database
                for issue in issues:
                    await self.database_adapter.save_issue(issue)

                # Log monitoring event
                await self.database_adapter.log_monitoring_event(
                    event_type="quality_check",
                    details={"summary": summary, "tool_results": results},
                )

                # Save quality metrics
                await self.database_adapter.save_metric(
                    metric_name="quality_issues_total",
                    value=len(issues),
                    metadata={"timestamp": timestamp, "tool_breakdown": summary["issues_by_tool"]},
                )

                await self.database_adapter.save_metric(
                    metric_name="quality_issues_fixed",
                    value=len(fixed_issues),
                    metadata={"timestamp": timestamp},
                )

                logger.info("📊 Quality monitoring results stored to database")

            except Exception as e:
                logger.error("Failed to store results to database: %s", e)
                # Fallback to file storage
                await self._store_to_file(results, issues, summary)
        else:
            # Fallback to file storage
            await self._store_to_file(results, issues, summary)

        self.results_history.append(summary)

    async def _store_to_file(
        self, results: Dict[str, Any], issues: List[CodeIssue], summary: Dict[str, Any]
    ) -> None:
        """Fallback file storage for monitoring results"""
        results_file = self.project_path / "data" / "quality_monitoring_results.json"
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "detailed_results": results,
                    "issues": [
                        {
                            "id": issue.id,
                            "type": issue.type.value,
                            "severity": issue.severity,
                            "file": issue.file_path,
                            "line": issue.line_number,
                            "description": issue.description,
                            "fix_status": issue.fix_status.value,
                        }
                        for issue in issues
                    ],
                },
                f,
                indent=2,
            )

        logger.info("📊 Quality monitoring results stored to file")

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get current quality summary"""
        if not self.results_history:
            return {"status": "no_data"}

        latest = self.results_history[-1]

        # Calculate trends
        trend = "stable"
        if len(self.results_history) >= 2:
            previous = self.results_history[-2]
            if latest["total_issues"] < previous["total_issues"]:
                trend = "improving"
            elif latest["total_issues"] > previous["total_issues"]:
                trend = "declining"

        return {
            "timestamp": latest["timestamp"],
            "total_issues": latest["total_issues"],
            "fixed_issues": latest["fixed_issues"],
            "trend": trend,
            "issues_by_tool": latest["issues_by_tool"],
            "issues_by_severity": latest["issues_by_severity"],
        }
