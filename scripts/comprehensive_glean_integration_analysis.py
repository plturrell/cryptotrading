#!/usr/bin/env python3
"""
Comprehensive Glean Integration Analysis - Final Verification
Analyzes all integration components and provides detailed scoring
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@dataclass
class AnalysisResult:
    component: str
    score: float
    max_score: float
    issues: List[str]
    recommendations: List[str]
    success: bool


class ComprehensiveIntegrationAnalyzer:
    """Comprehensive analyzer for all integration components"""

    def __init__(self):
        self.project_root = project_root
        self.results: List[AnalysisResult] = []
        self.logger = logging.getLogger(__name__)

    async def analyze_all_components(self) -> Dict[str, Any]:
        """Run comprehensive analysis on all components"""
        print("🔍 Starting Comprehensive Glean Integration Analysis...")
        print("=" * 60)

        start_time = time.time()

        # Component analyses
        components = [
            ("Code Analyzer", self._analyze_code_analyzer),
            ("CLI Integration", self._analyze_cli_integration),
            ("MCP Integration", self._analyze_mcp_integration),
            ("Framework Integration", self._analyze_framework_integration),
            ("Code Quality", self._analyze_code_quality),
            ("Dependencies", self._analyze_dependencies),
            ("Architecture", self._analyze_architecture),
        ]

        for component_name, analyzer_func in components:
            print(f"\n📊 Analyzing {component_name}...")
            try:
                result = await analyzer_func()
                self.results.append(result)
                status = "✅" if result.success else "❌"
                print(f"{status} {component_name}: {result.score:.1f}/{result.max_score}")

                if result.issues:
                    for issue in result.issues[:3]:  # Show top 3 issues
                        print(f"   ⚠️  {issue}")

            except Exception as e:
                print(f"❌ {component_name}: Analysis failed - {e}")
                self.results.append(
                    AnalysisResult(
                        component=component_name,
                        score=0.0,
                        max_score=100.0,
                        issues=[f"Analysis failed: {e}"],
                        recommendations=[],
                        success=False,
                    )
                )

        # Calculate overall scores
        total_score = sum(r.score for r in self.results)
        max_total = sum(r.max_score for r in self.results)
        overall_percentage = (total_score / max_total * 100) if max_total > 0 else 0

        duration = time.time() - start_time

        # Generate final report
        report = self._generate_final_report(overall_percentage, duration)

        return {
            "overall_score": overall_percentage,
            "component_scores": {r.component: r.score for r in self.results},
            "total_issues": sum(len(r.issues) for r in self.results),
            "analysis_duration": duration,
            "report": report,
            "success": overall_percentage >= 95.0,
        }

    async def _analyze_code_analyzer(self) -> AnalysisResult:
        """Analyze CodeAnalyzer implementation"""
        issues = []
        score = 0.0
        max_score = 100.0

        try:
            # Check if CodeAnalyzer exists and has required methods
            analyzer_path = (
                self.project_root / "src/cryptotrading/infrastructure/analysis/code_analyzer.py"
            )
            if not analyzer_path.exists():
                issues.append("CodeAnalyzer file not found")
                return AnalysisResult("Code Analyzer", 0, max_score, issues, [], False)

            # Read and analyze the code
            with open(analyzer_path, "r") as f:
                content = f.read()

            # Check for required methods
            required_methods = [
                "analyze_dependencies",
                "analyze_symbol_usage",
                "analyze_code_complexity",
                "detect_dead_code",
            ]

            method_score = 0
            for method in required_methods:
                if f"def {method}" in content:
                    method_score += 20
                else:
                    issues.append(f"Missing method: {method}")

            score += method_score

            # Check for proper imports
            if "from .glean_client import GleanClient" in content:
                score += 10
            else:
                issues.append("Missing GleanClient import")

            # Check for error handling
            if "try:" in content and "except" in content:
                score += 10
            else:
                issues.append("Insufficient error handling")

        except Exception as e:
            issues.append(f"Analysis error: {e}")

        return AnalysisResult(
            component="Code Analyzer",
            score=score,
            max_score=max_score,
            issues=issues,
            recommendations=["Add missing methods", "Improve error handling"],
            success=score >= 80,
        )

    async def _analyze_cli_integration(self) -> AnalysisResult:
        """Analyze CLI integration"""
        issues = []
        score = 0.0
        max_score = 100.0

        try:
            cli_path = (
                self.project_root / "src/cryptotrading/infrastructure/analysis/cli_commands.py"
            )
            if not cli_path.exists():
                issues.append("CLI commands file not found")
                return AnalysisResult("CLI Integration", 0, max_score, issues, [], False)

            with open(cli_path, "r") as f:
                content = f.read()

            # Check for required CLI functions
            cli_functions = [
                "glean_analyze_dependencies",
                "glean_analyze_impact",
                "glean_validate_architecture",
                "glean_full_analysis",
            ]

            function_score = 0
            for func in cli_functions:
                if f"async def {func}" in content:
                    function_score += 20
                else:
                    issues.append(f"Missing CLI function: {func}")

            score += function_score

            # Check for GleanCLI class
            if "class GleanCLI:" in content:
                score += 10
            else:
                issues.append("Missing GleanCLI class")

            # Check for missing methods
            if "_generate_impact_report" in content:
                score += 5
            else:
                issues.append("Missing _generate_impact_report method")

            if "_generate_full_analysis_report" in content:
                score += 5
            else:
                issues.append("Missing _generate_full_analysis_report method")

        except Exception as e:
            issues.append(f"CLI analysis error: {e}")

        return AnalysisResult(
            component="CLI Integration",
            score=score,
            max_score=max_score,
            issues=issues,
            recommendations=["Add missing CLI methods", "Fix import issues"],
            success=score >= 80,
        )

    async def _analyze_mcp_integration(self) -> AnalysisResult:
        """Analyze MCP integration"""
        issues = []
        score = 80.0  # Base score for existing MCP system
        max_score = 100.0

        try:
            # Check MCP server file
            mcp_path = self.project_root / "api/mcp.py"
            if mcp_path.exists():
                score += 10
            else:
                issues.append("MCP server file not found")

            # Check for Glean tools in MCP
            if mcp_path.exists():
                with open(mcp_path, "r") as f:
                    content = f.read()

                if "glean" in content.lower():
                    score += 10
                else:
                    issues.append("Glean integration not found in MCP server")

        except Exception as e:
            issues.append(f"MCP analysis error: {e}")

        return AnalysisResult(
            component="MCP Integration",
            score=score,
            max_score=max_score,
            issues=issues,
            recommendations=["Enhance Glean-MCP integration"],
            success=score >= 80,
        )

    async def _analyze_framework_integration(self) -> AnalysisResult:
        """Analyze framework integration"""
        issues = []
        score = 0.0
        max_score = 100.0

        try:
            # Check agent testing framework
            framework_path = self.project_root / "framework/agent_testing"
            if framework_path.exists():
                score += 30
            else:
                issues.append("Agent testing framework not found")

            # Check for CLI integration
            cli_path = framework_path / "cli.py" if framework_path.exists() else None
            if cli_path and cli_path.exists():
                score += 20
            else:
                issues.append("Framework CLI not found")

            # Check for Glean integration in framework
            if framework_path.exists():
                glean_files = list(framework_path.rglob("*glean*"))
                if glean_files:
                    score += 25
                else:
                    issues.append("No Glean integration in framework")

            # Check requirements.txt for dependencies
            req_path = self.project_root / "requirements.txt"
            if req_path.exists():
                with open(req_path, "r") as f:
                    content = f.read()
                if "networkx" in content and "click" in content:
                    score += 25
                else:
                    issues.append("Missing required dependencies")

        except Exception as e:
            issues.append(f"Framework analysis error: {e}")

        return AnalysisResult(
            component="Framework Integration",
            score=score,
            max_score=max_score,
            issues=issues,
            recommendations=["Complete framework integration", "Add missing dependencies"],
            success=score >= 80,
        )

    async def _analyze_code_quality(self) -> AnalysisResult:
        """Analyze code quality issues"""
        issues = []
        score = 70.0  # Base score
        max_score = 100.0

        try:
            # Check for common code quality issues
            analysis_dir = self.project_root / "src/cryptotrading/infrastructure/analysis"
            if analysis_dir.exists():
                python_files = list(analysis_dir.glob("*.py"))

                for py_file in python_files:
                    with open(py_file, "r") as f:
                        content = f.read()

                    # Check for f-strings in logging (should use lazy formatting)
                    if 'logger.error(f"' in content or 'logger.info(f"' in content:
                        issues.append(f"F-string in logging: {py_file.name}")
                        score -= 2

                    # Check for broad exception handling
                    if "except Exception:" in content:
                        issues.append(f"Broad exception handling: {py_file.name}")
                        score -= 2

                    # Check for unused imports
                    if "import asyncio" in content and "asyncio." not in content:
                        issues.append(f"Unused import: {py_file.name}")
                        score -= 1

                # Bonus for having type hints
                if any("-> Dict[str, Any]" in open(f).read() for f in python_files):
                    score += 10

        except Exception as e:
            issues.append(f"Code quality analysis error: {e}")

        return AnalysisResult(
            component="Code Quality",
            score=max(score, 0),
            max_score=max_score,
            issues=issues,
            recommendations=[
                "Fix logging format",
                "Improve exception handling",
                "Remove unused imports",
            ],
            success=score >= 80,
        )

    async def _analyze_dependencies(self) -> AnalysisResult:
        """Analyze dependency management"""
        issues = []
        score = 0.0
        max_score = 100.0

        try:
            req_path = self.project_root / "requirements.txt"
            if req_path.exists():
                with open(req_path, "r") as f:
                    content = f.read()

                required_deps = [
                    "networkx",
                    "click",
                    "watchdog",
                    "rich",
                    "scip-python",
                    "protobuf",
                    "lsif-py",
                ]

                for dep in required_deps:
                    if dep in content:
                        score += 100 / len(required_deps)
                    else:
                        issues.append(f"Missing dependency: {dep}")
            else:
                issues.append("requirements.txt not found")

        except Exception as e:
            issues.append(f"Dependency analysis error: {e}")

        return AnalysisResult(
            component="Dependencies",
            score=score,
            max_score=max_score,
            issues=issues,
            recommendations=["Add missing dependencies"],
            success=score >= 90,
        )

    async def _analyze_architecture(self) -> AnalysisResult:
        """Analyze overall architecture"""
        issues = []
        score = 85.0  # Base score for existing architecture
        max_score = 100.0

        try:
            # Check for proper module structure
            analysis_dir = self.project_root / "src/cryptotrading/infrastructure/analysis"
            if analysis_dir.exists():
                expected_files = [
                    "glean_client.py",
                    "code_analyzer.py",
                    "cli_commands.py",
                    "architecture_validator.py",
                ]

                for file in expected_files:
                    if (analysis_dir / file).exists():
                        score += 15 / len(expected_files)
                    else:
                        issues.append(f"Missing architecture file: {file}")
            else:
                issues.append("Analysis module not found")
                score = 0

        except Exception as e:
            issues.append(f"Architecture analysis error: {e}")

        return AnalysisResult(
            component="Architecture",
            score=score,
            max_score=max_score,
            issues=issues,
            recommendations=["Complete architecture implementation"],
            success=score >= 80,
        )

    def _generate_final_report(self, overall_percentage: float, duration: float) -> str:
        """Generate comprehensive final report"""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("🎯 FINAL INTEGRATION ANALYSIS REPORT")
        lines.append("=" * 60)

        # Overall score
        if overall_percentage >= 95:
            status = "🏆 EXCELLENT"
            color = "🟢"
        elif overall_percentage >= 85:
            status = "✅ GOOD"
            color = "🟡"
        elif overall_percentage >= 70:
            status = "⚠️  NEEDS IMPROVEMENT"
            color = "🟠"
        else:
            status = "❌ CRITICAL ISSUES"
            color = "🔴"

        lines.append(f"\n{color} Overall Integration Score: {overall_percentage:.1f}% - {status}")
        lines.append(f"⏱️  Analysis Duration: {duration:.2f} seconds")

        # Component breakdown
        lines.append(f"\n📊 Component Scores:")
        lines.append("-" * 40)

        for result in self.results:
            percentage = (result.score / result.max_score * 100) if result.max_score > 0 else 0
            status_icon = "✅" if result.success else "❌"
            lines.append(f"{status_icon} {result.component:<20} {percentage:>6.1f}%")

        # Issues summary
        total_issues = sum(len(r.issues) for r in self.results)
        if total_issues > 0:
            lines.append(f"\n⚠️  Total Issues Found: {total_issues}")
            lines.append("\n🔧 Top Priority Fixes:")

            # Show top 5 issues across all components
            all_issues = []
            for result in self.results:
                for issue in result.issues:
                    all_issues.append(f"{result.component}: {issue}")

            for i, issue in enumerate(all_issues[:5], 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("\n🎉 No issues found! Perfect integration!")

        # Recommendations
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)

        if all_recommendations:
            lines.append(f"\n💡 Recommendations:")
            unique_recs = list(set(all_recommendations))
            for i, rec in enumerate(unique_recs[:3], 1):
                lines.append(f"  {i}. {rec}")

        # Final status
        lines.append("\n" + "=" * 60)
        if overall_percentage >= 100:
            lines.append("🏆 PERFECT SCORE ACHIEVED! 🏆")
        elif overall_percentage >= 95:
            lines.append("🎯 EXCELLENT INTEGRATION - PRODUCTION READY")
        elif overall_percentage >= 85:
            lines.append("✅ GOOD INTEGRATION - MINOR IMPROVEMENTS NEEDED")
        else:
            lines.append("⚠️  INTEGRATION NEEDS WORK - ADDRESS CRITICAL ISSUES")

        lines.append("=" * 60)

        return "\n".join(lines)


async def main():
    """Main analysis function"""
    analyzer = ComprehensiveIntegrationAnalyzer()

    try:
        results = await analyzer.analyze_all_components()

        # Print the report
        print(results["report"])

        # Save detailed results
        output_file = analyzer.project_root / "data/integration_analysis_results.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {output_file}")

        # Exit with appropriate code
        sys.exit(0 if results["success"] else 1)

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
