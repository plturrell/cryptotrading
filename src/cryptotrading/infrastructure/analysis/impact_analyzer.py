"""
Impact Analyzer - Production implementation for change impact analysis
Analyzes the impact of code changes using Glean integration
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .glean_client import CodeReference, CodeSymbol, GleanClient

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChangeImpactReport:
    """Comprehensive change impact report"""

    file_path: str
    changed_symbols: List[str]
    affected_files: List[str]
    affected_modules: List[str]
    affected_tests: List[str]
    risk_level: RiskLevel
    impact_score: float
    recommendations: List[str]
    breaking_changes: List[Dict[str, Any]]
    timestamp: datetime


@dataclass
class ImpactedComponent:
    """Represents a component impacted by changes"""

    name: str
    type: str  # module, class, function, test
    file_path: str
    impact_type: str  # direct, indirect, test
    risk_level: RiskLevel
    details: str


class ImpactAnalyzer:
    """Analyzes the impact of code changes"""

    def __init__(self, glean_client: GleanClient):
        self.glean = glean_client
        self.impact_cache: Dict[str, ChangeImpactReport] = {}

        # Critical modules that require extra attention
        self.critical_modules = {
            "cryptotrading.core.agents.base",
            "cryptotrading.core.protocols.mcp",
            "cryptotrading.core.protocols.a2a",
            "cryptotrading.data.database",
            "cryptotrading.infrastructure.security",
        }

    async def analyze_file_change_impact(
        self, file_path: str, changed_lines: Optional[List[int]] = None
    ) -> ChangeImpactReport:
        """Analyze impact of changes to a specific file"""
        logger.info(f"Analyzing change impact for {file_path}")

        # Get all symbols in the changed file
        symbols = await self.glean.get_file_symbols(file_path)

        if changed_lines:
            # Filter symbols to only those in changed lines
            symbols = [s for s in symbols if s.line in changed_lines]

        changed_symbol_names = [s.name for s in symbols]

        # Find all references to changed symbols
        all_references = []
        for symbol in symbols:
            refs = await self.glean.find_references(symbol.name)
            all_references.extend(refs)

        # Analyze impact
        affected_files = list(set(ref.file_path for ref in all_references))
        affected_modules = list(
            set(self._get_module_from_path(ref.file_path) for ref in all_references)
        )

        # Find affected tests
        affected_tests = await self._find_affected_tests(changed_symbol_names, affected_files)

        # Calculate risk level and impact score
        risk_level, impact_score = await self._calculate_risk_level(
            file_path, changed_symbol_names, affected_files, affected_modules
        )

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            file_path, symbols, affected_files, risk_level
        )

        # Detect breaking changes
        breaking_changes = await self._detect_breaking_changes(symbols, all_references)

        report = ChangeImpactReport(
            file_path=file_path,
            changed_symbols=changed_symbol_names,
            affected_files=affected_files,
            affected_modules=affected_modules,
            affected_tests=affected_tests,
            risk_level=risk_level,
            impact_score=impact_score,
            recommendations=recommendations,
            breaking_changes=breaking_changes,
            timestamp=datetime.now(),
        )

        # Cache the report
        self.impact_cache[file_path] = report

        return report

    async def analyze_symbol_change_impact(
        self, symbol_name: str, change_type: str = "modification"
    ) -> Dict[str, Any]:
        """Analyze impact of changes to a specific symbol"""
        logger.info(f"Analyzing impact for symbol: {symbol_name}")

        # Find all references to the symbol
        references = await self.glean.find_references(symbol_name)

        if not references:
            return {
                "symbol": symbol_name,
                "impact": "none",
                "risk_level": RiskLevel.LOW,
                "affected_components": [],
            }

        # Categorize impacts
        direct_impacts = []
        indirect_impacts = []
        test_impacts = []

        for ref in references:
            impact_component = ImpactedComponent(
                name=self._get_symbol_at_location(ref.file_path, ref.line),
                type=self._determine_component_type(ref.file_path),
                file_path=ref.file_path,
                impact_type="direct",
                risk_level=self._assess_reference_risk(ref, change_type),
                details=f"{change_type} of {symbol_name} affects {ref.reference_type} at line {ref.line}",
            )

            if "test" in ref.file_path.lower():
                test_impacts.append(impact_component)
            else:
                direct_impacts.append(impact_component)

        # Find indirect impacts (symbols that depend on directly impacted symbols)
        for impact in direct_impacts:
            indirect_refs = await self.glean.find_references(impact.name)
            for indirect_ref in indirect_refs:
                if indirect_ref.file_path not in [i.file_path for i in direct_impacts]:
                    indirect_impact = ImpactedComponent(
                        name=self._get_symbol_at_location(
                            indirect_ref.file_path, indirect_ref.line
                        ),
                        type=self._determine_component_type(indirect_ref.file_path),
                        file_path=indirect_ref.file_path,
                        impact_type="indirect",
                        risk_level=RiskLevel.MEDIUM,
                        details=f"Indirectly affected through {impact.name}",
                    )
                    indirect_impacts.append(indirect_impact)

        # Calculate overall risk
        all_impacts = direct_impacts + indirect_impacts + test_impacts
        overall_risk = self._calculate_overall_risk(all_impacts, change_type)

        return {
            "symbol": symbol_name,
            "change_type": change_type,
            "total_impacts": len(all_impacts),
            "direct_impacts": len(direct_impacts),
            "indirect_impacts": len(indirect_impacts),
            "test_impacts": len(test_impacts),
            "overall_risk": overall_risk,
            "affected_components": [
                {
                    "name": comp.name,
                    "type": comp.type,
                    "file": comp.file_path,
                    "impact_type": comp.impact_type,
                    "risk_level": comp.risk_level.value,
                    "details": comp.details,
                }
                for comp in all_impacts
            ],
        }

    async def analyze_pr_impact(
        self, changed_files: List[str], pr_diff: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze impact of a pull request"""
        logger.info(f"Analyzing PR impact for {len(changed_files)} files")

        all_reports = []
        overall_risk = RiskLevel.LOW
        total_affected_files = set()
        total_affected_modules = set()
        total_affected_tests = set()

        for file_path in changed_files:
            # Parse changed lines from diff if available
            changed_lines = self._parse_changed_lines(pr_diff, file_path) if pr_diff else None

            # Analyze each file
            report = await self.analyze_file_change_impact(file_path, changed_lines)
            all_reports.append(report)

            # Aggregate data
            total_affected_files.update(report.affected_files)
            total_affected_modules.update(report.affected_modules)
            total_affected_tests.update(report.affected_tests)

            # Update overall risk
            if report.risk_level.value == "critical":
                overall_risk = RiskLevel.CRITICAL
            elif report.risk_level.value == "high" and overall_risk != RiskLevel.CRITICAL:
                overall_risk = RiskLevel.HIGH
            elif report.risk_level.value == "medium" and overall_risk in [RiskLevel.LOW]:
                overall_risk = RiskLevel.MEDIUM

        # Generate PR-level recommendations
        pr_recommendations = self._generate_pr_recommendations(all_reports, overall_risk)

        return {
            "changed_files": changed_files,
            "file_reports": [
                {
                    "file": report.file_path,
                    "risk_level": report.risk_level.value,
                    "impact_score": report.impact_score,
                    "affected_files": len(report.affected_files),
                    "affected_modules": len(report.affected_modules),
                    "breaking_changes": len(report.breaking_changes),
                }
                for report in all_reports
            ],
            "overall_impact": {
                "risk_level": overall_risk.value,
                "total_affected_files": len(total_affected_files),
                "total_affected_modules": len(total_affected_modules),
                "total_affected_tests": len(total_affected_tests),
                "requires_full_test_suite": overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            },
            "recommendations": pr_recommendations,
            "test_strategy": self._generate_test_strategy(all_reports, overall_risk),
        }

    async def _find_affected_tests(
        self, changed_symbols: List[str], affected_files: List[str]
    ) -> List[str]:
        """Find test files that might be affected by changes"""
        affected_tests = []

        # Look for test files that reference changed symbols
        for symbol in changed_symbols:
            refs = await self.glean.find_references(symbol)
            test_refs = [ref for ref in refs if "test" in ref.file_path.lower()]
            affected_tests.extend([ref.file_path for ref in test_refs])

        # Look for test files that test affected modules
        for file_path in affected_files:
            module_name = self._get_module_from_path(file_path)
            test_file_patterns = [
                f"test_{module_name.split('.')[-1]}.py",
                f"test_{module_name.replace('.', '_')}.py",
                f"{module_name.split('.')[-1]}_test.py",
            ]

            # Search for corresponding test files
            for pattern in test_file_patterns:
                # This would need actual file system search or Glean query
                pass

        return list(set(affected_tests))

    async def _calculate_risk_level(
        self,
        file_path: str,
        changed_symbols: List[str],
        affected_files: List[str],
        affected_modules: List[str],
    ) -> tuple[RiskLevel, float]:
        """Calculate risk level and impact score"""
        risk_factors = []

        # Factor 1: Is this a critical module?
        module = self._get_module_from_path(file_path)
        if module in self.critical_modules:
            risk_factors.append(("critical_module", 0.4))

        # Factor 2: Number of affected files
        if len(affected_files) > 20:
            risk_factors.append(("high_file_impact", 0.3))
        elif len(affected_files) > 10:
            risk_factors.append(("medium_file_impact", 0.2))

        # Factor 3: Number of affected modules
        if len(affected_modules) > 5:
            risk_factors.append(("high_module_impact", 0.2))
        elif len(affected_modules) > 2:
            risk_factors.append(("medium_module_impact", 0.1))

        # Factor 4: Type of symbols changed
        for symbol in changed_symbols:
            if symbol.startswith("_"):  # Private symbols are lower risk
                risk_factors.append(("private_symbol", -0.1))
            else:  # Public symbols are higher risk
                risk_factors.append(("public_symbol", 0.1))

        # Factor 5: File type
        if file_path.endswith("__init__.py"):
            risk_factors.append(("init_file", 0.2))
        elif "base" in file_path.lower():
            risk_factors.append(("base_class", 0.3))

        # Calculate impact score
        impact_score = max(0.0, min(1.0, sum(factor[1] for factor in risk_factors)))

        # Determine risk level
        if impact_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif impact_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif impact_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return risk_level, impact_score

    async def _generate_recommendations(
        self,
        file_path: str,
        symbols: List[CodeSymbol],
        affected_files: List[str],
        risk_level: RiskLevel,
    ) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("🚨 CRITICAL: Run full test suite before merging")
            recommendations.append("🚨 CRITICAL: Require multiple code reviews")
            recommendations.append("🚨 CRITICAL: Consider feature flag for gradual rollout")

        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("⚠️ HIGH IMPACT: Run integration tests")
            recommendations.append("⚠️ HIGH IMPACT: Update documentation")
            recommendations.append("⚠️ HIGH IMPACT: Notify dependent teams")

        if len(affected_files) > 10:
            recommendations.append(
                f"📊 {len(affected_files)} files affected - consider breaking into smaller changes"
            )

        # Check for public API changes
        public_symbols = [s for s in symbols if not s.name.startswith("_")]
        if public_symbols:
            recommendations.append("🔄 Public API changes detected - check for breaking changes")

        # Module-specific recommendations
        module = self._get_module_from_path(file_path)
        if "agents" in module:
            recommendations.append("🤖 Agent changes - test agent interactions and workflows")
        elif "protocols" in module:
            recommendations.append("📡 Protocol changes - test message handling and compatibility")
        elif "database" in module:
            recommendations.append(
                "🗄️ Database changes - run migration tests and backup procedures"
            )

        return recommendations

    async def _detect_breaking_changes(
        self, symbols: List[CodeSymbol], references: List[CodeReference]
    ) -> List[Dict[str, Any]]:
        """Detect potential breaking changes"""
        breaking_changes = []

        for symbol in symbols:
            if symbol.type == "function" and symbol.signature:
                # Check if function signature might have changed
                # This would require more sophisticated analysis
                symbol_refs = [ref for ref in references if ref.symbol == symbol.name]

                if symbol_refs:
                    breaking_changes.append(
                        {
                            "type": "function_signature_change",
                            "symbol": symbol.name,
                            "file": symbol.file_path,
                            "line": symbol.line,
                            "impact": f"{len(symbol_refs)} references may be affected",
                            "severity": "high" if len(symbol_refs) > 5 else "medium",
                        }
                    )

        return breaking_changes

    def _get_module_from_path(self, file_path: str) -> str:
        """Extract module name from file path"""
        path = Path(file_path)
        if "src/cryptotrading" in str(path):
            parts = path.parts
            try:
                crypto_index = parts.index("cryptotrading")
                module_parts = parts[crypto_index:]
                if module_parts[-1].endswith(".py"):
                    module_parts = module_parts[:-1] + (module_parts[-1][:-3],)
                return ".".join(module_parts)
            except (ValueError, IndexError):
                return "unknown"
        return "external"

    def _get_symbol_at_location(self, file_path: str, line: int) -> str:
        """Get symbol name at a specific location (simplified)"""
        # This would need more sophisticated parsing
        return f"symbol_at_{Path(file_path).stem}_{line}"

    def _determine_component_type(self, file_path: str) -> str:
        """Determine the type of component based on file path"""
        if "test" in file_path.lower():
            return "test"
        elif "agents" in file_path:
            return "agent"
        elif "protocols" in file_path:
            return "protocol"
        elif "database" in file_path:
            return "database"
        else:
            return "module"

    def _assess_reference_risk(self, ref: CodeReference, change_type: str) -> RiskLevel:
        """Assess risk level for a specific reference"""
        if ref.reference_type == "call" and change_type == "deletion":
            return RiskLevel.CRITICAL
        elif ref.reference_type == "import" and change_type == "modification":
            return RiskLevel.HIGH
        elif ref.reference_type == "assignment":
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_overall_risk(
        self, impacts: List[ImpactedComponent], change_type: str
    ) -> RiskLevel:
        """Calculate overall risk from individual impacts"""
        if not impacts:
            return RiskLevel.LOW

        risk_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        max_risk = max(risk_scores[impact.risk_level.value] for impact in impacts)

        # Adjust based on change type
        if change_type == "deletion":
            max_risk += 1

        if max_risk >= 4:
            return RiskLevel.CRITICAL
        elif max_risk >= 3:
            return RiskLevel.HIGH
        elif max_risk >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _parse_changed_lines(self, pr_diff: str, file_path: str) -> List[int]:
        """Parse changed lines from PR diff"""
        # Simplified diff parsing - would need more robust implementation
        changed_lines = []
        lines = pr_diff.split("\n")

        current_file = None
        for line in lines:
            if line.startswith("+++") and file_path in line:
                current_file = file_path
            elif current_file == file_path and line.startswith("+") and not line.startswith("+++"):
                # This is a simplified approach - real diff parsing is more complex
                pass

        return changed_lines

    def _generate_pr_recommendations(
        self, reports: List[ChangeImpactReport], overall_risk: RiskLevel
    ) -> List[str]:
        """Generate PR-level recommendations"""
        recommendations = []

        if overall_risk == RiskLevel.CRITICAL:
            recommendations.extend(
                [
                    "🚨 CRITICAL PR: Requires architecture review",
                    "🚨 CRITICAL PR: Deploy to staging environment first",
                    "🚨 CRITICAL PR: Create rollback plan",
                ]
            )

        total_breaking_changes = sum(len(report.breaking_changes) for report in reports)
        if total_breaking_changes > 0:
            recommendations.append(
                f"💥 {total_breaking_changes} potential breaking changes detected"
            )

        critical_modules_affected = [
            report.file_path
            for report in reports
            if self._get_module_from_path(report.file_path) in self.critical_modules
        ]

        if critical_modules_affected:
            recommendations.append(
                f"⚠️ Critical modules affected: {', '.join(critical_modules_affected)}"
            )

        return recommendations

    def _generate_test_strategy(
        self, reports: List[ChangeImpactReport], overall_risk: RiskLevel
    ) -> Dict[str, Any]:
        """Generate testing strategy based on impact analysis"""
        all_affected_tests = []
        for report in reports:
            all_affected_tests.extend(report.affected_tests)

        strategy = {
            "affected_tests": list(set(all_affected_tests)),
            "run_full_suite": overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            "integration_tests_required": overall_risk != RiskLevel.LOW,
            "manual_testing_required": overall_risk == RiskLevel.CRITICAL,
        }

        if overall_risk == RiskLevel.CRITICAL:
            strategy["additional_requirements"] = [
                "Load testing",
                "Security testing",
                "Performance regression testing",
            ]

        return strategy
