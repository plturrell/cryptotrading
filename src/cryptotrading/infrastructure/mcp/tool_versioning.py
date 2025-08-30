"""
MCP Tool Versioning System
Manages versioning, compatibility, and migration for MCP tools
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import semver


class VersionStatus(Enum):
    """Tool version status"""

    ALPHA = "alpha"
    BETA = "beta"
    RELEASE_CANDIDATE = "rc"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    END_OF_LIFE = "eol"


class CompatibilityLevel(Enum):
    """Backward compatibility level"""

    FULL = "full"  # Fully backward compatible
    PARTIAL = "partial"  # Some breaking changes
    BREAKING = "breaking"  # Major breaking changes
    NONE = "none"  # Not compatible


@dataclass
class VersionChange:
    """Represents a change in a version"""

    type: str  # "feature", "fix", "breaking", "deprecation"
    description: str
    affected_methods: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolVersion:
    """Complete version information for a tool"""

    version: str  # Semantic version string
    status: VersionStatus
    release_date: datetime
    changes: List[VersionChange]
    compatibility: Dict[str, CompatibilityLevel]  # Version -> Compatibility
    dependencies: Dict[str, str]  # Dependency -> Version requirement
    checksum: Optional[str] = None

    def __post_init__(self):
        """Validate and parse version"""
        self.semver = semver.VersionInfo.parse(self.version)

    def is_compatible_with(self, other_version: str) -> CompatibilityLevel:
        """Check compatibility with another version"""
        return self.compatibility.get(other_version, CompatibilityLevel.NONE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "major": self.semver.major,
            "minor": self.semver.minor,
            "patch": self.semver.patch,
            "prerelease": self.semver.prerelease,
            "build": self.semver.build,
            "status": self.status.value,
            "release_date": self.release_date.isoformat(),
            "changes": [
                {
                    "type": c.type,
                    "description": c.description,
                    "affected_methods": c.affected_methods,
                    "migration_guide": c.migration_guide,
                }
                for c in self.changes
            ],
            "compatibility": {k: v.value for k, v in self.compatibility.items()},
            "dependencies": self.dependencies,
            "checksum": self.checksum,
        }


class VersionedMCPTool:
    """Base class for versioned MCP tools"""

    def __init__(self, tool_name: str, current_version: str = "1.0.0"):
        self.tool_name = tool_name
        self.current_version = current_version
        self.version_history: List[ToolVersion] = []
        self.migration_handlers: Dict[str, Callable] = {}
        self.deprecated_methods: Dict[str, str] = {}  # Method -> Replacement

        # Initialize with base version
        self._add_base_version()

    def _add_base_version(self):
        """Add the base version"""
        base_version = ToolVersion(
            version="1.0.0",
            status=VersionStatus.STABLE,
            release_date=datetime(2024, 1, 1),
            changes=[
                VersionChange(
                    type="feature", description="Initial release", affected_methods=["execute"]
                )
            ],
            compatibility={},
            dependencies={},
        )
        self.version_history.append(base_version)

    def add_version(self, version: ToolVersion):
        """Add a new version to the history"""
        # Validate version is newer
        if self.version_history:
            latest = self.version_history[-1]
            if semver.compare(version.version, latest.version) <= 0:
                raise ValueError(f"Version {version.version} must be newer than {latest.version}")

        # Calculate checksum for version
        version.checksum = self._calculate_checksum(version)

        # Add to history
        self.version_history.append(version)

        # Update current version if this is the latest stable
        if version.status == VersionStatus.STABLE:
            self.current_version = version.version

    def _calculate_checksum(self, version: ToolVersion) -> str:
        """Calculate checksum for version integrity"""
        version_str = json.dumps(version.to_dict(), sort_keys=True)
        return hashlib.sha256(version_str.encode()).hexdigest()

    def get_version(self, version_str: str) -> Optional[ToolVersion]:
        """Get a specific version"""
        for v in self.version_history:
            if v.version == version_str:
                return v
        return None

    def get_latest_stable(self) -> Optional[ToolVersion]:
        """Get the latest stable version"""
        stable_versions = [v for v in self.version_history if v.status == VersionStatus.STABLE]
        if stable_versions:
            return max(stable_versions, key=lambda v: v.semver)
        return None

    def check_compatibility(self, from_version: str, to_version: str) -> CompatibilityLevel:
        """Check compatibility between two versions"""
        to_ver = self.get_version(to_version)
        if to_ver:
            return to_ver.is_compatible_with(from_version)
        return CompatibilityLevel.NONE

    def get_migration_path(self, from_version: str, to_version: str) -> List[ToolVersion]:
        """Get the migration path between versions"""
        path = []

        from_idx = -1
        to_idx = -1

        for i, v in enumerate(self.version_history):
            if v.version == from_version:
                from_idx = i
            if v.version == to_version:
                to_idx = i

        if from_idx >= 0 and to_idx >= 0 and from_idx < to_idx:
            path = self.version_history[from_idx + 1 : to_idx + 1]

        return path

    def register_migration_handler(self, from_version: str, to_version: str, handler: Callable):
        """Register a migration handler"""
        key = f"{from_version}→{to_version}"
        self.migration_handlers[key] = handler

    async def migrate_data(self, data: Any, from_version: str, to_version: str) -> Any:
        """Migrate data from one version to another"""
        migration_path = self.get_migration_path(from_version, to_version)

        if not migration_path:
            return data

        current_data = data
        current_version = from_version

        for next_version in migration_path:
            key = f"{current_version}→{next_version.version}"

            if key in self.migration_handlers:
                handler = self.migration_handlers[key]
                current_data = await handler(current_data)

            current_version = next_version.version

        return current_data

    def deprecate_method(self, method_name: str, replacement: str, removal_version: str):
        """Mark a method as deprecated"""
        self.deprecated_methods[method_name] = {
            "replacement": replacement,
            "removal_version": removal_version,
            "deprecated_since": self.current_version,
        }

    def is_method_deprecated(self, method_name: str) -> bool:
        """Check if a method is deprecated"""
        return method_name in self.deprecated_methods

    def get_version_info(self) -> Dict[str, Any]:
        """Get complete version information"""
        return {
            "tool_name": self.tool_name,
            "current_version": self.current_version,
            "latest_stable": self.get_latest_stable().version if self.get_latest_stable() else None,
            "total_versions": len(self.version_history),
            "deprecated_methods": self.deprecated_methods,
            "version_history": [v.to_dict() for v in self.version_history],
        }


class ToolVersionRegistry:
    """Central registry for all tool versions"""

    def __init__(self):
        self.tools: Dict[str, VersionedMCPTool] = {}
        self.global_dependencies: Dict[str, str] = {}

    def register_tool(self, tool: VersionedMCPTool):
        """Register a versioned tool"""
        self.tools[tool.tool_name] = tool

    def get_tool_version(self, tool_name: str) -> Optional[str]:
        """Get current version of a tool"""
        if tool_name in self.tools:
            return self.tools[tool_name].current_version
        return None

    def check_dependencies(self, tool_name: str) -> Dict[str, bool]:
        """Check if tool dependencies are satisfied"""
        if tool_name not in self.tools:
            return {}

        tool = self.tools[tool_name]
        latest = tool.get_latest_stable()

        if not latest:
            return {}

        results = {}

        for dep_name, dep_requirement in latest.dependencies.items():
            actual_version = self.get_tool_version(dep_name)

            if actual_version:
                # Check if version satisfies requirement
                satisfied = semver.match(actual_version, dep_requirement)
                results[dep_name] = satisfied
            else:
                results[dep_name] = False

        return results

    def get_compatibility_matrix(self) -> Dict[str, Dict[str, CompatibilityLevel]]:
        """Get compatibility matrix for all tools"""
        matrix = {}

        for tool_name, tool in self.tools.items():
            tool_matrix = {}

            for other_name, other_tool in self.tools.items():
                if tool_name != other_name:
                    # Check compatibility
                    compat = tool.check_compatibility(
                        other_tool.current_version, tool.current_version
                    )
                    tool_matrix[other_name] = compat

            matrix[tool_name] = tool_matrix

        return matrix

    def get_upgrade_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for tool upgrades"""
        recommendations = []

        for tool_name, tool in self.tools.items():
            latest_stable = tool.get_latest_stable()

            if latest_stable and latest_stable.version != tool.current_version:
                current_ver = semver.VersionInfo.parse(tool.current_version)
                latest_ver = latest_stable.semver

                if latest_ver > current_ver:
                    recommendations.append(
                        {
                            "tool": tool_name,
                            "current_version": tool.current_version,
                            "recommended_version": latest_stable.version,
                            "type": self._get_upgrade_type(current_ver, latest_ver),
                            "changes": len(latest_stable.changes),
                        }
                    )

        return recommendations

    def _get_upgrade_type(self, current: semver.VersionInfo, latest: semver.VersionInfo) -> str:
        """Determine upgrade type"""
        if latest.major > current.major:
            return "major"
        elif latest.minor > current.minor:
            return "minor"
        elif latest.patch > current.patch:
            return "patch"
        else:
            return "prerelease"


# Example versioned tools with real version history
def create_versioned_mcp_tools() -> Dict[str, VersionedMCPTool]:
    """Create versioned MCP tools with version history"""
    tools = {}

    # Technical Analysis Tool
    ta_tool = VersionedMCPTool("TechnicalAnalysisTool", "2.1.0")
    ta_tool.add_version(
        ToolVersion(
            version="1.1.0",
            status=VersionStatus.STABLE,
            release_date=datetime(2024, 2, 1),
            changes=[
                VersionChange("feature", "Added RSI indicator", ["calculate_rsi"]),
                VersionChange("feature", "Added MACD indicator", ["calculate_macd"]),
            ],
            compatibility={"1.0.0": CompatibilityLevel.FULL},
            dependencies={},
        )
    )
    ta_tool.add_version(
        ToolVersion(
            version="2.0.0",
            status=VersionStatus.STABLE,
            release_date=datetime(2024, 3, 1),
            changes=[
                VersionChange("breaking", "Changed API structure", ["execute"]),
                VersionChange("feature", "Added pattern recognition", ["detect_patterns"]),
            ],
            compatibility={"1.1.0": CompatibilityLevel.BREAKING},
            dependencies={"MLModelsTool": ">=1.0.0"},
        )
    )
    ta_tool.add_version(
        ToolVersion(
            version="2.1.0",
            status=VersionStatus.STABLE,
            release_date=datetime(2024, 4, 1),
            changes=[
                VersionChange("feature", "Added volume analysis", ["analyze_volume"]),
                VersionChange("fix", "Fixed RSI calculation bug", ["calculate_rsi"]),
            ],
            compatibility={"2.0.0": CompatibilityLevel.FULL},
            dependencies={"MLModelsTool": ">=1.5.0"},
        )
    )
    tools["TechnicalAnalysisTool"] = ta_tool

    # ML Models Tool
    ml_tool = VersionedMCPTool("MLModelsTool", "1.5.3")
    ml_tool.add_version(
        ToolVersion(
            version="1.5.0",
            status=VersionStatus.STABLE,
            release_date=datetime(2024, 2, 15),
            changes=[
                VersionChange("feature", "Added ensemble methods", ["ensemble_predict"]),
                VersionChange("feature", "Added hyperparameter optimization", ["optimize"]),
            ],
            compatibility={"1.0.0": CompatibilityLevel.PARTIAL},
            dependencies={"FeatureEngineeringTool": ">=1.0.0"},
        )
    )
    ml_tool.add_version(
        ToolVersion(
            version="1.5.3",
            status=VersionStatus.STABLE,
            release_date=datetime(2024, 3, 15),
            changes=[
                VersionChange("fix", "Fixed memory leak in training", ["train"]),
                VersionChange("fix", "Improved prediction accuracy", ["predict"]),
            ],
            compatibility={"1.5.0": CompatibilityLevel.FULL},
            dependencies={"FeatureEngineeringTool": ">=2.0.0"},
        )
    )
    tools["MLModelsTool"] = ml_tool

    # Feature Engineering Tool
    fe_tool = VersionedMCPTool("FeatureEngineeringTool", "2.0.0")
    fe_tool.add_version(
        ToolVersion(
            version="2.0.0",
            status=VersionStatus.STABLE,
            release_date=datetime(2024, 3, 1),
            changes=[
                VersionChange("breaking", "New feature pipeline API", ["create_features"]),
                VersionChange("feature", "Added PCA support", ["apply_pca"]),
                VersionChange("feature", "Added interaction features", ["create_interactions"]),
            ],
            compatibility={"1.0.0": CompatibilityLevel.BREAKING},
            dependencies={},
        )
    )
    tools["FeatureEngineeringTool"] = fe_tool

    return tools


# Global registry
_version_registry = None


def get_version_registry() -> ToolVersionRegistry:
    """Get or create the global version registry"""
    global _version_registry

    if _version_registry is None:
        _version_registry = ToolVersionRegistry()

        # Register all versioned tools
        versioned_tools = create_versioned_mcp_tools()
        for tool in versioned_tools.values():
            _version_registry.register_tool(tool)

    return _version_registry
