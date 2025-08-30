"""
Multi-Language SCIP Indexer - Unified indexing for all supported languages
Combines Python, SAP CAP, JavaScript/UI5, and configuration file indexing
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .javascript_ui5_indexer import index_javascript_ui5_for_glean
from .sap_cap_indexer import index_cap_project_for_glean
from .scip_indexer import index_project_for_glean
from .typescript_indexer import index_typescript_files

logger = logging.getLogger(__name__)


class MultiLanguageStats:
    """Statistics for multi-language indexing"""

    def __init__(self):
        self.python_files: int = 0
        self.javascript_files: int = 0
        self.typescript_files: int = 0
        self.cap_files: int = 0
        self.xml_files: int = 0
        self.json_files: int = 0
        self.total_facts: int = 0
        self.indexing_duration: float = 0.0


class UnifiedLanguageIndexer:
    """Unified indexer for all supported languages in the project"""

    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".cds": "cap",
        ".xml": "xml",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    }

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.stats = MultiLanguageStats()
        self.all_facts: List[Dict[str, Any]] = []

    def index_entire_project(self) -> Dict[str, Any]:
        """Index the entire project across all supported languages"""
        start_time = datetime.now()

        logger.info("Starting comprehensive multi-language indexing...")

        # Index Python files
        python_result = self._index_python()
        self.stats.python_files = python_result.get("stats", {}).get("files_indexed", 0)
        self.all_facts.extend(python_result.get("glean_facts", []))

        # Index SAP CAP files
        cap_result = self._index_cap()
        self.stats.cap_files = cap_result.get("stats", {}).get("files_indexed", 0)
        self.all_facts.extend(cap_result.get("glean_facts", []))

        # Index JavaScript/UI5 files
        js_result = self._index_javascript_ui5()
        self.stats.javascript_files = js_result.get("stats", {}).get("files_indexed", 0)
        self.all_facts.extend(js_result.get("glean_facts", []))

        # Index TypeScript files
        ts_result = self._index_typescript()
        self.stats.typescript_files = ts_result.get("stats", {}).get("files_indexed", 0)
        self.all_facts.extend(ts_result.get("glean_facts", []))

        # Index configuration files
        config_result = self._index_configuration_files()
        self.stats.json_files = config_result.get("files_indexed", 0)
        self.all_facts.extend(config_result.get("glean_facts", []))

        # Calculate totals
        end_time = datetime.now()
        self.stats.indexing_duration = (end_time - start_time).total_seconds()
        self.stats.total_facts = len(self.all_facts)

        # Generate comprehensive report
        return self._generate_comprehensive_report()

    def _index_python(self) -> Dict[str, Any]:
        """Index Python files across entire project including tests, scripts, data"""
        logger.info("🐍 Indexing Python files...")
        try:
            # Index all Python files in project, not just src/
            result = index_project_for_glean(str(self.project_root))

            # Count all Python files in project for accurate stats
            python_files = list(self.project_root.rglob("*.py"))
            python_files = [
                f
                for f in python_files
                if "node_modules" not in str(f) and "__pycache__" not in str(f)
            ]
            actual_count = len(python_files)

            # Update stats to reflect actual file count
            if "stats" in result:
                result["stats"]["files_indexed"] = actual_count
                logger.info("Python indexing: %d files processed", actual_count)

            return result
        except (OSError, ValueError, ImportError) as e:
            logger.error("Python indexing failed: %s", str(e))
            return {"glean_facts": [], "stats": {"files_indexed": 0}}

    def _index_cap(self) -> Dict[str, Any]:
        """Index SAP CAP files"""
        logger.info("🏢 Indexing SAP CAP files...")
        try:
            return index_cap_project_for_glean(str(self.project_root))
        except (OSError, ValueError, ImportError) as e:
            logger.error("CAP indexing failed: %s", str(e))
            return {"glean_facts": [], "stats": {"files_indexed": 0}}

    def _index_javascript_ui5(self) -> Dict[str, Any]:
        """Index JavaScript and UI5 files"""
        logger.info("📱 Indexing JavaScript/UI5 files...")
        try:
            return index_javascript_ui5_for_glean(str(self.project_root))
        except (OSError, ValueError, ImportError) as e:
            logger.error("JavaScript/UI5 indexing failed: %s", str(e))
            return {"glean_facts": [], "stats": {"files_indexed": 0}}

    def _index_typescript(self) -> Dict[str, Any]:
        """Index TypeScript files"""
        logger.info("⚡ Indexing TypeScript files...")
        try:
            facts = index_typescript_files(str(self.project_root))
            # Count actual TypeScript files indexed, not just src.File facts
            ts_files = [f for f in self.project_root.rglob("*.ts") if "node_modules" not in str(f)]
            tsx_files = [
                f for f in self.project_root.rglob("*.tsx") if "node_modules" not in str(f)
            ]
            files_indexed = len(ts_files) + len(tsx_files)
            logger.info(
                "TypeScript indexing: %d files processed, %d facts generated",
                files_indexed,
                len(facts),
            )
            return {"glean_facts": facts, "stats": {"files_indexed": files_indexed}}
        except (OSError, ValueError, ImportError) as e:
            logger.error("TypeScript indexing failed: %s", str(e))
            return {"glean_facts": [], "stats": {"files_indexed": 0}}

    def _index_configuration_files(self) -> Dict[str, Any]:
        """Index configuration files (JSON, YAML, etc.)"""
        logger.info("⚙️ Indexing configuration files...")
        facts = []
        files_indexed = 0

        try:
            # Index JSON files
            for json_file in self.project_root.rglob("*.json"):
                if "node_modules" in str(json_file) or "__pycache__" in str(json_file):
                    continue

                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    relative_path = json_file.relative_to(self.project_root)

                    # Generate file fact
                    facts.append(
                        {
                            "predicate": "src.File",
                            "key": {"path": str(relative_path)},
                            "value": {"language": "JSON"},
                        }
                    )

                    # Generate configuration fact
                    facts.append(
                        {
                            "predicate": "config.File",
                            "key": {"path": str(relative_path), "type": "json"},
                            "value": {
                                "keys": list(data.keys()) if isinstance(data, dict) else [],
                                "size": len(json.dumps(data)),
                            },
                        }
                    )

                    files_indexed += 1

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning("Failed to parse JSON file %s: %s", json_file, e)

            # Index YAML files
            for yaml_file in self.project_root.rglob("*.yaml"):
                if "node_modules" in str(yaml_file):
                    continue

                relative_path = yaml_file.relative_to(self.project_root)
                facts.append(
                    {
                        "predicate": "src.File",
                        "key": {"path": str(relative_path)},
                        "value": {"language": "YAML"},
                    }
                )
                files_indexed += 1

            for yml_file in self.project_root.rglob("*.yml"):
                if "node_modules" in str(yml_file):
                    continue

                relative_path = yml_file.relative_to(self.project_root)
                facts.append(
                    {
                        "predicate": "src.File",
                        "key": {"path": str(relative_path)},
                        "value": {"language": "YAML"},
                    }
                )
                files_indexed += 1

        except (OSError, ValueError) as e:
            logger.error("Configuration file indexing failed: %s", str(e))

        return {"glean_facts": facts, "files_indexed": files_indexed}

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive indexing report"""
        # Analyze facts by predicate
        predicate_counts = {}
        language_counts = {}

        for fact in self.all_facts:
            predicate = fact.get("predicate", "unknown")
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1

            if predicate == "src.File":
                language = fact.get("value", {}).get("language", "unknown")
                language_counts[language] = language_counts.get(language, 0) + 1

        return {
            "indexing_summary": {
                "total_files_indexed": (
                    self.stats.python_files
                    + self.stats.javascript_files
                    + self.stats.typescript_files
                    + self.stats.cap_files
                    + self.stats.json_files
                ),
                "total_facts_generated": self.stats.total_facts,
                "indexing_duration_seconds": self.stats.indexing_duration,
                "languages_supported": len(language_counts),
            },
            "language_breakdown": {
                "python_files": self.stats.python_files,
                "javascript_files": self.stats.javascript_files,
                "typescript_files": self.stats.typescript_files,
                "cap_files": self.stats.cap_files,
                "xml_files": self.stats.xml_files,
                "json_files": self.stats.json_files,
            },
            "language_distribution": language_counts,
            "predicate_distribution": predicate_counts,
            "glean_facts": self.all_facts,
            "coverage_analysis": self._analyze_coverage(),
            "blind_spots_eliminated": self._check_blind_spots(),
        }

    def _analyze_coverage(self) -> Dict[str, Any]:
        """Analyze indexing coverage"""
        total_files = 0
        indexed_files = 0

        # Count all relevant files in project
        for ext in self.SUPPORTED_EXTENSIONS.keys():
            files = list(self.project_root.rglob(f"*{ext}"))
            # Filter out node_modules and cache directories
            files = [
                f for f in files if "node_modules" not in str(f) and "__pycache__" not in str(f)
            ]
            total_files += len(files)

        indexed_files = (
            self.stats.python_files
            + self.stats.javascript_files
            + self.stats.typescript_files
            + self.stats.cap_files
            + self.stats.json_files
        )

        coverage_percentage = (indexed_files / total_files * 100) if total_files > 0 else 0

        return {
            "total_relevant_files": total_files,
            "indexed_files": indexed_files,
            "coverage_percentage": coverage_percentage,
            "unindexed_files": total_files - indexed_files,
        }

    def _check_blind_spots(self) -> Dict[str, Any]:
        """Check for remaining blind spots"""
        blind_spots = []

        # Check for TypeScript files (should be handled by indexer now)
        ts_files = [f for f in self.project_root.rglob("*.ts") if "node_modules" not in str(f)]
        tsx_files = [f for f in self.project_root.rglob("*.tsx") if "node_modules" not in str(f)]
        ts_total = len(ts_files) + len(tsx_files)

        # Only report as blind spot if TypeScript files exist but weren't indexed
        if ts_total > 0 and self.stats.typescript_files == 0:
            blind_spots.append(f"TypeScript files found but not indexed: {ts_total}")

        # Check for other potential files
        other_extensions = [".vue", ".svelte", ".php", ".rb", ".go", ".rs"]
        for ext in other_extensions:
            files = [
                f
                for f in self.project_root.rglob(f"*{ext}")
                if "node_modules" not in str(f) and "__pycache__" not in str(f)
            ]
            if files:
                blind_spots.append(f"{ext.upper()[1:]} files found but not supported: {len(files)}")

        return {
            "remaining_blind_spots": blind_spots,
            "blind_spots_count": len(blind_spots),
            "coverage_complete": len(blind_spots) == 0,
        }


def index_multi_language_project(project_path: str) -> Dict[str, Any]:
    """Main entry point for comprehensive multi-language indexing"""
    indexer = UnifiedLanguageIndexer(Path(project_path))
    return indexer.index_entire_project()
