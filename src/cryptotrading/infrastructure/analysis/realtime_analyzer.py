"""
Realtime Code Analyzer - Production implementation for continuous code monitoring
Watches for file changes and triggers real-time analysis using Glean integration
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    Observer = None
    FileSystemEventHandler = None

from .architecture_validator import ArchitectureValidator
from .code_analyzer import CodeAnalyzer
from .glean_client import GleanClient
from .impact_analyzer import ImpactAnalyzer

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    IMPACT_ANALYSIS = "impact_analysis"
    ARCHITECTURE_VALIDATION = "architecture_validation"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    DEAD_CODE_DETECTION = "dead_code_detection"


@dataclass
class AnalysisResult:
    """Result of real-time analysis"""

    analysis_type: AnalysisType
    file_path: str
    timestamp: float
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class FileChangeEvent:
    """Represents a file change event"""

    file_path: str
    event_type: str  # created, modified, deleted, moved
    timestamp: float
    file_hash: Optional[str] = None


class CodeChangeHandler(FileSystemEventHandler):
    """Handles file system events for code changes"""

    def __init__(self, analyzer: "RealtimeCodeAnalyzer"):
        self.analyzer = analyzer
        self.debounce_delay = 1.0  # seconds
        self.pending_changes: Dict[str, float] = {}

    def on_modified(self, event):
        if not event.is_directory and self._is_python_file(event.src_path):
            self._schedule_analysis(event.src_path, "modified")

    def on_created(self, event):
        if not event.is_directory and self._is_python_file(event.src_path):
            self._schedule_analysis(event.src_path, "created")

    def on_deleted(self, event):
        if not event.is_directory and self._is_python_file(event.src_path):
            self._schedule_analysis(event.src_path, "deleted")

    def on_moved(self, event):
        if not event.is_directory:
            if self._is_python_file(event.src_path):
                self._schedule_analysis(event.src_path, "moved_from")
            if self._is_python_file(event.dest_path):
                self._schedule_analysis(event.dest_path, "moved_to")

    def _is_python_file(self, file_path: str) -> bool:
        """Check if file is a Python file"""
        return file_path.endswith(".py")

    def _schedule_analysis(self, file_path: str, event_type: str):
        """Schedule analysis with debouncing"""
        current_time = time.time()
        self.pending_changes[file_path] = current_time

        # Schedule debounced analysis
        asyncio.create_task(self._debounced_analysis(file_path, event_type, current_time))

    async def _debounced_analysis(self, file_path: str, event_type: str, schedule_time: float):
        """Perform debounced analysis"""
        await asyncio.sleep(self.debounce_delay)

        # Check if this is still the latest change for this file
        if self.pending_changes.get(file_path) == schedule_time:
            # Remove from pending
            self.pending_changes.pop(file_path, None)

            # Trigger analysis
            await self.analyzer.analyze_file_change(
                FileChangeEvent(
                    file_path=file_path,
                    event_type=event_type,
                    timestamp=schedule_time,
                    file_hash=self._calculate_file_hash(file_path)
                    if Path(file_path).exists()
                    else None,
                )
            )

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


class RealtimeCodeAnalyzer:
    """Real-time code analysis with file watching and continuous monitoring"""

    def __init__(self, project_root: str, glean_client: GleanClient):
        self.project_root = Path(project_root)
        self.glean = glean_client

        # Initialize analyzers
        self.code_analyzer = CodeAnalyzer(glean_client)
        self.impact_analyzer = ImpactAnalyzer(glean_client)
        self.architecture_validator = ArchitectureValidator(glean_client)

        # File watching
        self.observer = None
        self.event_handler = None
        self.is_watching = False

        # Analysis configuration
        self.enabled_analyses = {
            AnalysisType.DEPENDENCY_ANALYSIS,
            AnalysisType.IMPACT_ANALYSIS,
            AnalysisType.COMPLEXITY_ANALYSIS,
        }

        # Results storage
        self.recent_results: List[AnalysisResult] = []
        self.max_recent_results = 100

        # Callbacks for real-time notifications
        self.analysis_callbacks: List[Callable[[AnalysisResult], None]] = []

        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_duration_ms": 0.0,
        }

    async def start_watching(self) -> bool:
        """Start watching for file changes"""
        if Observer is None:
            logger.error("watchdog not available - install with: pip install watchdog")
            return False

        if self.is_watching:
            logger.warning("Already watching for file changes")
            return True

        try:
            # Ensure Glean server is running
            if not await self.glean.ensure_server_running():
                logger.error("Failed to start Glean server")
                return False

            # Initial indexing
            logger.info("Performing initial code indexing...")
            await self.glean.index_codebase()

            # Set up file watching
            self.event_handler = CodeChangeHandler(self)
            self.observer = Observer()

            # Watch the src directory
            src_path = self.project_root / "src"
            if src_path.exists():
                self.observer.schedule(self.event_handler, str(src_path), recursive=True)
                logger.info(f"Watching for changes in: {src_path}")

            # Start observer
            self.observer.start()
            self.is_watching = True

            logger.info("Real-time code analysis started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
            return False

    def stop_watching(self):
        """Stop watching for file changes"""
        if self.observer and self.is_watching:
            self.observer.stop()
            self.observer.join()
            self.is_watching = False
            logger.info("Stopped watching for file changes")

    async def analyze_file_change(self, change_event: FileChangeEvent):
        """Analyze a specific file change"""
        logger.info(f"Analyzing file change: {change_event.file_path} ({change_event.event_type})")

        # Skip analysis for deleted files (except for impact analysis)
        if change_event.event_type == "deleted":
            await self._analyze_file_deletion(change_event)
            return

        # Re-index the changed file
        try:
            await self.glean.index_file(change_event.file_path)
        except Exception as e:
            logger.warning(f"Failed to re-index {change_event.file_path}: {e}")

        # Run enabled analyses
        for analysis_type in self.enabled_analyses:
            await self._run_analysis(analysis_type, change_event.file_path)

    async def _analyze_file_deletion(self, change_event: FileChangeEvent):
        """Handle file deletion analysis"""
        # For deleted files, we mainly care about impact analysis
        if AnalysisType.IMPACT_ANALYSIS in self.enabled_analyses:
            result = await self._run_analysis(AnalysisType.IMPACT_ANALYSIS, change_event.file_path)

            # Add deletion context to result
            if result and result.success:
                result.data["deletion_event"] = True
                result.data["deleted_file"] = change_event.file_path

    async def _run_analysis(
        self, analysis_type: AnalysisType, file_path: str
    ) -> Optional[AnalysisResult]:
        """Run a specific type of analysis"""
        start_time = time.time()

        try:
            data = {}

            if analysis_type == AnalysisType.DEPENDENCY_ANALYSIS:
                # Get module name from file path
                module_name = self._file_path_to_module(file_path)
                if module_name:
                    deps = await self.code_analyzer.analyze_dependencies(module_name)
                    data = {"dependencies": deps}

            elif analysis_type == AnalysisType.IMPACT_ANALYSIS:
                changes = [{"file": file_path, "type": "modified"}]
                impact = await self.impact_analyzer.analyze_change_impact(changes)
                data = {"impact": impact}

            elif analysis_type == AnalysisType.COMPLEXITY_ANALYSIS:
                complexity = await self.code_analyzer.analyze_complexity([file_path])
                data = {"complexity": complexity}

            elif analysis_type == AnalysisType.DEAD_CODE_DETECTION:
                dead_code = await self.code_analyzer.detect_dead_code([file_path])
                data = {"dead_code": dead_code}

            elif analysis_type == AnalysisType.ARCHITECTURE_VALIDATION:
                # Run full architecture validation (expensive)
                violations = await self.architecture_validator.validate_architecture()
                # Filter to violations related to this file
                file_violations = [
                    v
                    for v in violations
                    if v.file_path and Path(v.file_path).samefile(Path(file_path))
                ]
                data = {
                    "violations": [
                        {
                            "type": v.violation_type.value,
                            "severity": v.severity,
                            "description": v.description,
                            "recommendation": v.recommendation,
                        }
                        for v in file_violations
                    ]
                }

            duration_ms = (time.time() - start_time) * 1000

            result = AnalysisResult(
                analysis_type=analysis_type,
                file_path=file_path,
                timestamp=time.time(),
                success=True,
                data=data,
                duration_ms=duration_ms,
            )

            # Store result and update stats
            self._store_result(result)
            self._update_stats(result)

            # Notify callbacks
            for callback in self.analysis_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.warning(f"Analysis callback failed: {e}")

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            result = AnalysisResult(
                analysis_type=analysis_type,
                file_path=file_path,
                timestamp=time.time(),
                success=False,
                data={},
                error=str(e),
                duration_ms=duration_ms,
            )

            self._store_result(result)
            self._update_stats(result)

            logger.error(f"Analysis failed for {file_path} ({analysis_type.value}): {e}")
            return result

    def _file_path_to_module(self, file_path: str) -> Optional[str]:
        """Convert file path to module name"""
        try:
            path = Path(file_path)

            # Find the src directory
            src_index = -1
            for i, part in enumerate(path.parts):
                if part == "src":
                    src_index = i
                    break

            if src_index == -1:
                return None

            # Get parts after src
            module_parts = path.parts[src_index + 1 :]

            # Remove .py extension from last part
            if module_parts and module_parts[-1].endswith(".py"):
                module_parts = module_parts[:-1] + (module_parts[-1][:-3],)

            # Skip __init__ modules
            if module_parts and module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]

            return ".".join(module_parts) if module_parts else None

        except Exception:
            return None

    def _store_result(self, result: AnalysisResult):
        """Store analysis result"""
        self.recent_results.append(result)

        # Keep only recent results
        if len(self.recent_results) > self.max_recent_results:
            self.recent_results = self.recent_results[-self.max_recent_results :]

    def _update_stats(self, result: AnalysisResult):
        """Update analysis statistics"""
        self.analysis_stats["total_analyses"] += 1

        if result.success:
            self.analysis_stats["successful_analyses"] += 1
        else:
            self.analysis_stats["failed_analyses"] += 1

        # Update average duration
        total = self.analysis_stats["total_analyses"]
        current_avg = self.analysis_stats["average_duration_ms"]
        new_avg = ((current_avg * (total - 1)) + result.duration_ms) / total
        self.analysis_stats["average_duration_ms"] = new_avg

    def add_analysis_callback(self, callback: Callable[[AnalysisResult], None]):
        """Add callback for analysis results"""
        self.analysis_callbacks.append(callback)

    def remove_analysis_callback(self, callback: Callable[[AnalysisResult], None]):
        """Remove analysis callback"""
        if callback in self.analysis_callbacks:
            self.analysis_callbacks.remove(callback)

    def configure_analyses(self, enabled_analyses: Set[AnalysisType]):
        """Configure which analyses to run"""
        self.enabled_analyses = enabled_analyses
        logger.info(f"Configured analyses: {[a.value for a in enabled_analyses]}")

    def get_recent_results(self, limit: Optional[int] = None) -> List[AnalysisResult]:
        """Get recent analysis results"""
        results = self.recent_results
        if limit:
            results = results[-limit:]
        return results

    def get_results_for_file(
        self, file_path: str, limit: Optional[int] = None
    ) -> List[AnalysisResult]:
        """Get analysis results for a specific file"""
        file_results = [r for r in self.recent_results if r.file_path == file_path]
        if limit:
            file_results = file_results[-limit:]
        return file_results

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return {
            **self.analysis_stats,
            "is_watching": self.is_watching,
            "enabled_analyses": [a.value for a in self.enabled_analyses],
            "recent_results_count": len(self.recent_results),
        }

    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run full analysis on entire codebase"""
        logger.info("Running full codebase analysis...")

        start_time = time.time()
        results = {}

        try:
            # Re-index entire codebase
            await self.glean.index_codebase()

            # Run dependency analysis
            if AnalysisType.DEPENDENCY_ANALYSIS in self.enabled_analyses:
                logger.info("Running dependency analysis...")
                deps = await self.code_analyzer.build_dependency_graph()
                results["dependencies"] = deps

            # Run impact analysis (for recent changes)
            if AnalysisType.IMPACT_ANALYSIS in self.enabled_analyses:
                logger.info("Running impact analysis...")
                # Get recent file changes from results
                recent_changes = []
                for result in self.recent_results[-10:]:  # Last 10 changes
                    if result.success:
                        recent_changes.append({"file": result.file_path, "type": "modified"})

                if recent_changes:
                    impact = await self.impact_analyzer.analyze_change_impact(recent_changes)
                    results["impact"] = impact

            # Run architecture validation
            if AnalysisType.ARCHITECTURE_VALIDATION in self.enabled_analyses:
                logger.info("Running architecture validation...")
                violations = await self.architecture_validator.validate_architecture()
                results["architecture"] = self.architecture_validator.generate_violation_report()

            # Run complexity analysis
            if AnalysisType.COMPLEXITY_ANALYSIS in self.enabled_analyses:
                logger.info("Running complexity analysis...")
                src_files = list(Path(self.project_root / "src").rglob("*.py"))
                file_paths = [str(f) for f in src_files if f.name != "__init__.py"]
                complexity = await self.code_analyzer.analyze_complexity(
                    file_paths[:20]
                )  # Limit for performance
                results["complexity"] = complexity

            # Run dead code detection
            if AnalysisType.DEAD_CODE_DETECTION in self.enabled_analyses:
                logger.info("Running dead code detection...")
                src_files = list(Path(self.project_root / "src").rglob("*.py"))
                file_paths = [str(f) for f in src_files if f.name != "__init__.py"]
                dead_code = await self.code_analyzer.detect_dead_code(
                    file_paths[:20]
                )  # Limit for performance
                results["dead_code"] = dead_code

            duration = time.time() - start_time

            results["analysis_summary"] = {
                "duration_seconds": duration,
                "timestamp": time.time(),
                "analyses_run": [a.value for a in self.enabled_analyses],
                "success": True,
            }

            logger.info(f"Full analysis completed in {duration:.2f} seconds")
            return results

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Full analysis failed after {duration:.2f} seconds: {e}")

            return {
                "analysis_summary": {
                    "duration_seconds": duration,
                    "timestamp": time.time(),
                    "analyses_run": [a.value for a in self.enabled_analyses],
                    "success": False,
                    "error": str(e),
                }
            }

    async def cleanup(self):
        """Cleanup resources"""
        self.stop_watching()
        await self.glean.cleanup()
        logger.info("Real-time analyzer cleanup completed")
