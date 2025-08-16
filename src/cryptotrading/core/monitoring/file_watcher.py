"""
Real-time File Watching and Analysis System
Monitors code changes and triggers incremental analysis with AI insights
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# File watching dependencies
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object
    print("Warning: watchdog not available. Install with: pip install watchdog")

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of file changes"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChange:
    """Represents a file change event"""
    file_path: str
    change_type: ChangeType
    timestamp: datetime = field(default_factory=datetime.now)
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "file_size": self.file_size,
            "checksum": self.checksum
        }


@dataclass
class AnalysisResult:
    """Result of file analysis"""
    file_path: str
    analysis_type: str
    symbols_found: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    complexity_score: float = 0.0
    ai_insights: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "analysis_type": self.analysis_type,
            "symbols_found": self.symbols_found,
            "dependencies": self.dependencies,
            "complexity_score": self.complexity_score,
            "ai_insights": self.ai_insights,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time
        }


class CodeFileHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """Handles file system events for code files"""
    
    def __init__(self, file_watcher: 'RealtimeFileWatcher'):
        self.file_watcher = file_watcher
        self.supported_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp'}
        self.ignore_dirs = {'__pycache__', '.git', 'node_modules', '.venv', 'venv', '.env'}
        
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed"""
        path = Path(file_path)
        
        # Check extension
        if path.suffix not in self.supported_extensions:
            return False
        
        # Check ignore directories
        if any(ignore_dir in path.parts for ignore_dir in self.ignore_dirs):
            return False
        
        # Check file size (ignore very large files)
        try:
            if path.exists() and path.stat().st_size > 1024 * 1024:  # 1MB limit
                return False
        except:
            pass
        
        return True
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and self.should_process_file(event.src_path):
            change = FileChange(
                file_path=event.src_path,
                change_type=ChangeType.MODIFIED
            )
            self._schedule_async_task(change)
    
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory and self.should_process_file(event.src_path):
            change = FileChange(
                file_path=event.src_path,
                change_type=ChangeType.CREATED
            )
            self._schedule_async_task(change)
    
    def on_deleted(self, event):
        """Handle file deletion"""
        if not event.is_directory and self.should_process_file(event.src_path):
            change = FileChange(
                file_path=event.src_path,
                change_type=ChangeType.DELETED
            )
            self._schedule_async_task(change)
    
    def _schedule_async_task(self, change: FileChange):
        """Schedule async task in a thread-safe way"""
        try:
            # Try to get the running loop
            loop = asyncio.get_running_loop()
            # Schedule the coroutine on the event loop
            asyncio.run_coroutine_threadsafe(
                self.file_watcher.handle_file_change(change), 
                loop
            )
        except RuntimeError:
            # No running loop, store change for later processing
            if hasattr(self.file_watcher, 'pending_changes'):
                self.file_watcher.pending_changes.append(change)
            else:
                logger.warning(f"No event loop available to process file change: {change.file_path}")


class AnalysisQueue:
    """Queue for managing analysis tasks with prioritization"""
    
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = False
        self.stats = {
            "queued": 0,
            "processed": 0,
            "failed": 0,
            "avg_processing_time": 0.0
        }
    
    async def add_task(self, change: FileChange, priority: int = 1):
        """Add analysis task to queue"""
        try:
            await self.queue.put((priority, change, time.time()))
            self.stats["queued"] += 1
        except asyncio.QueueFull:
            logger.warning("Analysis queue full, dropping task")
    
    async def get_next_task(self) -> Optional[FileChange]:
        """Get next task from queue"""
        try:
            priority, change, queued_time = await self.queue.get()
            return change
        except:
            return None
    
    def update_stats(self, success: bool, processing_time: float):
        """Update processing statistics"""
        if success:
            self.stats["processed"] += 1
        else:
            self.stats["failed"] += 1
        
        # Update average processing time
        total_processed = self.stats["processed"] + self.stats["failed"]
        if total_processed > 0:
            current_avg = self.stats["avg_processing_time"]
            self.stats["avg_processing_time"] = (current_avg * (total_processed - 1) + processing_time) / total_processed


class RealtimeFileWatcher:
    """Real-time file watching system with incremental analysis"""
    
    def __init__(
        self,
        watch_directories: List[str],
        glean_client=None,
        grok_client=None,
        enable_ai_analysis: bool = True
    ):
        self.watch_directories = [Path(d) for d in watch_directories]
        self.glean_client = glean_client
        self.grok_client = grok_client
        self.enable_ai_analysis = enable_ai_analysis
        
        # For handling changes from non-async threads
        self.pending_changes: List[FileChange] = []
        
        # File watching components
        self.observer: Optional[Observer] = None
        self.handlers: List[CodeFileHandler] = []
        self.is_watching = False
        
        # Analysis components
        self.analysis_queue = AnalysisQueue()
        self.analysis_tasks: Set[asyncio.Task] = set()
        self.change_buffer: Dict[str, FileChange] = {}
        self.buffer_timeout = 2.0  # seconds
        
        # Change callbacks
        self.change_callbacks: List[Callable[[FileChange], None]] = []
        self.analysis_callbacks: List[Callable[[AnalysisResult], None]] = []
        
        # Statistics
        self.stats = {
            "files_watched": 0,
            "changes_detected": 0,
            "analyses_completed": 0,
            "ai_insights_generated": 0,
            "start_time": None
        }
    
    async def start_watching(self):
        """Start the file watching system"""
        if not WATCHDOG_AVAILABLE:
            logger.error("Watchdog not available. Cannot start file watching.")
            return False
        
        try:
            self.observer = Observer()
            
            # Add handlers for each directory
            for directory in self.watch_directories:
                if directory.exists():
                    handler = CodeFileHandler(self)
                    self.handlers.append(handler)
                    self.observer.schedule(handler, str(directory), recursive=True)
                    
                    # Count files being watched
                    file_count = sum(1 for p in directory.rglob("*.py") if p.is_file())
                    self.stats["files_watched"] += file_count
                    logger.info(f"Watching {file_count} Python files in {directory}")
                else:
                    logger.warning(f"Directory not found: {directory}")
            
            # Start observer
            self.observer.start()
            self.is_watching = True
            
            # Start analysis worker
            asyncio.create_task(self._analysis_worker())
            
            # Start buffer flush worker
            asyncio.create_task(self._buffer_flush_worker())
            
            self.stats["start_time"] = datetime.now()
            logger.info(f"File watching started for {len(self.watch_directories)} directories")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
            return False
    
    async def stop_watching(self):
        """Stop the file watching system"""
        self.is_watching = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        # Cancel analysis tasks
        for task in self.analysis_tasks:
            task.cancel()
        
        logger.info("File watching stopped")
    
    async def handle_file_change(self, change: FileChange):
        """Handle a file change event"""
        try:
            # Add checksum for change detection
            if change.change_type != ChangeType.DELETED:
                try:
                    with open(change.file_path, 'rb') as f:
                        content = f.read()
                        change.checksum = hashlib.md5(content).hexdigest()
                        change.file_size = len(content)
                except:
                    pass
            
            # Buffer changes to avoid duplicate processing
            self.change_buffer[change.file_path] = change
            
            self.stats["changes_detected"] += 1
            
            # Notify change callbacks
            for callback in self.change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Change callback failed: {e}")
            
            logger.debug(f"File change buffered: {change.file_path} ({change.change_type.value})")
            
        except Exception as e:
            logger.error(f"Error handling file change: {e}")
    
    async def _buffer_flush_worker(self):
        """Worker that flushes the change buffer periodically"""
        while self.is_watching:
            try:
                await asyncio.sleep(self.buffer_timeout)
                
                if self.change_buffer:
                    # Process buffered changes
                    changes_to_process = list(self.change_buffer.values())
                    self.change_buffer.clear()
                    
                    for change in changes_to_process:
                        await self.analysis_queue.add_task(change)
                    
                    logger.debug(f"Flushed {len(changes_to_process)} changes to analysis queue")
                
            except Exception as e:
                logger.error(f"Buffer flush worker error: {e}")
    
    async def _analysis_worker(self):
        """Worker that processes analysis tasks"""
        while self.is_watching:
            try:
                change = await self.analysis_queue.get_next_task()
                if change:
                    # Create analysis task
                    task = asyncio.create_task(self._analyze_file_change(change))
                    self.analysis_tasks.add(task)
                    
                    # Clean up completed tasks
                    self.analysis_tasks = {t for t in self.analysis_tasks if not t.done()}
                
            except Exception as e:
                logger.error(f"Analysis worker error: {e}")
                await asyncio.sleep(1)
    
    async def _analyze_file_change(self, change: FileChange):
        """Analyze a file change"""
        start_time = time.time()
        
        try:
            if change.change_type == ChangeType.DELETED:
                # Handle file deletion
                result = AnalysisResult(
                    file_path=change.file_path,
                    analysis_type="deletion",
                    processing_time=time.time() - start_time
                )
            else:
                # Analyze file content
                result = await self._perform_file_analysis(change)
            
            # Update statistics
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self.analysis_queue.update_stats(True, processing_time)
            self.stats["analyses_completed"] += 1
            
            # Notify analysis callbacks
            for callback in self.analysis_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Analysis callback failed: {e}")
            
            logger.debug(f"Analysis completed: {change.file_path} ({processing_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"File analysis failed for {change.file_path}: {e}")
            self.analysis_queue.update_stats(False, time.time() - start_time)
    
    async def _perform_file_analysis(self, change: FileChange) -> AnalysisResult:
        """Perform comprehensive file analysis"""
        result = AnalysisResult(
            file_path=change.file_path,
            analysis_type="incremental"
        )
        
        try:
            # Basic file analysis
            if self.glean_client:
                # Get file symbols using Glean
                try:
                    from ...infrastructure.analysis.angle_parser import create_query
                    query = create_query("file_symbols", file=change.file_path)
                    symbols = await self.glean_client.query(query)
                    
                    if isinstance(symbols, list):
                        result.symbols_found = symbols[:20]  # Limit for performance
                    
                    # Update Glean index incrementally
                    await self.glean_client.index_project("realtime", force_reindex=False)
                    
                except Exception as e:
                    logger.debug(f"Glean analysis failed for {change.file_path}: {e}")
            
            # AI analysis (if enabled and available)
            if self.enable_ai_analysis and self.grok_client:
                try:
                    # Prepare analysis data for AI
                    analysis_data = {
                        "file_path": change.file_path,
                        "change_type": change.change_type.value,
                        "symbols": result.symbols_found,
                        "file_size": change.file_size
                    }
                    
                    # Get AI insights
                    ai_result = await self.grok_client.analyze_code_structure(
                        analysis_data, 
                        focus="real_time_change"
                    )
                    
                    if ai_result.get("status") == "success":
                        result.ai_insights = ai_result
                        self.stats["ai_insights_generated"] += 1
                    
                except Exception as e:
                    logger.debug(f"AI analysis failed for {change.file_path}: {e}")
            
            # Calculate complexity score (simple heuristic)
            if change.file_size:
                result.complexity_score = min(change.file_size / 1000.0, 10.0)  # 0-10 scale
            
        except Exception as e:
            logger.error(f"Analysis error for {change.file_path}: {e}")
        
        return result
    
    def add_change_callback(self, callback: Callable[[FileChange], None]):
        """Add callback for file changes"""
        self.change_callbacks.append(callback)
    
    def add_analysis_callback(self, callback: Callable[[AnalysisResult], None]):
        """Add callback for analysis results"""
        self.analysis_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get watching and analysis statistics"""
        uptime = 0.0
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            "is_watching": self.is_watching,
            "uptime_seconds": uptime,
            "directories_watched": len(self.watch_directories),
            "files_watched": self.stats["files_watched"],
            "changes_detected": self.stats["changes_detected"],
            "analyses_completed": self.stats["analyses_completed"],
            "ai_insights_generated": self.stats["ai_insights_generated"],
            "queue_stats": self.analysis_queue.stats,
            "active_analysis_tasks": len(self.analysis_tasks),
            "buffered_changes": len(self.change_buffer)
        }


# Factory function for easy creation
async def create_file_watcher(
    watch_directories: List[str],
    project_root: str = None,
    enable_ai: bool = True
) -> RealtimeFileWatcher:
    """Create and configure a real-time file watcher"""
    
    # Import Glean client
    glean_client = None
    try:
        from ...infrastructure.analysis.vercel_glean_client import VercelGleanClient
        glean_client = VercelGleanClient(project_root=project_root)
    except Exception as e:
        logger.warning(f"Glean client not available: {e}")
    
    # Import Grok client
    grok_client = None
    if enable_ai:
        try:
            from ...core.ai.grok_client import GrokClient
            grok_client = GrokClient()
        except Exception as e:
            logger.warning(f"Grok client not available: {e}")
    
    watcher = RealtimeFileWatcher(
        watch_directories=watch_directories,
        glean_client=glean_client,
        grok_client=grok_client,
        enable_ai_analysis=enable_ai and grok_client is not None
    )
    
    return watcher


# Simple CLI for testing
async def test_file_watcher():
    """Test the file watcher system"""
    print("🔍 TESTING REAL-TIME FILE WATCHER")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent.parent.parent
    watch_dirs = [str(project_root / "src" / "cryptotrading")]
    
    # Create watcher
    watcher = await create_file_watcher(watch_dirs, str(project_root), enable_ai=False)
    
    # Add callbacks
    def on_change(change: FileChange):
        print(f"📝 File {change.change_type.value}: {Path(change.file_path).name}")
    
    def on_analysis(result: AnalysisResult):
        print(f"🔬 Analysis: {Path(result.file_path).name} - {len(result.symbols_found)} symbols")
    
    watcher.add_change_callback(on_change)
    watcher.add_analysis_callback(on_analysis)
    
    # Start watching
    if await watcher.start_watching():
        print("✅ File watcher started. Make some file changes...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                await asyncio.sleep(5)
                stats = watcher.get_statistics()
                print(f"📊 Stats: {stats['changes_detected']} changes, {stats['analyses_completed']} analyses")
        except KeyboardInterrupt:
            print("\n🛑 Stopping watcher...")
    else:
        print("❌ Failed to start file watcher")
    
    await watcher.stop_watching()
    print("👋 File watcher stopped")


if __name__ == "__main__":
    asyncio.run(test_file_watcher())