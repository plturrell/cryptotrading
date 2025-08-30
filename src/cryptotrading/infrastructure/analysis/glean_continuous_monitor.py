"""
Glean Continuous Monitoring MCP Tool
Real-time monitoring of code changes and blind spot detection
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .glean_zero_blindspots_mcp_tool import GleanValidationResult, GleanZeroBlindSpotsMCPTool

logger = logging.getLogger(__name__)


@dataclass
class FileChangeEvent:
    """File change event for monitoring"""

    file_path: str
    change_type: str  # created, modified, deleted
    timestamp: datetime
    file_hash: Optional[str] = None
    language: Optional[str] = None


@dataclass
class MonitoringSession:
    """Continuous monitoring session"""

    session_id: str
    project_path: str
    start_time: datetime
    last_validation: Optional[datetime] = None
    validation_interval: int = 300  # 5 minutes
    change_events: List[FileChangeEvent] = None
    current_score: float = 0.0
    baseline_score: float = 0.0

    def __post_init__(self):
        if self.change_events is None:
            self.change_events = []


class GleanContinuousMonitor:
    """Continuous monitoring for Glean knowledge completeness"""

    def __init__(self):
        self.name = "glean_continuous_monitor"
        self.description = (
            "Continuously monitors code changes and validates Glean knowledge completeness"
        )
        self.active_sessions: Dict[str, MonitoringSession] = {}
        self.validator = GleanZeroBlindSpotsMCPTool()

    async def execute(self, parameters: Dict[str, Any], agent_context=None) -> Dict[str, Any]:
        """Execute continuous monitoring command"""
        try:
            command = parameters.get("command", "status")
            project_path = parameters.get("project_path", ".")

            if command == "start":
                return await self._start_monitoring(project_path, parameters)
            elif command == "stop":
                return await self._stop_monitoring(project_path)
            elif command == "status":
                return await self._get_monitoring_status(project_path)
            elif command == "validate_now":
                return await self._validate_now(project_path)
            elif command == "list_sessions":
                return await self._list_sessions()
            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {command}",
                    "available_commands": [
                        "start",
                        "stop",
                        "status",
                        "validate_now",
                        "list_sessions",
                    ],
                }

        except Exception as e:
            logger.error(f"Continuous monitoring failed: {e}")
            return {"success": False, "error": str(e)}

    async def _start_monitoring(
        self, project_path: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start continuous monitoring session"""
        session_id = self._generate_session_id(project_path)

        # Check if already monitoring
        if session_id in self.active_sessions:
            return {
                "success": False,
                "error": "Monitoring already active for this project",
                "session_id": session_id,
            }

        # Create monitoring session
        session = MonitoringSession(
            session_id=session_id,
            project_path=project_path,
            start_time=datetime.now(),
            validation_interval=parameters.get("validation_interval", 300),
        )

        # Run initial validation
        logger.info(f"🔍 Starting continuous monitoring for {project_path}")
        initial_validation = await self.validator.execute({"project_path": project_path})

        if initial_validation["success"]:
            session.baseline_score = initial_validation["validation_result"]["validation_score"]
            session.current_score = session.baseline_score
            session.last_validation = datetime.now()

        self.active_sessions[session_id] = session

        # Start background monitoring task
        asyncio.create_task(self._monitor_loop(session_id))

        return {
            "success": True,
            "session_id": session_id,
            "baseline_score": session.baseline_score,
            "monitoring_started": session.start_time.isoformat(),
            "validation_interval": session.validation_interval,
            "message": f"Continuous monitoring started for {project_path}",
        }

    async def _stop_monitoring(self, project_path: str) -> Dict[str, Any]:
        """Stop continuous monitoring session"""
        session_id = self._generate_session_id(project_path)

        if session_id not in self.active_sessions:
            return {"success": False, "error": "No active monitoring session for this project"}

        session = self.active_sessions.pop(session_id)
        duration = datetime.now() - session.start_time

        return {
            "success": True,
            "session_id": session_id,
            "monitoring_stopped": datetime.now().isoformat(),
            "session_duration": str(duration),
            "total_change_events": len(session.change_events),
            "final_score": session.current_score,
            "score_change": session.current_score - session.baseline_score,
        }

    async def _get_monitoring_status(self, project_path: str) -> Dict[str, Any]:
        """Get current monitoring status"""
        session_id = self._generate_session_id(project_path)

        if session_id not in self.active_sessions:
            return {
                "success": True,
                "monitoring_active": False,
                "message": "No active monitoring session",
            }

        session = self.active_sessions[session_id]
        uptime = datetime.now() - session.start_time

        # Get recent change events
        recent_changes = [asdict(event) for event in session.change_events[-10:]]

        return {
            "success": True,
            "monitoring_active": True,
            "session_id": session_id,
            "uptime": str(uptime),
            "current_score": session.current_score,
            "baseline_score": session.baseline_score,
            "score_trend": session.current_score - session.baseline_score,
            "last_validation": session.last_validation.isoformat()
            if session.last_validation
            else None,
            "total_change_events": len(session.change_events),
            "recent_changes": recent_changes,
            "next_validation_in": self._time_until_next_validation(session),
        }

    async def _validate_now(self, project_path: str) -> Dict[str, Any]:
        """Force immediate validation"""
        session_id = self._generate_session_id(project_path)

        # Run validation
        validation_result = await self.validator.execute({"project_path": project_path})

        # Update session if monitoring is active
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if validation_result["success"]:
                session.current_score = validation_result["validation_result"]["validation_score"]
                session.last_validation = datetime.now()

        return {
            "success": True,
            "validation_result": validation_result,
            "monitoring_active": session_id in self.active_sessions,
        }

    async def _list_sessions(self) -> Dict[str, Any]:
        """List all active monitoring sessions"""
        sessions_info = []

        for session_id, session in self.active_sessions.items():
            uptime = datetime.now() - session.start_time
            sessions_info.append(
                {
                    "session_id": session_id,
                    "project_path": session.project_path,
                    "uptime": str(uptime),
                    "current_score": session.current_score,
                    "change_events": len(session.change_events),
                    "last_validation": session.last_validation.isoformat()
                    if session.last_validation
                    else None,
                }
            )

        return {
            "success": True,
            "active_sessions": len(self.active_sessions),
            "sessions": sessions_info,
        }

    async def _monitor_loop(self, session_id: str):
        """Background monitoring loop"""
        while session_id in self.active_sessions:
            try:
                session = self.active_sessions[session_id]

                # Check if validation is due
                if self._should_validate(session):
                    logger.info(f"🔍 Running scheduled validation for {session.project_path}")

                    validation_result = await self.validator.execute(
                        {"project_path": session.project_path, "mode": "quick"}
                    )

                    if validation_result["success"]:
                        old_score = session.current_score
                        session.current_score = validation_result["validation_result"][
                            "validation_score"
                        ]
                        session.last_validation = datetime.now()

                        # Check for significant score changes
                        score_change = session.current_score - old_score
                        if abs(score_change) > 5.0:
                            logger.warning(
                                f"📊 Significant score change detected: {score_change:+.1f} points"
                            )

                            # Could trigger alerts here
                            await self._handle_score_change(
                                session, score_change, validation_result
                            )

                # Scan for file changes
                await self._scan_for_changes(session)

                # Sleep until next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def _should_validate(self, session: MonitoringSession) -> bool:
        """Check if validation is due"""
        if session.last_validation is None:
            return True

        time_since_validation = datetime.now() - session.last_validation
        return time_since_validation.total_seconds() >= session.validation_interval

    async def _scan_for_changes(self, session: MonitoringSession):
        """Scan for file changes in the project"""
        # This is a simplified implementation
        # In production, you'd use file system watchers like watchdog
        project_path = Path(session.project_path)

        # Get current file hashes
        current_files = {}
        for ext in [".py", ".js", ".ts", ".tsx", ".cds", ".xml", ".json", ".yaml"]:
            for file_path in project_path.rglob(f"*{ext}"):
                if file_path.is_file() and "node_modules" not in str(file_path):
                    try:
                        with open(file_path, "rb") as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        current_files[str(file_path)] = file_hash
                    except Exception:
                        continue

        # Compare with previous scan (simplified - would need to store previous state)
        # For now, just log the scan
        logger.debug(f"Scanned {len(current_files)} files in {session.project_path}")

    async def _handle_score_change(
        self, session: MonitoringSession, score_change: float, validation_result: Dict[str, Any]
    ):
        """Handle significant score changes"""
        if score_change < -10:
            logger.error(f"🚨 Critical score drop: {score_change:+.1f} points")
        elif score_change < -5:
            logger.warning(f"⚠️ Score decreased: {score_change:+.1f} points")
        elif score_change > 5:
            logger.info(f"📈 Score improved: {score_change:+.1f} points")

    def _generate_session_id(self, project_path: str) -> str:
        """Generate unique session ID for project"""
        return hashlib.md5(project_path.encode()).hexdigest()[:8]

    def _time_until_next_validation(self, session: MonitoringSession) -> str:
        """Calculate time until next validation"""
        if session.last_validation is None:
            return "Now"

        next_validation = session.last_validation + timedelta(seconds=session.validation_interval)
        time_remaining = next_validation - datetime.now()

        if time_remaining.total_seconds() <= 0:
            return "Now"

        return str(time_remaining).split(".")[0]  # Remove microseconds


# MCP tool registration
async def glean_continuous_monitor_tool(
    parameters: Dict[str, Any], agent_context=None
) -> Dict[str, Any]:
    """MCP tool entry point for continuous monitoring"""
    monitor = GleanContinuousMonitor()
    return await monitor.execute(parameters, agent_context)


# Tool metadata for MCP server registration
GLEAN_CONTINUOUS_MONITOR_TOOL_METADATA = {
    "name": "glean_continuous_monitor",
    "description": "Continuously monitors code changes and validates Glean knowledge completeness",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["start", "stop", "status", "validate_now", "list_sessions"],
                "description": "Monitoring command to execute",
                "default": "status",
            },
            "project_path": {
                "type": "string",
                "description": "Path to the project to monitor",
                "default": ".",
            },
            "validation_interval": {
                "type": "integer",
                "description": "Validation interval in seconds",
                "default": 300,
            },
        },
    },
    "required": [],
}
