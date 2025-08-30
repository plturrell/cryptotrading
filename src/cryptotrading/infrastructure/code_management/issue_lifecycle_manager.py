"""
Issue Lifecycle Management System
Comprehensive system for managing issue states, transitions, and lifecycle tracking
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .database_adapter import CodeManagementDatabaseAdapter
from .intelligent_code_manager import CodeIssue, FixStatus, IssueType

logger = logging.getLogger(__name__)


class IssueState(Enum):
    """Issue lifecycle states"""

    DETECTED = "detected"
    TRIAGED = "triaged"
    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    FIXING = "fixing"
    FIXED = "fixed"
    TESTING = "testing"
    VERIFIED = "verified"
    CLOSED = "closed"
    REJECTED = "rejected"


class IssuePriority(Enum):
    """Issue priority levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class IssueTransition:
    """Represents a state transition in issue lifecycle"""

    issue_id: str
    from_state: IssueState
    to_state: IssueState
    timestamp: str
    reason: str
    automated: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class IssueLifecycleMetrics:
    """Metrics for issue lifecycle analysis"""

    total_issues: int
    by_state: Dict[str, int]
    by_priority: Dict[str, int]
    by_type: Dict[str, int]
    average_resolution_time: float
    backlog_age_distribution: Dict[str, int]
    fix_success_rate: float
    automated_fix_rate: float


class IssueLifecycleManager:
    """Manages issue lifecycle states and transitions"""

    def __init__(self, database_adapter: CodeManagementDatabaseAdapter):
        self.database_adapter = database_adapter
        self.state_transitions: List[IssueTransition] = []

        # Define valid state transitions
        self.valid_transitions = {
            IssueState.DETECTED: [IssueState.TRIAGED, IssueState.IN_PROGRESS, IssueState.REJECTED],
            IssueState.TRIAGED: [IssueState.BACKLOG, IssueState.IN_PROGRESS, IssueState.REJECTED],
            IssueState.BACKLOG: [IssueState.IN_PROGRESS, IssueState.REJECTED],
            IssueState.IN_PROGRESS: [IssueState.FIXING, IssueState.BACKLOG],
            IssueState.FIXING: [IssueState.FIXED, IssueState.IN_PROGRESS],
            IssueState.FIXED: [IssueState.TESTING, IssueState.IN_PROGRESS],
            IssueState.TESTING: [IssueState.VERIFIED, IssueState.IN_PROGRESS],
            IssueState.VERIFIED: [IssueState.CLOSED],
            IssueState.CLOSED: [],  # Terminal state
            IssueState.REJECTED: [],  # Terminal state
        }

    async def transition_issue(
        self,
        issue_id: str,
        to_state: IssueState,
        reason: str,
        automated: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Transition an issue to a new state"""
        try:
            # Get current issue
            issues_data = await self.database_adapter.get_issues()
            issue_data = next((i for i in issues_data if i.get("id") == issue_id), None)

            if not issue_data:
                logger.error("Issue not found: %s", issue_id)
                return False

            # Convert dict to CodeIssue object
            issue = CodeIssue(
                id=issue_data.get("id"),
                type=IssueType(issue_data.get("issue_type", "critical")),
                severity=issue_data.get("severity", 5),
                description=issue_data.get("description", ""),
                file_path=issue_data.get("file_path", ""),
                line_number=issue_data.get("line_number", 0),
                fix_status=FixStatus(issue_data.get("fix_status", "pending")),
                auto_fixable=issue_data.get("auto_fixable", False),
                suggested_fix=issue_data.get("suggested_fix", ""),
                detected_at=issue_data.get("detected_at", ""),
            )

            # Get current state from database metadata or default to DETECTED
            metadata = (
                json.loads(issue_data.get("metadata", "{}"))
                if isinstance(issue_data.get("metadata"), str)
                else issue_data.get("metadata", {})
            )
            current_state_str = metadata.get("lifecycle_state", "detected")
            current_state = IssueState(current_state_str)

            # Validate transition
            if not self._is_valid_transition(current_state, to_state):
                logger.error(
                    "Invalid transition from %s to %s for issue %s",
                    current_state.value,
                    to_state.value,
                    issue_id,
                )
                return False

            # Create transition record
            transition = IssueTransition(
                issue_id=issue_id,
                from_state=current_state,
                to_state=to_state,
                timestamp=datetime.now().isoformat(),
                reason=reason,
                automated=automated,
                metadata=metadata,
            )

            # Update metadata dictionary (not issue.metadata which doesn't exist)
            updated_metadata = metadata.copy()
            updated_metadata["lifecycle_state"] = to_state.value
            updated_metadata["last_transition"] = transition.timestamp
            updated_metadata["transition_reason"] = reason

            # Update fix status based on state
            new_fix_status = issue.fix_status
            if to_state == IssueState.FIXED:
                new_fix_status = FixStatus.COMPLETED
            elif to_state in [IssueState.FIXING, IssueState.IN_PROGRESS]:
                new_fix_status = FixStatus.IN_PROGRESS
            elif to_state == IssueState.REJECTED:
                new_fix_status = FixStatus.FAILED

            # Create updated issue with new fix status
            updated_issue = CodeIssue(
                id=issue.id,
                type=issue.type,
                severity=issue.severity,
                description=issue.description,
                file_path=issue.file_path,
                line_number=issue.line_number,
                fix_status=new_fix_status,
                auto_fixable=issue.auto_fixable,
                suggested_fix=issue.suggested_fix,
                detected_at=issue.detected_at,
            )

            # Save updated issue
            await self.database_adapter.save_issue(updated_issue)

            # Update the database metadata directly
            await self.database_adapter.update_issue_metadata(issue_id, updated_metadata)

            # Log transition
            await self.database_adapter.log_monitoring_event(
                event_type="issue_transition",
                component="lifecycle_manager",
                severity="info",
                message=f"Issue {issue_id} transitioned from {current_state.value} to {to_state.value}",
                details=asdict(transition),
            )

            self.state_transitions.append(transition)

            logger.info(
                "Issue %s transitioned from %s to %s: %s",
                issue_id,
                current_state.value,
                to_state.value,
                reason,
            )

            return True

        except Exception as e:
            logger.error("Error transitioning issue %s: %s", issue_id, e)
            return False

    def _is_valid_transition(self, from_state: IssueState, to_state: IssueState) -> bool:
        """Check if a state transition is valid"""
        return to_state in self.valid_transitions.get(from_state, [])

    async def auto_triage_issues(self) -> int:
        """Automatically triage newly detected issues"""
        triaged_count = 0

        try:
            # Get all detected issues
            issues_data = await self.database_adapter.get_issues()
            detected_issues = [
                i
                for i in issues_data
                if i.get("metadata")
                and (
                    json.loads(i.get("metadata", "{}"))
                    if isinstance(i.get("metadata"), str)
                    else i.get("metadata", {})
                ).get("lifecycle_state")
                == "detected"
            ]

            for issue_data in detected_issues:
                # Convert dict to CodeIssue object
                issue = CodeIssue(
                    id=issue_data.get("id"),
                    type=IssueType(issue_data.get("issue_type", "critical")),
                    severity=issue_data.get("severity", 5),
                    description=issue_data.get("description", ""),
                    file_path=issue_data.get("file_path", ""),
                    line_number=issue_data.get("line_number", 0),
                    fix_status=FixStatus(issue_data.get("fix_status", "pending")),
                    auto_fixable=issue_data.get("auto_fixable", False),
                    suggested_fix=issue_data.get("suggested_fix", ""),
                    detected_at=issue_data.get("detected_at", ""),
                )
                # Auto-triage based on severity and type
                priority = self._determine_priority(issue)

                # Update issue metadata in database
                metadata = (
                    json.loads(issue_data.get("metadata", "{}"))
                    if isinstance(issue_data.get("metadata"), str)
                    else issue_data.get("metadata", {})
                )
                metadata["priority"] = priority.value

                # Transition to triaged
                success = await self.transition_issue(
                    issue.id,
                    IssueState.TRIAGED,
                    f"Auto-triaged with priority: {priority.value}",
                    automated=True,
                    metadata={"priority": priority.value},
                )

                if success:
                    triaged_count += 1

            logger.info("Auto-triaged %d issues", triaged_count)

        except Exception as e:
            logger.error("Error in auto-triage: %s", e)

        return triaged_count

    def _determine_priority(self, issue: CodeIssue) -> IssuePriority:
        """Determine issue priority based on severity and type"""
        if issue.severity >= 9:
            return IssuePriority.CRITICAL
        elif issue.severity >= 7:
            return IssuePriority.HIGH
        elif issue.severity >= 5:
            return IssuePriority.MEDIUM
        elif issue.severity >= 3:
            return IssuePriority.LOW
        else:
            return IssuePriority.INFORMATIONAL

    async def prioritize_backlog(self) -> List[CodeIssue]:
        """Get prioritized backlog of issues"""
        try:
            issues_data = await self.database_adapter.get_issues()

            # Filter and convert backlog issues
            backlog_issues = []
            for issue_data in issues_data:
                metadata = (
                    json.loads(issue_data.get("metadata", "{}"))
                    if isinstance(issue_data.get("metadata"), str)
                    else issue_data.get("metadata", {})
                )
                if metadata.get("lifecycle_state") in ["triaged", "in_progress"]:
                    issue = CodeIssue(
                        id=issue_data.get("id"),
                        type=IssueType(issue_data.get("issue_type", "critical")),
                        severity=issue_data.get("severity", 5),
                        description=issue_data.get("description", ""),
                        file_path=issue_data.get("file_path", ""),
                        line_number=issue_data.get("line_number", 0),
                        fix_status=FixStatus(issue_data.get("fix_status", "pending")),
                        auto_fixable=issue_data.get("auto_fixable", False),
                        suggested_fix=issue_data.get("suggested_fix", ""),
                        detected_at=issue_data.get("detected_at", ""),
                    )
                    backlog_issues.append(issue)

            # Sort by priority and severity
            priority_order = {
                IssuePriority.CRITICAL.value: 0,
                IssuePriority.HIGH.value: 1,
                IssuePriority.MEDIUM.value: 2,
                IssuePriority.LOW.value: 3,
                IssuePriority.INFORMATIONAL.value: 4,
            }

            def sort_key(issue):
                priority = issue.metadata.get("priority", "low") if issue.metadata else "low"
                return (priority_order.get(priority, 4), -issue.severity)

            return sorted(backlog_issues, key=sort_key)

        except Exception as e:
            logger.error("Error prioritizing backlog: %s", e)
            return []

    async def auto_progress_fixable_issues(self) -> int:
        """Automatically progress auto-fixable issues"""
        progressed_count = 0

        try:
            # Get triaged auto-fixable issues
            issues = await self.database_adapter.get_issues()
            auto_fixable = [
                i
                for i in issues
                if (
                    i.auto_fixable and i.metadata and i.metadata.get("lifecycle_state") == "triaged"
                )
            ]

            for issue in auto_fixable:
                # Move to in_progress
                success = await self.transition_issue(
                    issue.id,
                    IssueState.IN_PROGRESS,
                    "Auto-progressed: issue is auto-fixable",
                    automated=True,
                )

                if success:
                    progressed_count += 1

            logger.info("Auto-progressed %d fixable issues", progressed_count)

        except Exception as e:
            logger.error("Error in auto-progress: %s", e)

        return progressed_count

    async def get_lifecycle_metrics(self) -> IssueLifecycleMetrics:
        """Get comprehensive lifecycle metrics"""
        try:
            issues_data = await self.database_adapter.get_issues()

            # Count by state
            by_state = {}
            for state in IssueState:
                by_state[state.value] = 0

            # Initialize counters
            by_priority = {}
            by_type = {}
            resolution_times = []

            for issue_data in issues_data:
                metadata = (
                    json.loads(issue_data.get("metadata", "{}"))
                    if isinstance(issue_data.get("metadata"), str)
                    else issue_data.get("metadata", {})
                )
                state = metadata.get("lifecycle_state", "detected")
                if state in by_state:
                    by_state[state] += 1

                # Priority counts
                priority = metadata.get("priority", "low")
                by_priority[priority] = by_priority.get(priority, 0) + 1

                # Type counts
                issue_type = issue_data.get("issue_type", "bug")
                by_type[issue_type] = by_type.get(issue_type, 0) + 1

                # Resolution time calculation
                if state == "closed" and metadata:
                    detected_time = datetime.fromisoformat(
                        issue_data.get("created_at", datetime.now().isoformat())
                    )
                    closed_time = datetime.fromisoformat(
                        metadata.get(
                            "last_transition",
                            issue_data.get("created_at", datetime.now().isoformat()),
                        )
                    )
                    resolution_time = (closed_time - detected_time).total_seconds() / 3600  # hours
                    resolution_times.append(resolution_time)

            # Calculate backlog age distribution
            backlog_age_distribution = await self._get_backlog_age_distribution(issues_data)

            # Calculate success rates
            total_fixed = (
                by_state.get("fixed", 0) + by_state.get("verified", 0) + by_state.get("closed", 0)
            )
            total_attempted = total_fixed + by_state.get("rejected", 0)
            fix_success_rate = (total_fixed / total_attempted * 100) if total_attempted > 0 else 0

            auto_fixed = len(
                [
                    i
                    for i in issues_data
                    if i.get("auto_fixable", False)
                    and (
                        json.loads(i.get("metadata", "{}"))
                        if isinstance(i.get("metadata"), str)
                        else i.get("metadata", {})
                    ).get("lifecycle_state")
                    in ["fixed", "verified", "closed"]
                ]
            )
            automated_fix_rate = (auto_fixed / total_fixed * 100) if total_fixed > 0 else 0

            return IssueLifecycleMetrics(
                total_issues=len(issues_data),
                by_state=by_state,
                by_priority=by_priority,
                by_type=by_type,
                average_resolution_time=sum(resolution_times) / len(resolution_times)
                if resolution_times
                else 0,
                backlog_age_distribution=backlog_age_distribution,
                fix_success_rate=fix_success_rate,
                automated_fix_rate=automated_fix_rate,
            )

        except Exception as e:
            logger.error("Error calculating lifecycle metrics: %s", e)
            return IssueLifecycleMetrics(
                total_issues=0,
                by_state={},
                by_priority={},
                by_type={},
                average_resolution_time=0,
                backlog_age_distribution={},
                fix_success_rate=0,
                automated_fix_rate=0,
            )

    async def _get_backlog_age_distribution(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate backlog age distribution"""
        now = datetime.now()
        age_buckets = {"< 1 day": 0, "1-7 days": 0, "1-4 weeks": 0, "> 1 month": 0}

        backlog_issues = []
        for issue_data in issues:
            metadata = (
                json.loads(issue_data.get("metadata", "{}"))
                if isinstance(issue_data.get("metadata"), str)
                else issue_data.get("metadata", {})
            )
            if metadata.get("lifecycle_state") == "backlog":
                backlog_issues.append(issue_data)

        for issue_data in backlog_issues:
            detected_at = issue_data.get("created_at", datetime.now().isoformat())
            try:
                detected_time = datetime.fromisoformat(detected_at)
                age_hours = (now - detected_time).total_seconds() / 3600

                if age_hours < 24:
                    age_buckets["< 1 day"] += 1
                elif age_hours < 168:  # 7 days
                    age_buckets["1-7 days"] += 1
                elif age_hours < 672:  # 4 weeks
                    age_buckets["1-4 weeks"] += 1
                else:
                    age_buckets["> 1 month"] += 1
            except (ValueError, TypeError):
                # Skip issues with invalid timestamps
                continue

        return age_buckets

    async def purge_expired_issues(self, days_threshold: int = 90) -> int:
        """Purge expired closed/rejected issues"""
        cleaned_count = 0

        try:
            issues = await self.database_adapter.get_issues()
            cutoff_date = datetime.now() - timedelta(days=days_threshold)

            for issue in issues:
                if (
                    issue.metadata
                    and issue.metadata.get("lifecycle_state") in ["closed", "rejected"]
                    and issue.metadata.get("last_transition")
                ):
                    last_transition = datetime.fromisoformat(issue.metadata["last_transition"])
                    if last_transition < cutoff_date:
                        # Archive the issue (could implement archival logic here)
                        logger.info("Archiving old issue: %s", issue.id)
                        cleaned_count += 1

        except Exception as e:
            logger.error("Error cleaning up old issues: %s", e)

        return cleaned_count

    async def get_issue_history(self, issue_id: str) -> List[IssueTransition]:
        """Get transition history for a specific issue"""
        return [t for t in self.state_transitions if t.issue_id == issue_id]

    async def bulk_transition_issues(
        self, issue_ids: List[str], to_state: IssueState, reason: str
    ) -> Dict[str, bool]:
        """Bulk transition multiple issues"""
        results = {}

        for issue_id in issue_ids:
            success = await self.transition_issue(issue_id, to_state, reason)
            results[issue_id] = success

        return results
