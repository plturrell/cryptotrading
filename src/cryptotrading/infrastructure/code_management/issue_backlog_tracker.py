"""
Issue Backlog and Progress Tracking System
Comprehensive system for managing issue backlogs, sprint planning, and progress tracking
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .database_adapter import CodeManagementDatabaseAdapter
from .intelligent_code_manager import CodeIssue, FixStatus, IssueType
from .issue_lifecycle_manager import IssueLifecycleManager, IssuePriority, IssueState

logger = logging.getLogger(__name__)


class SprintStatus(Enum):
    """Sprint status values"""

    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Sprint:
    """Represents a development sprint for issue resolution"""

    id: str
    name: str
    start_date: str
    end_date: str
    status: SprintStatus
    capacity: int  # Number of issues that can be handled
    assigned_issues: List[str]  # Issue IDs
    goals: List[str]
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BacklogMetrics:
    """Metrics for backlog analysis"""

    total_backlog_size: int
    by_priority: Dict[str, int]
    by_type: Dict[str, int]
    by_component: Dict[str, int]
    average_age_days: float
    velocity_issues_per_week: float
    estimated_completion_weeks: float
    critical_issues: int
    auto_fixable_count: int


@dataclass
class ProgressReport:
    """Progress tracking report"""

    sprint_id: str
    sprint_name: str
    total_issues: int
    completed_issues: int
    in_progress_issues: int
    blocked_issues: int
    completion_percentage: float
    days_remaining: int
    velocity: float
    burndown_data: List[Dict[str, Any]]
    risk_factors: List[str]


class IssueBacklogTracker:
    """Manages issue backlogs and progress tracking"""

    def __init__(
        self,
        database_adapter: CodeManagementDatabaseAdapter,
        lifecycle_manager: IssueLifecycleManager,
    ):
        self.database_adapter = database_adapter
        self.lifecycle_manager = lifecycle_manager
        self.sprints: List[Sprint] = []
        self.current_sprint: Optional[Sprint] = None

    async def create_sprint(
        self,
        name: str,
        start_date: datetime = None,
        end_date: datetime = None,
        duration_weeks: int = 2,
    ) -> str:
        """Create a new sprint"""
        try:
            if start_date is None:
                start_date = datetime.now()
            if end_date is None:
                end_date = start_date + timedelta(weeks=duration_weeks)

            sprint = Sprint(
                id=f"sprint_{int(start_date.timestamp())}",
                name=name,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                status=SprintStatus.PLANNING,
                capacity=40,  # Default capacity
                assigned_issues=[],
                goals=[],  # Default empty goals
                created_at=start_date.isoformat(),
            )

            self.sprints.append(sprint)

            # Save sprint to database
            await self.database_adapter.log_monitoring_event(
                event_type="sprint_created",
                component="backlog_tracker",
                severity="info",
                message=f"Sprint '{name}' created successfully",
                details=asdict(sprint),
            )

            logger.info("Created sprint: %s", name)
            return sprint.id

        except Exception as e:
            logger.error("Error creating sprint: %s", e)
            raise

    async def assign_issue_to_sprint(self, issue_id: str, sprint_id: str) -> bool:
        """Assign an issue to a specific sprint"""
        try:
            sprint = next((s for s in self.sprints if s.id == sprint_id), None)
            if not sprint:
                raise ValueError(f"Sprint not found: {sprint_id}")

            if issue_id not in sprint.assigned_issues:
                sprint.assigned_issues.append(issue_id)

                # Transition issue to in_progress
                await self.lifecycle_manager.transition_issue(
                    issue_id,
                    IssueState.IN_PROGRESS,
                    f"Assigned to sprint: {sprint.name}",
                    automated=True,
                    metadata={"sprint_id": sprint_id},
                )

                logger.info("Assigned issue %s to sprint %s", issue_id, sprint_id)
                return True

            return False

        except Exception as e:
            logger.error("Error assigning issue to sprint: %s", e)
            raise

    async def populate_sprint_backlog(self, sprint_id: str, auto_select: bool = True) -> List[str]:
        """Populate sprint backlog with prioritized issues"""
        try:
            sprint = next((s for s in self.sprints if s.id == sprint_id), None)
            if not sprint:
                raise ValueError(f"Sprint not found: {sprint_id}")

            if auto_select:
                # Auto-select highest priority issues
                prioritized_backlog = await self.lifecycle_manager.prioritize_backlog()

                # Select issues up to sprint capacity
                selected_issues = []
                for issue in prioritized_backlog[: sprint.capacity]:
                    selected_issues.append(issue.id)

                    # Transition to in_progress
                    await self.lifecycle_manager.transition_issue(
                        issue.id,
                        IssueState.IN_PROGRESS,
                        f"Added to sprint: {sprint.name}",
                        automated=True,
                        metadata={"sprint_id": sprint_id},
                    )

                sprint.assigned_issues = selected_issues

                logger.info(
                    "Auto-populated sprint %s with %d issues", sprint.name, len(selected_issues)
                )

                return selected_issues

            return sprint.assigned_issues

        except Exception as e:
            logger.error("Error populating sprint backlog: %s", e)
            return []

    async def start_sprint(self, sprint_id: str) -> bool:
        """Start an active sprint"""
        try:
            sprint = next((s for s in self.sprints if s.id == sprint_id), None)
            if not sprint:
                return False

            # End current sprint if active
            if self.current_sprint and self.current_sprint.status == SprintStatus.ACTIVE:
                await self.end_sprint(self.current_sprint.id)

            sprint.status = SprintStatus.ACTIVE
            self.current_sprint = sprint

            # Log sprint start
            await self.database_adapter.log_monitoring_event(
                event_type="sprint_started",
                details={"sprint_id": sprint_id, "sprint_name": sprint.name},
            )

            logger.info("Started sprint: %s", sprint.name)
            return True

        except Exception as e:
            logger.error("Error starting sprint: %s", e)
            return False

    async def end_sprint(self, sprint_id: str) -> ProgressReport:
        """End a sprint and generate final report"""
        try:
            sprint = next((s for s in self.sprints if s.id == sprint_id), None)
            if not sprint:
                raise ValueError(f"Sprint not found: {sprint_id}")

            sprint.status = SprintStatus.COMPLETED

            # Generate final progress report
            report = await self.get_progress_report(sprint_id)

            # Log sprint completion
            await self.database_adapter.log_monitoring_event(
                event_type="sprint_completed",
                details={
                    "sprint_id": sprint_id,
                    "completion_percentage": report.completion_percentage,
                    "completed_issues": report.completed_issues,
                    "total_issues": report.total_issues,
                },
            )

            if self.current_sprint and self.current_sprint.id == sprint_id:
                self.current_sprint = None

            logger.info(
                "Completed sprint: %s (%.1f%% completion)",
                sprint.name,
                report.completion_percentage,
            )

            return report

        except Exception as e:
            logger.error("Error ending sprint: %s", e)
            raise

    async def get_backlog_metrics(self) -> BacklogMetrics:
        """Get comprehensive backlog metrics"""
        try:
            # Get prioritized backlog
            backlog_issues = await self.lifecycle_manager.prioritize_backlog()

            if not backlog_issues:
                return BacklogMetrics(
                    total_backlog_size=0,
                    by_priority={},
                    by_type={},
                    by_component={},
                    average_age_days=0,
                    velocity_issues_per_week=0,
                    estimated_completion_weeks=0,
                    critical_issues=0,
                    auto_fixable_count=0,
                )

            # Calculate metrics
            by_priority = {}
            by_type = {}
            by_component = {}
            ages = []
            auto_fixable_count = 0
            critical_issues = 0

            now = datetime.now()

            for issue in backlog_issues:
                # Priority distribution
                priority = issue.metadata.get("priority", "low") if issue.metadata else "low"
                by_priority[priority] = by_priority.get(priority, 0) + 1

                # Type distribution
                by_type[issue.type.value] = by_type.get(issue.type.value, 0) + 1

                # Component distribution (from file path)
                component = self._extract_component(issue.file_path)
                by_component[component] = by_component.get(component, 0) + 1

                # Age calculation
                detected_time = datetime.fromisoformat(issue.detected_at)
                age_days = (now - detected_time).days
                ages.append(age_days)

                # Count auto-fixable and critical
                if issue.auto_fixable:
                    auto_fixable_count += 1
                if priority == "critical":
                    critical_issues += 1

            # Calculate velocity (issues completed per week)
            velocity = await self._calculate_velocity()

            # Estimate completion time
            estimated_weeks = len(backlog_issues) / velocity if velocity > 0 else float("inf")

            return BacklogMetrics(
                total_backlog_size=len(backlog_issues),
                by_priority=by_priority,
                by_type=by_type,
                by_component=by_component,
                average_age_days=sum(ages) / len(ages) if ages else 0,
                velocity_issues_per_week=velocity,
                estimated_completion_weeks=estimated_weeks,
                critical_issues=critical_issues,
                auto_fixable_count=auto_fixable_count,
            )

        except Exception as e:
            logger.error("Error calculating backlog metrics: %s", e)
            return BacklogMetrics(
                total_backlog_size=0,
                by_priority={},
                by_type={},
                by_component={},
                average_age_days=0,
                velocity_issues_per_week=0,
                estimated_completion_weeks=0,
                critical_issues=0,
                auto_fixable_count=0,
            )

    def _extract_component(self, file_path: str) -> str:
        """Extract component name from file path"""
        if not file_path:
            return "unknown"

        parts = file_path.split("/")
        if "src" in parts:
            src_index = parts.index("src")
            if src_index + 1 < len(parts):
                return parts[src_index + 1]

        return parts[0] if parts else "unknown"

    async def _calculate_velocity(self) -> float:
        """Calculate team velocity (issues completed per week)"""
        try:
            # Get completed issues from last 4 weeks
            cutoff_date = datetime.now() - timedelta(weeks=4)

            issues = await self.database_adapter.get_issues()
            completed_issues = [
                i
                for i in issues
                if (
                    i.metadata
                    and i.metadata.get("lifecycle_state") in ["verified", "closed"]
                    and i.metadata.get("last_transition")
                    and datetime.fromisoformat(i.metadata["last_transition"]) >= cutoff_date
                )
            ]

            return len(completed_issues) / 4.0  # Issues per week

        except Exception as e:
            logger.error("Error calculating velocity: %s", e)
            return 0.0

    async def get_progress_report(self, sprint_id: str) -> ProgressReport:
        """Get detailed progress report for a sprint"""
        try:
            sprint = next((s for s in self.sprints if s.id == sprint_id), None)
            if not sprint:
                raise ValueError(f"Sprint not found: {sprint_id}")

            # Get sprint issues
            issues = await self.database_adapter.get_issues()
            sprint_issues = [
                i for i in issues if (i.metadata and i.metadata.get("sprint_id") == sprint_id)
            ]

            # Count by status
            completed = len(
                [
                    i
                    for i in sprint_issues
                    if i.metadata.get("lifecycle_state") in ["verified", "closed"]
                ]
            )
            in_progress = len(
                [
                    i
                    for i in sprint_issues
                    if i.metadata.get("lifecycle_state") in ["in_progress", "fixing"]
                ]
            )
            blocked = len([i for i in sprint_issues if i.metadata.get("blocked", False)])

            total = len(sprint_issues)
            completion_percentage = (completed / total * 100) if total > 0 else 0

            # Calculate days remaining
            end_date = datetime.fromisoformat(sprint.end_date)
            days_remaining = max(0, (end_date - datetime.now()).days)

            # Calculate current velocity
            start_date = datetime.fromisoformat(sprint.start_date)
            days_elapsed = (datetime.now() - start_date).days
            velocity = completed / max(1, days_elapsed) * 7  # Issues per week

            # Generate burndown data
            burndown_data = await self._generate_burndown_data(sprint_id)

            # Identify risk factors
            risk_factors = self._identify_risk_factors(
                sprint, completion_percentage, days_remaining, velocity
            )

            return ProgressReport(
                sprint_id=sprint_id,
                sprint_name=sprint.name,
                total_issues=total,
                completed_issues=completed,
                in_progress_issues=in_progress,
                blocked_issues=blocked,
                completion_percentage=completion_percentage,
                days_remaining=days_remaining,
                velocity=velocity,
                burndown_data=burndown_data,
                risk_factors=risk_factors,
            )

        except Exception as e:
            logger.error("Error generating progress report: %s", e)
            raise

    async def _generate_burndown_data(self, sprint_id: str) -> List[Dict[str, Any]]:
        """Generate burndown chart data"""
        try:
            sprint = next((s for s in self.sprints if s.id == sprint_id), None)
            if not sprint:
                return []

            start_date = datetime.fromisoformat(sprint.start_date)
            end_date = datetime.fromisoformat(sprint.end_date)

            # Generate daily data points
            burndown_data = []
            current_date = start_date

            while current_date <= min(datetime.now(), end_date):
                # Count remaining issues at this date
                issues = await self.database_adapter.get_issues()
                remaining = len(
                    [
                        i
                        for i in issues
                        if (
                            i.metadata
                            and i.metadata.get("sprint_id") == sprint_id
                            and i.metadata.get("lifecycle_state") not in ["verified", "closed"]
                        )
                    ]
                )

                burndown_data.append(
                    {
                        "date": current_date.isoformat(),
                        "remaining_issues": remaining,
                        "ideal_remaining": max(
                            0,
                            len(sprint.assigned_issues)
                            * (end_date - current_date).days
                            / (end_date - start_date).days,
                        ),
                    }
                )

                current_date += timedelta(days=1)

            return burndown_data

        except Exception as e:
            logger.error("Error generating burndown data: %s", e)
            return []

    def _identify_risk_factors(
        self, sprint: Sprint, completion_percentage: float, days_remaining: int, velocity: float
    ) -> List[str]:
        """Identify potential risk factors for sprint completion"""
        risks = []

        # Low completion rate
        sprint_duration = (
            datetime.fromisoformat(sprint.end_date) - datetime.fromisoformat(sprint.start_date)
        ).days
        expected_completion = ((sprint_duration - days_remaining) / sprint_duration) * 100

        if completion_percentage < expected_completion - 20:
            risks.append("Behind schedule - completion rate below expected")

        # Low velocity
        if velocity < 2.0:  # Less than 2 issues per week
            risks.append("Low team velocity detected")

        # High number of blocked issues
        if len([i for i in sprint.assigned_issues]) > len(sprint.assigned_issues) * 0.3:
            risks.append("High number of blocked issues")

        # Sprint overcommitment
        if len(sprint.assigned_issues) > sprint.capacity * 1.2:
            risks.append("Sprint appears overcommitted")

        # Time pressure
        if days_remaining < 3 and completion_percentage < 80:
            risks.append("Time pressure - sprint ending soon with low completion")

        return risks

    async def get_sprint_summary(self, sprint_id: str) -> Dict[str, Any]:
        """Get comprehensive sprint summary"""
        try:
            sprint = next((s for s in self.sprints if s.id == sprint_id), None)
            if not sprint:
                return {}

            progress_report = await self.get_progress_report(sprint_id)

            return {
                "sprint": asdict(sprint),
                "progress": asdict(progress_report),
                "health_score": self._calculate_sprint_health_score(progress_report),
                "recommendations": self._generate_sprint_recommendations(progress_report),
            }

        except Exception as e:
            logger.error("Error generating sprint summary: %s", e)
            return {}

    def _calculate_sprint_health_score(self, report: ProgressReport) -> float:
        """Calculate overall sprint health score (0-100)"""
        score = 100.0

        # Deduct for low completion rate
        if report.completion_percentage < 50:
            score -= 30
        elif report.completion_percentage < 70:
            score -= 15

        # Deduct for blocked issues
        if report.blocked_issues > 0:
            score -= min(20, report.blocked_issues * 5)

        # Deduct for risk factors
        score -= len(report.risk_factors) * 10

        # Deduct for low velocity
        if report.velocity < 2.0:
            score -= 15

        return max(0, score)

    def _generate_sprint_recommendations(self, report: ProgressReport) -> List[str]:
        """Generate actionable recommendations for sprint improvement"""
        recommendations = []

        if report.completion_percentage < 50:
            recommendations.append("Consider reducing sprint scope or extending timeline")

        if report.blocked_issues > 0:
            recommendations.append("Focus on unblocking issues to improve flow")

        if report.velocity < 2.0:
            recommendations.append("Investigate velocity bottlenecks and process improvements")

        if "Behind schedule" in [r for r in report.risk_factors]:
            recommendations.append("Increase daily standup frequency and remove impediments")

        if report.days_remaining < 3 and report.completion_percentage < 80:
            recommendations.append("Prioritize critical issues and defer non-essential work")

        return recommendations

    async def archive_completed_sprints(self, days_threshold: int = 30) -> int:
        """Archive old completed sprints"""
        archived_count = 0
        cutoff_date = datetime.now() - timedelta(days=days_threshold)

        for sprint in self.sprints[:]:
            if (
                sprint.status == SprintStatus.COMPLETED
                and datetime.fromisoformat(sprint.end_date) < cutoff_date
            ):
                # Log archival
                await self.database_adapter.log_monitoring_event(
                    event_type="sprint_archived",
                    details={"sprint_id": sprint.id, "sprint_name": sprint.name},
                )

                self.sprints.remove(sprint)
                archived_count += 1

        return archived_count
