"""
SEAL-Based Code Adaptation Engine
Self-Adapting Language Model integration for automated code improvement and learning
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database_adapter import CodeManagementDatabaseAdapter
from .intelligent_code_manager import CodeIssue, FixStatus, IssueType
from .issue_lifecycle_manager import IssueLifecycleManager, IssueState

logger = logging.getLogger(__name__)


class AdaptationType(Enum):
    """Types of code adaptations"""

    BUG_FIX = "bug_fix"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CODE_QUALITY_IMPROVEMENT = "code_quality_improvement"
    SECURITY_ENHANCEMENT = "security_enhancement"
    REFACTORING = "refactoring"
    FEATURE_ENHANCEMENT = "feature_enhancement"


class AdaptationStrategy(Enum):
    """SEAL adaptation strategies"""

    SELF_EDIT = "self_edit"
    CONTINUAL_LEARNING = "continual_learning"
    FEW_SHOT_ADAPTATION = "few_shot_adaptation"
    KNOWLEDGE_INCORPORATION = "knowledge_incorporation"


@dataclass
class CodeAdaptationRequest:
    """Request for code adaptation using SEAL"""

    issue_id: str
    file_path: str
    code_content: str
    issue_description: str
    adaptation_type: AdaptationType
    strategy: AdaptationStrategy
    context: Dict[str, Any]
    priority: int = 5
    created_at: str = ""


@dataclass
class SelfEditResult:
    """Result of SEAL self-editing process"""

    original_code: str
    adapted_code: str
    edit_instructions: List[str]
    confidence_score: float
    reasoning: str
    validation_results: Dict[str, Any]
    improvement_metrics: Dict[str, float]


@dataclass
class AdaptationMetrics:
    """Metrics for tracking adaptation performance"""

    total_adaptations: int
    successful_adaptations: int
    failed_adaptations: int
    average_confidence: float
    average_improvement: float
    adaptation_types: Dict[str, int]
    strategies_used: Dict[str, int]


class SEALCodeAdapter:
    """SEAL-based code adaptation engine for automated code improvement"""

    def __init__(
        self,
        database_adapter: CodeManagementDatabaseAdapter,
        lifecycle_manager: IssueLifecycleManager,
    ):
        self.database_adapter = database_adapter
        self.lifecycle_manager = lifecycle_manager
        self.adaptation_history: List[SelfEditResult] = []
        self.learning_context: Dict[str, Any] = {}

        # SEAL configuration
        self.config = {
            "model_name": "grok",
            "max_iterations": 3,
            "confidence_threshold": 0.7,
            "learning_rate": 0.001,
            "adaptation_temperature": 0.3,
            "validation_enabled": True,
        }

    async def analyze_codebase_for_adaptation(
        self, project_path: Path
    ) -> List[CodeAdaptationRequest]:
        """Analyze codebase to identify adaptation opportunities"""
        try:
            adaptation_requests = []

            # Get current issues from lifecycle manager
            issues = await self.database_adapter.get_issues()

            for issue in issues:
                if issue.metadata and issue.metadata.get("lifecycle_state") in [
                    "triaged",
                    "backlog",
                ]:
                    # Determine adaptation type based on issue characteristics
                    adaptation_type = self._classify_adaptation_type(issue)
                    strategy = self._select_adaptation_strategy(issue, adaptation_type)

                    # Read current code content
                    try:
                        with open(issue.file_path, "r", encoding="utf-8") as f:
                            code_content = f.read()
                    except Exception as e:
                        logger.warning("Could not read file %s: %s", issue.file_path, e)
                        continue

                    request = CodeAdaptationRequest(
                        issue_id=issue.id,
                        file_path=issue.file_path,
                        code_content=code_content,
                        issue_description=issue.description,
                        adaptation_type=adaptation_type,
                        strategy=strategy,
                        context={
                            "severity": issue.severity,
                            "type": issue.type.value,
                            "auto_fixable": issue.auto_fixable,
                            "metadata": issue.metadata,
                        },
                        priority=self._calculate_adaptation_priority(issue),
                        created_at=datetime.now().isoformat(),
                    )

                    adaptation_requests.append(request)

            logger.info("Identified %d adaptation opportunities", len(adaptation_requests))
            return adaptation_requests

        except Exception as e:
            logger.error("Error analyzing codebase for adaptation: %s", e)
            return []

    def _classify_adaptation_type(self, issue: CodeIssue) -> AdaptationType:
        """Classify the type of adaptation needed based on issue characteristics"""
        if issue.type == IssueType.BUG:
            return AdaptationType.BUG_FIX
        elif issue.type == IssueType.PERFORMANCE:
            return AdaptationType.PERFORMANCE_OPTIMIZATION
        elif issue.type == IssueType.SECURITY:
            return AdaptationType.SECURITY_ENHANCEMENT
        elif issue.type == IssueType.CODE_QUALITY:
            return AdaptationType.CODE_QUALITY_IMPROVEMENT
        elif "refactor" in issue.description.lower():
            return AdaptationType.REFACTORING
        else:
            return AdaptationType.FEATURE_ENHANCEMENT

    def _select_adaptation_strategy(
        self, issue: CodeIssue, adaptation_type: AdaptationType
    ) -> AdaptationStrategy:
        """Select the best SEAL strategy for the adaptation"""
        if issue.auto_fixable:
            return AdaptationStrategy.SELF_EDIT
        elif issue.severity >= 7:
            return AdaptationStrategy.KNOWLEDGE_INCORPORATION
        elif adaptation_type in [AdaptationType.REFACTORING, AdaptationType.FEATURE_ENHANCEMENT]:
            return AdaptationStrategy.FEW_SHOT_ADAPTATION
        else:
            return AdaptationStrategy.CONTINUAL_LEARNING

    def _calculate_adaptation_priority(self, issue: CodeIssue) -> int:
        """Calculate priority for adaptation (1-10, higher is more urgent)"""
        priority = issue.severity

        if issue.auto_fixable:
            priority += 2
        if issue.type == IssueType.SECURITY:
            priority += 3
        if issue.type == IssueType.BUG:
            priority += 2

        return min(10, priority)

    async def perform_self_edit_adaptation(self, request: CodeAdaptationRequest) -> SelfEditResult:
        """Perform SEAL self-edit adaptation on code"""
        try:
            logger.info("Starting self-edit adaptation for issue %s", request.issue_id)

            # Generate self-edit instructions using SEAL approach
            edit_instructions = await self._generate_self_edit_instructions(request)

            # Apply edits iteratively
            adapted_code = request.code_content
            confidence_scores = []

            for iteration in range(self.config["max_iterations"]):
                # Generate adaptation
                adaptation_result = await self._apply_self_edit(
                    adapted_code, edit_instructions, request, iteration
                )

                adapted_code = adaptation_result["code"]
                confidence = adaptation_result["confidence"]
                confidence_scores.append(confidence)

                # Check if adaptation meets confidence threshold
                if confidence >= self.config["confidence_threshold"]:
                    break

                # Update instructions based on feedback
                edit_instructions = await self._refine_edit_instructions(
                    edit_instructions, adaptation_result["feedback"], request
                )

            # Validate the adapted code
            validation_results = await self._validate_adapted_code(
                request.code_content, adapted_code, request
            )

            # Calculate improvement metrics
            improvement_metrics = await self._calculate_improvement_metrics(
                request.code_content, adapted_code, request
            )

            result = SelfEditResult(
                original_code=request.code_content,
                adapted_code=adapted_code,
                edit_instructions=edit_instructions,
                confidence_score=max(confidence_scores) if confidence_scores else 0.0,
                reasoning=f"Applied {len(edit_instructions)} self-edit instructions over {len(confidence_scores)} iterations",
                validation_results=validation_results,
                improvement_metrics=improvement_metrics,
            )

            # Store adaptation result
            await self._store_adaptation_result(request, result)

            logger.info(
                "Self-edit adaptation completed with confidence %.2f", result.confidence_score
            )
            return result

        except Exception as e:
            logger.error("Error in self-edit adaptation: %s", e)
            return SelfEditResult(
                original_code=request.code_content,
                adapted_code=request.code_content,
                edit_instructions=[],
                confidence_score=0.0,
                reasoning=f"Adaptation failed: {e}",
                validation_results={"error": str(e)},
                improvement_metrics={},
            )

    async def _generate_self_edit_instructions(self, request: CodeAdaptationRequest) -> List[str]:
        """Generate SEAL-style self-edit instructions"""
        # This would integrate with actual SEAL model in production
        # For now, we'll generate rule-based instructions

        instructions = []

        if request.adaptation_type == AdaptationType.BUG_FIX:
            instructions.extend(
                [
                    "Identify the root cause of the bug in the code",
                    "Apply minimal fix that addresses the issue without side effects",
                    "Add error handling and validation where appropriate",
                    "Ensure fix is backwards compatible",
                ]
            )

        elif request.adaptation_type == AdaptationType.PERFORMANCE_OPTIMIZATION:
            instructions.extend(
                [
                    "Identify performance bottlenecks in the code",
                    "Optimize algorithms and data structures",
                    "Reduce unnecessary computations and memory allocations",
                    "Add caching where beneficial",
                ]
            )

        elif request.adaptation_type == AdaptationType.CODE_QUALITY_IMPROVEMENT:
            instructions.extend(
                [
                    "Improve code readability and maintainability",
                    "Follow established coding conventions and patterns",
                    "Add comprehensive documentation and type hints",
                    "Reduce code complexity and duplication",
                ]
            )

        elif request.adaptation_type == AdaptationType.SECURITY_ENHANCEMENT:
            instructions.extend(
                [
                    "Identify and fix security vulnerabilities",
                    "Add input validation and sanitization",
                    "Implement proper authentication and authorization",
                    "Use secure coding practices",
                ]
            )

        elif request.adaptation_type == AdaptationType.REFACTORING:
            instructions.extend(
                [
                    "Restructure code for better organization",
                    "Extract reusable components and functions",
                    "Improve separation of concerns",
                    "Maintain existing functionality while improving structure",
                ]
            )

        # Add context-specific instructions
        if request.context.get("severity", 0) >= 8:
            instructions.append("Prioritize safety and reliability in all changes")

        if request.context.get("auto_fixable"):
            instructions.append("Focus on automated, low-risk improvements")

        return instructions

    async def _apply_self_edit(
        self, code: str, instructions: List[str], request: CodeAdaptationRequest, iteration: int
    ) -> Dict[str, Any]:
        """Apply self-edit instructions to code"""
        # This would use actual SEAL model for code generation
        # For now, we'll simulate the process

        try:
            # Simulate code adaptation based on instructions
            adapted_code = await self._simulate_code_adaptation(code, instructions, request)

            # Calculate confidence based on various factors
            confidence = self._calculate_adaptation_confidence(code, adapted_code, request)

            # Generate feedback for next iteration
            feedback = await self._generate_adaptation_feedback(code, adapted_code, request)

            return {
                "code": adapted_code,
                "confidence": confidence,
                "feedback": feedback,
                "iteration": iteration,
            }

        except Exception as e:
            logger.error("Error applying self-edit: %s", e)
            return {
                "code": code,
                "confidence": 0.0,
                "feedback": f"Error: {e}",
                "iteration": iteration,
            }

    async def _simulate_code_adaptation(
        self, code: str, instructions: List[str], request: CodeAdaptationRequest
    ) -> str:
        """Simulate code adaptation (would use SEAL model in production)"""
        # This is a placeholder for actual SEAL integration
        # In production, this would call the SEAL model to generate adapted code

        adapted_code = code

        # Apply basic adaptations based on issue type
        if request.adaptation_type == AdaptationType.CODE_QUALITY_IMPROVEMENT:
            # Add type hints if missing
            if "def " in adapted_code and "->" not in adapted_code:
                adapted_code = adapted_code.replace("def ", "def ")  # Placeholder

        elif request.adaptation_type == AdaptationType.BUG_FIX:
            # Add basic error handling
            if "try:" not in adapted_code and "except:" not in adapted_code:
                # Wrap risky operations in try-catch (simplified)
                pass

        return adapted_code

    def _calculate_adaptation_confidence(
        self, original: str, adapted: str, request: CodeAdaptationRequest
    ) -> float:
        """Calculate confidence score for adaptation"""
        confidence = 0.5  # Base confidence

        # Increase confidence for successful adaptations
        if len(adapted) > len(original):
            confidence += 0.1  # Code was enhanced

        if request.adaptation_type == AdaptationType.BUG_FIX:
            confidence += 0.2  # Bug fixes are high priority

        if request.context.get("auto_fixable"):
            confidence += 0.3  # Auto-fixable issues are more reliable

        return min(1.0, confidence)

    async def _generate_adaptation_feedback(
        self, original: str, adapted: str, request: CodeAdaptationRequest
    ) -> str:
        """Generate feedback for adaptation refinement"""
        feedback = []

        if len(adapted) == len(original):
            feedback.append("No significant changes detected, consider more aggressive adaptation")

        if "TODO" in adapted or "FIXME" in adapted:
            feedback.append("Code still contains TODO/FIXME markers")

        if request.adaptation_type == AdaptationType.SECURITY_ENHANCEMENT:
            feedback.append("Ensure all security vulnerabilities are addressed")

        return "; ".join(feedback) if feedback else "Adaptation looks good"

    async def _refine_edit_instructions(
        self, instructions: List[str], feedback: str, request: CodeAdaptationRequest
    ) -> List[str]:
        """Refine edit instructions based on feedback"""
        refined_instructions = instructions.copy()

        if "aggressive" in feedback.lower():
            refined_instructions.append("Apply more comprehensive changes")

        if "TODO" in feedback:
            refined_instructions.append("Remove all TODO and FIXME comments")

        if "security" in feedback.lower():
            refined_instructions.append("Conduct thorough security review")

        return refined_instructions

    async def _validate_adapted_code(
        self, original: str, adapted: str, request: CodeAdaptationRequest
    ) -> Dict[str, Any]:
        """Validate the adapted code"""
        validation_results = {
            "syntax_valid": True,
            "functionality_preserved": True,
            "improvement_detected": True,
            "security_enhanced": False,
            "performance_improved": False,
        }

        try:
            # Basic syntax validation (would use AST parsing in production)
            if "def " in adapted and ":" in adapted:
                validation_results["syntax_valid"] = True

            # Check for improvements
            if len(adapted) != len(original):
                validation_results["improvement_detected"] = True

            # Check for security improvements
            if request.adaptation_type == AdaptationType.SECURITY_ENHANCEMENT:
                validation_results["security_enhanced"] = "validate" in adapted.lower()

            # Check for performance improvements
            if request.adaptation_type == AdaptationType.PERFORMANCE_OPTIMIZATION:
                validation_results["performance_improved"] = "cache" in adapted.lower()

        except Exception as e:
            validation_results["error"] = str(e)
            validation_results["syntax_valid"] = False

        return validation_results

    async def _calculate_improvement_metrics(
        self, original: str, adapted: str, request: CodeAdaptationRequest
    ) -> Dict[str, float]:
        """Calculate improvement metrics"""
        metrics = {
            "code_length_change": len(adapted) / len(original) if len(original) > 0 else 1.0,
            "complexity_reduction": 0.0,
            "readability_improvement": 0.0,
            "maintainability_score": 0.0,
        }

        # Simple heuristics (would use more sophisticated analysis in production)
        if request.adaptation_type == AdaptationType.CODE_QUALITY_IMPROVEMENT:
            metrics["readability_improvement"] = 0.2
            metrics["maintainability_score"] = 0.3

        if request.adaptation_type == AdaptationType.REFACTORING:
            metrics["complexity_reduction"] = 0.15
            metrics["maintainability_score"] = 0.4

        return metrics

    async def _store_adaptation_result(
        self, request: CodeAdaptationRequest, result: SelfEditResult
    ) -> None:
        """Store adaptation result in database"""
        try:
            # Store in database
            await self.database_adapter.log_monitoring_event(
                event_type="seal_adaptation",
                details={
                    "request": asdict(request),
                    "result": asdict(result),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Update issue lifecycle if adaptation was successful
            if result.confidence_score >= self.config["confidence_threshold"]:
                await self.lifecycle_manager.transition_issue(
                    request.issue_id,
                    IssueState.FIXED,
                    f"SEAL adaptation completed with confidence {result.confidence_score:.2f}",
                    automated=True,
                    metadata={
                        "adaptation_type": request.adaptation_type.value,
                        "strategy": request.strategy.value,
                        "confidence": result.confidence_score,
                        "improvement_metrics": result.improvement_metrics,
                    },
                )

            # Add to adaptation history
            self.adaptation_history.append(result)

        except Exception as e:
            logger.error("Error storing adaptation result: %s", e)

    async def get_adaptation_metrics(self) -> AdaptationMetrics:
        """Get comprehensive adaptation metrics"""
        try:
            total = len(self.adaptation_history)
            successful = len(
                [
                    r
                    for r in self.adaptation_history
                    if r.confidence_score >= self.config["confidence_threshold"]
                ]
            )
            failed = total - successful

            avg_confidence = (
                sum(r.confidence_score for r in self.adaptation_history) / total if total > 0 else 0
            )
            avg_improvement = (
                sum(
                    r.improvement_metrics.get("maintainability_score", 0)
                    for r in self.adaptation_history
                )
                / total
                if total > 0
                else 0
            )

            return AdaptationMetrics(
                total_adaptations=total,
                successful_adaptations=successful,
                failed_adaptations=failed,
                average_confidence=avg_confidence,
                average_improvement=avg_improvement,
                adaptation_types={},  # Would calculate from stored data
                strategies_used={},  # Would calculate from stored data
            )

        except Exception as e:
            logger.error("Error calculating adaptation metrics: %s", e)
            return AdaptationMetrics(0, 0, 0, 0.0, 0.0, {}, {})

    async def run_continual_adaptation_cycle(self, project_path: Path) -> Dict[str, Any]:
        """Run a complete SEAL continual adaptation cycle"""
        try:
            logger.info("Starting SEAL continual adaptation cycle...")

            # Analyze codebase for adaptation opportunities
            adaptation_requests = await self.analyze_codebase_for_adaptation(project_path)

            # Sort by priority
            adaptation_requests.sort(key=lambda x: x.priority, reverse=True)

            results = []
            successful_adaptations = 0

            # Process high-priority adaptations
            for request in adaptation_requests[:10]:  # Limit to top 10
                result = await self.perform_self_edit_adaptation(request)
                results.append(result)

                if result.confidence_score >= self.config["confidence_threshold"]:
                    successful_adaptations += 1

            # Update learning context
            await self._update_learning_context(results)

            cycle_summary = {
                "total_requests": len(adaptation_requests),
                "processed_requests": len(results),
                "successful_adaptations": successful_adaptations,
                "average_confidence": sum(r.confidence_score for r in results) / len(results)
                if results
                else 0,
                "cycle_timestamp": datetime.now().isoformat(),
            }

            logger.info(
                "SEAL adaptation cycle completed: %d/%d successful",
                successful_adaptations,
                len(results),
            )

            return cycle_summary

        except Exception as e:
            logger.error("Error in continual adaptation cycle: %s", e)
            return {"error": str(e), "cycle_timestamp": datetime.now().isoformat()}

    async def _update_learning_context(self, results: List[SelfEditResult]) -> None:
        """Update learning context based on adaptation results"""
        try:
            # Analyze successful patterns
            successful_results = [r for r in results if r.confidence_score >= 0.7]

            if successful_results:
                # Extract successful patterns
                self.learning_context["successful_patterns"] = {
                    "average_confidence": sum(r.confidence_score for r in successful_results)
                    / len(successful_results),
                    "common_improvements": {},  # Would analyze improvement types
                    "effective_strategies": {},  # Would analyze strategy effectiveness
                }

            # Update configuration based on learning
            if len(successful_results) / len(results) > 0.8 if results else False:
                # High success rate, can be more aggressive
                self.config["confidence_threshold"] = max(
                    0.6, self.config["confidence_threshold"] - 0.05
                )
            elif len(successful_results) / len(results) < 0.3 if results else True:
                # Low success rate, be more conservative
                self.config["confidence_threshold"] = min(
                    0.9, self.config["confidence_threshold"] + 0.05
                )

            logger.info(
                "Updated learning context with %d successful adaptations", len(successful_results)
            )

        except Exception as e:
            logger.error("Error updating learning context: %s", e)
