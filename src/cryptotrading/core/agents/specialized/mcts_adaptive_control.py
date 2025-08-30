"""
Adaptive Control and Dynamic Parameters for MCTS
Implements convergence detection and dynamic exploration adjustment
"""
import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Tracks convergence metrics for adaptive control"""

    window_size: int = 100
    value_history: deque = field(default_factory=lambda: deque(maxlen=100))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=100))
    best_action_changes: int = 0
    last_best_action: Optional[str] = None
    convergence_threshold: float = 0.01
    stability_threshold: float = 0.005
    min_iterations: int = 200

    def update(self, value: float, confidence: float, best_action: str, iteration: int):
        """Update convergence metrics"""
        self.value_history.append(value)
        self.confidence_history.append(confidence)

        # Track best action changes
        if self.last_best_action != best_action:
            self.best_action_changes += 1
            self.last_best_action = best_action

    def check_convergence(self, iteration: int) -> Tuple[bool, str]:
        """Check if MCTS has converged"""
        if iteration < self.min_iterations:
            return False, "minimum_iterations_not_reached"

        if len(self.value_history) < self.window_size:
            return False, "insufficient_history"

        # Calculate value stability
        recent_values = list(self.value_history)[-50:]  # Last 50 iterations
        if len(recent_values) < 10:
            return False, "insufficient_recent_data"

        value_std = statistics.stdev(recent_values)
        value_mean = statistics.mean(recent_values)

        # Relative stability check
        relative_std = value_std / abs(value_mean) if abs(value_mean) > 1e-6 else value_std

        # Confidence stability
        recent_confidence = list(self.confidence_history)[-50:]
        confidence_trend = self._calculate_trend(recent_confidence)

        # Check convergence criteria
        value_stable = relative_std < self.convergence_threshold
        confidence_stable = abs(confidence_trend) < self.stability_threshold

        if value_stable and confidence_stable:
            return True, "value_and_confidence_stable"
        elif value_stable:
            return False, "value_stable_confidence_changing"
        elif confidence_stable:
            return False, "confidence_stable_value_changing"
        else:
            return False, "neither_stable"

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values using linear regression slope"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))

        # Calculate linear regression slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_convergence_confidence(self) -> float:
        """Get confidence in convergence assessment"""
        if len(self.value_history) < self.window_size // 2:
            return 0.0

        recent_values = list(self.value_history)[-20:]
        if len(recent_values) < 5:
            return 0.0

        # Stability-based confidence
        value_std = statistics.stdev(recent_values)
        value_mean = statistics.mean(recent_values)
        relative_std = value_std / abs(value_mean) if abs(value_mean) > 1e-6 else value_std

        # Convert stability to confidence (inverse relationship)
        stability_confidence = max(0, 1 - (relative_std / self.convergence_threshold))

        # Action stability confidence
        action_changes_ratio = self.best_action_changes / len(self.value_history)
        action_confidence = max(0, 1 - action_changes_ratio * 2)

        return min(stability_confidence * action_confidence, 1.0)


@dataclass
class DynamicExplorationParams:
    """Manages dynamic exploration parameters"""

    initial_c_param: float = 1.4
    min_c_param: float = 0.5
    max_c_param: float = 2.5
    current_c_param: float = 1.4

    # Adaptive parameters
    exploration_decay_rate: float = 0.95
    exploitation_boost_rate: float = 1.05
    convergence_sensitivity: float = 0.1

    # Phase tracking
    exploration_phase: bool = True
    phase_switch_threshold: float = 0.3

    def update(
        self, convergence_metrics: ConvergenceMetrics, iteration: int, total_iterations: int
    ) -> float:
        """Update exploration parameter based on search progress"""

        # Calculate search progress
        progress = iteration / total_iterations if total_iterations > 0 else 0

        # Get convergence confidence
        convergence_confidence = convergence_metrics.get_convergence_confidence()

        # Determine if we should switch from exploration to exploitation
        if self.exploration_phase and convergence_confidence > self.phase_switch_threshold:
            self.exploration_phase = False
            logger.info(f"Switching to exploitation phase at iteration {iteration}")

        # Update c_param based on phase and convergence
        if self.exploration_phase:
            # Early exploration: maintain higher exploration
            target_c_param = self.initial_c_param * (1 + 0.5 * (1 - progress))
        else:
            # Exploitation phase: reduce exploration as we converge
            exploitation_factor = 1 - (convergence_confidence * self.convergence_sensitivity)
            target_c_param = self.initial_c_param * exploitation_factor

        # Apply bounds
        target_c_param = max(self.min_c_param, min(self.max_c_param, target_c_param))

        # Smooth transition
        adjustment_rate = 0.1  # How fast to adjust
        self.current_c_param = (
            self.current_c_param * (1 - adjustment_rate) + target_c_param * adjustment_rate
        )

        return self.current_c_param

    def get_rave_weight(self, convergence_confidence: float) -> float:
        """Calculate RAVE weighting based on convergence"""
        # Higher RAVE weight when we're less converged (more exploration)
        base_weight = 0.5
        convergence_adjustment = (1 - convergence_confidence) * 0.3
        return min(base_weight + convergence_adjustment, 0.8)


class AdaptiveIterationController:
    """Controls MCTS iterations adaptively based on convergence"""

    def __init__(
        self,
        min_iterations: int = 100,
        max_iterations: int = 10000,
        convergence_window: int = 50,
        early_stop_confidence: float = 0.95,
    ):
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.convergence_window = convergence_window
        self.early_stop_confidence = early_stop_confidence

        self.convergence_metrics = ConvergenceMetrics(window_size=convergence_window)
        self.dynamic_params = DynamicExplorationParams()

        # Iteration control
        self.current_iteration = 0
        self.should_continue = True
        self.convergence_reason = ""

        # Performance tracking
        self.start_time = time.time()
        self.iteration_times = deque(maxlen=100)

    def should_continue_search(
        self, current_value: float, current_confidence: float, best_action: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Determine if MCTS search should continue"""

        self.current_iteration += 1
        iteration_start = time.time()

        # Update convergence metrics
        self.convergence_metrics.update(
            current_value, current_confidence, best_action, self.current_iteration
        )

        # Update dynamic parameters
        new_c_param = self.dynamic_params.update(
            self.convergence_metrics, self.current_iteration, self.max_iterations
        )

        # Check stopping conditions

        # 1. Maximum iterations reached
        if self.current_iteration >= self.max_iterations:
            self.should_continue = False
            self.convergence_reason = "max_iterations_reached"

        # 2. Early convergence (only after minimum iterations)
        elif self.current_iteration >= self.min_iterations:
            converged, reason = self.convergence_metrics.check_convergence(self.current_iteration)
            convergence_confidence = self.convergence_metrics.get_convergence_confidence()

            if converged and convergence_confidence >= self.early_stop_confidence:
                self.should_continue = False
                self.convergence_reason = f"early_convergence_{reason}"
            elif self.current_iteration > self.max_iterations * 0.8:
                # In final 20%, be more lenient about stopping
                if convergence_confidence >= self.early_stop_confidence * 0.8:
                    self.should_continue = False
                    self.convergence_reason = "late_stage_convergence"

        # Track iteration timing
        iteration_time = time.time() - iteration_start
        self.iteration_times.append(iteration_time)

        # Prepare status information
        status = {
            "iteration": self.current_iteration,
            "should_continue": self.should_continue,
            "convergence_reason": self.convergence_reason,
            "convergence_confidence": self.convergence_metrics.get_convergence_confidence(),
            "exploration_param": new_c_param,
            "exploration_phase": self.dynamic_params.exploration_phase,
            "estimated_time_remaining": self._estimate_time_remaining(),
            "progress": self.current_iteration / self.max_iterations,
        }

        return self.should_continue, self.convergence_reason, status

    def _estimate_time_remaining(self) -> float:
        """Estimate remaining time based on current performance"""
        if len(self.iteration_times) < 5:
            return 0.0

        avg_iteration_time = statistics.mean(list(self.iteration_times)[-20:])
        remaining_iterations = self.max_iterations - self.current_iteration

        # Account for potential early stopping
        convergence_confidence = self.convergence_metrics.get_convergence_confidence()
        if convergence_confidence > 0.5:
            # Likely to stop early
            estimated_remaining = remaining_iterations * (1 - convergence_confidence * 0.5)
        else:
            estimated_remaining = remaining_iterations

        return avg_iteration_time * estimated_remaining

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current adaptive parameters"""
        return {
            "c_param": self.dynamic_params.current_c_param,
            "rave_weight": self.dynamic_params.get_rave_weight(
                self.convergence_metrics.get_convergence_confidence()
            ),
            "exploration_phase": self.dynamic_params.exploration_phase,
            "convergence_confidence": self.convergence_metrics.get_convergence_confidence(),
            "iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
        }

    def reset(self):
        """Reset controller for new search"""
        self.convergence_metrics = ConvergenceMetrics(window_size=self.convergence_window)
        self.dynamic_params = DynamicExplorationParams()
        self.current_iteration = 0
        self.should_continue = True
        self.convergence_reason = ""
        self.start_time = time.time()
        self.iteration_times.clear()

    def get_final_report(self) -> Dict[str, Any]:
        """Generate final convergence report"""
        total_time = time.time() - self.start_time
        avg_iteration_time = (
            statistics.mean(list(self.iteration_times)) if self.iteration_times else 0
        )

        return {
            "total_iterations": self.current_iteration,
            "convergence_reason": self.convergence_reason,
            "total_time": total_time,
            "average_iteration_time": avg_iteration_time,
            "iterations_per_second": self.current_iteration / total_time if total_time > 0 else 0,
            "convergence_confidence": self.convergence_metrics.get_convergence_confidence(),
            "final_exploration_param": self.dynamic_params.current_c_param,
            "exploration_phase_ended": not self.dynamic_params.exploration_phase,
            "best_action_changes": self.convergence_metrics.best_action_changes,
            "efficiency_gain": max(
                0, (self.max_iterations - self.current_iteration) / self.max_iterations
            ),
        }


class MemoryOptimizedMCTSNode:
    """Memory-optimized MCTS node with compression and pruning"""

    def __init__(self, state_hash: str, action: Optional[Dict[str, Any]] = None):
        # Use hash instead of full state for memory efficiency
        self.state_hash = state_hash
        self.action = action
        self.parent = None
        self.children = {}  # Use dict for O(1) lookup

        # Core MCTS data
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []

        # RAVE data with lazy initialization
        self._rave_visits = None
        self._rave_values = None

        # Memory optimization flags
        self.is_terminal = False
        self.is_pruned = False
        self.last_accessed = time.time()

    @property
    def rave_visits(self):
        if self._rave_visits is None:
            self._rave_visits = {}
        return self._rave_visits

    @property
    def rave_values(self):
        if self._rave_values is None:
            self._rave_values = {}
        return self._rave_values

    def add_child(self, action_key: str, child_node: "MemoryOptimizedMCTSNode"):
        """Add child with memory tracking"""
        self.children[action_key] = child_node
        child_node.parent = self
        self.last_accessed = time.time()

    def prune_subtree(self, keep_best_n: int = 3):
        """Prune less promising subtrees to save memory"""
        if len(self.children) <= keep_best_n:
            return

        # Sort children by promise (UCB1 value)
        child_scores = []
        for action_key, child in self.children.items():
            if child.visits > 0:
                ucb_score = (child.value / child.visits) + math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
            else:
                ucb_score = float("inf")
            child_scores.append((ucb_score, action_key, child))

        # Keep top N children
        child_scores.sort(reverse=True)
        children_to_keep = {
            action_key: child for _, action_key, child in child_scores[:keep_best_n]
        }

        # Mark pruned children
        for action_key, child in self.children.items():
            if action_key not in children_to_keep:
                child.is_pruned = True
                child._recursive_cleanup()

        self.children = children_to_keep

    def _recursive_cleanup(self):
        """Recursively cleanup pruned subtree"""
        for child in self.children.values():
            child._recursive_cleanup()
        self.children.clear()
        self._rave_visits = None
        self._rave_values = None

    def get_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage of subtree"""

        def count_nodes(node):
            count = 1
            for child in node.children.values():
                if not child.is_pruned:
                    count += count_nodes(child)
            return count

        node_count = count_nodes(self)
        estimated_bytes = node_count * 200  # Rough estimate per node

        return {
            "node_count": node_count,
            "estimated_bytes": estimated_bytes,
            "estimated_mb": estimated_bytes / (1024 * 1024),
        }
