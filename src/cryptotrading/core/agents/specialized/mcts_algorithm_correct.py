"""
Algorithmically Correct MCTS Implementation
Following the mathematical foundations from Browne et al. (2012) and Silver et al. (2016)
"""
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """
    Algorithmically correct MCTS node implementation
    """

    state: Any
    parent: Optional["MCTSNode"] = None
    action: Optional[Any] = None

    # Core MCTS statistics
    visits: int = 0
    total_value: float = 0.0

    # Children management
    children: Dict[Any, "MCTSNode"] = field(default_factory=dict)
    untried_actions: List[Any] = field(default_factory=list)

    # RAVE statistics (All-Moves-As-First)
    rave_visits: Dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    rave_values: Dict[Any, float] = field(default_factory=lambda: defaultdict(float))

    # Virtual loss for parallel MCTS
    virtual_loss: int = 0

    # Prior probability for actions (e.g., from policy network)
    action_priors: Dict[Any, float] = field(default_factory=dict)

    # Progressive widening parameters
    pw_alpha: float = 0.5  # Controls widening rate
    pw_k: float = 1.0  # Initial widening constant

    @property
    def q_value(self) -> float:
        """Average value (Q-value) of this node"""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def effective_visits(self) -> int:
        """Visits including virtual loss for parallel MCTS"""
        return self.visits + self.virtual_loss

    def uct_value(self, c_param: float = math.sqrt(2), parent_visits: int = None) -> float:
        """
        Calculate UCT (Upper Confidence bounds applied to Trees) value
        UCT = Q + c * sqrt(ln(N_parent) / N_child)
        """
        if self.visits == 0:
            return float("inf")

        if parent_visits is None:
            parent_visits = self.parent.visits if self.parent else 1

        exploitation = self.q_value
        exploration = c_param * math.sqrt(math.log(parent_visits) / self.visits)

        return exploitation + exploration

    def puct_value(self, c_param: float = 1.0, parent_visits: int = None) -> float:
        """
        Calculate PUCT (Polynomial Upper Confidence Trees) value - used in AlphaGo
        PUCT = Q + c * P * sqrt(N_parent) / (1 + N_child)
        where P is the prior probability
        """
        if parent_visits is None:
            parent_visits = self.parent.visits if self.parent else 1

        prior = (
            self.action_priors.get(self.action, 1.0 / len(self.parent.children))
            if self.action
            else 0
        )

        exploitation = self.q_value
        exploration = c_param * prior * math.sqrt(parent_visits) / (1 + self.visits)

        return exploitation + exploration

    def rave_adjusted_value(
        self,
        c_param: float = math.sqrt(2),
        parent_visits: int = None,
        rave_c: float = 1.0,
        equiv_param: float = 1000,
    ) -> float:
        """
        Calculate RAVE-adjusted UCT value
        Uses the schedule from Gelly & Silver (2007)
        """
        if self.visits == 0:
            return float("inf")

        # Standard UCT value
        uct = self.uct_value(c_param, parent_visits)

        # RAVE value
        action_key = self.action
        if action_key in self.parent.rave_visits and self.parent.rave_visits[action_key] > 0:
            rave_q = self.parent.rave_values[action_key] / self.parent.rave_visits[action_key]

            # RAVE exploration bonus
            rave_exploration = rave_c * math.sqrt(
                math.log(parent_visits) / self.parent.rave_visits[action_key]
            )
            rave_value = rave_q + rave_exploration

            # Weighted average using equivalence parameter
            beta = math.sqrt(equiv_param / (3 * self.visits + equiv_param))

            return beta * rave_value + (1 - beta) * uct

        return uct

    def add_virtual_loss(self, loss: int = 1):
        """Add virtual loss for parallel MCTS"""
        self.virtual_loss += loss

    def remove_virtual_loss(self, loss: int = 1):
        """Remove virtual loss after MCTS iteration completes"""
        self.virtual_loss = max(0, self.virtual_loss - loss)

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 0.0, use_rave: bool = False) -> Optional["MCTSNode"]:
        """
        Select best child for given exploration constant
        c_param = 0 means pure exploitation (for final selection)
        """
        if not self.children:
            return None

        if use_rave and c_param > 0:
            # Use RAVE-adjusted values during tree policy
            return max(
                self.children.values(), key=lambda n: n.rave_adjusted_value(c_param, self.visits)
            )
        else:
            # Standard UCT or pure exploitation
            return max(self.children.values(), key=lambda n: n.uct_value(c_param, self.visits))

    def robust_child(self) -> Optional["MCTSNode"]:
        """Select most visited child (robust selection for final move)"""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda n: n.visits)

    def secure_child(self) -> Optional["MCTSNode"]:
        """Select child with highest lower confidence bound (secure selection)"""
        if not self.children:
            return None

        def lcb(node: "MCTSNode", c: float = 1.0) -> float:
            if node.visits == 0:
                return float("-inf")
            return node.q_value - c * math.sqrt(math.log(self.visits) / node.visits)

        return max(self.children.values(), key=lambda n: lcb(n))

    def max_child(self) -> Optional["MCTSNode"]:
        """Select child with highest average value"""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda n: n.q_value)

    def update_rave(self, action_sequence: List[Any], value: float):
        """Update RAVE statistics for all actions in the sequence"""
        for action in action_sequence:
            if action != self.action:  # Don't double-count the actual action
                self.rave_visits[action] += 1
                self.rave_values[action] += value


class ProgressiveWidening:
    """
    Progressive Widening for continuous or large action spaces
    Based on Coulom (2007) and Chaslot et al. (2008)
    """

    def __init__(self, alpha: float = 0.5, k: float = 1.0, c: float = 1.0):
        """
        Initialize progressive widening parameters

        Args:
            alpha: Controls the widening rate (0.5 is common)
            k: Widening constant
            c: Constant multiplier
        """
        self.alpha = alpha
        self.k = k
        self.c = c

    def max_children(self, visits: int) -> int:
        """
        Calculate maximum number of children for a node with given visits
        max_children = c * k * visits^alpha
        """
        return int(self.c * self.k * math.pow(visits, self.alpha))

    def should_expand(self, node: MCTSNode) -> bool:
        """Check if node should expand a new child"""
        current_children = len(node.children)
        max_allowed = self.max_children(node.visits)
        return current_children < max_allowed and node.untried_actions


class MCTSAlgorithm:
    """
    Algorithmically correct MCTS implementation with all modern enhancements
    """

    def __init__(
        self,
        exploration_constant: float = math.sqrt(2),
        rave_c: float = 1.0,
        rave_equiv: float = 1000,
        use_rave: bool = True,
        use_progressive_widening: bool = False,
        pw_alpha: float = 0.5,
        pw_k: float = 1.0,
        virtual_loss: int = 1,
        selection_method: str = "robust",
    ):  # robust, secure, max
        """
        Initialize MCTS algorithm with parameters

        Args:
            exploration_constant: UCT exploration parameter (c)
            rave_c: RAVE exploration parameter
            rave_equiv: RAVE equivalence parameter (controls RAVE influence decay)
            use_rave: Whether to use RAVE
            use_progressive_widening: For large/continuous action spaces
            pw_alpha: Progressive widening alpha parameter
            pw_k: Progressive widening k parameter
            virtual_loss: Virtual loss for parallel MCTS
            selection_method: Method for final move selection
        """
        self.exploration_constant = exploration_constant
        self.rave_c = rave_c
        self.rave_equiv = rave_equiv
        self.use_rave = use_rave
        self.use_progressive_widening = use_progressive_widening
        self.progressive_widening = ProgressiveWidening(pw_alpha, pw_k)
        self.virtual_loss = virtual_loss
        self.selection_method = selection_method

        # Statistics
        self.iterations_completed = 0
        self.max_depth_reached = 0

    def search(
        self, root_state: Any, num_iterations: int, environment: Any, parallel: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run MCTS search from root state

        Returns:
            (best_action, statistics)
        """
        # Create root node
        root = MCTSNode(state=root_state, untried_actions=environment.get_actions(root_state))

        # Set up action priors if available
        if hasattr(environment, "get_action_priors"):
            root.action_priors = environment.get_action_priors(root_state)

        # Run MCTS iterations
        for i in range(num_iterations):
            # For parallel MCTS, we would launch multiple iterations here
            self._run_iteration(root, environment)
            self.iterations_completed += 1

        # Select best action based on selection method
        best_child = self._select_best_child(root)
        best_action = best_child.action if best_child else None

        # Gather statistics
        stats = self._gather_statistics(root)

        return best_action, stats

    def _run_iteration(self, root: MCTSNode, environment: Any):
        """Run a single MCTS iteration"""
        node = root
        state = root.state

        # Track path for backpropagation
        path = [node]
        action_sequence = []

        # 1. SELECTION - traverse tree using tree policy
        while not environment.is_terminal(state) and node.is_fully_expanded():
            # Apply virtual loss for parallel MCTS
            node.add_virtual_loss(self.virtual_loss)

            # Progressive widening check
            if self.use_progressive_widening and self.progressive_widening.should_expand(node):
                break

            # Select child using tree policy
            node = node.best_child(self.exploration_constant, self.use_rave)
            if node is None:
                break

            state = environment.apply_action(state, node.action)
            path.append(node)
            action_sequence.append(node.action)

        # 2. EXPANSION - add new child to tree
        if not environment.is_terminal(state) and node.untried_actions:
            # Choose action (could use prior probabilities here)
            if hasattr(environment, "get_action_priors") and node.action_priors:
                # Sample according to priors
                actions = list(node.action_priors.keys())
                probs = list(node.action_priors.values())
                action = np.random.choice(actions, p=probs)
            else:
                # Random selection
                action = random.choice(node.untried_actions)

            # Create new child
            new_state = environment.apply_action(state, action)
            child = MCTSNode(
                state=new_state,
                parent=node,
                action=action,
                untried_actions=environment.get_actions(new_state),
            )

            # Set up action priors for child
            if hasattr(environment, "get_action_priors"):
                child.action_priors = environment.get_action_priors(new_state)

            node.children[action] = child
            node.untried_actions.remove(action)

            # Add virtual loss
            child.add_virtual_loss(self.virtual_loss)

            path.append(child)
            action_sequence.append(action)
            state = new_state
            node = child

        # 3. SIMULATION - play out to terminal state
        simulation_actions = []
        depth = len(path)

        while not environment.is_terminal(state):
            actions = environment.get_actions(state)
            if not actions:
                break

            # Default policy (random)
            # In practice, this could be a learned policy or heuristic
            action = random.choice(actions)
            state = environment.apply_action(state, action)
            simulation_actions.append(action)
            depth += 1

        # Track maximum depth
        self.max_depth_reached = max(self.max_depth_reached, depth)

        # 4. EVALUATION - get value of terminal state
        value = environment.evaluate(state)

        # 5. BACKPROPAGATION - update statistics in path
        for node in reversed(path):
            node.visits += 1
            node.total_value += value

            # Remove virtual loss
            node.remove_virtual_loss(self.virtual_loss)

            # Update RAVE statistics
            if self.use_rave:
                node.update_rave(action_sequence + simulation_actions, value)

            # Flip value for opponent in two-player games
            if hasattr(environment, "is_two_player") and environment.is_two_player:
                value = -value

    def _select_best_child(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Select best child based on selection method"""
        if self.selection_method == "robust":
            return root.robust_child()
        elif self.selection_method == "secure":
            return root.secure_child()
        elif self.selection_method == "max":
            return root.max_child()
        else:
            # Default to robust
            return root.robust_child()

    def _gather_statistics(self, root: MCTSNode) -> Dict[str, Any]:
        """Gather search statistics"""
        stats = {
            "iterations": self.iterations_completed,
            "root_visits": root.visits,
            "max_depth": self.max_depth_reached,
            "num_children": len(root.children),
            "best_child_visits": 0,
            "best_child_value": 0,
            "action_visits": {},
        }

        # Get best child stats
        best_child = self._select_best_child(root)
        if best_child:
            stats["best_child_visits"] = best_child.visits
            stats["best_child_value"] = best_child.q_value

        # Action visit distribution
        for action, child in root.children.items():
            stats["action_visits"][str(action)] = child.visits

        return stats


class ParallelMCTS(MCTSAlgorithm):
    """
    Parallel MCTS implementation using virtual loss
    Based on Chaslot et al. (2008)
    """

    def __init__(self, num_threads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads

    async def search_parallel(
        self, root_state: Any, num_iterations: int, environment: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run parallel MCTS search

        Note: This is a simplified version. Real implementation would use
        threading or multiprocessing with shared memory for the tree.
        """
        import asyncio

        # Create root node
        root = MCTSNode(state=root_state, untried_actions=environment.get_actions(root_state))

        # Batch iterations
        batch_size = min(self.num_threads, num_iterations)

        for i in range(0, num_iterations, batch_size):
            # Launch parallel iterations
            tasks = []
            for j in range(min(batch_size, num_iterations - i)):
                # In real implementation, these would run in parallel
                # Here we simulate with async tasks
                task = self._run_iteration_async(root, environment)
                tasks.append(task)

            # Wait for all to complete
            await asyncio.gather(*tasks)
            self.iterations_completed += len(tasks)

        # Select best action
        best_child = self._select_best_child(root)
        best_action = best_child.action if best_child else None

        # Gather statistics
        stats = self._gather_statistics(root)
        stats["parallel_threads"] = self.num_threads

        return best_action, stats

    async def _run_iteration_async(self, root: MCTSNode, environment: Any):
        """Async wrapper for iteration (simulates parallel execution)"""
        # In real implementation, this would be in a separate thread/process
        self._run_iteration(root, environment)
