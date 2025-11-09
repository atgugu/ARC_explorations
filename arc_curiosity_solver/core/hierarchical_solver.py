"""
Hierarchical Solver Architecture

Three-tier cognitive architecture:
1. Generator: Strategic/long-term (what to explore)
2. Workspace: Tactical/working memory (which hypotheses to think about)
3. Navigator: Operational/executive (where to explore)
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


@dataclass
class Task:
    """Represents an ARC task."""
    id: str
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    test_input: np.ndarray
    schema_family: str = "unknown"
    difficulty: float = 0.5
    curiosity_score: float = 0.0


@dataclass
class WorkspaceItem:
    """Item in the workspace (hypothesis being considered)."""
    hypothesis: Any
    fit_score: float
    curiosity_score: float
    stability_score: float
    activation: float
    priority: float = 0.0

    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


class Generator:
    """
    Strategic curiosity: decides what types of problems/schemas to explore.

    Responsibilities:
    - Task selection and curriculum
    - Schema family exploration
    - Multi-armed bandit for learning progress
    """

    def __init__(self,
                 exploration_bonus: float = 1.0,
                 ucb_c: float = 1.41):
        self.exploration_bonus = exploration_bonus
        self.ucb_c = ucb_c

        # Track performance by schema family
        self.schema_stats = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'learning_progress': [],
            'avg_reward': 0.0
        })

        self.total_attempts = 0
        self.task_history = []

    def select_task(self,
                   available_tasks: List[Task],
                   method: str = 'ucb') -> Optional[Task]:
        """
        Select next task to work on based on curiosity and learning progress.

        Args:
            available_tasks: List of available tasks
            method: 'ucb', 'epsilon_greedy', or 'curiosity'

        Returns:
            Selected task
        """
        if not available_tasks:
            return None

        if method == 'ucb':
            return self._select_ucb(available_tasks)
        elif method == 'epsilon_greedy':
            return self._select_epsilon_greedy(available_tasks)
        elif method == 'curiosity':
            return self._select_by_curiosity(available_tasks)
        else:
            return np.random.choice(available_tasks)

    def _select_ucb(self, tasks: List[Task]) -> Task:
        """Select task using Upper Confidence Bound."""
        self.total_attempts += 1

        best_score = -np.inf
        best_task = tasks[0]

        for task in tasks:
            stats = self.schema_stats[task.schema_family]
            n_attempts = stats['attempts']

            if n_attempts == 0:
                # Explore unvisited schema families first
                return task

            # UCB score: avg_reward + c * sqrt(ln(N) / n)
            avg_reward = stats['avg_reward']
            exploration_bonus = self.ucb_c * np.sqrt(
                np.log(self.total_attempts) / n_attempts
            )

            ucb_score = avg_reward + exploration_bonus + task.curiosity_score

            if ucb_score > best_score:
                best_score = ucb_score
                best_task = task

        return best_task

    def _select_epsilon_greedy(self,
                              tasks: List[Task],
                              epsilon: float = 0.2) -> Task:
        """Epsilon-greedy selection."""
        if np.random.random() < epsilon:
            # Explore
            return np.random.choice(tasks)
        else:
            # Exploit
            best_task = max(tasks, key=lambda t: (
                self.schema_stats[t.schema_family]['avg_reward'] +
                t.curiosity_score
            ))
            return best_task

    def _select_by_curiosity(self, tasks: List[Task]) -> Task:
        """Select task with highest curiosity score."""
        return max(tasks, key=lambda t: t.curiosity_score)

    def update_statistics(self,
                         task: Task,
                         success: bool,
                         performance: float):
        """
        Update statistics after attempting a task.

        Args:
            task: The attempted task
            success: Whether it was solved
            performance: Performance metric (e.g., accuracy)
        """
        stats = self.schema_stats[task.schema_family]
        stats['attempts'] += 1
        stats['successes'] += int(success)

        # Update average reward
        n = stats['attempts']
        stats['avg_reward'] = (
            (stats['avg_reward'] * (n - 1) + performance) / n
        )

        # Track learning progress
        stats['learning_progress'].append(performance)

        self.task_history.append({
            'task_id': task.id,
            'schema': task.schema_family,
            'success': success,
            'performance': performance
        })

    def get_learning_progress(self, schema: str, window: int = 5) -> float:
        """Compute recent learning progress for a schema family."""
        stats = self.schema_stats[schema]
        progress_history = stats['learning_progress']

        if len(progress_history) < 2:
            return 0.0

        # Compare recent window to previous window
        recent = progress_history[-window:] if len(progress_history) >= window else progress_history

        if len(progress_history) > window:
            previous = progress_history[-2*window:-window]
        else:
            previous = progress_history[:len(recent)]

        if len(previous) == 0:
            return 0.0

        return np.mean(recent) - np.mean(previous)


class Workspace:
    """
    Tactical curiosity: manages active hypotheses and attention.

    Responsibilities:
    - Hypothesis competition and selection
    - Working memory management (7±2 items)
    - Attention allocation
    """

    def __init__(self,
                 capacity: int = 7,
                 fit_weight: float = 1.0,
                 curiosity_weight: float = 0.5,
                 stability_weight: float = 0.3):
        self.capacity = capacity
        self.fit_weight = fit_weight
        self.curiosity_weight = curiosity_weight
        self.stability_weight = stability_weight

        # Priority queue for active hypotheses
        self.active_items: List[WorkspaceItem] = []
        self.item_history = []

    def add_hypothesis(self,
                      hypothesis: Any,
                      fit_score: float,
                      curiosity_score: float,
                      stability_score: float) -> bool:
        """
        Add a hypothesis to the workspace.

        Returns:
            True if added, False if rejected
        """
        # Compute priority
        priority = (
            self.fit_weight * fit_score +
            self.curiosity_weight * curiosity_score -
            self.stability_weight * (1.0 - stability_score)  # Penalize instability
        )

        item = WorkspaceItem(
            hypothesis=hypothesis,
            fit_score=fit_score,
            curiosity_score=curiosity_score,
            stability_score=stability_score,
            activation=1.0,
            priority=priority
        )

        # Add to workspace
        heapq.heappush(self.active_items, item)

        # Enforce capacity limit
        while len(self.active_items) > self.capacity:
            removed = heapq.heappop(self.active_items)
            self.item_history.append(removed)

        return True

    def get_top_k(self, k: int = 2) -> List[WorkspaceItem]:
        """Get top k hypotheses by priority."""
        # Get k highest priority items without removing them
        return heapq.nlargest(k, self.active_items)

    def update_activations(self, decay_rate: float = 0.1):
        """Decay activations over time (temporal dynamics)."""
        for item in self.active_items:
            item.activation *= (1.0 - decay_rate)

        # Remove items with very low activation
        self.active_items = [
            item for item in self.active_items
            if item.activation > 0.01
        ]
        heapq.heapify(self.active_items)

    def apply_lateral_inhibition(self, similarity_fn: callable):
        """
        Similar hypotheses inhibit each other.

        Args:
            similarity_fn: Function to compute similarity between hypotheses
        """
        if len(self.active_items) < 2:
            return

        # Compute pairwise similarities
        n = len(self.active_items)
        for i in range(n):
            for j in range(i+1, n):
                sim = similarity_fn(
                    self.active_items[i].hypothesis,
                    self.active_items[j].hypothesis
                )

                # Inhibit the lower priority item
                if self.active_items[i].priority > self.active_items[j].priority:
                    self.active_items[j].activation *= (1.0 - 0.3 * sim)
                else:
                    self.active_items[i].activation *= (1.0 - 0.3 * sim)

        heapq.heapify(self.active_items)

    def get_statistics(self) -> Dict[str, Any]:
        """Get workspace statistics."""
        if not self.active_items:
            return {
                'n_active': 0,
                'mean_fit': 0.0,
                'mean_curiosity': 0.0,
                'mean_stability': 0.0,
                'capacity_usage': 0.0
            }

        return {
            'n_active': len(self.active_items),
            'mean_fit': np.mean([item.fit_score for item in self.active_items]),
            'mean_curiosity': np.mean([item.curiosity_score for item in self.active_items]),
            'mean_stability': np.mean([item.stability_score for item in self.active_items]),
            'capacity_usage': len(self.active_items) / self.capacity,
            'top_priority': self.active_items[0].priority if self.active_items else 0.0
        }


class Navigator:
    """
    Operational curiosity: guides moment-to-moment exploration.

    Responsibilities:
    - Stability-aware search
    - Basin and edge curiosity
    - Trajectory control (avoid chaos, exploit stable regions)
    """

    def __init__(self,
                 stability_threshold: float = 0.3,
                 novelty_weight: float = 0.5,
                 progress_weight: float = 0.5):
        self.stability_threshold = stability_threshold
        self.novelty_weight = novelty_weight
        self.progress_weight = progress_weight

        # Track explored regions
        self.visited_states: Set[int] = set()
        self.state_statistics = defaultdict(lambda: {
            'visits': 0,
            'stability': 0.0,
            'rewards': []
        })

        self.trajectory = []

    def compute_basin_curiosity(self,
                               state_id: int,
                               stability_variance: float,
                               novelty: float,
                               learning_progress: float) -> float:
        """
        Compute curiosity for exploring a basin (stable region).

        C_basin(b) = exp(-Var_stab(b)) · Novelty(b) · LP(b)

        Args:
            state_id: Identifier for the state/basin
            stability_variance: Variance of stability in this region
            novelty: How novel/unexplored this region is
            learning_progress: Recent learning progress in this region

        Returns:
            Basin curiosity score
        """
        stability_bonus = np.exp(-stability_variance)
        curiosity = stability_bonus * novelty * (1.0 + learning_progress)

        return float(curiosity)

    def compute_edge_curiosity(self,
                              from_state: int,
                              to_state: int,
                              info_gain: float,
                              surprise: float,
                              coverage_gap: float) -> float:
        """
        Compute curiosity for exploring a transition edge.

        C_{i→j} = ω₁·IG_{i→j} + ω₂·Surprise_{i→j} + ω₃·CoverageGap

        Args:
            from_state: Source state
            to_state: Target state
            info_gain: Expected information gain
            surprise: Surprise of this transition
            coverage_gap: How underexplored this transition type is

        Returns:
            Edge curiosity score
        """
        omega1, omega2, omega3 = 0.4, 0.3, 0.3

        curiosity = (
            omega1 * info_gain +
            omega2 * surprise +
            omega3 * coverage_gap
        )

        return float(curiosity)

    def should_explore(self,
                      state_id: int,
                      instability: float,
                      curiosity: float) -> bool:
        """
        Decide whether to explore a state based on stability and curiosity.

        Args:
            state_id: State identifier
            instability: Instability measure
            curiosity: Curiosity score

        Returns:
            True if should explore
        """
        # Don't explore highly unstable regions unless curiosity is very high
        if instability > self.stability_threshold:
            # Only explore if curiosity justifies the risk
            return curiosity > 2.0 * instability

        # For stable regions, explore if novel or curious
        stats = self.state_statistics[state_id]
        novelty = 1.0 / (1.0 + stats['visits'])

        return novelty > 0.3 or curiosity > 0.5

    def record_transition(self,
                         from_state: int,
                         to_state: int,
                         reward: float,
                         stability: float):
        """Record a state transition."""
        self.visited_states.add(to_state)

        stats = self.state_statistics[to_state]
        stats['visits'] += 1
        stats['rewards'].append(reward)

        # Update stability estimate
        n = stats['visits']
        stats['stability'] = (
            (stats['stability'] * (n - 1) + stability) / n
        )

        self.trajectory.append({
            'from': from_state,
            'to': to_state,
            'reward': reward,
            'stability': stability
        })

    def get_unexplored_coverage(self, all_states: List[int]) -> float:
        """Compute fraction of states not yet explored."""
        if not all_states:
            return 0.0

        n_unexplored = sum(1 for s in all_states if s not in self.visited_states)
        return n_unexplored / len(all_states)

    def compute_trajectory_stability(self, window: int = 10) -> float:
        """
        Compute stability of recent trajectory.

        Returns:
            Stability score (lower variance = more stable)
        """
        if len(self.trajectory) < window:
            return 0.5  # Default moderate stability

        recent = self.trajectory[-window:]
        stabilities = [t['stability'] for t in recent]

        return float(1.0 - np.std(stabilities))  # High std = low stability


class HierarchicalSolver:
    """
    Integrated hierarchical solver combining Generator, Workspace, and Navigator.
    """

    def __init__(self,
                 workspace_capacity: int = 7,
                 exploration_bonus: float = 1.0):
        self.generator = Generator(exploration_bonus=exploration_bonus)
        self.workspace = Workspace(capacity=workspace_capacity)
        self.navigator = Navigator()

        self.current_task = None
        self.solve_history = []

    def select_and_load_task(self, available_tasks: List[Task]) -> Optional[Task]:
        """Generator selects next task to work on."""
        task = self.generator.select_task(available_tasks, method='ucb')
        self.current_task = task
        return task

    def add_hypothesis_to_workspace(self,
                                   hypothesis: Any,
                                   fit: float,
                                   curiosity: float,
                                   stability: float):
        """Add hypothesis to workspace for consideration."""
        self.workspace.add_hypothesis(hypothesis, fit, curiosity, stability)

    def get_top_predictions(self, k: int = 2) -> List[Any]:
        """Get top k predictions from workspace."""
        top_items = self.workspace.get_top_k(k)
        return [item.hypothesis for item in top_items]

    def update_after_solve(self,
                          success: bool,
                          performance: float):
        """Update all components after solving attempt."""
        if self.current_task is not None:
            self.generator.update_statistics(
                self.current_task,
                success,
                performance
            )

        # Decay workspace activations
        self.workspace.update_activations()

        # Record solve attempt
        self.solve_history.append({
            'task_id': self.current_task.id if self.current_task else None,
            'success': success,
            'performance': performance,
            'workspace_stats': self.workspace.get_statistics()
        })

    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        return {
            'generator': {
                'total_attempts': self.generator.total_attempts,
                'n_schemas_explored': len(self.generator.schema_stats)
            },
            'workspace': self.workspace.get_statistics(),
            'navigator': {
                'n_visited_states': len(self.navigator.visited_states),
                'trajectory_length': len(self.navigator.trajectory),
                'recent_stability': self.navigator.compute_trajectory_stability()
            }
        }
