"""
Controller for stability-aware beam search over node paths.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import heapq


@dataclass
class SearchNode:
    """
    Node in the search tree.

    Attributes:
        path: List of node names in current path
        score: Current score
        artifacts: Accumulated artifacts
        depth: Path depth
        stability_penalty: Penalty for instability
    """
    path: List[str]
    score: float
    artifacts: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    stability_penalty: float = 0.0
    variance: float = 0.0

    def __lt__(self, other):
        """For priority queue - higher score is better."""
        return (self.score - self.stability_penalty) > (other.score - other.stability_penalty)

    @property
    def total_score(self) -> float:
        """Get total score including penalties."""
        return self.score - self.stability_penalty


class Controller:
    """
    Stability-aware controller using beam search.
    """

    def __init__(
        self,
        node_registry,
        edge_registry,
        beam_width: int = 5,
        max_depth: int = 10,
        stability_weight: float = 0.3,
        exploration_bonus: float = 0.1
    ):
        """
        Initialize controller.

        Args:
            node_registry: NodeRegistry for executing nodes
            edge_registry: EdgeRegistry for graph structure
            beam_width: Width of beam search
            max_depth: Maximum path depth
            stability_weight: Weight for stability penalty
            exploration_bonus: Bonus for exploring new paths
        """
        self.node_registry = node_registry
        self.edge_registry = edge_registry
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.stability_weight = stability_weight
        self.exploration_bonus = exploration_bonus

    def search_paths(
        self,
        start_nodes: List[str],
        task_data: Dict[str, Any],
        target_category: str = "reasoner",
        use_stability: bool = True
    ) -> List[SearchNode]:
        """
        Search for best paths using beam search.

        Args:
            start_nodes: Initial nodes to start from
            task_data: Task data to process
            target_category: Target node category to reach
            use_stability: Whether to use stability-aware scoring

        Returns:
            List of best search nodes (paths)
        """
        # Initialize beam with start nodes
        beam = []

        for node_name in start_nodes:
            # Execute start node
            output = self.node_registry.execute(node_name, task_data)

            search_node = SearchNode(
                path=[node_name],
                score=self._score_output(output),
                artifacts=output.artifacts.copy(),
                depth=1
            )

            beam.append(search_node)

        # Beam search
        for depth in range(1, self.max_depth):
            candidates = []

            for current in beam:
                # Get possible next nodes
                next_nodes = self._get_next_nodes(current.path[-1], target_category)

                for next_node in next_nodes:
                    # Execute next node with accumulated artifacts
                    combined_data = {**task_data, **current.artifacts}
                    output = self.node_registry.execute(next_node, combined_data)

                    # Create new search node
                    new_path = current.path + [next_node]
                    new_artifacts = {**current.artifacts, **output.artifacts}

                    new_node = SearchNode(
                        path=new_path,
                        score=current.score + self._score_output(output),
                        artifacts=new_artifacts,
                        depth=depth + 1
                    )

                    # Compute stability penalty if requested
                    if use_stability:
                        new_node.stability_penalty = self._compute_stability_penalty(new_path)

                    # Add exploration bonus for new combinations
                    if self._is_novel_path(new_path):
                        new_node.score += self.exploration_bonus

                    candidates.append(new_node)

            # Keep top-k candidates
            candidates.sort(reverse=True)
            beam = candidates[:self.beam_width]

            # Early exit if we reached target category
            if any(self._has_target_category(node.path, target_category) for node in beam):
                break

        return beam

    def _get_next_nodes(self, current_node: str, target_category: str) -> List[str]:
        """
        Get possible next nodes from current node.

        Args:
            current_node: Current node name
            target_category: Target category we're trying to reach

        Returns:
            List of next node names
        """
        # Get outgoing edges
        edges = self.edge_registry.get_outgoing_edges(current_node)

        if edges:
            # Use edges with best utility
            edges.sort(key=lambda e: e.compute_cost())
            return [e.target for e in edges[:5]]  # Top 5 edges

        # Fallback: get nodes from target category
        target_nodes = self.node_registry.get_nodes_by_category(target_category)
        return [node.name for node in target_nodes[:5]]

    def _score_output(self, output) -> float:
        """
        Score a node output.

        Args:
            output: NodeOutput object

        Returns:
            Score value
        """
        base_score = 1.0 if output.telemetry.get('success', False) else 0.0

        # Bonus for generating useful artifacts
        if output.artifacts:
            base_score += len(output.artifacts) * 0.1

        # Specific bonuses
        if 'hypotheses' in output.artifacts:
            base_score += len(output.artifacts['hypotheses']) * 0.2

        if 'programs' in output.artifacts:
            base_score += len(output.artifacts['programs']) * 0.3

        if 'scores' in output.artifacts:
            # Use actual IoU scores
            base_score += np.mean(output.artifacts['scores']) * 2.0

        return base_score

    def _compute_stability_penalty(self, path: List[str]) -> float:
        """
        Compute stability penalty for a path.

        Args:
            path: Node path

        Returns:
            Penalty value (higher = more unstable)
        """
        penalty = 0.0

        # Check edges for high variance
        for i in range(len(path) - 1):
            edge = self.edge_registry.get_edge(path[i], path[i + 1])
            if edge:
                penalty += edge.variance * self.stability_weight

        return penalty

    def _is_novel_path(self, path: List[str]) -> bool:
        """
        Check if path represents a novel combination.

        Args:
            path: Node path

        Returns:
            True if novel
        """
        # Simple heuristic: path is novel if edges have low traversal counts
        if len(path) < 2:
            return False

        for i in range(len(path) - 1):
            edge = self.edge_registry.get_edge(path[i], path[i + 1])
            if edge and edge.traversal_count > 5:
                return False

        return True

    def _has_target_category(self, path: List[str], target_category: str) -> bool:
        """
        Check if path contains a node from target category.

        Args:
            path: Node path
            target_category: Target category

        Returns:
            True if target reached
        """
        for node_name in path:
            node = self.node_registry.get(node_name)
            if node and node.category == target_category:
                return True

        return False

    def select_best_path(self, candidates: List[SearchNode]) -> Optional[SearchNode]:
        """
        Select the best path from candidates.

        Args:
            candidates: List of search nodes

        Returns:
            Best search node or None
        """
        if not candidates:
            return None

        # Filter stable paths
        stable_candidates = [c for c in candidates if c.stability_penalty < 0.5]

        if stable_candidates:
            # Prefer stable paths
            return max(stable_candidates, key=lambda c: c.total_score)
        else:
            # Fallback to best overall
            return max(candidates, key=lambda c: c.total_score)

    def backtrack_to_stable_junction(self, path: List[str]) -> List[str]:
        """
        Backtrack to the nearest stable junction in a path.

        Args:
            path: Current path

        Returns:
            Truncated path to stable junction
        """
        # Walk backwards until we find a stable edge
        for i in range(len(path) - 2, 0, -1):
            edge = self.edge_registry.get_edge(path[i], path[i + 1])
            if edge and edge.variance < 0.1:
                # Found stable junction
                return path[:i + 1]

        # Fallback: return just the first node
        return path[:1] if path else []
