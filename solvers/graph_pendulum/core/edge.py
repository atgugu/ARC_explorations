"""
Edge abstraction: directed relations between nodes with geometry and utility.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Edge:
    """
    Directed edge between two nodes.

    Attributes:
        source: Source node name
        target: Target node name
        angle: Angle between behavior vectors (in radians)
        distance: Distance metric (1 - cosine similarity)
        utility: Accumulated causal credit
        traversal_count: Number of times this edge was used
        success_count: Number of successful traversals
        variance: Variance in outcomes when using this edge
    """
    source: str
    target: str
    angle: float = 0.0
    distance: float = 0.0
    utility: float = 0.0
    traversal_count: int = 0
    success_count: int = 0
    variance: float = 0.0
    metadata: dict = field(default_factory=dict)

    def update_utility(self, success_delta: float, variance_penalty: float = 0.1):
        """
        Update edge utility based on causal credit.

        Args:
            success_delta: Change in success when using this edge
            variance_penalty: Penalty coefficient for variance
        """
        self.utility += success_delta - variance_penalty * self.variance
        self.traversal_count += 1

    def record_traversal(self, success: bool):
        """Record a traversal of this edge."""
        self.traversal_count += 1
        if success:
            self.success_count += 1

    @property
    def success_rate(self) -> float:
        """Compute success rate for this edge."""
        if self.traversal_count == 0:
            return 0.0
        return self.success_count / self.traversal_count

    def compute_cost(self) -> float:
        """
        Compute edge cost for pathfinding.

        Lower cost = better edge (high utility, low distance, low variance).
        """
        # Base cost is the distance
        cost = self.distance

        # Adjust by utility (negative utility increases cost)
        cost -= self.utility * 0.1

        # Penalize high variance
        cost += self.variance * 0.2

        # Penalize low success rate if we have data
        if self.traversal_count > 0:
            cost += (1.0 - self.success_rate) * 0.3

        return max(0.01, cost)  # Ensure positive cost


class EdgeRegistry:
    """Registry for managing edges in the graph."""

    def __init__(self):
        self.edges: dict[tuple[str, str], Edge] = {}

    def add_edge(self, edge: Edge):
        """Add or update an edge."""
        key = (edge.source, edge.target)
        self.edges[key] = edge

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        """Get an edge between two nodes."""
        return self.edges.get((source, target))

    def get_outgoing_edges(self, source: str) -> list[Edge]:
        """Get all edges from a source node."""
        return [edge for (s, t), edge in self.edges.items() if s == source]

    def get_incoming_edges(self, target: str) -> list[Edge]:
        """Get all edges to a target node."""
        return [edge for (s, t), edge in self.edges.items() if t == target]

    def compute_angles_and_distances(self, behavior_vectors: dict[str, np.ndarray]):
        """
        Compute angles and distances for all edges based on behavior vectors.

        Args:
            behavior_vectors: Dictionary mapping node names to behavior vectors
        """
        for (source, target), edge in self.edges.items():
            if source in behavior_vectors and target in behavior_vectors:
                vec_s = behavior_vectors[source]
                vec_t = behavior_vectors[target]

                # Compute cosine similarity
                cos_sim = np.dot(vec_s, vec_t) / (np.linalg.norm(vec_s) * np.linalg.norm(vec_t) + 1e-8)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)

                # Angle in radians
                edge.angle = np.arccos(cos_sim)

                # Distance (1 - cosine similarity)
                edge.distance = 1.0 - cos_sim

    def get_best_edges_from(self, source: str, top_k: int = 5) -> list[Edge]:
        """
        Get the best outgoing edges from a source node.

        Args:
            source: Source node name
            top_k: Number of top edges to return

        Returns:
            List of best edges sorted by cost (ascending)
        """
        edges = self.get_outgoing_edges(source)
        edges.sort(key=lambda e: e.compute_cost())
        return edges[:top_k]
