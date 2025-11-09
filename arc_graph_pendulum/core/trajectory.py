"""
Trajectory abstraction: paths of activated nodes through the graph.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Trajectory:
    """
    A path through the node graph from perception to execution.

    Attributes:
        nodes: Ordered list of node names in the trajectory
        edges: List of edges traversed
        artifacts: Accumulated artifacts from nodes
        telemetry: Accumulated telemetry data
        final_score: Final success score (0-1)
        sensitivity: Measured sensitivity to perturbations
        timestamp: Creation timestamp
    """
    nodes: List[str] = field(default_factory=list)
    edges: List[tuple[str, str]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    final_score: float = 0.0
    sensitivity: float = 0.0
    variance: float = 0.0
    failure_taxonomy: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node_name: str, output: Any):
        """Add a node to the trajectory."""
        if self.nodes:
            # Record edge
            self.edges.append((self.nodes[-1], node_name))

        self.nodes.append(node_name)

        # Accumulate artifacts and telemetry
        if hasattr(output, 'artifacts'):
            self.artifacts.update(output.artifacts)
        if hasattr(output, 'telemetry'):
            for key, value in output.telemetry.items():
                if key not in self.telemetry:
                    self.telemetry[key] = []
                self.telemetry[key].append(value)

    def to_vector(self) -> np.ndarray:
        """
        Convert trajectory to a fixed-length summary vector for clustering.

        Vector includes:
        - Node visitation histogram (sparse, fixed vocab)
        - Average angle along path
        - Cumulative utility
        - Final scores
        - Failure taxonomy indicators
        """
        # This will be implemented based on the node registry vocabulary
        # For now, return a placeholder
        features = []

        # Length of trajectory
        features.append(len(self.nodes))

        # Final score
        features.append(self.final_score)

        # Sensitivity and variance
        features.append(self.sensitivity)
        features.append(self.variance)

        # Number of unique nodes visited
        features.append(len(set(self.nodes)))

        return np.array(features, dtype=np.float32)

    def get_node_histogram(self, vocab: List[str]) -> np.ndarray:
        """
        Get histogram of node visits for a given vocabulary.

        Args:
            vocab: List of all possible node names

        Returns:
            Histogram vector
        """
        hist = np.zeros(len(vocab), dtype=np.float32)
        for node in self.nodes:
            if node in vocab:
                idx = vocab.index(node)
                hist[idx] += 1

        # Normalize
        total = hist.sum()
        if total > 0:
            hist /= total

        return hist

    def compute_path_utility(self, edge_registry) -> float:
        """
        Compute cumulative utility along the path.

        Args:
            edge_registry: EdgeRegistry to look up edge utilities

        Returns:
            Total utility
        """
        total_utility = 0.0
        for source, target in self.edges:
            edge = edge_registry.get_edge(source, target)
            if edge:
                total_utility += edge.utility

        return total_utility

    def is_stable(self, threshold: float = 0.1) -> bool:
        """Check if trajectory is in a stable region."""
        return self.sensitivity < threshold and self.variance < threshold


@dataclass
class TrajectoryBatch:
    """
    A batch of trajectories for analysis.
    """
    trajectories: List[Trajectory] = field(default_factory=list)

    def add(self, trajectory: Trajectory):
        """Add a trajectory to the batch."""
        self.trajectories.append(trajectory)

    def to_matrix(self, vocab: List[str]) -> np.ndarray:
        """
        Convert all trajectories to a matrix for analysis.

        Args:
            vocab: Node vocabulary

        Returns:
            Matrix where each row is a trajectory vector
        """
        vectors = []
        for traj in self.trajectories:
            # Build extended vector
            hist = traj.get_node_histogram(vocab)
            scores = np.array([
                traj.final_score,
                traj.sensitivity,
                traj.variance,
                len(traj.nodes)
            ])
            vec = np.concatenate([hist, scores])
            vectors.append(vec)

        return np.array(vectors)

    def get_successful(self, threshold: float = 0.9) -> List[Trajectory]:
        """Get trajectories with high success scores."""
        return [t for t in self.trajectories if t.final_score >= threshold]

    def get_stable(self, threshold: float = 0.1) -> List[Trajectory]:
        """Get stable trajectories."""
        return [t for t in self.trajectories if t.is_stable(threshold)]
