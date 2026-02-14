"""
Basin abstraction: stable regions of the trajectory landscape.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
import numpy as np


@dataclass
class Basin:
    """
    A stable region of trajectories with consistent behavior.

    Attributes:
        id: Unique basin identifier
        centroid: Center point in trajectory space
        trajectories: Trajectories belonging to this basin
        node_motifs: Common node patterns in this basin
        success_rate: Overall success rate of trajectories
        avg_sensitivity: Average sensitivity to perturbations
        avg_variance: Average variance across trajectories
        description: Human-readable description
    """
    id: int
    centroid: np.ndarray
    trajectories: List = field(default_factory=list)  # Will hold Trajectory references
    node_motifs: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    avg_sensitivity: float = 0.0
    avg_variance: float = 0.0
    radius: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_trajectory(self, trajectory):
        """Add a trajectory to this basin."""
        self.trajectories.append(trajectory)
        self._recompute_statistics()

    def _recompute_statistics(self):
        """Recompute basin statistics from trajectories."""
        if not self.trajectories:
            return

        # Success rate
        scores = [t.final_score for t in self.trajectories]
        self.success_rate = np.mean([s >= 0.9 for s in scores])

        # Average sensitivity and variance
        self.avg_sensitivity = np.mean([t.sensitivity for t in self.trajectories])
        self.avg_variance = np.mean([t.variance for t in self.trajectories])

        # Node motifs (frequent node patterns)
        node_counts: Dict[str, int] = {}
        for traj in self.trajectories:
            for node in set(traj.nodes):
                node_counts[node] = node_counts.get(node, 0) + 1

        # Normalize to get frequencies
        total = len(self.trajectories)
        self.node_motifs = {
            node: count / total
            for node, count in node_counts.items()
        }

    def is_stable(self, sensitivity_threshold: float = 0.1, variance_threshold: float = 0.1) -> bool:
        """Check if this basin is stable."""
        return (self.avg_sensitivity < sensitivity_threshold and
                self.avg_variance < variance_threshold)

    def is_successful(self, threshold: float = 0.5) -> bool:
        """Check if this basin has good success rate."""
        return self.success_rate >= threshold

    def get_representative_trajectory(self):
        """Get the trajectory closest to the centroid."""
        if not self.trajectories:
            return None

        # Find trajectory with highest score and lowest sensitivity
        best = max(
            self.trajectories,
            key=lambda t: t.final_score - t.sensitivity
        )
        return best

    def get_common_patterns(self, min_frequency: float = 0.5) -> List[str]:
        """
        Get node patterns that appear frequently in this basin.

        Args:
            min_frequency: Minimum frequency threshold (0-1)

        Returns:
            List of common node names
        """
        return [
            node for node, freq in self.node_motifs.items()
            if freq >= min_frequency
        ]

    def compute_quality_score(self) -> float:
        """
        Compute overall quality score for this basin.

        Higher is better: high success, low sensitivity, low variance.
        """
        stability = 1.0 - (self.avg_sensitivity + self.avg_variance) / 2.0
        quality = self.success_rate * 0.6 + stability * 0.4
        return quality


class BasinRegistry:
    """Registry for managing discovered basins."""

    def __init__(self):
        self.basins: Dict[int, Basin] = {}
        self.next_id = 0

    def add_basin(self, basin: Basin) -> int:
        """Add a basin and return its ID."""
        if basin.id is None or basin.id < 0:
            basin.id = self.next_id
            self.next_id += 1

        self.basins[basin.id] = basin
        return basin.id

    def get_basin(self, basin_id: int) -> Basin:
        """Get a basin by ID."""
        return self.basins.get(basin_id)

    def get_stable_basins(self, sensitivity_threshold: float = 0.1) -> List[Basin]:
        """Get all stable basins."""
        return [
            basin for basin in self.basins.values()
            if basin.is_stable(sensitivity_threshold)
        ]

    def get_successful_basins(self, success_threshold: float = 0.5) -> List[Basin]:
        """Get basins with high success rates."""
        return [
            basin for basin in self.basins.values()
            if basin.is_successful(success_threshold)
        ]

    def get_best_basins(self, top_k: int = 5) -> List[Basin]:
        """
        Get the top-k best basins by quality score.

        Args:
            top_k: Number of basins to return

        Returns:
            List of best basins
        """
        basins = list(self.basins.values())
        basins.sort(key=lambda b: b.compute_quality_score(), reverse=True)
        return basins[:top_k]

    def find_basin_for_trajectory(self, trajectory_vector: np.ndarray, max_distance: float = 0.5) -> int:
        """
        Find the best basin for a trajectory vector.

        Args:
            trajectory_vector: Vector representation of trajectory
            max_distance: Maximum distance to consider a match

        Returns:
            Basin ID or -1 if no suitable basin found
        """
        best_basin_id = -1
        best_distance = float('inf')

        for basin_id, basin in self.basins.items():
            # Compute distance to centroid
            dist = np.linalg.norm(trajectory_vector - basin.centroid)

            if dist < best_distance and dist < max_distance:
                best_distance = dist
                best_basin_id = basin_id

        return best_basin_id
