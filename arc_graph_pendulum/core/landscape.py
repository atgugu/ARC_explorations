"""
Landscape analytics with UMAP clustering and basin discovery.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


@dataclass
class LandscapePoint:
    """
    A point in the trajectory landscape.

    Attributes:
        trajectory_id: Unique ID for the trajectory
        vector: High-dimensional feature vector
        embedding: Low-dimensional UMAP embedding
        cluster_id: Assigned cluster/basin ID
        metadata: Additional information
    """
    trajectory_id: str
    vector: np.ndarray
    embedding: Optional[np.ndarray] = None
    cluster_id: int = -1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LandscapeAnalyzer:
    """
    Analyzes trajectory landscape using dimensionality reduction and clustering.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = 'euclidean',
        random_state: int = 42
    ):
        """
        Initialize landscape analyzer.

        Args:
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            n_components: Number of embedding dimensions
            metric: Distance metric
            random_state: Random seed
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state

        self.umap_model = None
        self.cluster_model = None
        self.points: List[LandscapePoint] = []
        self.embeddings: Optional[np.ndarray] = None
        self.clusters: Optional[np.ndarray] = None

    def add_trajectory(self, trajectory, trajectory_id: str):
        """
        Add a trajectory to the landscape.

        Args:
            trajectory: Trajectory object
            trajectory_id: Unique identifier
        """
        # Convert trajectory to vector
        vector = trajectory.to_vector()

        # Create landscape point
        point = LandscapePoint(
            trajectory_id=trajectory_id,
            vector=vector,
            metadata={
                'final_score': trajectory.final_score,
                'sensitivity': trajectory.sensitivity,
                'variance': trajectory.variance,
                'num_nodes': len(trajectory.nodes),
                'nodes': trajectory.nodes,
            }
        )

        self.points.append(point)

    def add_trajectory_batch(self, trajectory_batch):
        """
        Add multiple trajectories from a TrajectoryBatch.

        Args:
            trajectory_batch: TrajectoryBatch object
        """
        for i, trajectory in enumerate(trajectory_batch.trajectories):
            self.add_trajectory(trajectory, f"traj_{i}")

    def compute_embeddings(self, use_umap: bool = True) -> np.ndarray:
        """
        Compute low-dimensional embeddings of trajectories.

        Args:
            use_umap: Whether to use UMAP (True) or PCA (False)

        Returns:
            Embedding matrix (n_trajectories x n_components)
        """
        if not self.points:
            raise ValueError("No trajectories added to landscape")

        # Get vectors
        vectors = np.array([p.vector for p in self.points])

        # For small datasets, use PCA instead of UMAP
        if len(self.points) < 10 or not SKLEARN_AVAILABLE:
            use_umap = False

        if use_umap and UMAP_AVAILABLE:
            # Use UMAP
            n_neighbors = min(self.n_neighbors, len(self.points) - 1)
            if n_neighbors < 2:
                n_neighbors = 2

            try:
                self.umap_model = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=self.min_dist,
                    n_components=self.n_components,
                    metric=self.metric,
                    random_state=self.random_state
                )

                embeddings = self.umap_model.fit_transform(vectors)
            except Exception as e:
                # Fall back to PCA if UMAP fails
                print(f"UMAP failed ({e}), falling back to PCA")
                use_umap = False

        if not use_umap and SKLEARN_AVAILABLE:
            # Fallback to PCA
            print("Using PCA for dimensionality reduction")
            n_comp = min(self.n_components, len(self.points), vectors.shape[1])
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            embeddings = pca.fit_transform(vectors)

            # Pad if needed
            if embeddings.shape[1] < self.n_components:
                padding = np.zeros((embeddings.shape[0], self.n_components - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])

        elif not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for dimensionality reduction")

        # Store embeddings in points
        for i, point in enumerate(self.points):
            point.embedding = embeddings[i]

        self.embeddings = embeddings
        return embeddings

    def discover_basins(
        self,
        method: str = 'dbscan',
        eps: float = 0.5,
        min_samples: int = 2,
        n_clusters: int = 5
    ) -> List[int]:
        """
        Discover basins (clusters) in the trajectory landscape.

        Args:
            method: Clustering method ('dbscan' or 'kmeans')
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            n_clusters: KMeans n_clusters parameter

        Returns:
            Cluster labels for each trajectory
        """
        if self.embeddings is None:
            raise ValueError("Must compute embeddings first")

        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for clustering")

        if method == 'dbscan':
            # Use DBSCAN for density-based clustering
            self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = self.cluster_model.fit_predict(self.embeddings)

        elif method == 'kmeans':
            # Use KMeans
            self.cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state
            )
            labels = self.cluster_model.fit_predict(self.embeddings)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Store cluster labels in points
        for i, point in enumerate(self.points):
            point.cluster_id = int(labels[i])

        self.clusters = labels
        return labels

    def get_basin_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Compute statistics for each discovered basin.

        Returns:
            Dictionary mapping basin IDs to statistics
        """
        if self.clusters is None:
            raise ValueError("Must discover basins first")

        basin_stats = {}

        for cluster_id in set(self.clusters):
            # Get points in this cluster
            cluster_points = [p for p in self.points if p.cluster_id == cluster_id]

            if not cluster_points:
                continue

            # Compute statistics
            scores = [p.metadata.get('final_score', 0.0) for p in cluster_points]
            sensitivities = [p.metadata.get('sensitivity', 0.0) for p in cluster_points]
            variances = [p.metadata.get('variance', 0.0) for p in cluster_points]

            # Node motifs (common node patterns)
            node_counts = {}
            for p in cluster_points:
                for node in p.metadata.get('nodes', []):
                    node_counts[node] = node_counts.get(node, 0) + 1

            # Most common nodes
            common_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            basin_stats[cluster_id] = {
                'size': len(cluster_points),
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'avg_sensitivity': np.mean(sensitivities),
                'avg_variance': np.mean(variances),
                'success_rate': np.mean([s >= 0.9 for s in scores]),
                'is_stable': np.mean(sensitivities) < 0.1 and np.mean(variances) < 0.1,
                'common_nodes': common_nodes,
                'quality_score': np.mean(scores) * 0.6 + (1.0 - np.mean(sensitivities)) * 0.4,
            }

        return basin_stats

    def find_best_basins(self, top_k: int = 3) -> List[int]:
        """
        Find the best basins by quality score.

        Args:
            top_k: Number of top basins to return

        Returns:
            List of basin IDs sorted by quality
        """
        stats = self.get_basin_statistics()

        # Sort by quality score
        ranked = sorted(
            stats.items(),
            key=lambda x: x[1]['quality_score'],
            reverse=True
        )

        return [basin_id for basin_id, _ in ranked[:top_k]]

    def get_basin_for_trajectory(self, trajectory) -> int:
        """
        Find the best basin for a new trajectory.

        Args:
            trajectory: Trajectory object

        Returns:
            Basin ID (-1 if no good match)
        """
        if self.embeddings is None or self.cluster_model is None:
            return -1

        # Convert trajectory to vector
        vector = trajectory.to_vector()

        if UMAP_AVAILABLE and self.umap_model is not None:
            # Transform using UMAP
            embedding = self.umap_model.transform([vector])[0]
        else:
            # Use raw vector (will be less accurate)
            embedding = vector[:self.n_components]

        # Find nearest cluster
        if hasattr(self.cluster_model, 'predict'):
            # KMeans
            cluster = self.cluster_model.predict([embedding])[0]
            return int(cluster)
        else:
            # DBSCAN - find nearest existing point
            min_dist = float('inf')
            best_cluster = -1

            for point in self.points:
                if point.embedding is not None:
                    dist = np.linalg.norm(embedding - point.embedding)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = point.cluster_id

            return best_cluster

    def visualize_landscape(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Create a visualization of the trajectory landscape.

        Args:
            output_path: Path to save visualization (optional)

        Returns:
            Path to saved visualization or None
        """
        if self.embeddings is None:
            raise ValueError("Must compute embeddings first")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Colored by cluster
        if self.clusters is not None:
            scatter1 = axes[0].scatter(
                self.embeddings[:, 0],
                self.embeddings[:, 1],
                c=self.clusters,
                cmap='tab10',
                alpha=0.6,
                s=100
            )
            axes[0].set_title('Trajectory Landscape (Colored by Basin)')
            plt.colorbar(scatter1, ax=axes[0], label='Basin ID')
        else:
            axes[0].scatter(
                self.embeddings[:, 0],
                self.embeddings[:, 1],
                alpha=0.6,
                s=100
            )
            axes[0].set_title('Trajectory Landscape (Unclustered)')

        axes[0].set_xlabel('UMAP Dimension 1')
        axes[0].set_ylabel('UMAP Dimension 2')

        # Plot 2: Colored by success score
        scores = [p.metadata.get('final_score', 0.0) for p in self.points]
        scatter2 = axes[1].scatter(
            self.embeddings[:, 0],
            self.embeddings[:, 1],
            c=scores,
            cmap='RdYlGn',
            alpha=0.6,
            s=100,
            vmin=0,
            vmax=1
        )
        axes[1].set_title('Trajectory Landscape (Colored by Success)')
        axes[1].set_xlabel('UMAP Dimension 1')
        axes[1].set_ylabel('UMAP Dimension 2')
        plt.colorbar(scatter2, ax=axes[1], label='Success Score')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Landscape visualization saved to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None

    def save_analysis(self, output_path: str):
        """
        Save landscape analysis to JSON.

        Args:
            output_path: Path to save JSON file
        """
        data = {
            'num_trajectories': len(self.points),
            'embedding_dimensions': self.n_components,
            'num_basins': len(set(self.clusters)) if self.clusters is not None else 0,
            'basin_statistics': self.get_basin_statistics() if self.clusters is not None else {},
            'trajectories': [
                {
                    'id': p.trajectory_id,
                    'cluster_id': p.cluster_id,
                    'embedding': p.embedding.tolist() if p.embedding is not None else None,
                    'metadata': p.metadata,
                }
                for p in self.points
            ],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Landscape analysis saved to {output_path}")

    def print_summary(self):
        """Print a summary of the landscape analysis."""
        print("\n" + "="*60)
        print("LANDSCAPE ANALYSIS SUMMARY")
        print("="*60)

        print(f"Total trajectories: {len(self.points)}")

        if self.embeddings is not None:
            print(f"Embedding dimensions: {self.n_components}")

        if self.clusters is not None:
            num_basins = len(set(self.clusters))
            print(f"Number of basins: {num_basins}")

            # Print basin statistics
            stats = self.get_basin_statistics()
            print("\nBasin Statistics:")
            print("-" * 60)

            for basin_id in sorted(stats.keys()):
                s = stats[basin_id]
                print(f"\nBasin {basin_id}:")
                print(f"  Size: {s['size']} trajectories")
                print(f"  Avg Score: {s['avg_score']:.3f} (±{s['std_score']:.3f})")
                print(f"  Success Rate: {s['success_rate']:.1%}")
                print(f"  Avg Sensitivity: {s['avg_sensitivity']:.3f}")
                print(f"  Stable: {'Yes' if s['is_stable'] else 'No'}")
                print(f"  Quality Score: {s['quality_score']:.3f}")
                print(f"  Common Nodes: {', '.join([n for n, _ in s['common_nodes'][:3]])}")

            # Best basins
            best = self.find_best_basins(top_k=3)
            print(f"\nBest Basins (by quality): {best}")

        print("="*60 + "\n")
