"""
ARC Graph Pendulum Solver V8 - With Ensemble Voting

Builds on V7 by using ensemble of multiple solver runs with voting.

Simpler approach than complex program-level ensemble:
- Run V7 solver multiple times with slight variations
- Combine predictions using weighted voting
- More robust to parameter sensitivity

Expected improvement:
- Near-miss tasks: Voting may fix slight errors
- Target: +0.5-1% evaluation solve rate
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter

from solver_v7 import ARCGraphPendulumSolverV7
from utils.arc_loader import ARCTask


class ARCGraphPendulumSolverV8(ARCGraphPendulumSolverV7):
    """
    V8 solver with ensemble voting.

    Uses multiple V7 runs with voting to improve robustness.
    """

    def __init__(self, beam_width: int = 5, use_stability: bool = True,
                 use_landscape_analytics: bool = False,
                 enable_refinement: bool = True,
                 ensemble_size: int = 3):
        """
        Initialize V8 solver with ensemble capabilities.

        Args:
            beam_width: Beam search width
            use_stability: Use stability-aware search
            use_landscape_analytics: Use landscape analytics
            enable_refinement: Enable execution refinement (from V7)
            ensemble_size: Number of ensemble members
        """
        # Initialize V7 base
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_landscape_analytics=use_landscape_analytics,
            enable_refinement=enable_refinement
        )

        self.ensemble_size = ensemble_size

        print("✓ Solver V8 Initialized (Ensemble Voting)")
        print(f"  - Ensemble size: {ensemble_size}")
        print(f"  - Weighted voting for robustness")
        print(f"  - Inherits V7 execution refinement")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve task using ensemble voting.

        Args:
            task: ARC task
            verbose: Print debug info

        Returns:
            List of ensemble predictions
        """
        if self.ensemble_size == 1:
            # Just use V7 directly
            return super().solve_task(task, verbose=verbose)

        if verbose:
            print(f"\n=== Solving task {task.task_id} (V8 - Ensemble Voting) ===")
            print(f"Train examples: {len(task.train)}, Test examples: {len(task.test)}")
            print(f"Ensemble size: {self.ensemble_size}")

        # Generate multiple predictions using V7
        all_predictions = []

        for i in range(self.ensemble_size):
            if verbose and i > 0:
                print(f"\n  Ensemble member {i+1}/{self.ensemble_size}...")

            # Get prediction from V7 (may have some randomness in beam search)
            predictions = super().solve_task(task, verbose=(verbose and i==0))

            all_predictions.append(predictions)

        # Combine predictions using voting
        ensemble_predictions = []

        for test_idx in range(len(task.test)):
            # Collect predictions for this test example
            test_predictions = [
                preds[test_idx] if test_idx < len(preds) else task.test[test_idx][0].copy()
                for preds in all_predictions
            ]

            # Vote
            ensemble_pred = self._vote_predictions(test_predictions, verbose=verbose)
            ensemble_predictions.append(ensemble_pred)

        if verbose:
            print(f"\n  Final ensemble predictions: {len(ensemble_predictions)}")

        return ensemble_predictions

    def _vote_predictions(self, predictions: List[np.ndarray],
                         verbose: bool = False) -> np.ndarray:
        """
        Combine predictions using majority voting.

        Args:
            predictions: List of prediction grids
            verbose: Print debug info

        Returns:
            Voted prediction
        """
        if len(predictions) == 1:
            return predictions[0]

        # Check if all predictions are identical
        all_same = all(np.array_equal(predictions[0], p) for p in predictions[1:])

        if all_same:
            if verbose:
                print(f"    All {len(predictions)} predictions agree")
            return predictions[0]

        # Check shapes
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) > 1:
            # Different shapes - use most common shape
            shape_counts = Counter(shapes)
            most_common_shape = shape_counts.most_common(1)[0][0]

            # Filter to predictions with most common shape
            valid_preds = [p for p in predictions if p.shape == most_common_shape]

            if not valid_preds:
                return predictions[0]

            predictions = valid_preds

            if verbose:
                print(f"    Shape variation, using most common: {most_common_shape}")

        # All same shape - pixel-wise voting
        result = self._pixel_wise_vote(predictions)

        if verbose:
            agreements = [np.sum(p == result) / p.size for p in predictions]
            print(f"    Voting agreement: {np.mean(agreements):.3f} avg")

        return result

    def _pixel_wise_vote(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Perform pixel-wise majority voting.

        Args:
            predictions: List of same-shape predictions

        Returns:
            Voted prediction
        """
        # Stack predictions
        stacked = np.stack(predictions, axis=0)

        height, width = predictions[0].shape
        result = np.zeros((height, width), dtype=np.int32)

        # Vote for each pixel
        for i in range(height):
            for j in range(width):
                pixel_values = stacked[:, i, j]

                # Most common value
                counts = Counter(pixel_values)
                result[i, j] = counts.most_common(1)[0][0]

        return result


def main():
    """Test V8 on near-miss tasks."""
    from utils.arc_loader import ARCLoader
    from pathlib import Path
    import json

    print("="*70)
    print("V8 SOLVER TEST - ENSEMBLE VOTING")
    print("="*70)

    # Load near-miss task IDs
    with open('v7_evaluation_results.json', 'r') as f:
        v7_results = json.load(f)

    near_miss = [r for r in v7_results['results'] if 0.95 <= r['avg_score'] < 0.99]
    near_miss.sort(key=lambda x: x['avg_score'], reverse=True)

    print(f"\nTesting on {min(3, len(near_miss))} near-miss tasks")

    loader = ARCLoader(cache_dir="./arc_data")

    for task_result in near_miss[:3]:
        task_id = task_result['task_id']
        v7_score = task_result['avg_score']

        print(f"\n{'='*70}")
        print(f"Task: {task_id} (V7 score: {v7_score:.4f})")
        print(f"{'='*70}")

        task_file = Path(f"./arc_data/evaluation/{task_id}.json")

        if not task_file.exists():
            print(f"Task file not found")
            continue

        task = loader.load_task(str(task_file))

        # Create V8 solver
        solver = ARCGraphPendulumSolverV8(
            beam_width=5,
            use_stability=True,
            use_landscape_analytics=False,
            enable_refinement=True,
            ensemble_size=3  # 3 ensemble members
        )

        # Solve
        result = solver.evaluate_on_task(task, verbose=True)

        status = "✓ SOLVED" if result['solved'] else f"IoU {result['avg_score']:.4f}"
        improvement = result['avg_score'] - v7_score

        print(f"\n{'='*70}")
        print(f"V7 Result: IoU {v7_score:.4f}")
        print(f"V8 Result: {status}")
        print(f"Change: {improvement:+.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
