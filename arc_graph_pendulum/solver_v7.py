"""
ARC Graph Pendulum Solver V7 - With Execution Refinement

Builds on V6 by adding execution refinement to fix common precision errors:
- Color-to-background substitution errors
- Edge/boundary handling issues
- Incomplete pixel transformations
- Small color mapping errors

Key improvements over V6:
- Wraps all program outputs with refinement post-processing
- Fixes 1-5% pixel errors in near-miss tasks
- Targets high-quality tasks (0.95+ IoU) for conversion to solves
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass

from solver_v6 import ARCGraphPendulumSolverV6
from nodes.execution_refiner import ExecutionRefiner
from utils.arc_loader import ARCTask


class ARCGraphPendulumSolverV7(ARCGraphPendulumSolverV6):
    """
    V7 solver with execution refinement.

    Extends V6 with post-processing refinement to fix common execution errors.
    """

    def __init__(self, beam_width: int = 5, use_stability: bool = True,
                 use_landscape_analytics: bool = False,
                 enable_refinement: bool = True):
        """
        Initialize V7 solver with execution refinement.

        Args:
            beam_width: Beam search width
            use_stability: Use stability-aware search
            use_landscape_analytics: Use landscape analytics
            enable_refinement: Enable execution refinement (default: True)
        """
        # Initialize V6 base
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_landscape_analytics=use_landscape_analytics
        )

        # Add execution refiner
        self.enable_refinement = enable_refinement
        self.execution_refiner = ExecutionRefiner()

        print("✓ Solver V7 Initialized (Execution Refinement)")
        if enable_refinement:
            print("  - Incomplete fill correction")
            print("  - Edge/boundary handling fixes")
            print("  - Color leakage correction")
            print("  - Object boundary refinement")

    def _evaluate_programs(self, programs: List[Dict[str, Any]],
                          train_examples: List[Tuple[np.ndarray, np.ndarray]],
                          verbose: bool = False) -> Tuple[Callable, float]:
        """
        Evaluate programs and select the best one.

        Overrides V6 to add refinement wrapping.

        Args:
            programs: List of program dictionaries
            train_examples: Training examples
            verbose: Print debug info

        Returns:
            (best_program_func, best_score)
        """
        if not programs:
            return None, 0.0

        best_score = -1
        best_program = None

        for prog in programs:
            # Get original program function
            original_func = prog['function']

            # Wrap with refinement if enabled
            if self.enable_refinement:
                refined_func = self._wrap_with_refinement(original_func, train_examples)
            else:
                refined_func = original_func

            # Evaluate on training examples
            score = self._evaluate_program_on_training(refined_func, train_examples)

            if score > best_score:
                best_score = score
                best_program = refined_func

        return best_program, best_score

    def _wrap_with_refinement(self, program_func: Callable,
                             train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Callable:
        """
        Wrap a program function to apply execution refinement.

        Args:
            program_func: Original program function
            train_examples: Training examples for pattern learning

        Returns:
            Wrapped function with refinement
        """
        return self.execution_refiner.create_refining_wrapper(program_func, train_examples)

    def _evaluate_program_on_training(self, program_func: Callable,
                                     train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Evaluate program on training examples.

        Args:
            program_func: Program to evaluate
            train_examples: Training input-output pairs

        Returns:
            Average IoU score
        """
        if not train_examples:
            return 0.0

        scores = []

        for input_grid, output_grid in train_examples:
            try:
                predicted = program_func(input_grid)

                # Compute IoU
                if predicted.shape == output_grid.shape:
                    iou = self._compute_iou(predicted, output_grid)
                else:
                    iou = 0.0

                scores.append(iou)

            except Exception:
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def _compute_iou(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """Compute Intersection over Union."""
        if pred.shape != truth.shape:
            return 0.0

        intersection = np.sum(pred == truth)
        total = pred.size

        return intersection / total if total > 0 else 0.0

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve an ARC task.

        Overrides V6 to ensure refinement is applied.

        Args:
            task: ARC task to solve
            verbose: Print debug information

        Returns:
            List of predicted outputs for test examples
        """
        # Use V6's solve_task which will call our overridden _evaluate_programs
        return super().solve_task(task, verbose=verbose)

    def evaluate_on_task(self, task: ARCTask, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate solver on a task and return detailed results.

        Args:
            task: ARC task
            verbose: Print debug info

        Returns:
            Dictionary with evaluation results
        """
        predictions = self.solve_task(task, verbose=verbose)

        if not predictions:
            return {
                'solved': False,
                'avg_score': 0.0,
                'test_examples': len(task.test),
                'predictions': []
            }

        # Compute scores for each test example
        scores = []
        for i, (test_input, test_output) in enumerate(task.test):
            if i < len(predictions):
                pred = predictions[i]
                iou = self._compute_iou(pred, test_output)
                scores.append(iou)
            else:
                scores.append(0.0)

        avg_score = np.mean(scores) if scores else 0.0
        solved = avg_score >= 0.99

        return {
            'solved': solved,
            'avg_score': avg_score,
            'test_examples': len(task.test),
            'predictions': predictions,
            'scores': scores
        }


def main():
    """Test V7 on a sample task."""
    from utils.arc_loader import ARCLoader

    print("="*70)
    print("V7 SOLVER TEST - EXECUTION REFINEMENT")
    print("="*70)

    # Load a near-miss task
    loader = ARCLoader(cache_dir="./arc_data")

    # Test on a specific near-miss task
    task_id = "27a77e38"  # 0.9877 IoU - just 1 pixel error
    print(f"\nTesting on near-miss task: {task_id}")

    from pathlib import Path
    task_file = Path(f"./arc_data/evaluation/{task_id}.json")

    if task_file.exists():
        task = loader.load_task(str(task_file))

        # Create V7 solver
        solver = ARCGraphPendulumSolverV7(
            beam_width=5,
            use_stability=True,
            use_landscape_analytics=False,
            enable_refinement=True
        )

        # Solve
        result = solver.evaluate_on_task(task, verbose=True)

        print(f"\n{'='*70}")
        status = "✓ SOLVED" if result['solved'] else f"IoU {result['avg_score']:.4f}"
        print(f"Result: {status}")
        print(f"{'='*70}")

    else:
        print(f"Task file not found: {task_file}")


if __name__ == "__main__":
    main()
