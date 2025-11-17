"""
ARC Graph Pendulum Solver V9 - With 20 Extended Primitives

Builds on V7 by adding 20 new transformation primitives across 4 categories:
1. Spatial Operations (5): Positional object extraction and alignment
2. Object Operations (5): Multi-object manipulation
3. Color Operations (5): Advanced color transformations
4. Pattern Operations (5): Symmetry and pattern completion

Expected improvement: +2-3% evaluation solve rate (1.7% → 4-5%)
Target: Convert 8-12 of 37 high-quality tasks (0.80-0.95 IoU) to solves

This is Phase 1 of the revised synthesis roadmap, validated by V6-V8 negative results.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Callable

from solver_v7 import ARCGraphPendulumSolverV7
from nodes.extended_primitive_detector import ExtendedPrimitiveDetector
from nodes.extended_primitive_synthesizer import ExtendedPrimitiveSynthesizer
from utils.arc_loader import ARCTask


class ARCGraphPendulumSolverV9(ARCGraphPendulumSolverV7):
    """
    V9 solver with 20 extended primitives.

    Extends V7 with new transformation primitives identified from analysis
    of high-quality tasks.
    """

    def __init__(self, beam_width: int = 5, use_stability: bool = True,
                 use_landscape_analytics: bool = False,
                 enable_refinement: bool = True):
        """
        Initialize V9 solver with extended primitives.

        Args:
            beam_width: Beam search width
            use_stability: Use stability-aware search
            use_landscape_analytics: Use landscape analytics
            enable_refinement: Enable execution refinement (from V7)
        """
        # Initialize V7 base
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_landscape_analytics=use_landscape_analytics,
            enable_refinement=enable_refinement
        )

        # Add extended primitive components
        self.extended_detector = ExtendedPrimitiveDetector()
        self.extended_synthesizer = ExtendedPrimitiveSynthesizer()

        print("✓ Solver V9 Initialized (20 Extended Primitives)")
        print("  Category 1: Spatial operations (5 primitives)")
        print("    - Extract leftmost/rightmost/topmost/bottommost object")
        print("    - Align objects to grid")
        print("  Category 2: Object operations (5 primitives)")
        print("    - Copy object horizontally/vertically")
        print("    - Connect objects, intersection, union")
        print("  Category 3: Color operations (5 primitives)")
        print("    - Recolor by row/column, checkerboard")
        print("    - Swap by size, color propagation")
        print("  Category 4: Pattern operations (5 primitives)")
        print("    - Symmetry (horizontal/vertical/rotational)")
        print("    - Pattern completion, periodic tiling")
        print("  Expected: +2-3% solve rate (target: 37 high-quality tasks)")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve task using extended primitives + V7 capabilities.

        Strategy:
        1. Try extended primitives first (new transformations)
        2. Fall back to V7 (V4-V7 capabilities)
        3. Use best program found

        Args:
            task: ARC task
            verbose: Print debug info

        Returns:
            List of predicted outputs
        """
        if verbose:
            print(f"\n=== Solving task {task.task_id} (V9 - Extended Primitives) ===")
            print(f"Train examples: {len(task.train)}, Test examples: {len(task.test)}")

        # Phase 1: Try extended primitives
        extended_programs, extended_score = self._try_extended_primitives(
            task, verbose=verbose
        )

        # Phase 2: Evaluate V7 on training (to compare fairly)
        v7_score = self._evaluate_v7_on_training(task)

        if verbose:
            print(f"\n[Program Selection]")
            print(f"  Extended primitives score: {extended_score:.3f}")
            print(f"  V7 approach score: {v7_score:.3f}")

        # Use best approach (require extended to be significantly better)
        if extended_score > v7_score + 0.05 and extended_programs:
            if verbose:
                print(f"  → Using extended primitives (better score)")

            # Generate predictions using extended programs
            test_inputs = [test_input for test_input, _ in task.test]
            predictions = self._generate_predictions(extended_programs, test_inputs)

        else:
            if verbose:
                reason = "better score" if v7_score >= extended_score else "not significantly better"
                print(f"  → Using V7 approach ({reason})")
            # Generate V7 predictions
            predictions = super().solve_task(task, verbose=False)

        return predictions

    def _try_extended_primitives(self, task: ARCTask, verbose: bool = False) -> Tuple[List[Dict], float]:
        """
        Try extended primitive approach.

        Args:
            task: ARC task
            verbose: Print debug info

        Returns:
            (programs, training_score)
        """
        if verbose:
            print(f"\n[Extended Primitives]")
            print(f"  Detecting applicable primitives...")

        # Detect applicable primitives
        detections = self.extended_detector.detect(task.train)

        if not detections:
            if verbose:
                print(f"  No extended primitives detected")
            return [], 0.0

        if verbose:
            print(f"  Detected {len(detections)} potential primitives:")
            for det in detections[:3]:  # Show top 3
                print(f"    - {det['description']} (confidence={det['confidence']:.2f})")

        # Synthesize programs
        programs = self.extended_synthesizer.synthesize(detections)

        if not programs:
            if verbose:
                print(f"  No programs generated")
            return [], 0.0

        # Evaluate programs on training
        best_program = None
        best_score = 0.0

        for prog in programs:
            score = self._evaluate_program(prog['function'], task.train)

            if score > best_score:
                best_score = score
                best_program = prog

        if verbose and best_program:
            print(f"  Best program: {best_program['description']} (score={best_score:.3f})")

        if best_program:
            return [best_program], best_score
        else:
            return [], 0.0

    def _generate_predictions(self, programs: List[Dict], test_inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Generate predictions using programs."""
        if not programs:
            return [inp.copy() for inp in test_inputs]

        program = programs[0]  # Use best program
        program_func = program['function']

        predictions = []
        for test_input in test_inputs:
            try:
                pred = program_func(test_input)
                predictions.append(pred)
            except Exception:
                predictions.append(test_input.copy())

        return predictions

    def _evaluate_program(self, program_func: Callable,
                         train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate program on training examples."""
        if not train_examples:
            return 0.0

        scores = []

        for input_grid, output_grid in train_examples:
            try:
                predicted = program_func(input_grid)

                # Compute IoU
                if predicted.shape == output_grid.shape:
                    iou = np.sum(predicted == output_grid) / predicted.size
                else:
                    iou = 0.0

                scores.append(iou)

            except Exception:
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def _evaluate_v7_on_training(self, task: ARCTask) -> float:
        """Evaluate V7 approach on training examples."""
        if not task.train:
            return 0.0

        # Create a temporary task with training examples as test
        temp_task = ARCTask(
            task_id=task.task_id,
            train=task.train,
            test=task.train  # Use training as "test" to evaluate
        )

        # Get V7 predictions on training examples
        v7_predictions = super().solve_task(temp_task, verbose=False)

        # Compute IoU scores
        scores = []
        for pred, (_, expected) in zip(v7_predictions, task.train):
            if pred.shape == expected.shape:
                iou = np.sum(pred == expected) / pred.size
            else:
                iou = 0.0
            scores.append(iou)

        return np.mean(scores) if scores else 0.0


def main():
    """Test V9 on high-quality tasks."""
    from utils.arc_loader import ARCLoader
    from pathlib import Path
    import json

    print("="*70)
    print("V9 SOLVER TEST - EXTENDED PRIMITIVES")
    print("="*70)

    # Load high-quality tasks from V7 results
    with open('v7_evaluation_results.json', 'r') as f:
        v7_results = json.load(f)

    high_quality = [r for r in v7_results['results'] if 0.80 <= r['avg_score'] < 0.95]
    high_quality.sort(key=lambda x: x['avg_score'], reverse=False)  # Start with lowest scores

    print(f"\nTesting on {min(15, len(high_quality))} high-quality tasks")
    print("(V7 scores: 0.80-0.95 IoU, starting with lowest)\n")

    loader = ARCLoader(cache_dir="./arc_data")

    improvements = 0
    regressions = 0
    tested = 0

    for task_result in high_quality[:15]:
        task_id = task_result['task_id']
        v7_score = task_result['avg_score']

        print(f"{'='*70}")
        print(f"Task: {task_id} (V7 score: {v7_score:.4f})")
        print(f"{'='*70}")

        task_file = Path(f"./arc_data/evaluation/{task_id}.json")

        if not task_file.exists():
            print(f"Task file not found\n")
            continue

        task = loader.load_task(str(task_file))

        # Create V9 solver
        solver = ARCGraphPendulumSolverV9(
            beam_width=5,
            use_stability=True,
            use_landscape_analytics=False,
            enable_refinement=True
        )

        # Solve
        result = solver.evaluate_on_task(task, verbose=True)

        status = "✓ SOLVED" if result['solved'] else f"IoU {result['avg_score']:.4f}"
        improvement = result['avg_score'] - v7_score

        print(f"\n{'='*70}")
        print(f"V7 Result: IoU {v7_score:.4f}")
        print(f"V9 Result: {status}")
        print(f"Change: {improvement:+.4f}")
        print(f"{'='*70}\n")

        tested += 1
        if improvement > 0.01:
            improvements += 1
        elif improvement < -0.01:
            regressions += 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"V9 SUMMARY ({tested} tasks)")
    print(f"{'='*70}")
    print(f"Improvements: {improvements}/{tested}")
    print(f"Regressions:  {regressions}/{tested}")
    print(f"No change:    {tested - improvements - regressions}/{tested}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
