"""
ARC Graph Pendulum Solver V10: Constraint-Based Synthesis

Extends V7 with constraint-based program synthesis. Uses extracted constraints
to guide program search, significantly pruning the search space.

Key innovation: Instead of random program search, extract constraints from
training examples and only consider programs that provably satisfy those constraints.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import List, Tuple, Dict, Any
from utils.arc_loader import ARCTask
from solver_v7 import ARCGraphPendulumSolverV7
from nodes.constraint_extractor import ConstraintExtractor
from nodes.constraint_based_synthesizer import ConstraintBasedSynthesizer


class ARCGraphPendulumSolverV10(ARCGraphPendulumSolverV7):
    """
    V10 Solver: Constraint-Based Program Synthesis

    Approach:
    1. Extract formal constraints from training examples
    2. Match constraints to compatible primitive operations
    3. Generate candidate programs that satisfy constraints
    4. Verify candidates on training, select best
    5. Fall back to V7 if constraint-based synthesis fails

    Expected impact: +3-5% evaluation solve rate
    """

    def __init__(self,
                 beam_width: int = 5,
                 use_stability: bool = True,
                 use_landscape_analytics: bool = False,
                 enable_refinement: bool = True,
                 constraint_threshold: float = 0.95):
        """
        Initialize V10 solver with constraint-based synthesis.

        Args:
            beam_width: Beam search width
            use_stability: Use stability-aware search
            use_landscape_analytics: Use landscape analytics
            enable_refinement: Enable execution refinement
            constraint_threshold: Minimum IoU on training to use constraint-based program
        """
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_landscape_analytics=use_landscape_analytics,
            enable_refinement=enable_refinement
        )

        self.constraint_extractor = ConstraintExtractor()
        self.constraint_synthesizer = ConstraintBasedSynthesizer()
        self.constraint_threshold = constraint_threshold

        print("✓ Solver V10 Initialized (Constraint-Based Synthesis)")
        print("  Approach: Extract constraints → Guide program search")
        print("  Components:")
        print("    - Constraint extractor (shape, color, spatial, object)")
        print("    - Constraint-based synthesizer (guided search)")
        print("    - Fallback to V7 if constraints insufficient")
        print(f"  Confidence threshold: {constraint_threshold:.2f}")
        print("  Expected: +3-5% solve rate (1.7% → 5-7%)")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve task using constraint-based synthesis with V7 fallback.

        Args:
            task: ARC task to solve
            verbose: Print detailed information

        Returns:
            List of predictions for test examples
        """
        if verbose:
            print(f"\n=== Solving task {task.task_id} (V10 - Constraint-Based Synthesis) ===")
            print(f"Train examples: {len(task.train)}, Test examples: {len(task.test)}")

        # Phase 1: Extract constraints from training
        constraints = self.constraint_extractor.extract(task.train)

        if verbose:
            print(f"\n[Constraint Extraction]")
            print(f"  Shape relationship: {constraints.get('shape', {}).get('shape_relationship', 'unknown')}")
            print(f"  Color mapping: {'Yes' if constraints.get('color', {}).get('color_mapping') else 'No'}")
            print(f"  Spatial transform: {constraints.get('spatial', {}).get('has_reflection') or constraints.get('spatial', {}).get('has_rotation') or 'None'}")

        # Phase 2: Synthesize programs using constraints
        candidate_programs = self.constraint_synthesizer.synthesize(
            constraints,
            task.train,
            max_depth=3,
            max_candidates=20
        )

        best_score = 0.0
        best_program = None

        if candidate_programs:
            best_program = candidate_programs[0]
            best_score = best_program['score']

            if verbose:
                print(f"\n[Constraint-Based Synthesis]")
                print(f"  Candidates generated: {len(candidate_programs)}")
                print(f"  Best program: {best_program['description']}")
                print(f"  Training score: {best_score:.3f}")

        # Phase 3: Evaluate V7 on training (for comparison)
        v7_score = self._evaluate_v7_on_training(task)

        if verbose:
            print(f"\n[Program Selection]")
            print(f"  Constraint-based score: {best_score:.3f}")
            print(f"  V7 approach score: {v7_score:.3f}")

        # Phase 4: Select best approach
        if best_score >= self.constraint_threshold and best_score > v7_score:
            if verbose:
                print(f"  → Using constraint-based program (high confidence)")

            # Generate predictions using constraint-based program
            test_inputs = [test_input for test_input, _ in task.test]
            predictions = self._generate_predictions(best_program, test_inputs)

        else:
            if verbose:
                reason = "better score" if v7_score >= best_score else "low confidence"
                print(f"  → Using V7 approach ({reason})")

            # Fall back to V7
            predictions = super().solve_task(task, verbose=False)

        return predictions

    def _generate_predictions(self,
                             program: Dict[str, Any],
                             test_inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Generate predictions using a constraint-based program."""
        predictions = []

        for test_input in test_inputs:
            try:
                output = program['function'](test_input.copy())
                predictions.append(output)
            except Exception:
                # If program fails, return zero grid
                predictions.append(np.zeros_like(test_input))

        return predictions

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
    """Test V10 on high-quality tasks."""
    from utils.arc_loader import ARCLoader
    from pathlib import Path
    import json

    print("="*70)
    print("V10 SOLVER TEST - CONSTRAINT-BASED SYNTHESIS")
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
    constraint_used = 0

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

        # Create V10 solver
        solver = ARCGraphPendulumSolverV10(
            beam_width=5,
            use_stability=True,
            use_landscape_analytics=False,
            enable_refinement=True,
            constraint_threshold=0.95
        )

        # Solve
        result = solver.evaluate_on_task(task, verbose=True)

        status = "✓ SOLVED" if result['solved'] else f"IoU {result['avg_score']:.4f}"
        improvement = result['avg_score'] - v7_score

        print(f"\n{'='*70}")
        print(f"V7 Result: IoU {v7_score:.4f}")
        print(f"V10 Result: {status}")
        print(f"Change: {improvement:+.4f}")
        print(f"{'='*70}\n")

        tested += 1
        if improvement > 0.01:
            improvements += 1
        elif improvement < -0.01:
            regressions += 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"V10 SUMMARY ({tested} tasks)")
    print(f"{'='*70}")
    print(f"Improvements: {improvements}/{tested}")
    print(f"Regressions:  {regressions}/{tested}")
    print(f"No change:    {tested - improvements - regressions}/{tested}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
