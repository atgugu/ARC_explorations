"""
Version 6 of ARC Graph Pendulum Solver with meta-pattern learning.
V6: Priority 3+ implementation - conditional rules and test-time adaptation
"""

import numpy as np
from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_v5 import ARCGraphPendulumSolverV5
from nodes.meta_pattern_learner import MetaPatternLearner
from nodes.conditional_synthesizer import ConditionalSynthesizer
from utils.arc_loader import ARCTask
from utils.grid_utils import compute_iou


class ARCGraphPendulumSolverV6(ARCGraphPendulumSolverV5):
    """
    Version 6 solver with meta-pattern learning (Priority 3+).

    Key Improvements over V5:
    - Meta-pattern learner (finds conditional rules from variation)
    - Conditional synthesizer (generates adaptive programs)
    - Test-time adaptation (programs examine input and choose parameters)
    - Addresses training-test generalization gap

    Expected Impact:
    - Solve 10-15 high-quality tasks stuck at 0.80-0.99 IoU
    - Increase solve rate by ~10-15%
    - Improve average IoU significantly
    - Reduce training-test performance gap
    """

    def __init__(
        self,
        beam_width: int = 5,
        use_stability: bool = True,
        use_landscape_analytics: bool = False
    ):
        """Initialize V6 solver with meta-pattern learning."""
        # Initialize base V5 solver
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_landscape_analytics=use_landscape_analytics
        )

        # Add meta-pattern learning components
        self.meta_pattern_learner = MetaPatternLearner()
        self.conditional_synthesizer = ConditionalSynthesizer()

        print(f"✓ Solver V6 Initialized (Meta-Pattern Learning & Test-Time Adaptation)")
        print(f"  - Variation analysis across training examples")
        print(f"  - Conditional rule inference")
        print(f"  - Adaptive program synthesis")
        print(f"  - Test-time parameter adaptation")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve task with meta-pattern learning.

        Strategy:
        1. Try V5 approaches (single-step, compositional)
        2. If confidence is moderate (0.70-0.95), analyze variation
        3. Generate conditional programs if patterns found
        4. Choose best program (may be adaptive or fixed)
        """
        # Clear cache
        self.node_registry.clear_cache()

        if verbose:
            print(f"\n=== Solving task {task.task_id} (V6 - Meta-Pattern Learning) ===")
            print(f"Train examples: {task.num_train}, Test examples: {task.num_test}")

        task_data = {
            'task_id': task.task_id,
            'train': task.train,
            'test': task.test,
        }

        from core.trajectory import Trajectory
        trajectory = Trajectory()

        # Check for shape changes
        has_shape_change = False
        for input_grid, output_grid in task.train:
            if input_grid.shape != output_grid.shape:
                has_shape_change = True
                break

        if verbose:
            print(f"\n[Task Type: {'Shape-Changing' if has_shape_change else 'Same-Shape'}]")

        # Phase 1: Try V5 approaches
        best_program = None
        best_score = 0.0
        analyses = None

        if has_shape_change:
            if verbose:
                print("  Trying shape transformation approach...")

            # Get shape analyses for meta-learning
            shape_analyses = []
            for input_grid, output_grid in task.train:
                analysis = self.shape_analyzer.analyze(input_grid, output_grid)
                if analysis:
                    shape_analyses.append(analysis)

            analyses = shape_analyses if shape_analyses else None

            best_program, best_score = self._try_shape_transformation_approach(
                task, task_data, trajectory, verbose=False
            )

        else:
            if verbose:
                print("  Trying enhanced differential approach...")

            # Get enhanced analyses for meta-learning
            diff_output = self.node_registry.execute("enhanced_differential_analyzer", task_data)
            analyses = diff_output.result if diff_output else None

            best_program, best_score = self._try_v3plus_approach(
                task, task_data, trajectory, verbose=False
            )

        # Try compositional if needed
        if best_score < 0.8:
            if verbose:
                print(f"  Single-step confidence low ({best_score:.3f}), trying compositional...")

            comp_program, comp_score = self._try_compositional_approach(
                task, task_data, trajectory, verbose=False
            )

            if comp_score > best_score:
                best_program = comp_program
                best_score = comp_score

        # Phase 2: Meta-Pattern Learning (if confidence is moderate)
        if 0.70 <= best_score < 0.95 and analyses and len(analyses) >= 2:
            if verbose:
                print(f"\n  Moderate confidence ({best_score:.3f}), analyzing variation...")

            meta_program, meta_score = self._try_meta_pattern_approach(
                analyses, task, verbose
            )

            if meta_score > best_score:
                if verbose:
                    print(f"  ✓ Meta-pattern approach better! ({meta_score:.3f} > {best_score:.3f})")
                best_program = meta_program
                best_score = meta_score
            else:
                if verbose:
                    if meta_program:
                        print(f"  Standard approach better ({best_score:.3f} > {meta_score:.3f})")
                    else:
                        print(f"  No meta-pattern found")

        elif best_score >= 0.95:
            if verbose:
                print(f"  High confidence ({best_score:.3f}), skipping meta-pattern analysis")

        # Test Prediction
        if verbose:
            print(f"\n[Test Prediction]")
            is_adaptive = best_program and best_program.get('adaptive', False)
            prog_type = best_program['type'] if best_program else 'None'
            if is_adaptive:
                prog_type += " (ADAPTIVE)"
            print(f"Best program: {prog_type} (score={best_score:.3f})")

        predictions = []
        if best_program:
            for i, (test_input, _) in enumerate(task.test):
                try:
                    pred = best_program['function'](test_input)
                    predictions.append(pred)

                    if verbose:
                        print(f"  Test {i}: predicted shape {pred.shape}")

                except Exception as e:
                    if verbose:
                        print(f"  Test {i}: failed ({e})")
                    predictions.append(test_input.copy())
        else:
            predictions = [test_input.copy() for test_input, _ in task.test]

        # Update trajectory
        trajectory.final_score = best_score
        self.trajectory_batch.add(trajectory)

        if self.use_landscape_analytics:
            self.landscape_analyzer.add_trajectory(trajectory, task.task_id)

        return predictions

    def _try_meta_pattern_approach(
        self,
        analyses: List[Dict],
        task: ARCTask,
        verbose: bool
    ) -> tuple:
        """
        Try meta-pattern learning approach.

        Returns: (best_program, best_score)
        """
        if verbose:
            print("\n[Meta-Pattern Analysis]")
            print("  Analyzing variation across training examples...")

        # Analyze variation
        meta_pattern = self.meta_pattern_learner.analyze_variation(
            analyses,
            task.train
        )

        if not meta_pattern:
            if verbose:
                print("  No meta-pattern detected")
            return None, 0.0

        if verbose:
            print(f"  Meta-pattern: {meta_pattern.get('meta_pattern_type', 'unknown')}")
            print(f"  Rule: {meta_pattern.get('description', 'unknown')}")
            print(f"  Confidence: {meta_pattern.get('confidence', 0.0):.2f}")

        # Synthesize conditional programs
        if verbose:
            print("\n[Conditional Program Synthesis]")
            print("  Generating adaptive programs...")

        programs = self.conditional_synthesizer.synthesize(meta_pattern)

        if verbose:
            print(f"  Generated {len(programs)} adaptive programs")
            for i, prog in enumerate(programs[:3], 1):
                print(f"    {i}. {prog['description']} (conf: {prog.get('confidence', 0.0):.2f})")

        # Evaluate programs
        if verbose:
            print("\n[Adaptive Program Evaluation]")

        best_program = None
        best_score = 0.0

        for program in programs:
            try:
                # Evaluate on training examples
                ious = []
                for input_grid, output_grid in task.train:
                    pred = program['function'](input_grid)
                    iou = compute_iou(pred, output_grid)
                    ious.append(iou)

                score = np.mean(ious)

                if verbose and score > 0.7:
                    adaptive_label = " (ADAPTIVE)" if program.get('adaptive') else ""
                    print(f"  {program['type']}{adaptive_label}: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_program = program

                    if score >= 0.99:
                        if verbose:
                            print(f"    ★ EXCELLENT adaptive program found!")
                        break

            except Exception as e:
                if verbose and len(programs) <= 5:
                    print(f"  {program.get('description', '?')}: failed ({e})")
                continue

        return best_program, best_score


def main():
    """Demo V6 solver on challenging tasks."""
    print("="*70)
    print("ARC GRAPH PENDULUM SOLVER V6 - META-PATTERN LEARNING")
    print("="*70)
    print()

    from utils.arc_loader import ARCLoader

    loader = ARCLoader(cache_dir="./arc_data")
    print("Loading ARC dataset...")
    tasks = loader.load_all_tasks("training")

    if not tasks:
        print("No tasks loaded.")
        return

    print(f"Loaded {len(tasks)} tasks\n")

    # Create V6 solver
    solver = ARCGraphPendulumSolverV6(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False
    )

    # Test on high-quality tasks from V5 that should benefit from adaptation
    print("\n" + "="*70)
    print("TESTING ON HIGH-QUALITY TASKS (0.80-0.99 IoU in V5)")
    print("="*70)

    test_task_ids = [
        '11852cab',  # 0.970 in V5 - almost perfect
        '025d127b',  # 0.980 in V5 - almost perfect
        '0e206a2e',  # 0.950 in V5 - high quality
        '06df4c85',  # 0.947 in V5 - high quality
        '1f642eb9',  # 0.930 in V5 - high quality
        '22233c11',  # 0.920 in V5 - high quality
    ]

    results = []

    for task_id in test_task_ids:
        if task_id in tasks:
            task = tasks[task_id]
            print(f"\n{'='*70}")
            print(f"TASK: {task_id}")
            print(f"{'='*70}")

            result = solver.evaluate_on_task(task, verbose=True)

            results.append({
                'task_id': task_id,
                'solved': result['solved'],
                'avg_score': result['avg_score']
            })

            print(f"\n{'='*70}")
            if result['solved']:
                print(f"★ SUCCESS: Solved task {task_id}!")
            else:
                print(f"Result: {result['avg_score']:.3f} IoU")
            print(f"{'='*70}\n")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    solved_count = sum(1 for r in results if r['solved'])
    total_count = len(results)
    avg_iou = np.mean([r['avg_score'] for r in results])

    print(f"Solved: {solved_count}/{total_count} ({solved_count/total_count*100:.1f}%)")
    print(f"Average IoU: {avg_iou:.3f}")

    print("\nDetailed results:")
    for r in results:
        status = "✓ SOLVED" if r['solved'] else f"  {r['avg_score']:.3f}"
        print(f"  {r['task_id']}: {status}")


if __name__ == "__main__":
    main()
