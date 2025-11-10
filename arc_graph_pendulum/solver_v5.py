"""
Version 5 of ARC Graph Pendulum Solver with compositional transformation support.
V5: Priority 2 implementation - handles multi-step transformations
"""

import numpy as np
from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_v4 import ARCGraphPendulumSolverV4
from nodes.compositional_analyzer import CompositionalTransformationAnalyzer
from nodes.compositional_synthesizer import CompositionalSynthesizer
from utils.arc_loader import ARCTask
from utils.grid_utils import compute_iou


class ARCGraphPendulumSolverV5(ARCGraphPendulumSolverV4):
    """
    Version 5 solver with compositional transformation support (Priority 2).

    Key Improvements over V4:
    - Compositional transformation analyzer (2-step and 3-step)
    - Intermediate state search
    - Multi-step program synthesis
    - Falls back to V4 for single-step transformations

    Expected Impact:
    - Improve 5-10 medium-quality tasks (0.5-0.79 IoU)
    - Solve 2-3 additional failures
    - Increase overall solve rate by ~5-10%
    """

    def __init__(
        self,
        beam_width: int = 5,
        use_stability: bool = True,
        use_landscape_analytics: bool = False
    ):
        """Initialize V5 solver with compositional transformation support."""
        # Initialize base V4 solver
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_landscape_analytics=use_landscape_analytics
        )

        # Add compositional transformation components
        self.compositional_analyzer = CompositionalTransformationAnalyzer()
        self.compositional_synthesizer = CompositionalSynthesizer()

        print(f"✓ Solver V5 Initialized (Compositional Transformation Support)")
        print(f"  - 2-step composition detection")
        print(f"  - 3-step composition detection")
        print(f"  - Intermediate state search")
        print(f"  - Multi-step program synthesis")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve task with compositional transformation support.

        Strategy:
        1. Check if single-step approaches work (V4)
        2. If confidence is low, try compositional analysis
        3. Fall back gracefully through V4 → V3+ → identity
        """
        # Clear cache before each task
        self.node_registry.clear_cache()

        if verbose:
            print(f"\n=== Solving task {task.task_id} (V5 - Compositional Support) ===")
            print(f"Train examples: {task.num_train}, Test examples: {task.num_test}")

        # Prepare task data
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

        # Try V4 single-step approach first
        best_program = None
        best_score = 0.0

        if has_shape_change:
            if verbose:
                print("  Trying single-step shape transformation...")
            best_program, best_score = self._try_shape_transformation_approach(
                task, task_data, trajectory, verbose=False
            )
        else:
            if verbose:
                print("  Trying single-step enhanced differential analyzer...")
            best_program, best_score = self._try_v3plus_approach(
                task, task_data, trajectory, verbose=False
            )

        # If single-step confidence is low, try compositional
        if best_score < 0.8:
            if verbose:
                print(f"\n  Single-step confidence low ({best_score:.3f}), trying compositional...")

            comp_program, comp_score = self._try_compositional_approach(
                task, task_data, trajectory, verbose
            )

            if comp_score > best_score:
                if verbose:
                    print(f"  ✓ Compositional approach better! ({comp_score:.3f} > {best_score:.3f})")
                best_program = comp_program
                best_score = comp_score
            else:
                if verbose:
                    if comp_program:
                        print(f"  Single-step approach better ({best_score:.3f} > {comp_score:.3f})")
                    else:
                        print(f"  Compositional approach found nothing useful")
        else:
            if verbose:
                print(f"  Single-step approach succeeded with high confidence ({best_score:.3f})")

        # Test Prediction
        if verbose:
            print(f"\n[Test Prediction]")
            print(f"Best program: {best_program['type'] if best_program else 'None'} (score={best_score:.3f})")

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

        # Log trajectory
        self.trajectory_batch.add(trajectory)

        if self.use_landscape_analytics:
            self.landscape_analyzer.add_trajectory(trajectory, task.task_id)

        return predictions

    def _try_compositional_approach(
        self,
        task: ARCTask,
        task_data: Dict,
        trajectory,
        verbose: bool
    ) -> tuple:
        """
        Try compositional transformation approach.

        Returns: (best_program, best_score)
        """
        if verbose:
            print("\n[Compositional Analysis]")
            print("  Analyzing multi-step transformations...")

        # Analyze each training example
        compositional_analyses = []

        for i, (input_grid, output_grid) in enumerate(task.train):
            analysis = self.compositional_analyzer.analyze(input_grid, output_grid)

            if analysis:
                compositional_analyses.append(analysis)

                if verbose:
                    print(f"  Example {i}: {analysis['transformation_type']} "
                          f"(conf: {analysis['confidence']:.2f})")
                    print(f"    → {analysis['description']}")
            else:
                if verbose:
                    print(f"  Example {i}: No compositional pattern detected")

        if not compositional_analyses:
            if verbose:
                print("  No compositional transformations detected.")
            return None, 0.0

        # Find consensus composition
        from collections import Counter
        types = [a['transformation_type'] for a in compositional_analyses]
        most_common_type = Counter(types).most_common(1)[0][0]
        consensus_count = sum(1 for t in types if t == most_common_type)
        consensus_ratio = consensus_count / len(types)

        if verbose:
            print(f"  Consensus: {most_common_type} ({consensus_ratio*100:.0f}% agreement)")

        # Use the analysis with highest confidence
        best_analysis = max(compositional_analyses, key=lambda a: a['confidence'])

        # Synthesize programs
        if verbose:
            print("\n[Compositional Program Synthesis]")
            print("  Generating multi-step programs...")

        programs = self.compositional_synthesizer.synthesize(best_analysis)

        if verbose:
            print(f"  Generated {len(programs)} programs:")
            for i, prog in enumerate(programs[:3], 1):  # Show top 3
                print(f"    {i}. {prog['description']} (conf: {prog.get('confidence', 0.0):.2f})")

        # Evaluate programs
        if verbose:
            print("\n[Program Evaluation]")

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

                if verbose and score > 0.5:
                    print(f"  {program.get('composition', '?')}: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_program = program

                    if score >= 0.99:
                        if verbose:
                            print(f"    ★ EXCELLENT compositional program found!")
                        break

            except Exception as e:
                if verbose and len(programs) <= 5:
                    print(f"  {program.get('description', '?')}: failed ({e})")
                continue

        return best_program, best_score


def main():
    """Demo V5 solver on diverse tasks."""
    print("="*70)
    print("ARC GRAPH PENDULUM SOLVER V5 - COMPOSITIONAL SUPPORT")
    print("="*70)
    print()

    # Load ARC dataset
    from utils.arc_loader import ARCLoader

    loader = ARCLoader(cache_dir="./arc_data")
    print("Loading ARC dataset...")
    tasks = loader.load_all_tasks("training")

    if not tasks:
        print("No tasks loaded.")
        return

    print(f"Loaded {len(tasks)} tasks\n")

    # Create V5 solver
    solver = ARCGraphPendulumSolverV5(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False
    )

    # Test on a few tasks
    print("\n" + "="*70)
    print("TESTING ON SAMPLE TASKS")
    print("="*70)

    # Test on medium-quality tasks from V4 that might benefit from composition
    test_task_ids = [
        '017c7c7b',  # 0.704 in V4 - pattern extraction
        '1e0a9b12',  # 0.720 in V4 - might be compositional
        '178fcbfb',  # 0.591 in V4 - needs improvement
        '1bfc4729',  # 0.500 in V4 - borderline
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
