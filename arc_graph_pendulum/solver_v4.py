"""
Version 4 of ARC Graph Pendulum Solver with shape transformation support.
V4: Priority 1 implementation - handles object extraction, cropping, region selection
"""

import numpy as np
from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_v3_plus import ARCGraphPendulumSolverV3Plus
from nodes.shape_transformation_analyzer import ShapeTransformationAnalyzer
from nodes.shape_transformation_synthesizer import ShapeTransformationSynthesizer
from nodes.shape_rule_inferencer import ShapeRuleInferencer
from utils.arc_loader import ARCTask
from utils.grid_utils import compute_iou


class ARCGraphPendulumSolverV4(ARCGraphPendulumSolverV3Plus):
    """
    Version 4 solver with shape transformation support (Priority 1).

    Key Improvements over V3+:
    - Shape transformation analyzer (object extraction, cropping, region selection)
    - Shape transformation synthesizer (generates programs for shape-changing ops)
    - Intelligent routing: shape transformations vs same-shape transformations
    - Falls back to V3+ for same-shape transformations

    Expected Impact:
    - Solve 11/13 previously failing tasks (84.6%)
    - Increase solve rate: 10.9% → 35-40%
    - Increase avg IoU: 0.611 → 0.75-0.80
    """

    def __init__(
        self,
        beam_width: int = 5,
        use_stability: bool = True,
        use_landscape_analytics: bool = False
    ):
        """Initialize V4 solver with shape transformation support."""
        # Initialize base V3+ solver
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_landscape_analytics=use_landscape_analytics
        )

        # Add shape transformation components
        self.shape_analyzer = ShapeTransformationAnalyzer()
        self.shape_synthesizer = ShapeTransformationSynthesizer()
        self.shape_rule_inferencer = ShapeRuleInferencer()

        print(f"✓ Solver V4 Initialized (Shape Transformation Support)")
        print(f"  - Object extraction detector")
        print(f"  - Cropping/bounding box detector")
        print(f"  - Region selection detector")
        print(f"  - Color counting detector")
        print(f"  - Shape rule inferencer (finds abstract patterns)")
        print(f"  - Intelligent routing (shape-changing vs same-shape)")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve task with shape transformation support.

        Strategy:
        1. Check if any train example has shape change (input != output shape)
        2. If yes: Try shape transformation analyzer first
        3. If shape analyzer succeeds: Use shape synthesizer
        4. Otherwise: Fall back to V3+ enhanced differential analyzer
        """
        # Clear cache before each task to prevent stale results
        self.node_registry.clear_cache()

        if verbose:
            print(f"\n=== Solving task {task.task_id} (V4 - Shape Transformation Support) ===")
            print(f"Train examples: {task.num_train}, Test examples: {task.num_test}")

        # Prepare task data
        task_data = {
            'task_id': task.task_id,
            'train': task.train,
            'test': task.test,
        }

        from core.trajectory import Trajectory
        trajectory = Trajectory()

        # Check if task involves shape transformations
        has_shape_change = False
        for input_grid, output_grid in task.train:
            if input_grid.shape != output_grid.shape:
                has_shape_change = True
                break

        if verbose:
            if has_shape_change:
                print(f"\n[Shape-Changing Task Detected]")
                print(f"  Using shape transformation analyzer...")
            else:
                print(f"\n[Same-Shape Task Detected]")
                print(f"  Using enhanced differential analyzer...")

        # Try shape transformation analysis first if shapes change
        if has_shape_change:
            best_program, best_score = self._try_shape_transformation_approach(
                task, task_data, trajectory, verbose
            )

            if best_score >= 0.7:
                # Shape transformation approach succeeded
                if verbose:
                    print(f"\n  ★ Shape transformation approach succeeded! (score={best_score:.3f})")
            else:
                # Fall back to V3+ approach
                if verbose:
                    print(f"\n  Shape transformation approach had low confidence (score={best_score:.3f})")
                    print(f"  Falling back to V3+ enhanced differential analyzer...")

                best_program, best_score = self._try_v3plus_approach(
                    task, task_data, trajectory, verbose
                )
        else:
            # Same-shape task: use V3+ approach directly
            best_program, best_score = self._try_v3plus_approach(
                task, task_data, trajectory, verbose
            )

        # Phase: Test Prediction
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

    def _try_shape_transformation_approach(
        self,
        task: ARCTask,
        task_data: Dict,
        trajectory,
        verbose: bool
    ) -> tuple:
        """
        Try shape transformation approach.

        Returns: (best_program, best_score)
        """
        if verbose:
            print("\n[Phase 1: Shape Transformation Analysis]")
            print("  Analyzing shape transformations...")

        # Analyze each training example
        shape_analyses = []

        for i, (input_grid, output_grid) in enumerate(task.train):
            analysis = self.shape_analyzer.analyze(input_grid, output_grid)

            if analysis:
                shape_analyses.append(analysis)

                if verbose:
                    print(f"  Example {i}: {analysis['transformation_type']} "
                          f"(conf: {analysis['confidence']:.2f}) - {analysis['description']}")
            else:
                if verbose:
                    print(f"  Example {i}: No shape transformation detected")

        if not shape_analyses:
            if verbose:
                print("  No shape transformations detected.")
            return None, 0.0

        # Phase 2: Shape Rule Inference
        if verbose:
            print("\n[Phase 2: Shape Rule Inference]")
            print("  Inferring abstract transformation rule...")

        # Use rule inferencer to find abstract pattern
        inferred_rule = self.shape_rule_inferencer.infer_rule(shape_analyses, task.train)

        if verbose:
            print(f"  Rule: {inferred_rule['rule_type']}")
            print(f"  Confidence: {inferred_rule['confidence']:.2f}")
            print(f"  Description: {inferred_rule['description']}")

        # Phase 3: Shape Transformation Synthesis
        if verbose:
            print("\n[Phase 3: Shape Transformation Synthesis]")
            print("  Generating shape transformation programs...")

        programs = self.shape_synthesizer.synthesize(inferred_rule)

        if verbose:
            print(f"  Generated {len(programs)} programs:")
            for i, prog in enumerate(programs[:5], 1):  # Show top 5
                print(f"    {i}. {prog['type']}: {prog['description']} (conf: {prog.get('confidence', 0.0):.2f})")

        # Phase 4: Program Evaluation
        if verbose:
            print("\n[Phase 4: Program Evaluation]")

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
                    print(f"  {program['type']}: score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_program = program

                    if score >= 0.99:
                        if verbose:
                            print(f"    ★ EXCELLENT program found!")
                        break

            except Exception as e:
                if verbose and len(programs) <= 10:  # Only show errors for small program sets
                    print(f"  {program['type']}: failed ({e})")
                continue

        return best_program, best_score

    def _try_v3plus_approach(
        self,
        task: ARCTask,
        task_data: Dict,
        trajectory,
        verbose: bool
    ) -> tuple:
        """
        Try V3+ enhanced differential analyzer approach.

        Returns: (best_program, best_score)
        """
        # Phase 1: Enhanced Differential Analysis
        if verbose:
            print("\n[Phase 1: Enhanced Differential Analysis]")
            print("  Analyzing transformations (including complex patterns)...")

        diff_output = self.node_registry.execute("enhanced_differential_analyzer", task_data)
        trajectory.add_node("enhanced_differential_analyzer", diff_output)

        analyses = diff_output.result

        if verbose:
            for i, analysis in enumerate(analyses[:3]):  # Show first 3
                print(f"  Example {i}: {analysis['transformation_type']} "
                      f"(conf: {analysis['confidence']:.2f}) - {analysis['description']}")

            consensus = diff_output.artifacts.get('consensus_type', 'unknown')
            consensus_ratio = diff_output.artifacts.get('consensus_ratio', 0.0)
            print(f"  Consensus: {consensus} ({consensus_ratio*100:.0f}% agreement)")

        # Phase 2: Rule Inference
        if verbose:
            print("\n[Phase 2: Rule Inference]")
            print("  Inferring general transformation rule...")

        rule_input = {'transformation_analyses': analyses}
        rule_output = self.node_registry.execute("rule_inferencer", rule_input)
        trajectory.add_node("rule_inferencer", rule_output)

        rule = rule_output.result

        if verbose:
            print(f"  Rule: {rule['rule_type']}")
            print(f"  Consistency: {rule['consistency']:.2f}")
            print(f"  Confidence: {rule['confidence']:.2f}")

        # Phase 3: Enhanced Targeted Program Synthesis
        if verbose:
            print("\n[Phase 3: Enhanced Targeted Program Synthesis]")
            print("  Generating programs...")

        synth_input = {'rule': rule}
        synth_output = self.node_registry.execute("enhanced_targeted_synthesizer", synth_input)
        trajectory.add_node("enhanced_targeted_synthesizer", synth_output)

        programs = synth_output.result

        if verbose:
            print(f"  Generated {len(programs)} targeted programs")

        # Phase 4: Program Evaluation
        if verbose:
            print("\n[Phase 4: Program Evaluation]")

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
                    print(f"  {program['type']}: score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_program = program

                    if score >= 0.99:
                        if verbose:
                            print(f"    ★ EXCELLENT program found!")
                        break

            except Exception as e:
                continue

        # Phase 5: Iterative Refinement
        if best_program and best_score < 0.99 and best_score > 0.5:
            if verbose:
                print(f"\n[Phase 5: Iterative Refinement]")
                print(f"  Refining best program (current score: {best_score:.3f})...")

            refine_input = {
                'program': best_program,
                'train_examples': task.train
            }

            refine_output = self.node_registry.execute("iterative_refiner", refine_input)
            trajectory.add_node("iterative_refiner", refine_output)

            refined_result = refine_output.result
            refined_program = refined_result['program']
            refined_score = refined_result['score']

            if verbose:
                print(f"  Refinement: {best_score:.3f} → {refined_score:.3f}")

            if refined_score > best_score:
                best_program = refined_program
                best_score = refined_score
                if verbose:
                    print(f"  ✓ Improvement achieved!")

        return best_program, best_score


def main():
    """Demo V4 solver on failing tasks."""
    print("="*70)
    print("ARC GRAPH PENDULUM SOLVER V4 - SHAPE TRANSFORMATION SUPPORT")
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

    # Create V4 solver
    solver = ARCGraphPendulumSolverV4(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False
    )

    # Test on previously failing shape-changing tasks
    print("\n" + "="*70)
    print("TESTING ON PREVIOUSLY FAILING SHAPE-CHANGING TASKS")
    print("="*70)

    # These are from the failure analysis - all shape-changing tasks that failed
    failed_task_ids = [
        '1fad071e',  # 9x9 → 1x5 (extreme downsampling)
        '239be575',  # 5x5 → 1x1 (color counting)
        '0b148d64',  # 21x21 → 10x10 (downsampling)
        '23b5c85d',  # 10x10 → 3x3 (object extraction)
        '137eaa0f',  # 11x11 → 3x3 (downsampling)
    ]

    results = []

    for task_id in failed_task_ids:
        if task_id in tasks:
            task = tasks[task_id]
            print(f"\n{'='*70}")
            print(f"TASK: {task_id} (V3+ failed with 0.000)")
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
