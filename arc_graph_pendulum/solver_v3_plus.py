"""
Version 3+ of ARC Graph Pendulum Solver with enhanced differential analysis.
V3+: Adds support for complex transformations (pattern tiling, pattern extraction, object ops)
"""

import numpy as np
from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_v3 import ARCGraphPendulumSolverV3
from nodes.enhanced_differential_analyzer import create_enhanced_differential_analyzer_node
from nodes.enhanced_targeted_synthesizer import create_enhanced_targeted_synthesizer_node
from nodes.rule_inferencer import create_rule_inferencer_node
from nodes.iterative_refiner import create_iterative_refiner_node
from utils.arc_loader import ARCTask
from utils.grid_utils import compute_iou


class ARCGraphPendulumSolverV3Plus(ARCGraphPendulumSolverV3):
    """
    Version 3+ solver with enhanced transformation detection.

    Enhancements over V3:
    - Enhanced differential analyzer (pattern tiling, pattern extraction, object ops)
    - Enhanced targeted synthesizer (generates programs for complex transformations)
    - Same iterative refinement and rule inference as V3
    """

    def __init__(
        self,
        beam_width: int = 5,
        use_stability: bool = True,
        use_landscape_analytics: bool = False
    ):
        """Initialize V3+ solver with enhanced components."""
        # Initialize base V3 solver
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_landscape_analytics=use_landscape_analytics
        )

        # Replace with enhanced nodes
        self.node_registry.register(create_enhanced_differential_analyzer_node())
        self.node_registry.register(create_enhanced_targeted_synthesizer_node())

        print(f"✓ Solver V3+ Initialized (Enhanced Rule Inference)")
        print(f"  - Enhanced differential analyzer (pattern tiling, pattern extraction, object ops)")
        print(f"  - Enhanced targeted synthesizer (complex transformations)")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve task using enhanced rule inference.
        Same pipeline as V3 but with enhanced nodes.
        """
        # Clear cache before each task to prevent stale results
        self.node_registry.clear_cache()

        if verbose:
            print(f"\n=== Solving task {task.task_id} (V3+ - Enhanced Rule Inference) ===")
            print(f"Train examples: {task.num_train}, Test examples: {task.num_test}")

        # Prepare task data
        task_data = {
            'task_id': task.task_id,
            'train': task.train,
            'test': task.test,
        }

        from core.trajectory import Trajectory
        trajectory = Trajectory()

        # Phase 1: Enhanced Differential Analysis
        if verbose:
            print("\n[Phase 1: Enhanced Differential Analysis]")
            print("  Analyzing transformations (including complex patterns)...")

        diff_output = self.node_registry.execute("enhanced_differential_analyzer", task_data)
        trajectory.add_node("enhanced_differential_analyzer", diff_output)

        analyses = diff_output.result

        if verbose:
            for i, analysis in enumerate(analyses):
                print(f"  Example {i}: {analysis['transformation_type']} "
                      f"(conf: {analysis['confidence']:.2f}) - {analysis['description']}")

            consensus = diff_output.artifacts.get('consensus_type', 'unknown')
            consensus_ratio = diff_output.artifacts.get('consensus_ratio', 0.0)
            print(f"  Consensus: {consensus} ({consensus_ratio*100:.0f}% agreement)")

        # Phase 2: Rule Inference (same as V3)
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
            print(f"  Description: {rule['description']}")

        # Phase 3: Enhanced Targeted Program Synthesis
        if verbose:
            print("\n[Phase 3: Enhanced Targeted Program Synthesis]")
            print("  Generating programs (including complex transformations)...")

        synth_input = {'rule': rule}
        synth_output = self.node_registry.execute("enhanced_targeted_synthesizer", synth_input)
        trajectory.add_node("enhanced_targeted_synthesizer", synth_output)

        programs = synth_output.result

        if verbose:
            print(f"  Generated {len(programs)} targeted programs:")
            for i, prog in enumerate(programs, 1):
                print(f"    {i}. {prog['type']}: {prog['description']} (conf: {prog['confidence']:.2f})")

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

                if verbose:
                    print(f"  {program['type']}: score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_program = program

                    if score >= 0.99:
                        if verbose:
                            print(f"    ★ EXCELLENT program found!")
                        break

            except Exception as e:
                if verbose:
                    print(f"  {program['type']}: failed ({e})")
                continue

        # Phase 5: Iterative Refinement
        if best_program and best_score < 0.99:
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
                print(f"  Refinement complete: {best_score:.3f} → {refined_score:.3f}")

            if refined_score > best_score:
                best_program = refined_program
                best_score = refined_score
                if verbose:
                    print(f"  ✓ Improvement achieved!")
            else:
                if verbose:
                    print(f"  No improvement from refinement")

        # Phase 6: Test Prediction
        if verbose:
            print(f"\n[Phase 6: Test Prediction]")
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


def main():
    """Demo V3+ solver."""
    print("="*70)
    print("ARC GRAPH PENDULUM SOLVER V3+ - ENHANCED RULE INFERENCE")
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

    # Create V3+ solver
    solver = ARCGraphPendulumSolverV3Plus(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False
    )

    # Test on previously failed tasks
    print("\n" + "="*70)
    print("TESTING ON PREVIOUSLY FAILED TASKS")
    print("="*70)

    failed_task_ids = ['007bbfb7', '017c7c7b']

    for task_id in failed_task_ids:
        if task_id in tasks:
            task = tasks[task_id]
            print(f"\n{'='*70}")
            print(f"TASK: {task_id} (V3 failed with 0.000)")
            print(f"{'='*70}")

            result = solver.evaluate_on_task(task, verbose=True)

            print(f"\n{'='*70}")
            if result['solved']:
                print(f"★ SUCCESS: Solved task {task_id}!")
            else:
                print(f"Result: {result['avg_score']:.3f} IoU")
            print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
