"""
Version 2 of ARC Graph Pendulum Solver with advanced program synthesis.
"""

import numpy as np
from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_enhanced import EnhancedARCGraphPendulumSolver
from nodes.advanced_synthesizer import create_advanced_program_synthesizer_node
from utils.arc_loader import ARCTask
from utils.grid_utils import compute_iou


class ARCGraphPendulumSolverV2(EnhancedARCGraphPendulumSolver):
    """
    Version 2 solver with advanced program synthesis (100+ diverse programs).
    """

    def __init__(
        self,
        beam_width: int = 5,
        use_stability: bool = True,
        use_repair_loops: bool = True,
        use_landscape_analytics: bool = False
    ):
        """Initialize V2 solver."""
        # Call parent init
        super().__init__(
            beam_width=beam_width,
            use_stability=use_stability,
            use_repair_loops=use_repair_loops,
            use_landscape_analytics=use_landscape_analytics
        )

        # Replace program synthesizer with advanced version
        self.node_registry.register(create_advanced_program_synthesizer_node())

        print(f"✓ Solver V2 Initialized (Advanced Synthesis)")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve task using advanced synthesis.
        """
        if verbose:
            print(f"\n=== Solving task {task.task_id} (V2 - Advanced Synthesis) ===")
            print(f"Train examples: {task.num_train}, Test examples: {task.num_test}")

        # Prepare task data
        task_data = {
            'task_id': task.task_id,
            'train': task.train,
            'test': task.test,
        }

        # Phase 1: Feature Extraction
        if verbose:
            print("\n[Phase 1: Feature Extraction]")

        from core.trajectory import Trajectory
        trajectory = Trajectory()

        extractor_nodes = [
            "color_histogram",
            "object_detector",
            "symmetry_detector",
            "periodicity_detector",
            "shape_detector"
        ]

        facts = {}
        for node_name in extractor_nodes:
            output = self.node_registry.execute(node_name, task_data)
            trajectory.add_node(node_name, output)
            facts.update(output.artifacts)

            if verbose:
                print(f"  ✓ {node_name}: {len(output.artifacts)} artifacts")

        # Phase 2: Hypothesis Generation
        if verbose:
            print("\n[Phase 2: Hypothesis Generation]")

        output = self.node_registry.execute("hypothesis_generator", facts)
        trajectory.add_node("hypothesis_generator", output)

        hypotheses = output.result
        if verbose:
            print(f"  Generated {len(hypotheses)} hypotheses")

        # Phase 3: ADVANCED Program Synthesis
        if verbose:
            print("\n[Phase 3: Advanced Program Synthesis]")

        synthesis_data = {
            'hypotheses': hypotheses,
            'facts': facts,
            'task_data': task_data,
        }

        # Use advanced synthesizer
        output = self.node_registry.execute("advanced_program_synthesizer", synthesis_data)
        trajectory.add_node("advanced_program_synthesizer", output)

        programs = output.result
        if verbose:
            print(f"  Synthesized {len(programs)} programs (ADVANCED)")
            # Show top 5
            for i, prog in enumerate(programs[:5], 1):
                print(f"    {i}. {prog['type']}: {prog['description']} (conf: {prog['confidence']:.2f})")

        # Phase 4: Execution and Evaluation
        if verbose:
            print("\n[Phase 4: Execution & Evaluation]")

        best_program = None
        best_score = 0.0
        tested_count = 0

        for program in programs:
            try:
                # Execute on training examples
                train_predictions = []
                train_targets = []
                train_inputs = []

                for input_grid, output_grid in task.train:
                    pred = program['function'](input_grid)
                    train_predictions.append(pred)
                    train_targets.append(output_grid)
                    train_inputs.append(input_grid)

                # Evaluate
                critic_data = {
                    'predictions': train_predictions,
                    'targets': train_targets,
                    'inputs': train_inputs,
                }

                iou_output = self.node_registry.execute("iou_critic", critic_data)
                score = iou_output.result

                tested_count += 1

                if verbose and (score > 0.5 or tested_count <= 10):
                    print(f"  {program['type']}: score={score:.3f}")

                # Track best
                if score > best_score:
                    best_score = score
                    best_program = program

                    if score >= 0.99 and verbose:
                        print(f"    ★ EXCELLENT program found!")

            except Exception as e:
                if verbose and tested_count <= 10:
                    print(f"  {program['type']}: failed ({e})")
                continue

        if verbose:
            print(f"  Tested {tested_count} programs, best score: {best_score:.3f}")

        # Phase 4.5: Repair Loops
        if self.use_repair_loops and best_program and best_score < 0.95:
            if verbose:
                print("\n[Phase 4.5: Repair Loops]")

            repaired_program, repair_score = self._apply_repair_loops(
                best_program,
                task,
                verbose=verbose
            )

            if repair_score > best_score:
                if verbose:
                    print(f"  ✓ Repair improved score: {best_score:.3f} → {repair_score:.3f}")
                best_program = repaired_program
                best_score = repair_score
            else:
                if verbose:
                    print(f"  No improvement from repairs")

        # Phase 5: Generate Test Predictions
        if verbose:
            print(f"\n[Phase 5: Test Prediction]")
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
    """Demo V2 solver."""
    print("="*70)
    print("ARC GRAPH PENDULUM SOLVER V2 - ADVANCED SYNTHESIS DEMO")
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

    # Create V2 solver
    solver = ARCGraphPendulumSolverV2(
        beam_width=5,
        use_stability=True,
        use_repair_loops=True,
        use_landscape_analytics=False
    )

    # Solve tasks
    print("\n" + "="*70)
    print("SOLVING WITH ADVANCED SYNTHESIS")
    print("="*70)

    results = []
    num_to_solve = len(tasks)

    task_list = list(tasks.values())[:num_to_solve]

    for i, task in enumerate(task_list):
        print(f"\n{'='*70}")
        print(f"TASK {i+1}/{num_to_solve}")
        print(f"{'='*70}")

        result = solver.evaluate_on_task(task, verbose=True)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - V2 ADVANCED SYNTHESIS")
    print("="*70)

    solved_count = sum(1 for r in results if r['solved'])
    avg_score = np.mean([r['avg_score'] for r in results])
    high_quality = sum(1 for r in results if r['avg_score'] >= 0.8)

    print(f"\nPerformance Metrics:")
    print(f"  Perfect solves (100%): {solved_count}/{len(results)} ({solved_count/len(results)*100:.1f}%)")
    print(f"  High quality (>0.80): {high_quality}/{len(results)} ({high_quality/len(results)*100:.1f}%)")
    print(f"  Average IoU: {avg_score:.3f}")

    print("\nDetailed Results:")
    print("-" * 70)

    for i, result in enumerate(results):
        status = "✓ SOLVED" if result['solved'] else "✗ FAILED"
        quality = "HIGH" if result['avg_score'] >= 0.8 else "MED" if result['avg_score'] >= 0.5 else "LOW"

        print(f"{i+1:2d}. {result['task_id']:12s} {status} {quality:4s} (IoU: {result['avg_score']:.3f})")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
