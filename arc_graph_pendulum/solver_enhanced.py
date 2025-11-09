"""
Enhanced ARC Graph Pendulum Solver with repair loops and landscape analytics.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver import ARCGraphPendulumSolver
from core.landscape import LandscapeAnalyzer
from nodes.repairers import (
    create_placement_repairer_node,
    create_color_repairer_node,
    create_scale_repairer_node,
)
from utils.arc_loader import ARCTask
from utils.grid_utils import compute_iou


class EnhancedARCGraphPendulumSolver(ARCGraphPendulumSolver):
    """
    Enhanced solver with repair loops and landscape analytics.
    """

    def __init__(
        self,
        beam_width: int = 5,
        use_stability: bool = True,
        use_repair_loops: bool = True,
        use_landscape_analytics: bool = True
    ):
        """
        Initialize the enhanced solver.

        Args:
            beam_width: Width for beam search
            use_stability: Whether to use stability-aware search
            use_repair_loops: Whether to use repair loops
            use_landscape_analytics: Whether to use landscape analytics
        """
        super().__init__(beam_width=beam_width, use_stability=use_stability)

        self.use_repair_loops = use_repair_loops
        self.use_landscape_analytics = use_landscape_analytics

        # Register repair nodes
        if use_repair_loops:
            self.node_registry.register(create_placement_repairer_node())
            self.node_registry.register(create_color_repairer_node())
            self.node_registry.register(create_scale_repairer_node())

        # Initialize landscape analyzer
        if use_landscape_analytics:
            self.landscape_analyzer = LandscapeAnalyzer(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2
            )

        print(f"✓ Enhanced Solver Initialized")
        if use_repair_loops:
            print(f"  Repair loops: ENABLED")
        if use_landscape_analytics:
            print(f"  Landscape analytics: ENABLED")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve an ARC task with repair loops.

        Args:
            task: ARCTask to solve
            verbose: Whether to print progress

        Returns:
            List of predicted output grids for test inputs
        """
        if verbose:
            print(f"\n=== Solving task {task.task_id} (Enhanced) ===")
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

        # Phase 3: Program Synthesis
        if verbose:
            print("\n[Phase 3: Program Synthesis]")

        synthesis_data = {
            'hypotheses': hypotheses,
            'facts': facts,
            'task_data': task_data,
        }

        output = self.node_registry.execute("program_synthesizer", synthesis_data)
        trajectory.add_node("program_synthesizer", output)

        programs = output.result
        if verbose:
            print(f"  Synthesized {len(programs)} programs")

        # Phase 4: Execution and Evaluation
        if verbose:
            print("\n[Phase 4: Execution & Evaluation]")

        best_program = None
        best_score = 0.0

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

                if verbose:
                    print(f"  {program['type']}: score={score:.3f}")

                # Track best
                if score > best_score:
                    best_score = score
                    best_program = program

            except Exception as e:
                if verbose:
                    print(f"  {program['type']}: failed ({e})")
                continue

        # Phase 4.5: Repair Loops (NEW!)
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

        # Log trajectory for landscape analytics
        self.trajectory_batch.add(trajectory)

        if self.use_landscape_analytics:
            self.landscape_analyzer.add_trajectory(trajectory, task.task_id)

        return predictions

    def _apply_repair_loops(
        self,
        program: Dict[str, Any],
        task: ARCTask,
        verbose: bool = True
    ) -> tuple[Dict[str, Any], float]:
        """
        Apply repair loops to improve a program.

        Args:
            program: Program to repair
            task: ARCTask
            verbose: Whether to print progress

        Returns:
            (repaired_program, score)
        """
        # Execute program on training data
        train_predictions = []
        train_targets = []
        train_inputs = []

        for input_grid, output_grid in task.train:
            try:
                pred = program['function'](input_grid)
                train_predictions.append(pred)
                train_targets.append(output_grid)
                train_inputs.append(input_grid)
            except:
                return program, 0.0

        # Analyze failures
        failure_data = {
            'predictions': train_predictions,
            'targets': train_targets,
            'inputs': train_inputs,
        }

        failure_output = self.node_registry.execute("failure_analyzer", failure_data)
        taxonomy = failure_output.result

        if verbose:
            dominant_failure = failure_output.artifacts.get('dominant_failure', 'unknown')
            print(f"  Dominant failure type: {dominant_failure}")

        # Try appropriate repairer based on failure type
        repairers_to_try = []

        if taxonomy.get('placement_error', 0) > 0 or taxonomy.get('partial_correct', 0) > 0:
            repairers_to_try.append('placement_repairer')

        if taxonomy.get('color_error', 0) > 0:
            repairers_to_try.append('color_repairer')

        if taxonomy.get('shape_mismatch', 0) > 0:
            repairers_to_try.append('scale_repairer')

        # If no specific repairer, try all
        if not repairers_to_try:
            repairers_to_try = ['placement_repairer', 'color_repairer', 'scale_repairer']

        best_repaired = program
        best_repair_score = 0.0

        for repairer_name in repairers_to_try:
            if verbose:
                print(f"  Trying {repairer_name}...")

            # Prepare data for repairer
            repair_data = {
                'program': program,
                'predictions': train_predictions,
                'targets': train_targets,
                'inputs': train_inputs,
            }

            # Execute repairer
            repair_output = self.node_registry.execute(repairer_name, repair_data)

            if not repair_output.telemetry.get('repaired', False):
                continue

            repaired_program = repair_output.result

            # Test repaired program
            repaired_predictions = []
            try:
                for input_grid, _ in task.train:
                    pred = repaired_program['function'](input_grid)
                    repaired_predictions.append(pred)

                # Evaluate
                critic_data = {
                    'predictions': repaired_predictions,
                    'targets': train_targets,
                    'inputs': train_inputs,
                }

                iou_output = self.node_registry.execute("iou_critic", critic_data)
                repair_score = iou_output.result

                if verbose:
                    print(f"    {repairer_name}: score={repair_score:.3f}")

                if repair_score > best_repair_score:
                    best_repair_score = repair_score
                    best_repaired = repaired_program

            except Exception as e:
                if verbose:
                    print(f"    {repairer_name}: failed ({e})")
                continue

        return best_repaired, best_repair_score

    def analyze_landscape(self, verbose: bool = True) -> Optional[Dict[str, Any]]:
        """
        Analyze the trajectory landscape.

        Args:
            verbose: Whether to print progress

        Returns:
            Dictionary with landscape analysis results
        """
        if not self.use_landscape_analytics:
            print("Landscape analytics not enabled")
            return None

        if len(self.landscape_analyzer.points) < 3:
            print(f"Not enough trajectories for landscape analysis (need at least 3, have {len(self.landscape_analyzer.points)})")
            return None

        if verbose:
            print("\n=== Landscape Analytics ===")

        # Compute embeddings
        if verbose:
            print("Computing UMAP embeddings...")

        embeddings = self.landscape_analyzer.compute_embeddings(use_umap=True)

        # Discover basins
        if verbose:
            print("Discovering basins...")

        # Use DBSCAN for automatic basin discovery
        labels = self.landscape_analyzer.discover_basins(
            method='dbscan',
            eps=0.5,
            min_samples=2
        )

        # Get statistics
        stats = self.landscape_analyzer.get_basin_statistics()

        if verbose:
            self.landscape_analyzer.print_summary()

        return {
            'embeddings': embeddings,
            'labels': labels,
            'statistics': stats,
        }

    def visualize_landscape(self, output_path: str = "landscape.png"):
        """
        Visualize the trajectory landscape.

        Args:
            output_path: Path to save visualization
        """
        if not self.use_landscape_analytics:
            print("Landscape analytics not enabled")
            return

        self.landscape_analyzer.visualize_landscape(output_path)

    def save_landscape_analysis(self, output_path: str = "landscape_analysis.json"):
        """
        Save landscape analysis to JSON.

        Args:
            output_path: Path to save JSON
        """
        if not self.use_landscape_analytics:
            print("Landscape analytics not enabled")
            return

        self.landscape_analyzer.save_analysis(output_path)


def main():
    """Demo the enhanced solver."""
    print("=== Enhanced ARC Graph Pendulum Solver Demo ===\n")

    # Load ARC dataset
    from utils.arc_loader import ARCLoader

    loader = ARCLoader(cache_dir="./arc_data")
    print("Loading ARC dataset...")
    tasks = loader.load_all_tasks("training")

    if not tasks:
        print("No tasks loaded.")
        return

    print(f"Loaded {len(tasks)} tasks\n")

    # Create enhanced solver
    solver = EnhancedARCGraphPendulumSolver(
        beam_width=5,
        use_stability=True,
        use_repair_loops=True,
        use_landscape_analytics=True
    )

    # Solve tasks
    results = []
    num_to_solve = min(10, len(tasks))

    task_list = list(tasks.values())[:num_to_solve]

    for task in task_list:
        result = solver.evaluate_on_task(task, verbose=True)
        results.append(result)

    # Landscape analysis
    print("\n" + "="*60)
    landscape_results = solver.analyze_landscape(verbose=True)

    if landscape_results:
        # Visualize
        solver.visualize_landscape("arc_landscape.png")
        solver.save_landscape_analysis("arc_landscape_analysis.json")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    solved_count = sum(1 for r in results if r['solved'])
    avg_score = np.mean([r['avg_score'] for r in results])

    print(f"Tasks solved: {solved_count}/{len(results)}")
    print(f"Average score: {avg_score:.3f}")

    print("\nDetailed Results:")
    for result in results:
        status = "✓ SOLVED" if result['solved'] else "✗ FAILED"
        print(f"  {result['task_id']}: {status} (score: {result['avg_score']:.3f})")


if __name__ == "__main__":
    main()
