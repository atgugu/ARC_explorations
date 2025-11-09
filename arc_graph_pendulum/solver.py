"""
Main ARC Graph Pendulum Solver.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.node import NodeRegistry
from core.edge import EdgeRegistry, Edge
from core.trajectory import Trajectory, TrajectoryBatch
from core.basin import BasinRegistry
from core.behavior import BehaviorVectorBuilder, create_simple_probe_bank
from core.stability import StabilityMeter
from core.controller import Controller

from nodes.extractors import (
    create_color_histogram_node,
    create_object_detector_node,
    create_symmetry_detector_node,
    create_periodicity_detector_node,
    create_shape_detector_node,
)

from nodes.reasoners import (
    create_hypothesis_generator_node,
    create_program_synthesizer_node,
)

from nodes.critics import (
    create_iou_critic_node,
    create_failure_analyzer_node,
)

from utils.arc_loader import ARCTask, ARCLoader
from utils.grid_utils import compute_iou


class ARCGraphPendulumSolver:
    """
    Main solver implementing the Graph Pendulum System.
    """

    def __init__(self, beam_width: int = 5, use_stability: bool = True):
        """
        Initialize the solver.

        Args:
            beam_width: Width for beam search
            use_stability: Whether to use stability-aware search
        """
        self.beam_width = beam_width
        self.use_stability = use_stability

        # Initialize registries
        self.node_registry = NodeRegistry()
        self.edge_registry = EdgeRegistry()
        self.basin_registry = BasinRegistry()

        # Initialize components
        self.trajectory_batch = TrajectoryBatch()
        self.stability_meter = StabilityMeter(num_perturbations=3)

        # Build probe bank and behavior vector builder
        self.probe_bank = create_simple_probe_bank()
        self.behavior_builder = BehaviorVectorBuilder(self.probe_bank)

        # Register nodes
        self._register_nodes()

        # Build graph structure
        self._build_graph()

        # Initialize controller
        self.controller = Controller(
            self.node_registry,
            self.edge_registry,
            beam_width=beam_width,
            max_depth=8,
            stability_weight=0.3
        )

        print(f"✓ Initialized ARC Graph Pendulum Solver")
        print(f"  Nodes: {len(self.node_registry.nodes)}")
        print(f"  Edges: {len(self.edge_registry.edges)}")

    def _register_nodes(self):
        """Register all nodes in the system."""
        # Extractors
        self.node_registry.register(create_color_histogram_node())
        self.node_registry.register(create_object_detector_node())
        self.node_registry.register(create_symmetry_detector_node())
        self.node_registry.register(create_periodicity_detector_node())
        self.node_registry.register(create_shape_detector_node())

        # Reasoners
        self.node_registry.register(create_hypothesis_generator_node())
        self.node_registry.register(create_program_synthesizer_node())

        # Critics
        self.node_registry.register(create_iou_critic_node())
        self.node_registry.register(create_failure_analyzer_node())

    def _build_graph(self):
        """Build the graph structure with edges."""
        # Compute behavior vectors
        print("Computing behavior vectors...")
        behavior_vectors = self.behavior_builder.compute_all_vectors(
            self.node_registry,
            vector_dim=256
        )

        self.node_registry.behavior_vectors = behavior_vectors

        # Create edges between compatible nodes
        extractors = self.node_registry.get_nodes_by_category("extractor")
        reasoners = self.node_registry.get_nodes_by_category("reasoner")
        critics = self.node_registry.get_nodes_by_category("critic")

        # Extractors -> Reasoners
        for extractor in extractors:
            for reasoner in reasoners:
                edge = Edge(
                    source=extractor.name,
                    target=reasoner.name,
                    utility=0.5,  # Initial neutral utility
                )
                self.edge_registry.add_edge(edge)

        # Reasoners -> Critics
        for reasoner in reasoners:
            for critic in critics:
                edge = Edge(
                    source=reasoner.name,
                    target=critic.name,
                    utility=0.5,
                )
                self.edge_registry.add_edge(edge)

        # Compute angles and distances
        self.edge_registry.compute_angles_and_distances(behavior_vectors)

        print(f"✓ Built graph with {len(self.edge_registry.edges)} edges")

    def solve_task(self, task: ARCTask, verbose: bool = True) -> List[np.ndarray]:
        """
        Solve an ARC task.

        Args:
            task: ARCTask to solve
            verbose: Whether to print progress

        Returns:
            List of predicted output grids for test inputs
        """
        # Clear cache before each task to prevent stale results
        self.node_registry.clear_cache()

        if verbose:
            print(f"\n=== Solving task {task.task_id} ===")
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

        extractor_nodes = [
            "color_histogram",
            "object_detector",
            "symmetry_detector",
            "periodicity_detector",
            "shape_detector"
        ]

        facts = {}
        trajectory = Trajectory()

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
            print(f"  Generated {len(hypotheses)} hypotheses:")
            for hyp in hypotheses[:3]:
                print(f"    - {hyp['type']}: {hyp['description']} (conf: {hyp['confidence']:.2f})")

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
        predictions = []

        for program in programs:
            try:
                # Execute on training examples
                train_predictions = []
                train_targets = []

                for input_grid, output_grid in task.train:
                    pred = program['function'](input_grid)
                    train_predictions.append(pred)
                    train_targets.append(output_grid)

                # Evaluate
                critic_data = {
                    'predictions': train_predictions,
                    'targets': train_targets,
                    'inputs': [inp for inp, _ in task.train],
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

        # Phase 5: Generate Test Predictions
        if verbose:
            print(f"\n[Phase 5: Test Prediction]")
            print(f"Best program: {best_program['type'] if best_program else 'None'} (score={best_score:.3f})")

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
                    # Fallback: return input
                    predictions.append(test_input.copy())
        else:
            # No program worked, return inputs
            predictions = [test_input.copy() for test_input, _ in task.test]

        # Update trajectory
        trajectory.final_score = best_score

        # Log trajectory
        self.trajectory_batch.add(trajectory)

        return predictions

    def evaluate_on_task(self, task: ARCTask, verbose: bool = True) -> Dict[str, Any]:
        """
        Solve and evaluate a task with ground truth.

        Args:
            task: ARCTask with test outputs
            verbose: Whether to print progress

        Returns:
            Dictionary with evaluation results
        """
        predictions = self.solve_task(task, verbose=verbose)

        # Compute metrics
        scores = []
        for i, (pred, (test_input, test_output)) in enumerate(zip(predictions, task.test)):
            score = compute_iou(pred, test_output)
            scores.append(score)

            if verbose:
                print(f"  Test {i} IoU: {score:.3f}")

        avg_score = np.mean(scores) if scores else 0.0
        solved = avg_score >= 0.99

        if verbose:
            print(f"\n{'✓' if solved else '✗'} Average IoU: {avg_score:.3f} ({'SOLVED' if solved else 'FAILED'})")

        return {
            'task_id': task.task_id,
            'predictions': predictions,
            'scores': scores,
            'avg_score': avg_score,
            'solved': solved,
        }


def main():
    """Demo the solver on ARC tasks."""
    print("=== ARC Graph Pendulum Solver Demo ===\n")

    # Load ARC dataset
    loader = ARCLoader(cache_dir="./arc_data")
    print("Downloading ARC dataset...")
    loader.download_dataset("training")

    # Load tasks
    tasks = loader.load_all_tasks("training")

    if not tasks:
        print("No tasks loaded. Please check dataset download.")
        return

    print(f"\nLoaded {len(tasks)} tasks")

    # Create solver
    solver = ARCGraphPendulumSolver(beam_width=5, use_stability=True)

    # Solve a few tasks
    results = []
    num_to_solve = min(5, len(tasks))

    task_list = list(tasks.values())[:num_to_solve]

    for task in task_list:
        result = solver.evaluate_on_task(task, verbose=True)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    solved_count = sum(1 for r in results if r['solved'])
    avg_score = np.mean([r['avg_score'] for r in results])

    print(f"Tasks solved: {solved_count}/{len(results)}")
    print(f"Average score: {avg_score:.3f}")

    for result in results:
        status = "✓ SOLVED" if result['solved'] else "✗ FAILED"
        print(f"  {result['task_id']}: {status} (score: {result['avg_score']:.3f})")


if __name__ == "__main__":
    main()
