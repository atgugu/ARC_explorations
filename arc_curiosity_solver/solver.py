"""
Main ARC Curiosity-Driven Active Inference Solver

Integrates all components:
- Curiosity signals
- Belief dynamics
- Active inference
- Hierarchical reasoning
- Transformation discovery
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import json
from pathlib import Path

from arc_curiosity_solver.curiosity.signals import (
    CuriositySignals, TaskCuriosityScorer, HypothesisCuriosityScorer
)
from arc_curiosity_solver.belief_dynamics.belief_space import (
    BeliefSpace, HierarchicalBeliefSpace, Hypothesis
)
from arc_curiosity_solver.active_inference.engine import (
    ActiveInferenceEngine, InferenceState
)
from arc_curiosity_solver.core.hierarchical_solver import (
    HierarchicalSolver, Task, WorkspaceItem
)
from arc_curiosity_solver.transformations.arc_primitives import (
    TransformLibrary, CompositeTransform, ARCPrimitives
)


class ARCCuriositySolver:
    """
    Complete ARC solver with curiosity-driven active inference.

    Always produces 2 best predictions.
    """

    def __init__(self,
                 workspace_capacity: int = 7,
                 learning_rate: float = 0.1,
                 exploration_bonus: float = 1.0,
                 n_hypotheses_to_explore: int = 50):
        """
        Initialize the solver.

        Args:
            workspace_capacity: Working memory capacity (7±2)
            learning_rate: Active inference learning rate
            exploration_bonus: Exploration bonus for Generator
            n_hypotheses_to_explore: Number of transformation hypotheses to generate
        """
        # Core components
        self.curiosity = CuriositySignals()
        self.task_curiosity = TaskCuriosityScorer(self.curiosity)
        self.hypothesis_curiosity = HypothesisCuriosityScorer(self.curiosity)

        self.belief_space = HierarchicalBeliefSpace()
        self.active_inference = ActiveInferenceEngine(learning_rate=learning_rate)

        self.hierarchical_solver = HierarchicalSolver(
            workspace_capacity=workspace_capacity,
            exploration_bonus=exploration_bonus
        )

        self.transform_library = TransformLibrary()

        self.n_hypotheses_to_explore = n_hypotheses_to_explore
        self.solve_attempts = 0

    def solve(self,
             train_pairs: List[Tuple[np.ndarray, np.ndarray]],
             test_input: np.ndarray,
             verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve an ARC task and return TWO predictions.

        Args:
            train_pairs: List of (input, output) training pairs
            test_input: Test input grid
            verbose: Whether to print progress

        Returns:
            Tuple of (prediction1, prediction2)
        """
        self.solve_attempts += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"Solving ARC Task (Attempt {self.solve_attempts})")
            print(f"Train pairs: {len(train_pairs)}")
            print(f"Test input shape: {test_input.shape}")
            print(f"{'='*60}\n")

        # Phase 1: Generate transformation hypotheses
        if verbose:
            print("Phase 1: Generating hypotheses...")

        hypotheses = self._generate_hypotheses(train_pairs, test_input)

        if verbose:
            print(f"  Generated {len(hypotheses)} hypotheses\n")

        # Phase 2: Active inference - iteratively update beliefs
        if verbose:
            print("Phase 2: Active inference with belief updating...")

        for i, (inp, out) in enumerate(train_pairs):
            if verbose:
                print(f"  Processing training pair {i+1}/{len(train_pairs)}")

            self._active_inference_step(hypotheses, inp, out, verbose=verbose)

        # Phase 3: Select top 2 hypotheses
        if verbose:
            print("\nPhase 3: Selecting top 2 predictions...")

        top_hypotheses = self._select_top_k_hypotheses(hypotheses, k=2)

        # Phase 4: Generate predictions
        predictions = []
        for i, (hyp, belief) in enumerate(top_hypotheses):
            if verbose:
                print(f"  Prediction {i+1}: {hyp.name} (belief={belief:.3f})")

            try:
                pred = self._apply_hypothesis(hyp, test_input)
                predictions.append(pred)
            except Exception as e:
                if verbose:
                    print(f"    Failed to apply: {e}")
                # Fallback: return test input
                predictions.append(test_input.copy())

        # Ensure we always return exactly 2 predictions
        while len(predictions) < 2:
            predictions.append(test_input.copy())

        if verbose:
            print(f"\n{'='*60}")
            print(f"Solution Complete!")
            print(f"{'='*60}\n")

        return predictions[0], predictions[1]

    def _generate_hypotheses(self,
                           train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                           test_input: np.ndarray) -> List[Hypothesis]:
        """Generate candidate transformation hypotheses."""
        hypotheses = []

        # Strategy 1: Try single primitive transformations
        for name, transform in self.transform_library.library.items():
            # Test if this transform works on training data
            fit_score = self._evaluate_transform_fit(transform, train_pairs)

            hyp = Hypothesis(
                program=transform,
                name=name,
                parameters=transform.parameters.copy(),
                activation=0.0,
                evidence_count=0,
                success_count=0
            )

            hypotheses.append(hyp)

        # Strategy 2: Try composite transformations (2-step)
        spatial_transforms = self.transform_library.get_all_spatial()

        # Generate some composite hypotheses
        n_composites = min(20, len(spatial_transforms) * 2)
        for _ in range(n_composites):
            # Randomly combine 2 transforms
            if len(spatial_transforms) >= 2:
                t1, t2 = np.random.choice(spatial_transforms, 2, replace=False)

                composite = CompositeTransform(f"{t1.name}+{t2.name}")
                composite.add(t1)
                composite.add(t2)

                fit_score = self._evaluate_composite_fit(composite, train_pairs)

                hyp = Hypothesis(
                    program=composite,
                    name=composite.name,
                    parameters={},
                    activation=0.0,
                    evidence_count=0,
                    success_count=0
                )

                hypotheses.append(hyp)

        # Strategy 3: Add identity transform (baseline)
        identity = Hypothesis(
            program=None,
            name="identity",
            parameters={},
            activation=0.0,
            evidence_count=0,
            success_count=0
        )
        hypotheses.append(identity)

        # Limit total number of hypotheses
        if len(hypotheses) > self.n_hypotheses_to_explore:
            # Keep most promising ones
            hypotheses = sorted(hypotheses, key=lambda h: h.success_count, reverse=True)
            hypotheses = hypotheses[:self.n_hypotheses_to_explore]

        return hypotheses

    def _evaluate_transform_fit(self,
                               transform,
                               train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate how well a transform fits the training data."""
        if not train_pairs:
            return 0.0

        total_score = 0.0

        for inp, out in train_pairs:
            try:
                pred = transform.function(inp, **transform.parameters)

                # Compute similarity
                if pred.shape == out.shape:
                    match = np.sum(pred == out)
                    total = out.size
                    score = match / total
                else:
                    # Shape mismatch - low score but not zero
                    score = 0.1

                total_score += score

            except Exception:
                # Transform failed
                total_score += 0.0

        return total_score / len(train_pairs)

    def _evaluate_composite_fit(self,
                               composite: CompositeTransform,
                               train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate composite transform fit."""
        if not train_pairs:
            return 0.0

        total_score = 0.0

        for inp, out in train_pairs:
            try:
                pred = composite.apply(inp)

                if pred.shape == out.shape:
                    match = np.sum(pred == out)
                    total = out.size
                    score = match / total
                else:
                    score = 0.1

                total_score += score

            except Exception:
                total_score += 0.0

        return total_score / len(train_pairs)

    def _active_inference_step(self,
                              hypotheses: List[Hypothesis],
                              input_grid: np.ndarray,
                              output_grid: np.ndarray,
                              verbose: bool = False):
        """
        Perform one step of active inference: update beliefs based on evidence.
        """
        # Evaluate each hypothesis on this example
        likelihoods = []

        for hyp in hypotheses:
            try:
                # Apply hypothesis to input
                if hyp.name == "identity":
                    pred = input_grid.copy()
                elif isinstance(hyp.program, CompositeTransform):
                    pred = hyp.program.apply(input_grid)
                else:
                    pred = hyp.program.function(input_grid, **hyp.program.parameters)

                # Compute likelihood based on match
                if pred.shape == output_grid.shape:
                    match = np.sum(pred == output_grid)
                    total = output_grid.size
                    likelihood = match / total

                    if match == total:
                        hyp.success_count += 1
                else:
                    likelihood = 0.01  # Shape mismatch

                likelihoods.append(likelihood)
                hyp.evidence_count += 1

            except Exception:
                likelihoods.append(0.01)  # Failed to apply

        # Update beliefs using active inference
        likelihoods = np.array(likelihoods)

        # Bayesian update
        activations = np.array([h.activation for h in hypotheses])

        # Initialize if first update
        if activations.sum() == 0:
            activations = np.ones(len(hypotheses)) / len(hypotheses)

        # Bayesian update
        new_activations = activations * (likelihoods + 0.01)  # Add small epsilon

        if new_activations.sum() > 0:
            new_activations = new_activations / new_activations.sum()

        # Update hypothesis activations
        for i, hyp in enumerate(hypotheses):
            hyp.activation = new_activations[i]

        # Compute curiosity signals
        surprise = self.curiosity.bayesian_surprise(
            prior_params={'mean': activations, 'cov': np.eye(len(activations)) * 0.1},
            posterior_params={'mean': new_activations, 'cov': np.eye(len(new_activations)) * 0.1}
        )

        if verbose:
            print(f"    Surprise: {surprise:.4f}, Top belief: {np.max(new_activations):.4f}")

    def _select_top_k_hypotheses(self,
                                hypotheses: List[Hypothesis],
                                k: int = 2) -> List[Tuple[Hypothesis, float]]:
        """Select top k hypotheses by activation/belief."""
        # Sort by activation
        sorted_hyps = sorted(hypotheses, key=lambda h: h.activation, reverse=True)

        # Return top k with their beliefs
        return [(h, h.activation) for h in sorted_hyps[:k]]

    def _apply_hypothesis(self,
                         hypothesis: Hypothesis,
                         input_grid: np.ndarray) -> np.ndarray:
        """Apply a hypothesis to generate prediction."""
        if hypothesis.name == "identity":
            return input_grid.copy()

        if isinstance(hypothesis.program, CompositeTransform):
            return hypothesis.program.apply(input_grid)

        return hypothesis.program.function(input_grid, **hypothesis.program.parameters)

    def get_solver_state(self) -> Dict[str, Any]:
        """Get complete solver state for analysis."""
        return {
            'solve_attempts': self.solve_attempts,
            'hierarchical_system': self.hierarchical_solver.get_system_state(),
            'belief_entropy': self.belief_space.overall_entropy(),
        }


def load_arc_task(task_file: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Load an ARC task from JSON file.

    Args:
        task_file: Path to ARC task JSON file

    Returns:
        Tuple of (train_pairs, test_input)
    """
    with open(task_file, 'r') as f:
        data = json.load(f)

    # Extract training pairs
    train_pairs = []
    for example in data['train']:
        inp = np.array(example['input'])
        out = np.array(example['output'])
        train_pairs.append((inp, out))

    # Extract test input (use first test case)
    test_input = np.array(data['test'][0]['input'])

    return train_pairs, test_input


def visualize_grids(grids: List[np.ndarray], titles: List[str] = None):
    """Simple text visualization of grids."""
    for i, grid in enumerate(grids):
        if titles:
            print(f"\n{titles[i]}:")
        else:
            print(f"\nGrid {i+1}:")

        print("-" * (grid.shape[1] * 2 + 1))
        for row in grid:
            print("|" + " ".join(str(int(x)) for x in row) + "|")
        print("-" * (grid.shape[1] * 2 + 1))


if __name__ == "__main__":
    # Example usage
    print("ARC Curiosity-Driven Active Inference Solver")
    print("=" * 60)

    # Create solver
    solver = ARCCuriositySolver(
        workspace_capacity=7,
        learning_rate=0.1,
        exploration_bonus=1.0
    )

    # Create a simple test task
    # Task: translate right by 1
    train_input_1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    train_output_1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    train_input_2 = np.array([[2, 2, 0], [0, 0, 0], [0, 0, 0]])
    train_output_2 = np.array([[0, 2, 2], [0, 0, 0], [0, 0, 0]])

    test_input = np.array([[3, 3, 3], [0, 0, 0], [0, 0, 0]])

    train_pairs = [(train_input_1, train_output_1), (train_input_2, train_output_2)]

    print("\nSolving example task...")
    print("\nTraining pairs:")
    visualize_grids([train_input_1, train_output_1], ["Input 1", "Output 1"])
    visualize_grids([train_input_2, train_output_2], ["Input 2", "Output 2"])

    print("\nTest input:")
    visualize_grids([test_input], ["Test"])

    # Solve
    pred1, pred2 = solver.solve(train_pairs, test_input, verbose=True)

    print("\nPredictions:")
    visualize_grids([pred1, pred2], ["Prediction 1", "Prediction 2"])

    print("\nSolver state:")
    print(json.dumps(solver.get_solver_state(), indent=2))
