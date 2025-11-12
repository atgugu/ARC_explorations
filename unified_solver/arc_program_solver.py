"""
ARC Program Synthesis Solver
============================

Unified solver using compositional program synthesis instead of fixed primitives.
Extends the Active Inference framework with program generation.
"""

import numpy as np
from typing import List, Dict, Tuple
import time

from arc_active_inference_solver import (
    ARCTask, Grid, Hypothesis, BeliefState,
    ActiveInferenceEngine, StabilityFilter, WorkspaceController
)
from arc_program_synthesis import (
    ProgramSynthesisHypothesisGenerator,
    Program, ProgramSynthesizer
)


class ARCProgramSolver:
    """
    ARC Solver with Program Synthesis

    Replaces fixed primitive library with compositional program synthesis.
    Keeps Active Inference framework for hypothesis management.
    """

    def __init__(self,
                 workspace_capacity: int = 20,
                 n_perturbations: int = 5,
                 max_synthesis_depth: int = 2,
                 max_programs: int = 100,
                 verbose: bool = False):
        """
        Initialize solver with program synthesis

        Args:
            workspace_capacity: Max hypotheses in working memory
            n_perturbations: Number of perturbations for stability testing
            max_synthesis_depth: Max depth for program synthesis (1-3)
            max_programs: Max programs to generate
            verbose: Print detailed progress
        """
        # Program synthesis generator (replaces fixed primitive generator)
        self.generator = ProgramSynthesisHypothesisGenerator(
            max_depth=max_synthesis_depth,
            max_programs=max_programs
        )

        # Keep existing Active Inference components
        self.active_inference = ActiveInferenceEngine()
        self.stability_filter = StabilityFilter(n_perturbations)
        self.workspace = WorkspaceController(workspace_capacity)

        self.verbose = verbose
        self.max_synthesis_depth = max_synthesis_depth
        self.max_programs = max_programs

    def solve(self, task: ARCTask, verbose: bool = None) -> List[Grid]:
        """
        Solve ARC task using program synthesis

        Returns: List of exactly 2 predictions (top-2 diverse programs)
        """
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print(f"\n{'='*80}")
            print(f"ARC Program Synthesis Solver")
            print(f"{'='*80}")
            print(f"Training examples: {len(task.train_pairs)}")
            print(f"Test input shape: {task.test_input.shape}")
            print(f"Synthesis depth: {self.max_synthesis_depth}")
            print(f"Max programs: {self.max_programs}")

        # Step 1: Perceive patterns (minimal - features not heavily used)
        features = {}
        if verbose:
            print(f"\n[Step 1] Perception: Extracting features")

        # Step 2: Generate hypotheses via program synthesis
        if verbose:
            print(f"\n[Step 2] Program Synthesis: Generating compositional programs")

        start_time = time.time()
        hypotheses = self.generator.generate_hypotheses(task, features, verbose=verbose)
        synthesis_time = time.time() - start_time

        if verbose:
            print(f"  Generated {len(hypotheses)} programs in {synthesis_time:.2f}s")

        if len(hypotheses) == 0:
            if verbose:
                print("  ⚠ No programs generated, returning input as-is")
            return [task.test_input.copy(), task.test_input.copy()]

        # Step 3: Initialize beliefs
        if verbose:
            print(f"\n[Step 3] Active Inference: Initializing belief distribution")

        belief = self.active_inference.initialize_beliefs(hypotheses)

        if verbose:
            print(f"  Uniform prior over {len(hypotheses)} programs")

        # Step 4: Update beliefs based on training examples (Active Inference)
        if verbose:
            print(f"\n[Step 4] Active Inference: Updating beliefs from training")

        for i, (input_grid, output_grid) in enumerate(task.train_pairs):
            if verbose:
                print(f"  Training example {i+1}/{len(task.train_pairs)}")

            belief = self.active_inference.update_beliefs(
                belief, (input_grid, output_grid), hypotheses
            )

        # Step 5: Assess stability
        if verbose:
            print(f"\n[Step 5] Stability: Assessing robustness")

        for h in hypotheses:
            if belief.stability_scores.get(h) is None:
                stability = self.stability_filter.assess_stability(
                    h, task, belief
                )
                belief.stability_scores[h] = stability

        # Step 6: Select hypotheses for workspace (attention)
        if verbose:
            print(f"\n[Step 6] Workspace: Selecting attended programs")

        attended_hypotheses = self.workspace.select_hypotheses(
            hypotheses, belief
        )

        if verbose:
            print(f"  Selected {len(attended_hypotheses)} programs for workspace")

        # Step 7: Rank by posterior × stability
        if verbose:
            print(f"\n[Step 7] Ranking: Scoring programs")

        final_scores = {}
        for h in attended_hypotheses:
            posterior = belief.probabilities.get(h, 0.0)
            stability = belief.stability_scores.get(h, None)
            if stability is None:
                stability = 0.0
            final_scores[h] = posterior * stability

        # Step 8: Select diverse top-2
        if verbose:
            print(f"\n[Step 8] Selection: Choosing diverse top-2 predictions")

        top_2_hypotheses = self._select_diverse_top_2(
            attended_hypotheses, final_scores, task.test_input, verbose=verbose
        )

        # Step 9: Apply top-2 to test input
        if verbose:
            print(f"\n[Step 9] Execution: Generating predictions")

        predictions = []
        for i, h in enumerate(top_2_hypotheses):
            prediction = h.apply(task.test_input)
            predictions.append(prediction)
            if verbose:
                print(f"  Prediction {i+1}: shape={prediction.shape}, program={h.name[:60]}")

        # Ensure exactly 2 predictions
        while len(predictions) < 2:
            predictions.append(task.test_input.copy())

        if verbose:
            print(f"\n{'='*80}")
            print(f"✓ Solving complete")
            print(f"{'='*80}\n")

        return predictions[:2]

    def _select_diverse_top_2(self,
                             hypotheses: List[Hypothesis],
                             scores: Dict[Hypothesis, float],
                             test_input: Grid,
                             verbose: bool = False) -> List[Hypothesis]:
        """
        Select top-2 hypotheses ensuring different outputs

        Same diversity enforcement as baseline solver
        """
        if len(hypotheses) == 0:
            return []
        if len(hypotheses) == 1:
            return [hypotheses[0], hypotheses[0]]

        # Get top-1 by score
        ranked = sorted(hypotheses, key=lambda h: scores.get(h, 0.0), reverse=True)
        top_1 = ranked[0]

        try:
            output_1 = top_1.apply(test_input)
        except Exception as e:
            if verbose:
                print(f"  Warning: Top-1 hypothesis failed: {e}")
            return ranked[:2] if len(ranked) >= 2 else [ranked[0], ranked[0]]

        # Find best hypothesis with different output
        best_different = None
        best_different_score = -1.0

        for h in ranked[1:]:
            try:
                output_h = h.apply(test_input)

                # Check if output is different
                if not np.array_equal(output_h.data, output_1.data):
                    score_h = scores.get(h, 0.0)
                    if score_h > best_different_score:
                        best_different = h
                        best_different_score = score_h

            except Exception:
                continue

        # Select top-2
        if best_different is not None:
            top_2 = best_different
            if verbose:
                print(f"  ✓ Diversity enforced: Different outputs")
        else:
            # All produce same output (edge case)
            top_2 = ranked[1] if len(ranked) > 1 else ranked[0]
            if verbose:
                print(f"  Note: All programs produce same output")

        return [top_1, top_2]


if __name__ == "__main__":
    # Test program synthesis solver
    print("Testing ARC Program Synthesis Solver")
    print("=" * 80)

    # Test 1: Simple flip horizontal
    print("\nTest 1: Horizontal Flip")
    print("-" * 40)

    train_pairs = [
        (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
        (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
    ]
    test_input = Grid([[9, 0], [1, 2]])
    expected_output = Grid([[0, 9], [2, 1]])

    task = ARCTask(train_pairs, test_input)
    solver = ARCProgramSolver(max_synthesis_depth=2, max_programs=50, verbose=True)

    predictions = solver.solve(task, verbose=True)

    print(f"\nExpected output:")
    print(expected_output.data)
    print(f"\nPrediction 1:")
    print(predictions[0].data)
    print(f"\nPrediction 2:")
    print(predictions[1].data)

    # Check if correct
    match_1 = np.array_equal(predictions[0].data, expected_output.data)
    match_2 = np.array_equal(predictions[1].data, expected_output.data)

    if match_1 or match_2:
        print(f"\n✓ Test PASSED - Correct solution found!")
    else:
        print(f"\n✗ Test FAILED - No correct solution")

    # Test 2: Zoom 2x
    print("\n\nTest 2: Zoom 2x")
    print("-" * 40)

    train_pairs = [
        (Grid([[1, 2]]), Grid([[1, 1, 2, 2]])),
        (Grid([[3], [4]]), Grid([[3, 3], [3, 3], [4, 4], [4, 4]])),
    ]
    test_input = Grid([[5, 6]])
    expected_output = Grid([[5, 5, 6, 6]])

    task = ARCTask(train_pairs, test_input)
    predictions = solver.solve(task, verbose=False)

    print(f"Expected: {expected_output.data.tolist()}")
    print(f"Prediction 1: {predictions[0].data.tolist()}")
    print(f"Prediction 2: {predictions[1].data.tolist()}")

    match_1 = np.array_equal(predictions[0].data, expected_output.data)
    match_2 = np.array_equal(predictions[1].data, expected_output.data)

    if match_1 or match_2:
        print(f"✓ Test PASSED")
    else:
        print(f"✗ Test FAILED")

    print("\n" + "=" * 80)
    print("✓ Testing complete")
    print("=" * 80)
