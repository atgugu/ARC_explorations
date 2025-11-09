"""
Enhanced ARC Solver with Diverse Dual Predictions

Ensures 100% diversity between prediction 1 and prediction 2 using:
1. Schema-based diversity (force different program types)
2. Parameter diversity (different parameters if same schema)
3. Stochastic sampling from posterior
4. Explicit diversity verification
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from arc_generative_solver import (
    ARCGenerativeSolver, Program, Executor, ActiveInferenceEngine,
    ProgramGenerator, evaluate_predictions
)
import copy


class DiverseARCSolver(ARCGenerativeSolver):
    """
    Enhanced solver that guarantees diverse dual predictions
    """

    def __init__(self,
                 max_candidates: int = 100,
                 beam_width: int = 15,
                 active_inference_steps: int = 5,
                 diversity_strategy: str = "schema_first"):
        """
        Initialize solver with diversity strategy

        diversity_strategy options:
        - "schema_first": Force different schemas
        - "stochastic": Sample from posterior distribution
        - "hybrid": Combine both approaches
        """
        super().__init__(max_candidates, beam_width, active_inference_steps)
        self.diversity_strategy = diversity_strategy

    def solve(self, task: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve ARC task with guaranteed diverse dual predictions

        Returns:
            prediction1: Best prediction
            prediction2: Second-best prediction (guaranteed different)
            metadata: Solving metadata
        """
        # Generate candidate programs
        candidates = self.generator.generate_candidates(
            task, self.max_candidates
        )

        # Initialize beliefs
        self.active_inference.initialize_beliefs(candidates)

        # Active inference loop
        for step in range(self.active_inference_steps):
            # Evaluate programs on training examples
            likelihoods = []
            valid_programs = []

            for program in candidates:
                likelihood = self._evaluate_program(program, task)
                if likelihood > 0:
                    likelihoods.append(likelihood)
                    valid_programs.append(program)

            if not valid_programs:
                break

            # Update beliefs
            self.active_inference.update_beliefs(valid_programs, likelihoods)

            # Sample from posterior for next iteration
            if step < self.active_inference_steps - 1:
                candidates = self.active_inference.sample_programs(
                    n=self.beam_width
                )

        # Get diverse top-2 programs based on strategy
        top_programs = self._get_diverse_top_programs(
            strategy=self.diversity_strategy
        )

        if len(top_programs) == 0:
            # No valid program found
            test_input = np.array(task["test"][0]["input"])
            return test_input.copy(), test_input.copy(), {"error": "No valid program"}

        # Execute top programs on test input
        test_input = np.array(task["test"][0]["input"])

        pred1 = self.executor.execute(top_programs[0][0], test_input)

        if len(top_programs) > 1:
            pred2 = self.executor.execute(top_programs[1][0], test_input)
        else:
            # Only one program found, try to generate diverse alternative
            pred2 = self._generate_diverse_fallback(pred1, test_input, task)

        # Verify diversity - critical check!
        if np.array_equal(pred1, pred2):
            # Predictions are same, force diversity
            pred2 = self._force_diverse_prediction(
                pred1, test_input, task, top_programs[0][0]
            )

        # Metadata
        metadata = {
            "top_programs": [
                {
                    "schema": p[0].schema,
                    "parameters": p[0].parameters,
                    "probability": p[1],
                    "complexity": p[0].complexity
                }
                for p in top_programs
            ],
            "free_energy": self.active_inference.beliefs.free_energy,
            "entropy": self.active_inference.beliefs.entropy(),
            "n_candidates": len(candidates),
            "n_valid": len([p for p, l in zip(candidates, likelihoods) if l > 0]),
            "diversity_strategy": self.diversity_strategy,
            "predictions_equal": False  # Guaranteed different now
        }

        return pred1, pred2, metadata

    def _get_diverse_top_programs(self,
                                  strategy: str = "schema_first") -> List[Tuple[Program, float]]:
        """
        Get top-2 programs ensuring diversity
        """
        if self.active_inference.beliefs is None or \
           not self.active_inference.beliefs.program_posterior:
            return []

        all_programs = sorted(
            self.active_inference.beliefs.program_posterior.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if len(all_programs) == 0:
            return []

        if strategy == "schema_first":
            return self._schema_diverse_selection(all_programs)
        elif strategy == "stochastic":
            return self._stochastic_diverse_selection(all_programs)
        elif strategy == "hybrid":
            return self._hybrid_diverse_selection(all_programs)
        else:
            return all_programs[:2]

    def _schema_diverse_selection(self,
                                  programs: List[Tuple[Program, float]]) -> List[Tuple[Program, float]]:
        """
        Select top-2 programs with different schemas
        """
        if len(programs) == 0:
            return []

        # First: highest probability
        selected = [programs[0]]
        first_schema = programs[0][0].schema

        # Second: highest probability with different schema
        for prog, prob in programs[1:]:
            if prog.schema != first_schema:
                selected.append((prog, prob))
                break

        # If all have same schema, try different parameters
        if len(selected) < 2:
            first_params = str(sorted(programs[0][0].parameters.items()))
            for prog, prob in programs[1:]:
                param_str = str(sorted(prog.parameters.items()))
                if param_str != first_params:
                    selected.append((prog, prob))
                    break

        # Last resort: just take second highest
        if len(selected) < 2 and len(programs) > 1:
            selected.append(programs[1])

        return selected

    def _stochastic_diverse_selection(self,
                                     programs: List[Tuple[Program, float]]) -> List[Tuple[Program, float]]:
        """
        Stochastically sample 2 diverse programs from posterior
        """
        if len(programs) == 0:
            return []

        # First: deterministic (highest probability)
        selected = [programs[0]]

        if len(programs) == 1:
            return selected

        # Second: stochastic sample (weighted by probability, excluding first)
        remaining = programs[1:]
        probs = np.array([p[1] for p in remaining])

        if probs.sum() > 0:
            probs = probs / probs.sum()
            idx = np.random.choice(len(remaining), p=probs)
            selected.append(remaining[idx])
        else:
            selected.append(remaining[0])

        return selected

    def _hybrid_diverse_selection(self,
                                  programs: List[Tuple[Program, float]]) -> List[Tuple[Program, float]]:
        """
        Hybrid: schema-diverse if possible, else stochastic
        """
        schema_diverse = self._schema_diverse_selection(programs)

        # Check if actually diverse
        if len(schema_diverse) >= 2:
            prog1, prog2 = schema_diverse[0][0], schema_diverse[1][0]
            if prog1.schema != prog2.schema:
                return schema_diverse

        # Fall back to stochastic
        return self._stochastic_diverse_selection(programs)

    def _generate_diverse_fallback(self,
                                   pred1: np.ndarray,
                                   test_input: np.ndarray,
                                   task: Dict[str, Any]) -> np.ndarray:
        """
        Generate a diverse fallback prediction when only one program found
        """
        # Try common alternative transformations
        alternatives = [
            ("identity", lambda x: x),
            ("flip_h", lambda x: np.fliplr(x)),
            ("flip_v", lambda x: np.flipud(x)),
            ("rot_90", lambda x: np.rot90(x, k=1)),
            ("rot_180", lambda x: np.rot90(x, k=2)),
            ("rot_270", lambda x: np.rot90(x, k=3)),
        ]

        for name, transform_fn in alternatives:
            try:
                alt_pred = transform_fn(test_input)
                # Return first one that's different from pred1
                if not np.array_equal(alt_pred, pred1):
                    return alt_pred
            except:
                continue

        # Last resort: return transposed if different
        try:
            transposed = test_input.T
            if not np.array_equal(transposed, pred1):
                return transposed
        except:
            pass

        # Very last resort: modify a few pixels
        pred2 = pred1.copy()
        if pred2.size > 0:
            # Change one pixel to make it different
            pred2.flat[0] = (pred2.flat[0] + 1) % 10

        return pred2

    def _force_diverse_prediction(self,
                                  pred1: np.ndarray,
                                  test_input: np.ndarray,
                                  task: Dict[str, Any],
                                  prog1: Program) -> np.ndarray:
        """
        Force a diverse second prediction by trying alternative programs
        """
        # Get all programs sorted by probability
        if self.active_inference.beliefs is None:
            return self._generate_diverse_fallback(pred1, test_input, task)

        all_programs = sorted(
            self.active_inference.beliefs.program_posterior.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Try each program until we find one that gives different output
        for prog, prob in all_programs[1:]:  # Skip first (already used)
            try:
                pred2 = self.executor.execute(prog, test_input)
                if not np.array_equal(pred1, pred2):
                    return pred2
            except:
                continue

        # If none work, use fallback
        return self._generate_diverse_fallback(pred1, test_input, task)


def compare_solvers(task: Dict[str, Any],
                   task_name: str,
                   original_solver: ARCGenerativeSolver,
                   diverse_solver: DiverseARCSolver) -> Dict[str, Any]:
    """
    Compare original vs diverse solver on a task
    """
    print(f"\n{'='*70}")
    print(f"Comparing on: {task_name}")
    print(f"{'='*70}")

    # Original solver
    print("\n[Original Solver]")
    pred1_orig, pred2_orig, meta_orig = original_solver.solve(task)

    target = np.array(task["test"][0]["output"])
    eval_orig = evaluate_predictions(pred1_orig, pred2_orig, target)

    diverse_orig = not np.array_equal(pred1_orig, pred2_orig)

    print(f"  Predictions differ: {diverse_orig}")
    print(f"  Pred1 correct: {eval_orig['exact_match_1']}")
    print(f"  Pred2 correct: {eval_orig['exact_match_2']}")
    print(f"  Success: {eval_orig['any_correct']}")

    # Diverse solver
    print("\n[Diverse Solver]")
    pred1_div, pred2_div, meta_div = diverse_solver.solve(task)

    eval_div = evaluate_predictions(pred1_div, pred2_div, target)

    diverse_div = not np.array_equal(pred1_div, pred2_div)

    print(f"  Predictions differ: {diverse_div}")
    print(f"  Pred1 correct: {eval_div['exact_match_1']}")
    print(f"  Pred2 correct: {eval_div['exact_match_2']}")
    print(f"  Success: {eval_div['any_correct']}")

    # Comparison
    print("\n[Comparison]")
    diversity_improved = diverse_div and not diverse_orig
    pred2_saves = eval_div['exact_match_2'] and not eval_div['exact_match_1']

    if diversity_improved:
        print("  ✓ Diversity improved!")
    if pred2_saves:
        print("  ✓ Pred2 saved the day!")

    return {
        "task_name": task_name,
        "original": {
            "diverse": diverse_orig,
            "pred1_correct": eval_orig['exact_match_1'],
            "pred2_correct": eval_orig['exact_match_2'],
            "success": eval_orig['any_correct']
        },
        "diverse_solver": {
            "diverse": diverse_div,
            "pred1_correct": eval_div['exact_match_1'],
            "pred2_correct": eval_div['exact_match_2'],
            "success": eval_div['any_correct']
        },
        "improvements": {
            "diversity": diversity_improved,
            "pred2_saves": pred2_saves
        }
    }


if __name__ == "__main__":
    print("="*70)
    print("DIVERSE DUAL PREDICTION STRATEGY")
    print("="*70)

    # Test on a simple task
    test_task = {
        "train": [
            {
                "input": [[1, 2, 3], [4, 5, 6]],
                "output": [[3, 2, 1], [6, 5, 4]]
            }
        ],
        "test": [
            {
                "input": [[7, 8, 9]],
                "output": [[9, 8, 7]]
            }
        ]
    }

    print("\nTesting diversity strategies...")

    for strategy in ["schema_first", "stochastic", "hybrid"]:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy}")
        print(f"{'='*70}")

        solver = DiverseARCSolver(
            max_candidates=100,
            beam_width=15,
            active_inference_steps=5,
            diversity_strategy=strategy
        )

        pred1, pred2, metadata = solver.solve(test_task)
        target = np.array(test_task["test"][0]["output"])

        diverse = not np.array_equal(pred1, pred2)
        eval_res = evaluate_predictions(pred1, pred2, target)

        print(f"\nResults:")
        print(f"  Predictions differ: {diverse}")
        print(f"  Pred1 correct: {eval_res['exact_match_1']}")
        print(f"  Pred2 correct: {eval_res['exact_match_2']}")
        print(f"  Success: {eval_res['any_correct']}")
        print(f"  Top programs: {[p['schema'] for p in metadata['top_programs'][:2]]}")
