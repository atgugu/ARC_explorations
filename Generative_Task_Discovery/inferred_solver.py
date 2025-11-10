"""
Inferred Parameter Solver

Integrates parameter inference into the compositional solver.
Uses training examples to learn correct parameters instead of guessing.

Expected improvement: 1% → 8-12% success rate
"""

import numpy as np
from typing import List, Dict, Any
from compositional_solver import CompositionalARCSolver, CompositionalProgram
from advanced_solver import AdvancedProgramGenerator, AdvancedExecutor
from parameter_inference import ParameterInference, InferredParameters
from arc_generative_solver import Program


class InferredProgramGenerator(AdvancedProgramGenerator):
    """
    Program generator that uses parameter inference

    Instead of hardcoded parameter guesses, analyzes training examples
    to infer correct parameters.
    """

    def generate_candidates(self, task: Dict[str, Any], max_candidates: int) -> List[Program]:
        """
        Generate program candidates with inferred parameters

        Strategy:
        1. Infer parameters from training examples
        2. Generate candidates using inferred parameters (high priority)
        3. Generate candidates using default parameters (lower priority)
        4. Return merged list up to max_candidates
        """
        # Infer parameters from training examples
        inferred = ParameterInference.infer_all_parameters(task)

        # Generate candidates with inferred parameters
        inferred_candidates = self._generate_with_inferred_params(task, inferred)

        # Generate default candidates
        default_candidates = super().generate_candidates(task, max_candidates)

        # Merge: prioritize inferred candidates
        all_candidates = inferred_candidates + default_candidates

        # Deduplicate based on schema + parameters
        seen = set()
        unique_candidates = []

        for prog in all_candidates:
            # Create hashable key from parameters
            param_items = []
            for k, v in prog.parameters.items():
                if isinstance(v, dict):
                    param_items.append((k, frozenset(v.items())))
                elif isinstance(v, list):
                    param_items.append((k, tuple(v)))
                else:
                    param_items.append((k, v))

            key = (prog.schema, frozenset(param_items))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(prog)

        return unique_candidates[:max_candidates]

    def _generate_with_inferred_params(self,
                                       task: Dict[str, Any],
                                       inferred: InferredParameters) -> List[Program]:
        """
        Generate candidates using inferred parameters

        Args:
            task: ARC task
            inferred: InferredParameters from training examples

        Returns:
            List of Program objects with inferred parameters
        """
        candidates = []

        # Color mapping candidates
        for color_map in inferred.color_mappings:
            candidates.append(Program(
                schema="color_remap",
                primitives=["color_remap"],
                parameters={"mapping": color_map},
                selectors={},
                complexity=1.5
            ))

            # Also try color_swap for simple bidirectional swaps
            if len(color_map) == 2:
                colors = list(color_map.keys())
                if len(colors) == 2 and color_map[colors[0]] == colors[1] and color_map[colors[1]] == colors[0]:
                    candidates.append(Program(
                        schema="color_swap",
                        primitives=["color_swap"],
                        parameters={"color1": colors[0], "color2": colors[1]},
                        selectors={},
                        complexity=1.0
                    ))

        # Translation candidates
        for dx, dy in inferred.translations:
            candidates.append(Program(
                schema="translation",
                primitives=["translation"],
                parameters={"dx": dx, "dy": dy},
                selectors={},
                complexity=1.5
            ))

        # Rotation candidates
        for k in inferred.rotations:
            # Map k to schema name
            if k == 1:
                schema = "rotate_90_ccw"  # np.rot90 rotates CCW
            elif k == 2:
                schema = "rotate_180"
            elif k == 3:
                schema = "rotate_90_cw"
            else:
                continue

            candidates.append(Program(
                schema=schema,
                primitives=[schema],
                parameters={},
                selectors={},
                complexity=1.5
            ))

        # Reflection candidates
        for axis in inferred.reflections:
            candidates.append(Program(
                schema="reflection",
                primitives=["reflection"],
                parameters={"axis": axis},
                selectors={},
                complexity=1.5
            ))

        # Scaling candidates
        for factor in inferred.scale_factors:
            if factor > 0:
                # Upscaling
                candidates.append(Program(
                    schema="upscale",
                    primitives=["upscale"],
                    parameters={"factor": factor},
                    selectors={},
                    complexity=1.5
                ))
            else:
                # Downscaling
                candidates.append(Program(
                    schema="downscale",
                    primitives=["downscale"],
                    parameters={"factor": -factor},
                    selectors={},
                    complexity=1.5
                ))

        # Morphology candidates
        for iterations in inferred.morphology_iterations:
            candidates.append(Program(
                schema="dilate",
                primitives=["dilate"],
                parameters={"iterations": iterations},
                selectors={},
                complexity=2.0
            ))
            candidates.append(Program(
                schema="erode",
                primitives=["erode"],
                parameters={"iterations": iterations},
                selectors={},
                complexity=2.0
            ))

        return candidates


class InferredCompositionalSolver(CompositionalARCSolver):
    """
    Compositional solver with parameter inference

    Combines:
    - Compositional reasoning (2-step program chains)
    - Parameter inference (learn parameters from training)

    Expected improvement: 1% → 8-12% success rate
    """

    def __init__(self,
                 max_candidates: int = 150,
                 beam_width: int = 20,
                 active_inference_steps: int = 3,
                 diversity_strategy: str = "schema_first",
                 max_depth: int = 2,
                 composition_beam_width: int = 10):
        super().__init__(
            max_candidates, beam_width,
            active_inference_steps, diversity_strategy,
            max_depth, composition_beam_width
        )

        # Replace generator with inferred version
        self.generator = InferredProgramGenerator()


def test_inferred_solver():
    """Test inferred solver on sample tasks"""

    print("="*70)
    print("INFERRED SOLVER - TESTING PARAMETER INFERENCE")
    print("="*70)

    solver = InferredCompositionalSolver(
        max_depth=2,
        composition_beam_width=10
    )

    # Test 1: Color swapping task
    print("\n1. Color Swap Task (Identity + Color Remap)")
    print("-"*70)

    # Create task where output = color_swap(input, 1, 2)
    input_grid = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [2, 2, 1, 1],
        [2, 2, 1, 1]
    ])

    # Swap 1 and 2
    output_grid = np.array([
        [2, 2, 1, 1],
        [2, 2, 1, 1],
        [1, 1, 2, 2],
        [1, 1, 2, 2]
    ])

    task = {
        "train": [
            {"input": input_grid.tolist(), "output": output_grid.tolist()}
        ],
        "test": [
            {"input": input_grid.tolist(), "output": output_grid.tolist()}
        ]
    }

    pred1, pred2, metadata = solver.solve(task)

    accuracy = np.mean(pred1 == output_grid) if pred1.shape == output_grid.shape else 0.0
    print(f"Accuracy: {accuracy:.1%}")

    if metadata.get('composition_found'):
        print(f"\nTop programs:")
        for i, prog in enumerate(metadata.get('top_programs', [])[:5]):
            schema = prog.get('schema', 'unknown')
            score = prog.get('probability', 0)
            print(f"  {i+1}. {schema} (score: {score:.3f})")

    # Test 2: Translation task
    print("\n2. Translation Task")
    print("-"*70)

    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    # Shifted right by 2
    output_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ])

    task = {
        "train": [
            {"input": input_grid.tolist(), "output": output_grid.tolist()}
        ],
        "test": [
            {"input": input_grid.tolist(), "output": output_grid.tolist()}
        ]
    }

    pred1, pred2, metadata = solver.solve(task)

    accuracy = np.mean(pred1 == output_grid) if pred1.shape == output_grid.shape else 0.0
    print(f"Accuracy: {accuracy:.1%}")

    if metadata.get('composition_found'):
        print(f"\nTop programs:")
        for i, prog in enumerate(metadata.get('top_programs', [])[:5]):
            schema = prog.get('schema', 'unknown')
            score = prog.get('probability', 0)
            print(f"  {i+1}. {schema} (score: {score:.3f})")

    print("\n" + "="*70)
    print("✓ Inferred solver tests complete!")
    print("="*70)


if __name__ == "__main__":
    test_inferred_solver()
