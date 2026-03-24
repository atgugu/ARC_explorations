"""
Enhanced ARC Solver with Near-Miss Primitives

Adds support for:
1. Object selection by size (extract_largest)
2. Object connection (connect_nearest)
3. Object alignment (align_objects)

Expected improvement: 40% → 48.6% (+3 tasks)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from diverse_solver import DiverseARCSolver
from arc_generative_solver import (
    Program, ARCObject, TRGPrimitives, Executor, ProgramGenerator
)
from near_miss_primitives import NearMissPrimitives


class EnhancedExecutor(Executor):
    """Enhanced executor with near-miss primitives"""

    def __init__(self):
        super().__init__()
        self.near_miss = NearMissPrimitives()

    def execute(self, program: Program, input_grid: np.ndarray) -> np.ndarray:
        """Execute program with enhanced primitives"""
        self.execution_trace = []
        grid = input_grid.copy()

        try:
            # Original schemas
            if program.schema == "identity":
                return grid

            elif program.schema == "rotation":
                k = program.parameters.get("k", 1)
                grid = self.primitives.rotate(grid, k)

            elif program.schema == "reflection":
                axis = program.parameters.get("axis", "h")
                grid = self.primitives.reflect(grid, axis)

            elif program.schema == "translation":
                dx = program.parameters.get("dx", 0)
                dy = program.parameters.get("dy", 0)
                grid = self.primitives.translate(grid, dx, dy)

            elif program.schema == "color_remap":
                mapping = program.parameters.get("mapping", {})
                grid = self.primitives.remap_color(grid, mapping)

            elif program.schema == "composite":
                # Execute sequence
                for prim_name in program.primitives:
                    if prim_name == "rotate":
                        k = program.parameters.get("k", 1)
                        grid = self.primitives.rotate(grid, k)
                    elif prim_name == "reflect":
                        axis = program.parameters.get("axis", "h")
                        grid = self.primitives.reflect(grid, axis)
                    elif prim_name == "translate":
                        dx = program.parameters.get("dx", 0)
                        dy = program.parameters.get("dy", 0)
                        grid = self.primitives.translate(grid, dx, dy)

            # NEW SCHEMAS
            elif program.schema == "extract_largest":
                # Extract only the largest object(s)
                objects = self.primitives.components(grid)
                grid = self.near_miss.extract_largest_to_grid(grid, objects)

            elif program.schema == "connect_objects":
                # Connect objects with lines
                objects = self.primitives.components(grid)
                line_color = program.parameters.get("line_color", 3)
                grid = self.near_miss.connect_nearest_objects(
                    grid, objects, line_color
                )

            elif program.schema == "align_horizontal":
                # Align objects horizontally
                objects = self.primitives.components(grid)
                alignment = program.parameters.get("alignment", "top")
                grid = self.near_miss.align_objects_horizontal(
                    grid, objects, alignment
                )

            elif program.schema == "align_vertical":
                # Align objects vertically
                objects = self.primitives.components(grid)
                alignment = program.parameters.get("alignment", "left")
                grid = self.near_miss.align_objects_vertical(
                    grid, objects, alignment
                )

            elif program.schema == "align_to_row":
                # Align objects to specific row
                objects = self.primitives.components(grid)
                row = program.parameters.get("row", 0)
                grid = self.near_miss.align_objects_to_row(grid, objects, row)

            elif program.schema == "pack_horizontal":
                # Pack objects horizontally at row without overlap
                objects = self.primitives.components(grid)
                row = program.parameters.get("row", 0)
                spacing = program.parameters.get("spacing", 0)
                grid = self.near_miss.pack_objects_horizontal(grid, objects, row, spacing)

            self.execution_trace.append(("final", grid.copy()))

        except Exception as e:
            self.execution_trace.append(("error", str(e)))

        return grid


class EnhancedProgramGenerator(ProgramGenerator):
    """Enhanced program generator with new schemas"""

    def _define_schemas(self) -> List[Dict[str, Any]]:
        """Define schemas including near-miss operations"""
        base_schemas = super()._define_schemas()

        # Add new schemas
        new_schemas = [
            {
                "name": "extract_largest",
                "params": {},
                "complexity": 2.5
            },
            {
                "name": "connect_objects",
                "params": {"line_color": [1, 2, 3, 4, 5]},
                "complexity": 2.5
            },
            {
                "name": "align_horizontal",
                "params": {"alignment": ["top", "center", "bottom"]},
                "complexity": 2.5
            },
            {
                "name": "align_vertical",
                "params": {"alignment": ["left", "center", "right"]},
                "complexity": 2.5
            },
            {
                "name": "align_to_row",
                "params": {"row": range(5)},
                "complexity": 2.5
            },
            {
                "name": "pack_horizontal",
                "params": {"row": range(5), "spacing": [0, 1]},
                "complexity": 2.5
            }
        ]

        return base_schemas + new_schemas

    def generate_candidates(self, task: Dict[str, Any],
                          max_candidates: int = 100) -> List[Program]:
        """Generate candidates including near-miss schemas"""
        candidates = super().generate_candidates(task, max_candidates)

        # Add extract_largest candidates
        candidates.append(Program(
            schema="extract_largest",
            primitives=["extract_largest"],
            parameters={},
            selectors={},
            complexity=2.5
        ))

        # Add connect_objects candidates
        for line_color in [1, 2, 3, 4, 5]:
            candidates.append(Program(
                schema="connect_objects",
                primitives=["connect_objects"],
                parameters={"line_color": line_color},
                selectors={},
                complexity=2.5
            ))

        # Add align_horizontal candidates
        for alignment in ["top", "center", "bottom"]:
            candidates.append(Program(
                schema="align_horizontal",
                primitives=["align_horizontal"],
                parameters={"alignment": alignment},
                selectors={},
                complexity=2.5
            ))

        # Add align_vertical candidates
        for alignment in ["left", "center", "right"]:
            candidates.append(Program(
                schema="align_vertical",
                primitives=["align_vertical"],
                parameters={"alignment": alignment},
                selectors={},
                complexity=2.5
            ))

        # Add align_to_row candidates
        for row in range(5):
            candidates.append(Program(
                schema="align_to_row",
                primitives=["align_to_row"],
                parameters={"row": row},
                selectors={},
                complexity=2.5
            ))

        # Add pack_horizontal candidates
        for row in range(5):
            for spacing in [0, 1]:
                candidates.append(Program(
                    schema="pack_horizontal",
                    primitives=["pack_horizontal"],
                    parameters={"row": row, "spacing": spacing},
                    selectors={},
                    complexity=2.5
                ))

        return candidates[:max_candidates + 30]  # Allow some extra


class EnhancedARCSolver(DiverseARCSolver):
    """Enhanced ARC solver with near-miss primitives"""

    def __init__(self,
                 max_candidates: int = 120,  # Increased for new schemas
                 beam_width: int = 15,
                 active_inference_steps: int = 5,
                 diversity_strategy: str = "schema_first"):
        super().__init__(
            max_candidates, beam_width,
            active_inference_steps, diversity_strategy
        )

        # Replace with enhanced components
        self.generator = EnhancedProgramGenerator()
        self.executor = EnhancedExecutor()


def test_enhanced_solver():
    """Test enhanced solver on near-miss tasks"""

    print("="*70)
    print("ENHANCED SOLVER - TESTING NEAR-MISS TASKS")
    print("="*70)

    solver = EnhancedARCSolver()

    # Test 1: Extract largest
    print("\n1. Extract Largest Object")
    print("-"*70)

    task1 = {
        "train": [
            {
                "input": [[1, 0, 2, 2], [0, 0, 2, 2], [3, 0, 0, 0]],
                "output": [[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 0, 0]]
            }
        ],
        "test": [
            {
                "input": [[4, 4, 4, 0], [0, 5, 0, 0], [0, 0, 0, 6]],
                "output": [[4, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            }
        ]
    }

    pred1, pred2, metadata = solver.solve(task1)
    target1 = np.array(task1["test"][0]["output"])

    from arc_generative_solver import evaluate_predictions
    result1 = evaluate_predictions(pred1, pred2, target1)

    print(f"Target:\n{target1}")
    print(f"Prediction 1:\n{pred1}")
    print(f"Exact match: {result1['any_correct']}")
    print(f"Pixel accuracy: {result1['pixel_accuracy_1']:.1%}")
    print(f"Top program: {metadata['top_programs'][0]['schema']}")

    # Test 2: Connect objects
    print("\n2. Connect Objects")
    print("-"*70)

    task2 = {
        "train": [
            {
                "input": [[1, 0, 0, 0, 2], [0, 0, 0, 0, 0]],
                "output": [[1, 3, 3, 3, 2], [0, 0, 0, 0, 0]]
            }
        ],
        "test": [
            {
                "input": [[4, 0, 0, 5], [0, 0, 0, 0]],
                "output": [[4, 3, 3, 5], [0, 0, 0, 0]]
            }
        ]
    }

    pred2_1, pred2_2, metadata2 = solver.solve(task2)
    target2 = np.array(task2["test"][0]["output"])

    result2 = evaluate_predictions(pred2_1, pred2_2, target2)

    print(f"Target:\n{target2}")
    print(f"Prediction 1:\n{pred2_1}")
    print(f"Exact match: {result2['any_correct']}")
    print(f"Pixel accuracy: {result2['pixel_accuracy_1']:.1%}")
    print(f"Top program: {metadata2['top_programs'][0]['schema']}")

    # Test 3: Align objects
    print("\n3. Align Objects")
    print("-"*70)

    task3 = {
        "train": [
            {
                "input": [[1, 0], [0, 0], [0, 2]],
                "output": [[1, 2], [0, 0], [0, 0]]
            }
        ],
        "test": [
            {
                "input": [[3, 0], [0, 0], [0, 0], [4, 0]],
                "output": [[3, 4], [0, 0], [0, 0], [0, 0]]
            }
        ]
    }

    pred3_1, pred3_2, metadata3 = solver.solve(task3)
    target3 = np.array(task3["test"][0]["output"])

    result3 = evaluate_predictions(pred3_1, pred3_2, target3)

    print(f"Target:\n{target3}")
    print(f"Prediction 1:\n{pred3_1}")
    print(f"Prediction 2:\n{pred3_2}")
    print(f"Pred1 correct: {result3['exact_match_1']}")
    print(f"Pred2 correct: {result3['exact_match_2']}")
    print(f"Any correct: {result3['any_correct']}")
    print(f"Pred1 accuracy: {result3['pixel_accuracy_1']:.1%}")
    print(f"Pred2 accuracy: {result3['pixel_accuracy_2']:.1%}")
    print(f"Top programs: {[p['schema'] for p in metadata3['top_programs'][:2]]}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Extract largest: {'✓' if result1['any_correct'] else '✗'} ({result1['pixel_accuracy_1']:.1%})")
    print(f"Connect objects: {'✓' if result2['any_correct'] else '✗'} ({result2['pixel_accuracy_1']:.1%})")
    print(f"Align objects: {'✓' if result3['any_correct'] else '✗'} ({result3['pixel_accuracy_1']:.1%})")

    total_success = sum([result1['any_correct'], result2['any_correct'], result3['any_correct']])
    print(f"\nTotal: {total_success}/3 ({total_success/3:.1%})")


if __name__ == "__main__":
    test_enhanced_solver()
