"""
Advanced ARC Solver with Near-Miss + Advanced Primitives

Integrates:
1. Near-miss primitives (extract_largest, connect, align)
2. Advanced primitives (rotation fix, pattern tiling, morphology)

Expected improvement: 40% → 55-60% (+15-20 percentage points)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from enhanced_solver import EnhancedARCSolver, EnhancedExecutor, EnhancedProgramGenerator
from arc_generative_solver import Program, ARCObject
from near_miss_primitives import NearMissPrimitives
from advanced_primitives import AdvancedPrimitives


class AdvancedExecutor(EnhancedExecutor):
    """Advanced executor with near-miss + advanced primitives"""

    def __init__(self):
        super().__init__()
        self.advanced = AdvancedPrimitives()

    def execute(self, program: Program, input_grid: np.ndarray) -> np.ndarray:
        """Execute program with all primitives"""
        self.execution_trace = []
        grid = input_grid.copy()

        try:
            # Near-miss schemas (from parent class)
            if program.schema in ["extract_largest", "connect_objects",
                                 "align_horizontal", "align_vertical",
                                 "align_to_row", "pack_horizontal"]:
                return super().execute(program, input_grid)

            # Original schemas
            elif program.schema == "identity":
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

            # ADVANCED SCHEMAS: IMPROVED ROTATION
            elif program.schema == "rotate_90_cw":
                grid = self.advanced.rotate_90_cw(grid)

            elif program.schema == "rotate_90_ccw":
                grid = self.advanced.rotate_90_ccw(grid)

            elif program.schema == "rotate_180":
                grid = self.advanced.rotate_180(grid)

            # ADVANCED SCHEMAS: PATTERN TILING
            elif program.schema == "tile_pattern":
                pattern_h = program.parameters.get("pattern_h", 1)
                pattern_w = program.parameters.get("pattern_w", 1)
                pattern = grid[:pattern_h, :pattern_w]
                grid = self.advanced.tile_pattern(pattern, grid.shape)

            elif program.schema == "repeat_horizontal":
                n_times = program.parameters.get("n_times", 2)
                grid = self.advanced.repeat_pattern_horizontal(grid, n_times)

            elif program.schema == "repeat_vertical":
                n_times = program.parameters.get("n_times", 2)
                grid = self.advanced.repeat_pattern_vertical(grid, n_times)

            elif program.schema == "complete_symmetry_h":
                grid = self.advanced.complete_symmetry_horizontal(grid)

            elif program.schema == "complete_symmetry_v":
                grid = self.advanced.complete_symmetry_vertical(grid)

            # ADVANCED SCHEMAS: MORPHOLOGY
            elif program.schema == "dilate":
                iterations = program.parameters.get("iterations", 1)
                grid = self.advanced.dilate_objects_enhanced(grid, iterations)

            elif program.schema == "erode":
                iterations = program.parameters.get("iterations", 1)
                grid = self.advanced.erode_objects_enhanced(grid, iterations)

            elif program.schema == "fill_holes":
                grid = self.advanced.fill_holes_in_objects(grid)

            elif program.schema == "hollow":
                thickness = program.parameters.get("thickness", 1)
                grid = self.advanced.hollow_objects(grid, thickness)

            elif program.schema == "find_boundaries":
                grid = self.advanced.find_object_boundaries(grid)

            elif program.schema == "gravity":
                direction = program.parameters.get("direction", "down")
                grid = self.advanced.gravity_transform(grid, direction)

            # OBJECT OPERATIONS
            elif program.schema == "move_object":
                objects = self.primitives.components(grid)
                if objects:
                    target_y = program.parameters.get("target_y", 0)
                    target_x = program.parameters.get("target_x", 0)
                    grid = self.advanced.move_object_to_position(grid, objects[0], target_y, target_x)

            elif program.schema == "scale_object":
                objects = self.primitives.components(grid)
                if objects:
                    scale = program.parameters.get("scale", 2.0)
                    grid = self.advanced.scale_object(grid, objects[0], scale)

            elif program.schema == "duplicate_object":
                objects = self.primitives.components(grid)
                if objects:
                    offset_y = program.parameters.get("offset_y", 0)
                    offset_x = program.parameters.get("offset_x", 2)
                    grid = self.advanced.duplicate_object(grid, objects[0], offset_y, offset_x)

            elif program.schema == "distribute_objects":
                objects = self.primitives.components(grid)
                axis = program.parameters.get("axis", "horizontal")
                grid = self.advanced.distribute_objects_evenly(grid, objects, axis)

            # PHYSICS-BASED
            elif program.schema == "gravity_objects":
                objects = self.primitives.components(grid)
                direction = program.parameters.get("direction", "down")
                grid = self.advanced.gravity_objects(grid, objects, direction)

            elif program.schema == "stack_objects":
                objects = self.primitives.components(grid)
                direction = program.parameters.get("direction", "vertical")
                grid = self.advanced.stack_objects(grid, objects, direction)

            elif program.schema == "compress":
                direction = program.parameters.get("direction", "down")
                grid = self.advanced.compress_objects(grid, direction)

            # REGION OPERATIONS (FLOOD FILL)
            elif program.schema == "flood_fill":
                start_y = program.parameters.get("start_y", 0)
                start_x = program.parameters.get("start_x", 0)
                fill_color = program.parameters.get("fill_color", 1)
                grid = self.advanced.flood_fill(grid, start_y, start_x, fill_color)

            elif program.schema == "fill_enclosed":
                fill_color = program.parameters.get("fill_color", 1)
                grid = self.advanced.fill_enclosed_regions(grid, fill_color)

            elif program.schema == "fill_all_background":
                fill_color = program.parameters.get("fill_color", 1)
                grid = self.advanced.flood_fill_all_regions(grid, fill_color, target_color=0)

            self.execution_trace.append(("final", grid.copy()))

        except Exception as e:
            self.execution_trace.append(("error", str(e)))

        return grid


class AdvancedProgramGenerator(EnhancedProgramGenerator):
    """Advanced program generator with all schemas"""

    def _define_schemas(self) -> List[Dict[str, Any]]:
        """Define all schemas including advanced ones"""
        base_schemas = super()._define_schemas()

        # Add advanced schemas
        advanced_schemas = [
            # Improved rotation
            {
                "name": "rotate_90_cw",
                "params": {},
                "complexity": 1.5
            },
            {
                "name": "rotate_90_ccw",
                "params": {},
                "complexity": 1.5
            },
            {
                "name": "rotate_180",
                "params": {},
                "complexity": 1.5
            },
            # Pattern tiling
            {
                "name": "tile_pattern",
                "params": {"pattern_h": [1, 2, 3], "pattern_w": [1, 2, 3]},
                "complexity": 3.0
            },
            {
                "name": "repeat_horizontal",
                "params": {"n_times": [2, 3, 4]},
                "complexity": 2.0
            },
            {
                "name": "repeat_vertical",
                "params": {"n_times": [2, 3, 4]},
                "complexity": 2.0
            },
            {
                "name": "complete_symmetry_h",
                "params": {},
                "complexity": 2.0
            },
            {
                "name": "complete_symmetry_v",
                "params": {},
                "complexity": 2.0
            },
            # Morphology
            {
                "name": "dilate",
                "params": {"iterations": [1, 2]},
                "complexity": 2.5
            },
            {
                "name": "erode",
                "params": {"iterations": [1, 2]},
                "complexity": 2.5
            },
            {
                "name": "fill_holes",
                "params": {},
                "complexity": 2.0
            },
            {
                "name": "hollow",
                "params": {"thickness": [1, 2]},
                "complexity": 2.5
            },
            {
                "name": "find_boundaries",
                "params": {},
                "complexity": 2.0
            },
            {
                "name": "gravity",
                "params": {"direction": ["down", "up", "left", "right"]},
                "complexity": 2.5
            },
            # Object operations
            {
                "name": "move_object",
                "params": {"target_y": [0, 1, 2], "target_x": [0, 1, 2]},
                "complexity": 3.0
            },
            {
                "name": "scale_object",
                "params": {"scale": [0.5, 2.0, 3.0]},
                "complexity": 3.0
            },
            {
                "name": "duplicate_object",
                "params": {"offset_y": [0, 1, 2], "offset_x": [1, 2, 3]},
                "complexity": 2.5
            },
            {
                "name": "distribute_objects",
                "params": {"axis": ["horizontal", "vertical"]},
                "complexity": 2.5
            },
            # Physics-based
            {
                "name": "gravity_objects",
                "params": {"direction": ["down", "up"]},
                "complexity": 2.5
            },
            {
                "name": "stack_objects",
                "params": {"direction": ["vertical", "horizontal"]},
                "complexity": 2.5
            },
            {
                "name": "compress",
                "params": {"direction": ["down", "up", "left", "right"]},
                "complexity": 2.0
            },
            # Region operations (flood fill)
            {
                "name": "flood_fill",
                "params": {"start_y": [0, 1, 2], "start_x": [0, 1, 2], "fill_color": [1, 2, 3]},
                "complexity": 2.5
            },
            {
                "name": "fill_enclosed",
                "params": {"fill_color": [1, 2, 3, 4]},
                "complexity": 2.5
            },
            {
                "name": "fill_all_background",
                "params": {"fill_color": [1, 2, 3, 4]},
                "complexity": 2.0
            }
        ]

        return base_schemas + advanced_schemas

    def generate_candidates(self, task: Dict[str, Any],
                          max_candidates: int = 100) -> List[Program]:
        """Generate candidates including advanced schemas"""
        candidates = super().generate_candidates(task, max_candidates)

        # Add improved rotation candidates
        candidates.extend([
            Program(
                schema="rotate_90_cw",
                primitives=["rotate_90_cw"],
                parameters={},
                selectors={},
                complexity=1.5
            ),
            Program(
                schema="rotate_90_ccw",
                primitives=["rotate_90_ccw"],
                parameters={},
                selectors={},
                complexity=1.5
            ),
            Program(
                schema="rotate_180",
                primitives=["rotate_180"],
                parameters={},
                selectors={},
                complexity=1.5
            )
        ])

        # Add pattern tiling candidates
        for h in [1, 2]:
            for w in [1, 2]:
                candidates.append(Program(
                    schema="tile_pattern",
                    primitives=["tile_pattern"],
                    parameters={"pattern_h": h, "pattern_w": w},
                    selectors={},
                    complexity=3.0
                ))

        for n in [2, 3]:
            candidates.append(Program(
                schema="repeat_horizontal",
                primitives=["repeat_horizontal"],
                parameters={"n_times": n},
                selectors={},
                complexity=2.0
            ))
            candidates.append(Program(
                schema="repeat_vertical",
                primitives=["repeat_vertical"],
                parameters={"n_times": n},
                selectors={},
                complexity=2.0
            ))

        # Add symmetry candidates
        candidates.extend([
            Program(
                schema="complete_symmetry_h",
                primitives=["complete_symmetry_h"],
                parameters={},
                selectors={},
                complexity=2.0
            ),
            Program(
                schema="complete_symmetry_v",
                primitives=["complete_symmetry_v"],
                parameters={},
                selectors={},
                complexity=2.0
            )
        ])

        # Add morphology candidates
        for iterations in [1, 2]:
            candidates.append(Program(
                schema="dilate",
                primitives=["dilate"],
                parameters={"iterations": iterations},
                selectors={},
                complexity=2.5
            ))
            candidates.append(Program(
                schema="erode",
                primitives=["erode"],
                parameters={"iterations": iterations},
                selectors={},
                complexity=2.5
            ))

        candidates.extend([
            Program(
                schema="fill_holes",
                primitives=["fill_holes"],
                parameters={},
                selectors={},
                complexity=2.0
            ),
            Program(
                schema="find_boundaries",
                primitives=["find_boundaries"],
                parameters={},
                selectors={},
                complexity=2.0
            )
        ])

        # Add gravity candidates
        for direction in ["down", "up", "left", "right"]:
            candidates.append(Program(
                schema="gravity",
                primitives=["gravity"],
                parameters={"direction": direction},
                selectors={},
                complexity=2.5
            ))

        # Add object operation candidates
        candidates.append(Program(
            schema="duplicate_object",
            primitives=["duplicate_object"],
            parameters={"offset_y": 0, "offset_x": 2},
            selectors={},
            complexity=2.5
        ))

        candidates.append(Program(
            schema="distribute_objects",
            primitives=["distribute_objects"],
            parameters={"axis": "horizontal"},
            selectors={},
            complexity=2.5
        ))

        candidates.append(Program(
            schema="distribute_objects",
            primitives=["distribute_objects"],
            parameters={"axis": "vertical"},
            selectors={},
            complexity=2.5
        ))

        # Add physics-based candidates
        for direction in ["down", "up"]:
            candidates.append(Program(
                schema="gravity_objects",
                primitives=["gravity_objects"],
                parameters={"direction": direction},
                selectors={},
                complexity=2.5
            ))

        candidates.append(Program(
            schema="stack_objects",
            primitives=["stack_objects"],
            parameters={"direction": "vertical"},
            selectors={},
            complexity=2.5
        ))

        for direction in ["down", "up", "left", "right"]:
            candidates.append(Program(
                schema="compress",
                primitives=["compress"],
                parameters={"direction": direction},
                selectors={},
                complexity=2.0
            ))

        # Add flood fill candidates
        for fill_color in [1, 2, 3]:
            candidates.append(Program(
                schema="fill_enclosed",
                primitives=["fill_enclosed"],
                parameters={"fill_color": fill_color},
                selectors={},
                complexity=2.5
            ))

            candidates.append(Program(
                schema="fill_all_background",
                primitives=["fill_all_background"],
                parameters={"fill_color": fill_color},
                selectors={},
                complexity=2.0
            ))

        return candidates[:max_candidates + 70]  # Allow more for all operations


class AdvancedARCSolver(EnhancedARCSolver):
    """Advanced ARC solver with all primitives"""

    def __init__(self,
                 max_candidates: int = 150,  # Increased for more schemas
                 beam_width: int = 20,
                 active_inference_steps: int = 5,
                 diversity_strategy: str = "schema_first"):
        super().__init__(
            max_candidates, beam_width,
            active_inference_steps, diversity_strategy
        )

        # Replace with advanced components
        self.generator = AdvancedProgramGenerator()
        self.executor = AdvancedExecutor()


def test_advanced_solver():
    """Test advanced solver on rotation task"""

    print("="*70)
    print("ADVANCED SOLVER - TESTING ON ROTATION TASK")
    print("="*70)

    solver = AdvancedARCSolver()

    # Test: 90° rotation on non-square grid (rotation_90_large from gap analysis)
    print("\nTest: 90° Rotation (Non-Square Grid)")
    print("-"*70)

    task = {
        "train": [
            {
                "input": [[1, 2, 3, 4], [5, 6, 7, 8]],
                "output": [[5, 1], [6, 2], [7, 3], [8, 4]]
            }
        ],
        "test": [
            {
                "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "output": [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
            }
        ]
    }

    pred1, pred2, metadata = solver.solve(task)
    target = np.array(task["test"][0]["output"])

    from arc_generative_solver import evaluate_predictions
    result = evaluate_predictions(pred1, pred2, target)

    print(f"Target:\n{target}")
    print(f"\nPrediction 1:\n{pred1}")
    print(f"Pred1 correct: {result['exact_match_1']}")
    print(f"Pred1 accuracy: {result['pixel_accuracy_1']:.1%}")
    print(f"Top program: {metadata['top_programs'][0]['schema']}")

    print(f"\nPrediction 2:\n{pred2}")
    print(f"Pred2 correct: {result['exact_match_2']}")
    print(f"Pred2 accuracy: {result['pixel_accuracy_2']:.1%}")

    if len(metadata['top_programs']) > 1:
        print(f"Second program: {metadata['top_programs'][1]['schema']}")

    print(f"\nSuccess: {result['any_correct']}")

    return result['any_correct']


if __name__ == "__main__":
    success = test_advanced_solver()
    print("\n" + "="*70)
    if success:
        print("✓ ROTATION TEST PASSED!")
    else:
        print("✗ Rotation test failed")
