"""
Comprehensive ARC Task Suite - Simple to Complex

Tests 35+ diverse tasks covering:
- Basic geometric transformations
- Object-based operations
- Pattern operations
- Color transformations
- Relational reasoning
- Composite operations
- Advanced operations

Tracks exact requirements for pixel-perfect matches.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from diverse_solver import DiverseARCSolver
from arc_generative_solver import evaluate_predictions


class ComprehensiveARCTestSuite:
    """Comprehensive test suite from simple to complex"""

    def __init__(self):
        self.solver = DiverseARCSolver(
            max_candidates=100,
            beam_width=15,
            active_inference_steps=5,
            diversity_strategy="schema_first"
        )
        self.results = []

    def create_comprehensive_tasks(self) -> List[Dict[str, Any]]:
        """Create 35+ diverse tasks covering all categories"""

        tasks = []

        # ====================================================================
        # CATEGORY 1: BASIC GEOMETRIC (Should work well)
        # ====================================================================

        # Task 1: Simple horizontal flip
        tasks.append({
            "name": "basic_h_flip",
            "category": "basic_geometric",
            "complexity": "simple",
            "required_capability": "horizontal_reflection",
            "task": {
                "train": [
                    {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
                    {"input": [[4, 5]], "output": [[5, 4]]}
                ],
                "test": [{"input": [[6, 7, 8, 9]], "output": [[9, 8, 7, 6]]}]
            }
        })

        # Task 2: 90-degree rotation
        tasks.append({
            "name": "basic_rot_90",
            "category": "basic_geometric",
            "complexity": "simple",
            "required_capability": "rotation_90",
            "task": {
                "train": [
                    {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}
                ],
                "test": [{"input": [[5, 6, 7]], "output": [[7], [6], [5]]}]
            }
        })

        # Task 3: Translation (shift down)
        tasks.append({
            "name": "basic_translate",
            "category": "basic_geometric",
            "complexity": "simple",
            "required_capability": "translation",
            "task": {
                "train": [
                    {"input": [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
                     "output": [[0, 0, 0], [1, 1, 0], [0, 0, 0]]}
                ],
                "test": [
                    {"input": [[2, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     "output": [[0, 0, 0], [2, 2, 0], [0, 0, 0], [0, 0, 0]]}
                ]
            }
        })

        # Task 4: Vertical flip
        tasks.append({
            "name": "basic_v_flip",
            "category": "basic_geometric",
            "complexity": "simple",
            "required_capability": "vertical_reflection",
            "task": {
                "train": [
                    {"input": [[1, 2], [3, 4], [5, 6]],
                     "output": [[5, 6], [3, 4], [1, 2]]}
                ],
                "test": [
                    {"input": [[7, 8, 9], [1, 2, 3]],
                     "output": [[1, 2, 3], [7, 8, 9]]}
                ]
            }
        })

        # Task 5: 180-degree rotation
        tasks.append({
            "name": "basic_rot_180",
            "category": "basic_geometric",
            "complexity": "simple",
            "required_capability": "rotation_180",
            "task": {
                "train": [
                    {"input": [[1, 2, 3], [4, 5, 6]],
                     "output": [[6, 5, 4], [3, 2, 1]]}
                ],
                "test": [
                    {"input": [[7, 8], [9, 0]],
                     "output": [[0, 9], [8, 7]]}
                ]
            }
        })

        # ====================================================================
        # CATEGORY 2: OBJECT OPERATIONS (Partially supported)
        # ====================================================================

        # Task 6: Move single object
        tasks.append({
            "name": "move_object",
            "category": "object_ops",
            "complexity": "medium",
            "required_capability": "object_translation",
            "task": {
                "train": [
                    {"input": [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                     "output": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]]}
                ],
                "test": [
                    {"input": [[2, 2, 0, 0, 0], [2, 2, 0, 0, 0], [0, 0, 0, 0, 0]],
                     "output": [[0, 0, 0, 0, 0], [0, 2, 2, 0, 0], [0, 2, 2, 0, 0]]}
                ]
            }
        })

        # Task 7: Scale up object
        tasks.append({
            "name": "scale_up",
            "category": "object_ops",
            "complexity": "medium",
            "required_capability": "object_scaling",
            "task": {
                "train": [
                    {"input": [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                     "output": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]]}
                ],
                "test": [
                    {"input": [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0]],
                     "output": [[0, 0, 0, 0, 0], [0, 2, 2, 0, 0], [0, 2, 2, 0, 0]]}
                ]
            }
        })

        # Task 8: Duplicate object
        tasks.append({
            "name": "duplicate_object",
            "category": "object_ops",
            "complexity": "medium",
            "required_capability": "object_duplication",
            "task": {
                "train": [
                    {"input": [[1, 1, 0, 0], [1, 1, 0, 0]],
                     "output": [[1, 1, 1, 1], [1, 1, 1, 1]]}
                ],
                "test": [
                    {"input": [[2, 0, 0], [0, 0, 0]],
                     "output": [[2, 2, 0], [0, 0, 0]]}
                ]
            }
        })

        # Task 9: Fill enclosed region
        tasks.append({
            "name": "fill_enclosed",
            "category": "object_ops",
            "complexity": "hard",
            "required_capability": "region_filling",
            "task": {
                "train": [
                    {"input": [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
                     "output": [[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]}
                ],
                "test": [
                    {"input": [[3, 3, 3], [3, 0, 3], [3, 3, 3]],
                     "output": [[3, 3, 3], [3, 2, 3], [3, 3, 3]]}
                ]
            }
        })

        # Task 10: Extract largest object
        tasks.append({
            "name": "extract_largest",
            "category": "object_ops",
            "complexity": "hard",
            "required_capability": "object_selection_by_size",
            "task": {
                "train": [
                    {"input": [[1, 0, 2, 2], [0, 0, 2, 2], [3, 0, 0, 0]],
                     "output": [[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 0, 0]]}
                ],
                "test": [
                    {"input": [[4, 4, 4, 0], [0, 5, 0, 0], [0, 0, 0, 6]],
                     "output": [[4, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}
                ]
            }
        })

        # ====================================================================
        # CATEGORY 3: PATTERN OPERATIONS (Not well supported)
        # ====================================================================

        # Task 11: Tile pattern
        tasks.append({
            "name": "tile_pattern",
            "category": "pattern_ops",
            "complexity": "medium",
            "required_capability": "pattern_tiling",
            "task": {
                "train": [
                    {"input": [[1, 2], [3, 4]],
                     "output": [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]}
                ],
                "test": [
                    {"input": [[5, 6]],
                     "output": [[5, 6, 5, 6], [5, 6, 5, 6]]}
                ]
            }
        })

        # Task 12: Complete symmetry
        tasks.append({
            "name": "complete_symmetry",
            "category": "pattern_ops",
            "complexity": "medium",
            "required_capability": "symmetry_completion",
            "task": {
                "train": [
                    {"input": [[1, 2, 0], [3, 4, 0], [0, 0, 0]],
                     "output": [[1, 2, 1], [3, 4, 3], [1, 2, 1]]}
                ],
                "test": [
                    {"input": [[5, 0], [6, 0]],
                     "output": [[5, 5], [6, 6]]}
                ]
            }
        })

        # Task 13: Repeat pattern
        tasks.append({
            "name": "repeat_pattern",
            "category": "pattern_ops",
            "complexity": "medium",
            "required_capability": "pattern_repetition",
            "task": {
                "train": [
                    {"input": [[1, 0, 0, 0]],
                     "output": [[1, 1, 1, 1]]}
                ],
                "test": [
                    {"input": [[2, 3, 0, 0, 0, 0]],
                     "output": [[2, 3, 2, 3, 2, 3]]}
                ]
            }
        })

        # Task 14: Mirror and extend
        tasks.append({
            "name": "mirror_extend",
            "category": "pattern_ops",
            "complexity": "hard",
            "required_capability": "mirroring_extension",
            "task": {
                "train": [
                    {"input": [[1, 2, 3]],
                     "output": [[1, 2, 3, 3, 2, 1]]}
                ],
                "test": [
                    {"input": [[4, 5]],
                     "output": [[4, 5, 5, 4]]}
                ]
            }
        })

        # Task 15: Detect and continue sequence
        tasks.append({
            "name": "continue_sequence",
            "category": "pattern_ops",
            "complexity": "hard",
            "required_capability": "sequence_continuation",
            "task": {
                "train": [
                    {"input": [[1, 2, 3, 0]],
                     "output": [[1, 2, 3, 4]]}
                ],
                "test": [
                    {"input": [[2, 4, 6, 0]],
                     "output": [[2, 4, 6, 8]]}
                ]
            }
        })

        # ====================================================================
        # CATEGORY 4: COLOR OPERATIONS (Partially supported)
        # ====================================================================

        # Task 16: Simple color swap
        tasks.append({
            "name": "color_swap",
            "category": "color_ops",
            "complexity": "simple",
            "required_capability": "color_remapping",
            "task": {
                "train": [
                    {"input": [[1, 1, 2], [2, 1, 2]],
                     "output": [[2, 2, 1], [1, 2, 1]]}
                ],
                "test": [
                    {"input": [[1, 2, 1]],
                     "output": [[2, 1, 2]]}
                ]
            }
        })

        # Task 17: Color by position
        tasks.append({
            "name": "color_by_position",
            "category": "color_ops",
            "complexity": "medium",
            "required_capability": "positional_coloring",
            "task": {
                "train": [
                    {"input": [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
                     "output": [[0, 0, 0], [1, 2, 3], [0, 0, 0]]}
                ],
                "test": [
                    {"input": [[1, 1, 1, 1]],
                     "output": [[1, 2, 3, 4]]}
                ]
            }
        })

        # Task 18: Color by size
        tasks.append({
            "name": "color_by_size",
            "category": "color_ops",
            "complexity": "hard",
            "required_capability": "size_based_coloring",
            "task": {
                "train": [
                    {"input": [[1, 0, 2, 2], [0, 0, 2, 2]],
                     "output": [[9, 0, 3, 3], [0, 0, 3, 3]]}
                ],
                "test": [
                    {"input": [[4, 4, 4, 0], [5, 0, 0, 0]],
                     "output": [[3, 3, 3, 0], [9, 0, 0, 0]]}
                ]
            }
        })

        # Task 19: Conditional coloring
        tasks.append({
            "name": "conditional_color",
            "category": "color_ops",
            "complexity": "hard",
            "required_capability": "conditional_coloring",
            "task": {
                "train": [
                    {"input": [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
                     "output": [[3, 2, 3], [2, 3, 2], [3, 2, 3]]}
                ],
                "test": [
                    {"input": [[1, 1, 2], [2, 1, 1]],
                     "output": [[3, 3, 2], [2, 3, 3]]}
                ]
            }
        })

        # Task 20: Color palette change
        tasks.append({
            "name": "palette_change",
            "category": "color_ops",
            "complexity": "medium",
            "required_capability": "palette_transformation",
            "task": {
                "train": [
                    {"input": [[1, 2, 3], [1, 2, 3]],
                     "output": [[4, 5, 6], [4, 5, 6]]}
                ],
                "test": [
                    {"input": [[1, 1, 2, 3]],
                     "output": [[4, 4, 5, 6]]}
                ]
            }
        })

        # ====================================================================
        # CATEGORY 5: RELATIONAL REASONING (Not supported)
        # ====================================================================

        # Task 21: Connect nearest objects
        tasks.append({
            "name": "connect_nearest",
            "category": "relational",
            "complexity": "hard",
            "required_capability": "object_connection",
            "task": {
                "train": [
                    {"input": [[1, 0, 0, 0, 2], [0, 0, 0, 0, 0]],
                     "output": [[1, 3, 3, 3, 2], [0, 0, 0, 0, 0]]}
                ],
                "test": [
                    {"input": [[4, 0, 0, 5], [0, 0, 0, 0]],
                     "output": [[4, 3, 3, 5], [0, 0, 0, 0]]}
                ]
            }
        })

        # Task 22: Align objects
        tasks.append({
            "name": "align_objects",
            "category": "relational",
            "complexity": "hard",
            "required_capability": "object_alignment",
            "task": {
                "train": [
                    {"input": [[1, 0], [0, 0], [0, 2]],
                     "output": [[1, 2], [0, 0], [0, 0]]}
                ],
                "test": [
                    {"input": [[3, 0], [0, 0], [0, 0], [4, 0]],
                     "output": [[3, 4], [0, 0], [0, 0], [0, 0]]}
                ]
            }
        })

        # Task 23: Size-based ordering
        tasks.append({
            "name": "order_by_size",
            "category": "relational",
            "complexity": "hard",
            "required_capability": "size_based_ordering",
            "task": {
                "train": [
                    {"input": [[1, 1, 1, 0, 2, 0, 3, 3]],
                     "output": [[2, 0, 1, 1, 1, 0, 3, 3]]}
                ],
                "test": [
                    {"input": [[4, 4, 4, 4, 0, 5, 5, 0, 6]],
                     "output": [[6, 0, 5, 5, 0, 4, 4, 4, 4]]}
                ]
            }
        })

        # Task 24: Spatial relationships
        tasks.append({
            "name": "spatial_relations",
            "category": "relational",
            "complexity": "hard",
            "required_capability": "spatial_reasoning",
            "task": {
                "train": [
                    {"input": [[1, 0], [0, 2]],
                     "output": [[0, 2], [1, 0]]}
                ],
                "test": [
                    {"input": [[3, 0, 0], [0, 4, 0], [0, 0, 5]],
                     "output": [[0, 0, 5], [0, 4, 0], [3, 0, 0]]}
                ]
            }
        })

        # Task 25: Group by property
        tasks.append({
            "name": "group_by_property",
            "category": "relational",
            "complexity": "hard",
            "required_capability": "property_grouping",
            "task": {
                "train": [
                    {"input": [[1, 2, 1, 2, 1]],
                     "output": [[1, 1, 1, 2, 2]]}
                ],
                "test": [
                    {"input": [[3, 4, 3, 4, 3, 4]],
                     "output": [[3, 3, 3, 4, 4, 4]]}
                ]
            }
        })

        # ====================================================================
        # CATEGORY 6: COMPOSITE OPERATIONS (Partially supported)
        # ====================================================================

        # Task 26: Rotate then flip
        tasks.append({
            "name": "rotate_then_flip",
            "category": "composite",
            "complexity": "medium",
            "required_capability": "composite_geometric",
            "task": {
                "train": [
                    {"input": [[1, 2], [3, 4]],
                     "output": [[2, 4], [1, 3]]}
                ],
                "test": [
                    {"input": [[5, 6], [7, 8]],
                     "output": [[6, 8], [5, 7]]}
                ]
            }
        })

        # Task 27: Flip then translate
        tasks.append({
            "name": "flip_then_translate",
            "category": "composite",
            "complexity": "hard",
            "required_capability": "composite_flip_translate",
            "task": {
                "train": [
                    {"input": [[1, 2, 0], [0, 0, 0]],
                     "output": [[0, 0, 0], [0, 2, 1]]}
                ],
                "test": [
                    {"input": [[3, 4, 0, 0], [0, 0, 0, 0]],
                     "output": [[0, 0, 0, 0], [0, 0, 4, 3]]}
                ]
            }
        })

        # Task 28: Color remap then rotate
        tasks.append({
            "name": "remap_then_rotate",
            "category": "composite",
            "complexity": "hard",
            "required_capability": "composite_color_geometric",
            "task": {
                "train": [
                    {"input": [[1, 2], [1, 2]],
                     "output": [[3, 3], [4, 4]]}
                ],
                "test": [
                    {"input": [[1, 1], [2, 2]],
                     "output": [[3, 4], [3, 4]]}
                ]
            }
        })

        # Task 29: Multiple transformations
        tasks.append({
            "name": "multi_transform",
            "category": "composite",
            "complexity": "hard",
            "required_capability": "multi_step_transform",
            "task": {
                "train": [
                    {"input": [[1, 2, 3]],
                     "output": [[6, 5, 4], [6, 5, 4]]}
                ],
                "test": [
                    {"input": [[7, 8]],
                     "output": [[9, 8], [9, 8]]}
                ]
            }
        })

        # Task 30: Conditional composite
        tasks.append({
            "name": "conditional_composite",
            "category": "composite",
            "complexity": "hard",
            "required_capability": "conditional_transform",
            "task": {
                "train": [
                    {"input": [[1, 0, 2], [0, 0, 0]],
                     "output": [[0, 0, 0], [2, 0, 1]]}
                ],
                "test": [
                    {"input": [[3, 0, 4, 0]],
                     "output": [[0, 4, 0, 3]]}
                ]
            }
        })

        # ====================================================================
        # CATEGORY 7: ADVANCED OPERATIONS (Not supported)
        # ====================================================================

        # Task 31: Gravity (objects fall)
        tasks.append({
            "name": "gravity_down",
            "category": "advanced",
            "complexity": "hard",
            "required_capability": "gravity_simulation",
            "task": {
                "train": [
                    {"input": [[1, 0], [0, 0], [0, 2]],
                     "output": [[0, 0], [1, 0], [0, 2]]}
                ],
                "test": [
                    {"input": [[3, 0, 4], [0, 0, 0], [0, 0, 0]],
                     "output": [[0, 0, 0], [0, 0, 0], [3, 0, 4]]}
                ]
            }
        })

        # Task 32: Grow shape
        tasks.append({
            "name": "grow_shape",
            "category": "advanced",
            "complexity": "hard",
            "required_capability": "morphological_grow",
            "task": {
                "train": [
                    {"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     "output": [[0, 1, 0], [1, 1, 1], [0, 1, 0]]}
                ],
                "test": [
                    {"input": [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
                     "output": [[0, 2, 0, 0], [2, 2, 2, 0], [0, 2, 0, 0]]}
                ]
            }
        })

        # Task 33: Extend to boundary
        tasks.append({
            "name": "extend_to_boundary",
            "category": "advanced",
            "complexity": "hard",
            "required_capability": "boundary_extension",
            "task": {
                "train": [
                    {"input": [[1, 0, 0], [0, 0, 0], [0, 0, 2]],
                     "output": [[1, 1, 1], [0, 0, 0], [2, 2, 2]]}
                ],
                "test": [
                    {"input": [[3, 0], [0, 0], [0, 4]],
                     "output": [[3, 3], [0, 0], [4, 4]]}
                ]
            }
        })

        # Task 34: Flood fill from edges
        tasks.append({
            "name": "flood_fill",
            "category": "advanced",
            "complexity": "hard",
            "required_capability": "flood_filling",
            "task": {
                "train": [
                    {"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     "output": [[2, 2, 2], [2, 1, 2], [2, 2, 2]]}
                ],
                "test": [
                    {"input": [[0, 0, 0, 0], [0, 3, 3, 0], [0, 0, 0, 0]],
                     "output": [[2, 2, 2, 2], [2, 3, 3, 2], [2, 2, 2, 2]]}
                ]
            }
        })

        # Task 35: Maze/path finding
        tasks.append({
            "name": "find_path",
            "category": "advanced",
            "complexity": "very_hard",
            "required_capability": "path_finding",
            "task": {
                "train": [
                    {"input": [[1, 0, 0, 2], [0, 0, 0, 0]],
                     "output": [[1, 3, 3, 2], [0, 0, 0, 0]]}
                ],
                "test": [
                    {"input": [[4, 0, 0], [0, 0, 0], [0, 0, 5]],
                     "output": [[4, 3, 0], [0, 3, 0], [0, 0, 5]]}
                ]
            }
        })

        return tasks

    def test_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test solver on a task with detailed analysis"""

        task = task_data["task"]
        name = task_data["name"]

        print(f"\n[{name}] ", end="")

        try:
            pred1, pred2, metadata = self.solver.solve(task)
            target = np.array(task["test"][0]["output"])

            eval_results = evaluate_predictions(pred1, pred2, target)

            result = {
                "name": name,
                "category": task_data["category"],
                "complexity": task_data["complexity"],
                "required_capability": task_data["required_capability"],
                "success": bool(eval_results["any_correct"]),
                "pred1_exact": bool(eval_results["exact_match_1"]),
                "pred2_exact": bool(eval_results["exact_match_2"]),
                "pred1_pixel_acc": eval_results["pixel_accuracy_1"],
                "pred2_pixel_acc": eval_results["pixel_accuracy_2"],
                "best_pixel_acc": max(eval_results["pixel_accuracy_1"],
                                     eval_results["pixel_accuracy_2"]),
                "top_program": metadata["top_programs"][0]["schema"] if metadata["top_programs"] else None,
                "second_program": metadata["top_programs"][1]["schema"] if len(metadata["top_programs"]) > 1 else None,
                "pred1_shape": pred1.shape,
                "pred2_shape": pred2.shape,
                "target_shape": target.shape,
                "shape_match": pred1.shape == target.shape,
                "error": None
            }

            # Status indicator
            if result["success"]:
                print("✓", end="")
            else:
                acc = result["best_pixel_acc"]
                if acc >= 0.9:
                    print("~", end="")  # Close
                elif acc >= 0.5:
                    print("◐", end="")  # Partial
                else:
                    print("✗", end="")  # Failed

        except Exception as e:
            result = {
                "name": name,
                "category": task_data["category"],
                "complexity": task_data["complexity"],
                "required_capability": task_data["required_capability"],
                "success": False,
                "error": str(e)
            }
            print("E", end="")  # Error

        return result

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run complete comprehensive test"""

        print("\n" + "="*70)
        print("COMPREHENSIVE ARC TEST SUITE - SIMPLE TO COMPLEX")
        print("="*70)

        tasks = self.create_comprehensive_tasks()
        print(f"\nTesting {len(tasks)} diverse tasks...")
        print("\nLegend: ✓=Solved, ~=Close(>90%), ◐=Partial(>50%), ✗=Failed, E=Error")
        print("-"*70)

        # Test by category
        from collections import defaultdict
        by_category = defaultdict(list)

        for task_data in tasks:
            by_category[task_data["category"]].append(task_data)

        for category, category_tasks in sorted(by_category.items()):
            print(f"\n{category.upper().replace('_', ' ')}:", end=" ")
            for task_data in category_tasks:
                result = self.test_task(task_data)
                self.results.append(result)
            print()

        # Analyze results
        analysis = self.analyze_comprehensive_results()

        return analysis

    def analyze_comprehensive_results(self) -> Dict[str, Any]:
        """Detailed analysis of all results"""

        print("\n" + "="*70)
        print("COMPREHENSIVE ANALYSIS")
        print("="*70)

        # Overall stats
        total = len(self.results)
        solved = len([r for r in self.results if r.get("success")])

        print(f"\nOVERALL: {solved}/{total} ({solved/total:.1%}) solved")

        # By category
        print("\n" + "="*70)
        print("PERFORMANCE BY CATEGORY")
        print("="*70)

        from collections import defaultdict
        by_cat = defaultdict(list)
        for r in self.results:
            by_cat[r["category"]].append(r)

        for cat in sorted(by_cat.keys()):
            results = by_cat[cat]
            n_solved = len([r for r in results if r.get("success")])
            n_total = len(results)
            avg_acc = np.mean([r.get("best_pixel_acc", 0) for r in results])

            print(f"\n{cat.upper().replace('_', ' ')}:")
            print(f"  Solved: {n_solved}/{n_total} ({n_solved/n_total:.1%})")
            print(f"  Avg Pixel Accuracy: {avg_acc:.1%}")

            # List unsolved tasks
            unsolved = [r for r in results if not r.get("success")]
            if unsolved:
                print(f"  Missing capabilities:")
                for r in unsolved:
                    cap = r.get("required_capability", "unknown")
                    acc = r.get("best_pixel_acc", 0)
                    print(f"    - {cap} ({acc:.1%} accuracy)")

        # By complexity
        print("\n" + "="*70)
        print("PERFORMANCE BY COMPLEXITY")
        print("="*70)

        by_complexity = defaultdict(list)
        for r in self.results:
            by_complexity[r.get("complexity", "unknown")].append(r)

        for complexity in ["simple", "medium", "hard", "very_hard"]:
            if complexity in by_complexity:
                results = by_complexity[complexity]
                n_solved = len([r for r in results if r.get("success")])
                n_total = len(results)
                print(f"{complexity.upper()}: {n_solved}/{n_total} ({n_solved/n_total:.1%})")

        # Missing capabilities analysis
        print("\n" + "="*70)
        print("MISSING CAPABILITIES ANALYSIS")
        print("="*70)

        missing_caps = defaultdict(list)
        for r in self.results:
            if not r.get("success"):
                cap = r.get("required_capability", "unknown")
                missing_caps[cap].append(r)

        print(f"\nIdentified {len(missing_caps)} missing capabilities:")

        # Sort by number of affected tasks
        sorted_caps = sorted(missing_caps.items(),
                           key=lambda x: len(x[1]),
                           reverse=True)

        for cap, tasks in sorted_caps:
            print(f"\n  {cap} (affects {len(tasks)} tasks):")
            for t in tasks:
                acc = t.get("best_pixel_acc", 0)
                print(f"    - {t['name']} ({acc:.1%} accuracy)")

        # What would it take to reach 100%?
        print("\n" + "="*70)
        print("PATH TO 100% SUCCESS")
        print("="*70)

        print(f"\nCurrent: {solved}/{total} ({solved/total:.1%})")
        print(f"Gap: {total - solved} tasks")

        print("\nRequired capabilities to add:")
        for i, (cap, tasks) in enumerate(sorted_caps[:10], 1):
            print(f"  {i}. {cap} → +{len(tasks)} tasks")

        return {
            "total": total,
            "solved": solved,
            "success_rate": solved / total,
            "by_category": {cat: len([r for r in rs if r.get("success")]) / len(rs)
                          for cat, rs in by_cat.items()},
            "by_complexity": {comp: len([r for r in rs if r.get("success")]) / len(rs)
                            for comp, rs in by_complexity.items()},
            "missing_capabilities": {cap: len(tasks)
                                    for cap, tasks in missing_caps.items()},
            "results": self.results
        }


def main():
    """Run comprehensive test"""

    test_suite = ComprehensiveARCTestSuite()
    analysis = test_suite.run_comprehensive_test()

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return analysis


if __name__ == "__main__":
    analysis = main()
