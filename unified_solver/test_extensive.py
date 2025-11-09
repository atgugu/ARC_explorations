"""
Extensive Testing on Large Variety of ARC Tasks
================================================

Test on 50+ diverse ARC tasks with focus on exact match (100% pixel accuracy).
Ensures top-2 predictions are different.
"""

import numpy as np
from arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid
from arc_loader import ARCEvaluator
from typing import Dict, List, Tuple
import random


def create_extensive_test_suite() -> Dict[str, Tuple[ARCTask, Grid, str]]:
    """Create 50+ diverse ARC-style tasks"""
    tasks = {}

    # ==========================================================================
    # CATEGORY 1: GEOMETRIC TRANSFORMATIONS (10 tasks)
    # ==========================================================================

    # 1.1: Flip vertical
    tasks['geo_flip_v'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
            (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
        ], Grid([[9, 1], [2, 3]])),
        Grid([[1, 9], [3, 2]]),
        "Geometric"
    )

    # 1.2: Flip horizontal
    tasks['geo_flip_h'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[3, 4], [1, 2]])),
            (Grid([[5, 6], [7, 8]]), Grid([[7, 8], [5, 6]])),
        ], Grid([[9, 1], [2, 3]])),
        Grid([[2, 3], [9, 1]]),
        "Geometric"
    )

    # 1.3: Rotate 90 clockwise
    tasks['geo_rot90'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[3, 1], [4, 2]])),
            (Grid([[5, 6], [7, 8]]), Grid([[7, 5], [8, 6]])),
        ], Grid([[9, 1], [2, 3]])),
        Grid([[2, 9], [3, 1]]),
        "Geometric"
    )

    # 1.4: Rotate 180
    tasks['geo_rot180'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[4, 3], [2, 1]])),
            (Grid([[5, 6], [7, 8]]), Grid([[8, 7], [6, 5]])),
        ], Grid([[9, 1], [2, 3]])),
        Grid([[3, 2], [1, 9]]),
        "Geometric"
    )

    # 1.5: Rotate 270 (90 counter-clockwise)
    tasks['geo_rot270'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[2, 4], [1, 3]])),
            (Grid([[5, 6], [7, 8]]), Grid([[6, 8], [5, 7]])),
        ], Grid([[9, 1], [2, 3]])),
        Grid([[1, 3], [9, 2]]),
        "Geometric"
    )

    # 1.6: Transpose
    tasks['geo_transpose'] = (
        ARCTask([
            (Grid([[1, 2, 3], [4, 5, 6]]), Grid([[1, 4], [2, 5], [3, 6]])),
            (Grid([[7, 8], [9, 0]]), Grid([[7, 9], [8, 0]])),
        ], Grid([[1, 2], [3, 4], [5, 6]])),
        Grid([[1, 3, 5], [2, 4, 6]]),
        "Geometric"
    )

    # 1.7: Identity (no change)
    tasks['geo_identity'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[1, 2], [3, 4]])),
            (Grid([[5, 6], [7, 8]]), Grid([[5, 6], [7, 8]])),
        ], Grid([[9, 0], [1, 2]])),
        Grid([[9, 0], [1, 2]]),
        "Geometric"
    )

    # 1.8: Flip V then H
    tasks['geo_flip_both'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[4, 3], [2, 1]])),
            (Grid([[5, 6], [7, 8]]), Grid([[8, 7], [6, 5]])),
        ], Grid([[9, 1], [2, 3]])),
        Grid([[3, 2], [1, 9]]),
        "Geometric"
    )

    # 1.9: Rotate then flip
    tasks['geo_rot_flip'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[2, 4], [1, 3]])),
            (Grid([[5, 6], [7, 8]]), Grid([[6, 8], [5, 7]])),
        ], Grid([[9, 1], [2, 3]])),
        Grid([[1, 3], [9, 2]]),
        "Geometric"
    )

    # 1.10: Mirror horizontal
    tasks['geo_mirror_h'] = (
        ARCTask([
            (Grid([[1, 0]]), Grid([[1, 1]])),
            (Grid([[2, 0]]), Grid([[2, 2]])),
        ], Grid([[3, 0]])),
        Grid([[3, 3]]),
        "Geometric"
    )

    # ==========================================================================
    # CATEGORY 2: COLOR TRANSFORMATIONS (10 tasks)
    # ==========================================================================

    # 2.1-2.5: Replace specific colors
    for old, new in [(1, 5), (2, 6), (3, 7), (1, 9), (2, 8)]:
        tasks[f'color_replace_{old}_to_{new}'] = (
            ARCTask([
                (Grid([[old, 2, old], [old, 3, old]]), Grid([[new, 2, new], [new, 3, new]])),
                (Grid([[old, 0, 0], [old, old, 2]]), Grid([[new, 0, 0], [new, new, 2]])),
            ], Grid([[old, old, 2], [0, old, 0]])),
            Grid([[new, new, 2], [0, new, 0]]),
            "Color"
        )

    # 2.6: Swap colors 1<->2
    tasks['color_swap_1_2'] = (
        ARCTask([
            (Grid([[1, 2, 1], [2, 1, 2]]), Grid([[2, 1, 2], [1, 2, 1]])),
            (Grid([[1, 1, 2], [2, 2, 1]]), Grid([[2, 2, 1], [1, 1, 2]])),
        ], Grid([[1, 2, 0], [2, 1, 0]])),
        Grid([[2, 1, 0], [1, 2, 0]]),
        "Color"
    )

    # 2.7: Invert colors (simple case)
    tasks['color_invert'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[8, 7], [6, 5]])),
            (Grid([[5, 6], [7, 8]]), Grid([[4, 3], [2, 1]])),
        ], Grid([[9, 0], [1, 2]])),
        Grid([[0, 9], [8, 7]]),
        "Color"
    )

    # 2.8: Increment all colors by 1
    tasks['color_increment'] = (
        ARCTask([
            (Grid([[1, 2, 3]]), Grid([[2, 3, 4]])),
            (Grid([[4, 5, 6]]), Grid([[5, 6, 7]])),
        ], Grid([[7, 8, 9]])),
        Grid([[8, 9, 0]]),  # wraps around
        "Color"
    )

    # 2.9: Set all to single color
    tasks['color_set_all'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[5, 5], [5, 5]])),
            (Grid([[6, 7], [8, 9]]), Grid([[5, 5], [5, 5]])),
        ], Grid([[0, 1], [2, 3]])),
        Grid([[5, 5], [5, 5]]),
        "Color"
    )

    # 2.10: Keep only color 1
    tasks['color_keep_1'] = (
        ARCTask([
            (Grid([[1, 2, 1], [3, 1, 4]]), Grid([[1, 0, 1], [0, 1, 0]])),
            (Grid([[1, 1, 2], [3, 1, 3]]), Grid([[1, 1, 0], [0, 1, 0]])),
        ], Grid([[1, 2, 3], [1, 1, 2]])),
        Grid([[1, 0, 0], [1, 1, 0]]),
        "Color"
    )

    # ==========================================================================
    # CATEGORY 3: SCALING/TILING (8 tasks)
    # ==========================================================================

    # 3.1: Zoom 2x (2D)
    tasks['scale_zoom_2x'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])),
            (Grid([[5]]), Grid([[5, 5], [5, 5]])),
        ], Grid([[7, 8]])),
        Grid([[7, 7, 8, 8], [7, 7, 8, 8]]),
        "Scaling"
    )

    # 3.2: Zoom 3x (1D)
    tasks['scale_zoom_3x_1d'] = (
        ARCTask([
            (Grid([[1, 2]]), Grid([[1, 1, 1, 2, 2, 2]])),
            (Grid([[3]]), Grid([[3, 3, 3]])),
        ], Grid([[4, 5]])),
        Grid([[4, 4, 4, 5, 5, 5]]),
        "Scaling"
    )

    # 3.3: Tile 2x1
    tasks['scale_tile_2x1'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[1, 2, 1, 2], [3, 4, 3, 4]])),
            (Grid([[5]]), Grid([[5, 5]])),
        ], Grid([[6, 7]])),
        Grid([[6, 7, 6, 7]]),
        "Scaling"
    )

    # 3.4: Tile 1x2
    tasks['scale_tile_1x2'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[1, 2], [3, 4], [1, 2], [3, 4]])),
            (Grid([[5]]), Grid([[5], [5]])),
        ], Grid([[6, 7]])),
        Grid([[6, 7], [6, 7]]),
        "Scaling"
    )

    # 3.5: Tile 2x2
    tasks['scale_tile_2x2'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])),
            (Grid([[5]]), Grid([[5, 5], [5, 5]])),
        ], Grid([[6, 7]])),
        Grid([[6, 7, 6, 7], [6, 7, 6, 7]]),
        "Scaling"
    )

    # 3.6: Shrink 2x (take every other)
    tasks['scale_shrink_2x'] = (
        ARCTask([
            (Grid([[1, 2, 3, 4], [5, 6, 7, 8]]), Grid([[1, 3], [5, 7]])),
            (Grid([[9, 0, 1, 2]]), Grid([[9, 1]])),
        ], Grid([[3, 4, 5, 6], [7, 8, 9, 0]])),
        Grid([[3, 5], [7, 9]]),
        "Scaling"
    )

    # 3.7: Extend with border
    tasks['scale_extend'] = (
        ARCTask([
            (Grid([[1]]), Grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
            (Grid([[2]]), Grid([[0, 0, 0], [0, 2, 0], [0, 0, 0]])),
        ], Grid([[3]])),
        Grid([[0, 0, 0], [0, 3, 0], [0, 0, 0]]),
        "Scaling"
    )

    # 3.8: Crop to content
    tasks['scale_crop'] = (
        ARCTask([
            (Grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), Grid([[1]])),
            (Grid([[0, 2, 2, 0], [0, 2, 2, 0]]), Grid([[2, 2], [2, 2]])),
        ], Grid([[0, 0, 0, 0], [0, 3, 3, 0], [0, 3, 3, 0], [0, 0, 0, 0]])),
        Grid([[3, 3], [3, 3]]),
        "Scaling"
    )

    # ==========================================================================
    # CATEGORY 4: OBJECT-BASED (6 tasks)
    # ==========================================================================

    # 4.1: Keep largest
    tasks['obj_largest'] = (
        ARCTask([
            (Grid([[1, 1, 0, 2], [1, 1, 0, 0], [0, 0, 0, 3]]),
             Grid([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])),
            (Grid([[5, 0, 7, 7], [0, 0, 7, 7]]),
             Grid([[0, 0, 7, 7], [0, 0, 7, 7]])),
        ], Grid([[1, 0, 2, 2], [0, 0, 2, 2], [3, 3, 0, 0]])),
        Grid([[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 0, 0]]),
        "Object"
    )

    # 4.2: Keep smallest
    tasks['obj_smallest'] = (
        ARCTask([
            (Grid([[1, 1, 0, 2], [1, 1, 0, 0], [0, 0, 0, 3]]),
             Grid([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3]])),
            (Grid([[5, 0, 7, 7], [0, 0, 7, 7]]),
             Grid([[5, 0, 0, 0], [0, 0, 0, 0]])),
        ], Grid([[1, 0, 2, 2], [0, 0, 2, 2], [3, 3, 0, 0]])),
        Grid([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        "Object"
    )

    # 4.3: Count objects (output is number)
    tasks['obj_count'] = (
        ARCTask([
            (Grid([[1, 0, 2], [0, 0, 0], [3, 0, 4]]), Grid([[4]])),  # 4 objects
            (Grid([[1, 0, 2]]), Grid([[2]])),  # 2 objects
        ], Grid([[5, 0, 6], [0, 7, 0]])),
        Grid([[3]]),  # 3 objects
        "Object"
    )

    # 4.4: Remove isolated pixels
    tasks['obj_remove_isolated'] = (
        ARCTask([
            (Grid([[1, 0, 2], [0, 0, 0], [3, 0, 0]]), Grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])),
            (Grid([[1, 1, 0], [1, 1, 2]]), Grid([[1, 1, 0], [1, 1, 0]])),
        ], Grid([[3, 0, 4], [0, 5, 5], [0, 5, 5]])),
        Grid([[0, 0, 0], [0, 5, 5], [0, 5, 5]]),
        "Object"
    )

    # 4.5: Extract objects only (remove background)
    tasks['obj_extract'] = (
        ARCTask([
            (Grid([[0, 1, 0], [0, 1, 0]]), Grid([[1], [1]])),
            (Grid([[0, 2, 2, 0]]), Grid([[2, 2]])),
        ], Grid([[0, 0, 3, 0], [0, 0, 3, 0]])),
        Grid([[3], [3]]),
        "Object"
    )

    # 4.6: Dilate objects
    tasks['obj_dilate'] = (
        ARCTask([
            (Grid([[0, 1, 0], [0, 0, 0]]), Grid([[1, 1, 1], [1, 1, 1]])),
            (Grid([[2, 0], [0, 0]]), Grid([[2, 2], [2, 2]])),
        ], Grid([[0, 0, 0], [0, 3, 0], [0, 0, 0]])),
        Grid([[0, 3, 0], [3, 3, 3], [0, 3, 0]]),
        "Object"
    )

    # ==========================================================================
    # CATEGORY 5: PATTERN COMPLETION (6 tasks)
    # ==========================================================================

    # 5.1: Fill background with 1
    tasks['pattern_fill_bg'] = (
        ARCTask([
            (Grid([[0, 2, 0], [0, 0, 3]]), Grid([[1, 2, 1], [1, 1, 3]])),
            (Grid([[0, 0, 5], [0, 0, 0]]), Grid([[1, 1, 5], [1, 1, 1]])),
        ], Grid([[0, 0, 7], [8, 0, 0]])),
        Grid([[1, 1, 7], [8, 1, 1]]),
        "Pattern"
    )

    # 5.2: Complete horizontal line
    tasks['pattern_h_line'] = (
        ARCTask([
            (Grid([[1, 0, 1], [0, 0, 0]]), Grid([[1, 1, 1], [0, 0, 0]])),
            (Grid([[2, 0, 0, 2]]), Grid([[2, 2, 2, 2]])),
        ], Grid([[3, 0, 0, 3], [0, 0, 0, 0]])),
        Grid([[3, 3, 3, 3], [0, 0, 0, 0]]),
        "Pattern"
    )

    # 5.3: Complete vertical line
    tasks['pattern_v_line'] = (
        ARCTask([
            (Grid([[1], [0], [1]]), Grid([[1], [1], [1]])),
            (Grid([[2], [0], [0], [2]]), Grid([[2], [2], [2], [2]])),
        ], Grid([[3], [0], [0], [3]])),
        Grid([[3], [3], [3], [3]]),
        "Pattern"
    )

    # 5.4: Complete diagonal
    tasks['pattern_diagonal'] = (
        ARCTask([
            (Grid([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), Grid([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            (Grid([[2, 0], [0, 2]]), Grid([[2, 0], [0, 2]])),
        ], Grid([[3, 0, 0], [0, 0, 0], [0, 0, 3]])),
        Grid([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
        "Pattern"
    )

    # 5.5: Complete square
    tasks['pattern_square'] = (
        ARCTask([
            (Grid([[1, 1, 0], [1, 0, 0], [0, 0, 0]]), Grid([[1, 1, 1], [1, 0, 1], [1, 1, 1]])),
            (Grid([[2, 2], [2, 2]]), Grid([[2, 2], [2, 2]])),
        ], Grid([[3, 3, 3], [3, 0, 0], [0, 0, 0]])),
        Grid([[3, 3, 3], [3, 0, 3], [3, 3, 3]]),
        "Pattern"
    )

    # 5.6: Repeat pattern
    tasks['pattern_repeat'] = (
        ARCTask([
            (Grid([[1, 2]]), Grid([[1, 2, 1, 2]])),
            (Grid([[3]]), Grid([[3, 3]])),
        ], Grid([[4, 5]])),
        Grid([[4, 5, 4, 5]]),
        "Pattern"
    )

    # ==========================================================================
    # CATEGORY 6: COMPOSITE (5 tasks)
    # ==========================================================================

    # 6.1: Rotate then flip
    tasks['comp_rot_flip'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[2, 4], [1, 3]])),
            (Grid([[5, 6], [7, 8]]), Grid([[6, 8], [5, 7]])),
        ], Grid([[9, 0], [1, 2]])),
        Grid([[0, 2], [9, 1]]),
        "Composite"
    )

    # 6.2: Flip then rotate
    tasks['comp_flip_rot'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[4, 2], [3, 1]])),
            (Grid([[5, 6], [7, 8]]), Grid([[8, 6], [7, 5]])),
        ], Grid([[9, 0], [1, 2]])),
        Grid([[2, 0], [1, 9]]),
        "Composite"
    )

    # 6.3: Zoom then flip
    tasks['comp_zoom_flip'] = (
        ARCTask([
            (Grid([[1, 2]]), Grid([[2, 2, 1, 1]])),
            (Grid([[3]]), Grid([[3, 3]])),
        ], Grid([[4, 5]])),
        Grid([[5, 5, 4, 4]]),
        "Composite"
    )

    # 6.4: Replace then rotate
    tasks['comp_replace_rot'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[2, 5], [4, 5]])),  # Replace 1->5 then rotate
            (Grid([[1, 1], [5, 6]]), Grid([[1, 5], [5, 6]])),
        ], Grid([[1, 2], [3, 1]])),
        Grid([[2, 5], [1, 5]]),
        "Composite"
    )

    # 6.5: Flip twice (identity)
    tasks['comp_flip_twice'] = (
        ARCTask([
            (Grid([[1, 2], [3, 4]]), Grid([[1, 2], [3, 4]])),
            (Grid([[5, 6], [7, 8]]), Grid([[5, 6], [7, 8]])),
        ], Grid([[9, 0], [1, 2]])),
        Grid([[9, 0], [1, 2]]),
        "Composite"
    )

    # Add 5 more complex/edge cases to reach 50+
    # ==========================================================================
    # CATEGORY 7: EDGE CASES (5 tasks)
    # ==========================================================================

    # 7.1: Single pixel
    tasks['edge_single_pixel'] = (
        ARCTask([
            (Grid([[1]]), Grid([[1]])),
            (Grid([[2]]), Grid([[2]])),
        ], Grid([[3]])),
        Grid([[3]]),
        "Edge"
    )

    # 7.2: All zeros
    tasks['edge_all_zeros'] = (
        ARCTask([
            (Grid([[0, 0], [0, 0]]), Grid([[0, 0], [0, 0]])),
            (Grid([[0, 0, 0]]), Grid([[0, 0, 0]])),
        ], Grid([[0, 0], [0, 0], [0, 0]])),
        Grid([[0, 0], [0, 0], [0, 0]]),
        "Edge"
    )

    # 7.3: All same color
    tasks['edge_all_same'] = (
        ARCTask([
            (Grid([[3, 3], [3, 3]]), Grid([[3, 3], [3, 3]])),
            (Grid([[3, 3, 3]]), Grid([[3, 3, 3]])),
        ], Grid([[3, 3], [3, 3], [3, 3]])),
        Grid([[3, 3], [3, 3], [3, 3]]),
        "Edge"
    )

    # 7.4: Checkerboard pattern
    tasks['edge_checkerboard'] = (
        ARCTask([
            (Grid([[1, 2], [2, 1]]), Grid([[1, 2], [2, 1]])),
            (Grid([[3, 4, 3], [4, 3, 4]]), Grid([[3, 4, 3], [4, 3, 4]])),
        ], Grid([[5, 6], [6, 5]])),
        Grid([[5, 6], [6, 5]]),
        "Edge"
    )

    # 7.5: Large grid (stress test)
    tasks['edge_large_grid'] = (
        ARCTask([
            (Grid(np.ones((5, 5), dtype=int)), Grid(np.ones((5, 5), dtype=int))),
            (Grid(np.zeros((5, 5), dtype=int)), Grid(np.zeros((5, 5), dtype=int))),
        ], Grid(np.full((5, 5), 3, dtype=int))),
        Grid(np.full((5, 5), 3, dtype=int)),
        "Edge"
    )

    return tasks


def run_extensive_test(max_tasks: int = None):
    """Run extensive testing"""

    print("="*80)
    print("EXTENSIVE ARC-AGI SOLVER TESTING")
    print("="*80)
    print(f"\nTesting exact match (100% pixel accuracy) with 2 attempts per task")
    print("Ensuring top-2 predictions are different\n")

    tasks = create_extensive_test_suite()

    if max_tasks:
        task_items = list(tasks.items())[:max_tasks]
        tasks = dict(task_items)

    solver = ARCActiveInferenceSolver(workspace_capacity=20)

    # Results tracking
    results = {
        'exact_match': [],
        'first_correct': [],
        'second_correct': [],
        'both_wrong': [],
        'predictions_same': [],
        'by_category': {}
    }

    print(f"Testing {len(tasks)} tasks...\n")

    for task_name, (task, expected_output, category) in tasks.items():
        task.test_output = expected_output

        try:
            # Solve
            predictions = solver.solve(task, verbose=False)

            # Check if predictions are different
            same_predictions = np.array_equal(predictions[0].data, predictions[1].data)
            if same_predictions:
                results['predictions_same'].append(task_name)

            # Check exact matches
            match_1 = np.array_equal(predictions[0].data, expected_output.data)
            match_2 = np.array_equal(predictions[1].data, expected_output.data)

            # Track by category
            if category not in results['by_category']:
                results['by_category'][category] = {
                    'total': 0, 'solved': 0, 'first': 0, 'second': 0
                }

            results['by_category'][category]['total'] += 1

            if match_1 or match_2:
                results['exact_match'].append(task_name)
                results['by_category'][category]['solved'] += 1

                if match_1:
                    results['first_correct'].append(task_name)
                    results['by_category'][category]['first'] += 1
                    symbol = "✓1"
                else:
                    results['second_correct'].append(task_name)
                    results['by_category'][category]['second'] += 1
                    symbol = "✓2"
            else:
                results['both_wrong'].append(task_name)
                symbol = "✗ "

            # Print result
            same_marker = " [SAME!]" if same_predictions else ""
            print(f"{symbol} {task_name:30s} ({category:12s}){same_marker}")

        except Exception as e:
            print(f"✗  {task_name:30s} ({category:12s}) - ERROR: {str(e)[:40]}")
            results['both_wrong'].append(task_name)
            if category not in results['by_category']:
                results['by_category'][category] = {
                    'total': 0, 'solved': 0, 'first': 0, 'second': 0
                }
            results['by_category'][category]['total'] += 1

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total = len(tasks)
    solved = len(results['exact_match'])
    first = len(results['first_correct'])
    second = len(results['second_correct'])
    failed = len(results['both_wrong'])
    same_preds = len(results['predictions_same'])

    print(f"\nTotal Tasks:           {total}")
    print(f"Exact Match (either):  {solved} ({solved/total*100:.1f}%)")
    print(f"  - Attempt 1 correct: {first} ({first/total*100:.1f}%)")
    print(f"  - Attempt 2 correct: {second} ({second/total*100:.1f}%)")
    print(f"Both Wrong:            {failed} ({failed/total*100:.1f}%)")
    print(f"Predictions Same:      {same_preds} ({same_preds/total*100:.1f}%)")

    # Category breakdown
    print("\n" + "="*80)
    print("RESULTS BY CATEGORY")
    print("="*80)

    for category in sorted(results['by_category'].keys()):
        stats = results['by_category'][category]
        total_cat = stats['total']
        solved_cat = stats['solved']
        first_cat = stats['first']
        second_cat = stats['second']

        print(f"\n{category}:")
        print(f"  Success Rate: {solved_cat}/{total_cat} ({solved_cat/total_cat*100:.1f}%)")
        print(f"  Attempt 1:    {first_cat}/{total_cat} ({first_cat/total_cat*100:.1f}%)")
        print(f"  Attempt 2:    {second_cat}/{total_cat} ({second_cat/total_cat*100:.1f}%)")

    # Prediction diversity analysis
    print("\n" + "="*80)
    print("PREDICTION DIVERSITY ANALYSIS")
    print("="*80)

    if same_preds > 0:
        print(f"\nWARNING: {same_preds} tasks had identical top-2 predictions:")
        for task_name in results['predictions_same'][:10]:
            print(f"  - {task_name}")
        if len(results['predictions_same']) > 10:
            print(f"  ... and {len(results['predictions_same']) - 10} more")
    else:
        print("\n✓ All tasks produced different top-2 predictions")

    # Strengths and weaknesses
    print("\n" + "="*80)
    print("STRENGTHS & WEAKNESSES")
    print("="*80)

    print("\nSTRONG CATEGORIES (>80% success):")
    strong = [(cat, stats) for cat, stats in results['by_category'].items()
              if stats['solved'] / stats['total'] > 0.8]
    if strong:
        for cat, stats in sorted(strong, key=lambda x: x[1]['solved']/x[1]['total'], reverse=True):
            rate = stats['solved'] / stats['total']
            print(f"  {cat:15s}: {stats['solved']}/{stats['total']} ({rate*100:.1f}%)")
    else:
        print("  (None)")

    print("\nWEAK CATEGORIES (<50% success):")
    weak = [(cat, stats) for cat, stats in results['by_category'].items()
            if stats['solved'] / stats['total'] < 0.5]
    if weak:
        for cat, stats in sorted(weak, key=lambda x: x[1]['solved']/x[1]['total']):
            rate = stats['solved'] / stats['total']
            print(f"  {cat:15s}: {stats['solved']}/{stats['total']} ({rate*100:.1f}%)")
    else:
        print("  (None)")

    # Value of second attempt
    print("\n" + "="*80)
    print("VALUE OF SECOND ATTEMPT")
    print("="*80)

    only_second = second - 0  # Tasks solved only by second attempt
    print(f"\nTasks where second attempt saved us: {second}")
    print(f"Additional tasks solved by having 2 attempts: {second}")
    print(f"Improvement from 2 attempts: {second/total*100:.1f}% absolute")

    return results


if __name__ == "__main__":
    import sys

    max_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else None
    results = run_extensive_test(max_tasks)

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
