"""
Comprehensive Testing on Diverse ARC-AGI Tasks
==============================================

Test the solver on various task types to identify failure modes and strengths.
"""

import numpy as np
from arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid, visualize_grid
from arc_loader import ARCEvaluator
from typing import Dict, List, Tuple


def create_diverse_test_suite() -> Dict[str, Tuple[ARCTask, Grid, str]]:
    """
    Create a comprehensive test suite covering diverse ARC task types

    Returns:
        Dict mapping task_name -> (task, expected_output, task_type)
    """
    tasks = {}

    # =========================================================================
    # CATEGORY 1: SIMPLE GEOMETRIC TRANSFORMATIONS
    # =========================================================================

    # 1.1: Flip Vertical
    tasks['flip_vertical'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
                (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
            ],
            test_input=Grid([[1, 0], [2, 3]])
        ),
        Grid([[0, 1], [3, 2]]),
        "Simple geometric transformation"
    )

    # 1.2: Flip Horizontal
    tasks['flip_horizontal'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2], [3, 4]]), Grid([[3, 4], [1, 2]])),
                (Grid([[5, 6], [7, 8]]), Grid([[7, 8], [5, 6]])),
            ],
            test_input=Grid([[1, 0], [2, 3]])
        ),
        Grid([[2, 3], [1, 0]]),
        "Simple geometric transformation"
    )

    # 1.3: Rotate 90
    tasks['rotate_90'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2], [3, 4]]), Grid([[3, 1], [4, 2]])),
                (Grid([[5, 6], [7, 8]]), Grid([[7, 5], [8, 6]])),
            ],
            test_input=Grid([[1, 2], [3, 4]])
        ),
        Grid([[3, 1], [4, 2]]),
        "Simple geometric transformation"
    )

    # 1.4: Transpose
    tasks['transpose'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2, 3], [4, 5, 6]]), Grid([[1, 4], [2, 5], [3, 6]])),
                (Grid([[7, 8], [9, 0]]), Grid([[7, 9], [8, 0]])),
            ],
            test_input=Grid([[1, 2], [3, 4], [5, 6]])
        ),
        Grid([[1, 3, 5], [2, 4, 6]]),
        "Simple geometric transformation"
    )

    # =========================================================================
    # CATEGORY 2: COLOR TRANSFORMATIONS
    # =========================================================================

    # 2.1: Replace single color
    tasks['replace_color_1_to_5'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2, 1], [1, 3, 1]]), Grid([[5, 2, 5], [5, 3, 5]])),
                (Grid([[1, 0, 0], [1, 1, 2]]), Grid([[5, 0, 0], [5, 5, 2]])),
            ],
            test_input=Grid([[1, 1, 2], [0, 1, 0]])
        ),
        Grid([[5, 5, 2], [0, 5, 0]]),
        "Color transformation"
    )

    # 2.2: Swap two colors
    tasks['swap_colors_1_2'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2, 1], [2, 1, 2]]), Grid([[2, 1, 2], [1, 2, 1]])),
                (Grid([[1, 1, 2], [2, 2, 1]]), Grid([[2, 2, 1], [1, 1, 2]])),
            ],
            test_input=Grid([[1, 2, 0], [2, 1, 0]])
        ),
        Grid([[2, 1, 0], [1, 2, 0]]),
        "Color transformation"
    )

    # =========================================================================
    # CATEGORY 3: SCALING/TILING
    # =========================================================================

    # 3.1: Zoom 2x
    tasks['zoom_2x'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2]]), Grid([[1, 1, 2, 2]])),
                (Grid([[3, 4]]), Grid([[3, 3, 4, 4]])),
            ],
            test_input=Grid([[5, 6]])
        ),
        Grid([[5, 5, 6, 6]]),
        "Scaling transformation"
    )

    # 3.2: Zoom 2x (2D)
    tasks['zoom_2x_2d'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2], [3, 4]]), Grid([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])),
                (Grid([[5]]), Grid([[5, 5], [5, 5]])),
            ],
            test_input=Grid([[7, 8]])
        ),
        Grid([[7, 7, 8, 8], [7, 7, 8, 8]]),
        "Scaling transformation"
    )

    # 3.3: Tile 2x2
    tasks['tile_2x2'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2], [3, 4]]), Grid([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])),
                (Grid([[5]]), Grid([[5, 5], [5, 5]])),
            ],
            test_input=Grid([[7, 8]])
        ),
        Grid([[7, 8, 7, 8], [7, 8, 7, 8]]),
        "Scaling transformation"
    )

    # =========================================================================
    # CATEGORY 4: OBJECT-BASED TRANSFORMATIONS
    # =========================================================================

    # 4.1: Keep largest object
    tasks['keep_largest_object'] = (
        ARCTask(
            train_pairs=[
                (
                    Grid([
                        [1, 1, 0, 2],
                        [1, 1, 0, 0],
                        [0, 0, 0, 3],
                    ]),
                    Grid([
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                    ])
                ),
                (
                    Grid([
                        [5, 0, 7, 7],
                        [0, 0, 7, 7],
                    ]),
                    Grid([
                        [0, 0, 7, 7],
                        [0, 0, 7, 7],
                    ])
                ),
            ],
            test_input=Grid([
                [1, 0, 2, 2],
                [0, 0, 2, 2],
                [3, 3, 0, 0],
            ])
        ),
        Grid([
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ]),
        "Object-based transformation"
    )

    # =========================================================================
    # CATEGORY 5: COMPOSITE TRANSFORMATIONS
    # =========================================================================

    # 5.1: Rotate then flip
    tasks['rotate_then_flip'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 2], [3, 4]]), Grid([[2, 4], [1, 3]])),
                (Grid([[5, 6], [7, 8]]), Grid([[6, 8], [5, 7]])),
            ],
            test_input=Grid([[1, 0], [2, 3]])
        ),
        Grid([[0, 3], [1, 2]]),
        "Composite transformation"
    )

    # =========================================================================
    # CATEGORY 6: PATTERN COMPLETION
    # =========================================================================

    # 6.1: Fill background
    tasks['fill_background_with_1'] = (
        ARCTask(
            train_pairs=[
                (Grid([[0, 2, 0], [0, 0, 3]]), Grid([[1, 2, 1], [1, 1, 3]])),
                (Grid([[0, 0, 5], [0, 0, 0]]), Grid([[1, 1, 5], [1, 1, 1]])),
            ],
            test_input=Grid([[0, 0, 7], [8, 0, 0]])
        ),
        Grid([[1, 1, 7], [8, 1, 1]]),
        "Pattern completion"
    )

    # =========================================================================
    # CATEGORY 7: SYMMETRY-BASED
    # =========================================================================

    # 7.1: Mirror to create symmetry
    tasks['mirror_horizontal_symmetry'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 0]]), Grid([[1, 1]])),
                (Grid([[2, 0]]), Grid([[2, 2]])),
            ],
            test_input=Grid([[3, 0]])
        ),
        Grid([[3, 3]]),
        "Symmetry-based transformation"
    )

    # =========================================================================
    # CATEGORY 8: POSITIONAL/SPATIAL RULES
    # =========================================================================

    # 8.1: Move objects right
    tasks['shift_right'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 0, 0], [2, 0, 0]]), Grid([[0, 1, 0], [0, 2, 0]])),
                (Grid([[3, 0, 0]]), Grid([[0, 3, 0]])),
            ],
            test_input=Grid([[5, 0, 0], [6, 0, 0]])
        ),
        Grid([[0, 5, 0], [0, 6, 0]]),
        "Spatial transformation"
    )

    # =========================================================================
    # CATEGORY 9: SIZE CHANGES
    # =========================================================================

    # 9.1: Crop to bounding box
    tasks['crop_to_content'] = (
        ARCTask(
            train_pairs=[
                (Grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), Grid([[1]])),
                (Grid([[0, 2, 2, 0], [0, 2, 2, 0]]), Grid([[2, 2], [2, 2]])),
            ],
            test_input=Grid([[0, 0, 0, 0], [0, 3, 3, 0], [0, 3, 3, 0], [0, 0, 0, 0]])
        ),
        Grid([[3, 3], [3, 3]]),
        "Size transformation"
    )

    # =========================================================================
    # CATEGORY 10: COMPLEX PATTERNS (LIKELY TO FAIL)
    # =========================================================================

    # 10.1: Count-based transformation
    tasks['repeat_by_count'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1]]), Grid([[1]])),
                (Grid([[1, 1]]), Grid([[1, 1], [1, 1]])),
                (Grid([[1, 1, 1]]), Grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
            ],
            test_input=Grid([[1, 1, 1, 1]])
        ),
        Grid([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        "Complex pattern (count-based)"
    )

    # 10.2: Conditional color change based on position
    tasks['color_by_position'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 1], [1, 1]]), Grid([[1, 2], [1, 2]])),
                (Grid([[3, 3], [3, 3]]), Grid([[3, 4], [3, 4]])),
            ],
            test_input=Grid([[5, 5], [5, 5]])
        ),
        Grid([[5, 6], [5, 6]]),
        "Complex pattern (positional)"
    )

    # 10.3: Gravity/physics-based
    tasks['gravity_down'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 0], [0, 0], [0, 2]]), Grid([[0, 0], [1, 0], [0, 2]])),
                (Grid([[3, 0], [0, 0]]), Grid([[0, 0], [3, 0]])),
            ],
            test_input=Grid([[5, 0], [0, 0], [0, 0]])
        ),
        Grid([[0, 0], [0, 0], [5, 0]]),
        "Complex pattern (physics)"
    )

    # 10.4: Path drawing
    tasks['draw_path'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 0, 0], [0, 0, 2]]), Grid([[1, 3, 3], [3, 3, 2]])),
                (Grid([[4, 0], [0, 5]]), Grid([[4, 3], [3, 5]])),
            ],
            test_input=Grid([[6, 0, 0], [0, 0, 7]])
        ),
        Grid([[6, 3, 3], [3, 3, 7]]),
        "Complex pattern (path)"
    )

    # 10.5: Relational transformation
    tasks['connect_same_color'] = (
        ARCTask(
            train_pairs=[
                (Grid([[1, 0, 1], [0, 0, 0]]), Grid([[1, 1, 1], [0, 0, 0]])),
                (Grid([[2, 0], [0, 2]]), Grid([[2, 2], [2, 2]])),
            ],
            test_input=Grid([[3, 0, 0], [0, 0, 3]])
        ),
        Grid([[3, 3, 3], [3, 3, 3]]),
        "Complex pattern (relational)"
    )

    return tasks


def run_comprehensive_test(verbose: bool = True):
    """Run comprehensive testing and analyze results"""

    print("="*80)
    print("COMPREHENSIVE ARC-AGI SOLVER TESTING")
    print("="*80)
    print("\nTesting on diverse task types to identify strengths and weaknesses\n")

    # Create test suite
    tasks = create_diverse_test_suite()

    # Initialize solver
    solver = ARCActiveInferenceSolver(workspace_capacity=20)

    # Results tracking
    results = {
        'successes': [],
        'partial_successes': [],
        'failures': [],
        'by_category': {}
    }

    # Test each task
    for task_name, (task, expected_output, task_type) in tasks.items():
        if verbose:
            print("="*80)
            print(f"TASK: {task_name}")
            print(f"Type: {task_type}")
            print("="*80)

        # Add expected output for evaluation
        task.test_output = expected_output

        # Solve
        try:
            predictions = solver.solve(task, verbose=False)

            # Evaluate
            eval_result = ARCEvaluator.evaluate_task(solver, task, verbose=False)

            # Analyze result
            exact_match_1 = np.array_equal(predictions[0].data, expected_output.data)
            exact_match_2 = np.array_equal(predictions[1].data, expected_output.data)

            pixel_acc_1 = ARCEvaluator.evaluate_prediction(predictions[0], expected_output)['pixel_accuracy']
            pixel_acc_2 = ARCEvaluator.evaluate_prediction(predictions[1], expected_output)['pixel_accuracy']

            best_accuracy = max(pixel_acc_1, pixel_acc_2)

            # Categorize result
            if exact_match_1 or exact_match_2:
                result_category = 'SUCCESS'
                results['successes'].append(task_name)
                symbol = "✓"
            elif best_accuracy >= 0.7:
                result_category = 'PARTIAL'
                results['partial_successes'].append(task_name)
                symbol = "~"
            else:
                result_category = 'FAILURE'
                results['failures'].append(task_name)
                symbol = "✗"

            # Track by category
            if task_type not in results['by_category']:
                results['by_category'][task_type] = {'success': 0, 'partial': 0, 'failure': 0}

            if result_category == 'SUCCESS':
                results['by_category'][task_type]['success'] += 1
            elif result_category == 'PARTIAL':
                results['by_category'][task_type]['partial'] += 1
            else:
                results['by_category'][task_type]['failure'] += 1

            if verbose:
                print(f"\n{symbol} Result: {result_category}")
                print(f"Best Accuracy: {best_accuracy:.3f}")
                print(f"Prediction 1 Accuracy: {pixel_acc_1:.3f}")
                print(f"Prediction 2 Accuracy: {pixel_acc_2:.3f}")

                if not (exact_match_1 or exact_match_2):
                    print("\nTest Input:")
                    visualize_grid(task.test_input, "")
                    print("\nExpected Output:")
                    visualize_grid(expected_output, "")
                    print("\nPrediction 1:")
                    visualize_grid(predictions[0], "")
                    print("\nPrediction 2:")
                    visualize_grid(predictions[1], "")
            else:
                print(f"{symbol} {task_name:30s} ({task_type:30s}) - Acc: {best_accuracy:.3f}")

        except Exception as e:
            print(f"✗ {task_name:30s} - ERROR: {e}")
            results['failures'].append(task_name)
            if task_type not in results['by_category']:
                results['by_category'][task_type] = {'success': 0, 'partial': 0, 'failure': 0}
            results['by_category'][task_type]['failure'] += 1

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total = len(tasks)
    successes = len(results['successes'])
    partials = len(results['partial_successes'])
    failures = len(results['failures'])

    print(f"\nTotal Tasks: {total}")
    print(f"Successes:   {successes} ({successes/total*100:.1f}%)")
    print(f"Partial:     {partials} ({partials/total*100:.1f}%)")
    print(f"Failures:    {failures} ({failures/total*100:.1f}%)")

    print("\n" + "="*80)
    print("RESULTS BY CATEGORY")
    print("="*80)

    for category, stats in sorted(results['by_category'].items()):
        total_cat = stats['success'] + stats['partial'] + stats['failure']
        print(f"\n{category}:")
        print(f"  Success: {stats['success']}/{total_cat}")
        print(f"  Partial: {stats['partial']}/{total_cat}")
        print(f"  Failure: {stats['failure']}/{total_cat}")

    print("\n" + "="*80)
    print("DETAILED FAILURE ANALYSIS")
    print("="*80)

    if results['failures']:
        print("\nFailed Tasks:")
        for task_name in results['failures']:
            task, expected, task_type = tasks[task_name]
            print(f"  - {task_name} ({task_type})")
    else:
        print("\nNo complete failures!")

    if results['partial_successes']:
        print("\nPartial Successes:")
        for task_name in results['partial_successes']:
            task, expected, task_type = tasks[task_name]
            print(f"  - {task_name} ({task_type})")

    return results


def analyze_failure_patterns(results: dict, tasks: dict):
    """Analyze patterns in failures"""

    print("\n" + "="*80)
    print("FAILURE PATTERN ANALYSIS")
    print("="*80)

    failure_patterns = {
        'Lacks primitive': [],
        'Complex composition': [],
        'Size mismatch': [],
        'Relational reasoning': [],
        'Count/arithmetic': [],
        'Other': []
    }

    # Analyze each failure
    for task_name in results['failures']:
        task, expected, task_type = tasks[task_name]

        # Pattern detection
        if 'count' in task_name or 'repeat' in task_name:
            failure_patterns['Count/arithmetic'].append(task_name)
        elif 'connect' in task_name or 'path' in task_name or 'relational' in task_name:
            failure_patterns['Relational reasoning'].append(task_name)
        elif 'gravity' in task_name or 'physics' in task_name:
            failure_patterns['Lacks primitive'].append(task_name)
        elif expected.shape != task.test_input.shape:
            failure_patterns['Size mismatch'].append(task_name)
        elif 'position' in task_name:
            failure_patterns['Complex composition'].append(task_name)
        else:
            failure_patterns['Other'].append(task_name)

    print("\nFailure Patterns:")
    for pattern, task_names in failure_patterns.items():
        if task_names:
            print(f"\n{pattern}:")
            for name in task_names:
                print(f"  - {name}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\nBased on failure analysis:")

    if failure_patterns['Lacks primitive']:
        print("\n1. Missing DSL Primitives:")
        print("   - Add physics-based primitives (gravity, movement)")
        print("   - Add path-drawing primitives")
        print("   - Add connectivity primitives")

    if failure_patterns['Complex composition']:
        print("\n2. Complex Composition Issues:")
        print("   - Implement multi-level composition search")
        print("   - Add learned program templates")
        print("   - Increase search depth")

    if failure_patterns['Size mismatch']:
        print("\n3. Size Transformation Issues:")
        print("   - Better handling of output size prediction")
        print("   - Add adaptive sizing primitives")
        print("   - Infer size from pattern analysis")

    if failure_patterns['Relational reasoning']:
        print("\n4. Relational Reasoning Gaps:")
        print("   - Add graph-based reasoning")
        print("   - Implement object relationship detection")
        print("   - Add connectivity analysis")

    if failure_patterns['Count/arithmetic']:
        print("\n5. Arithmetic/Counting Limitations:")
        print("   - Add counting primitives")
        print("   - Implement arithmetic operations")
        print("   - Add repeat-by-count transformations")


if __name__ == "__main__":
    import sys

    verbose = '-v' in sys.argv or '--verbose' in sys.argv

    # Run comprehensive test
    results = run_comprehensive_test(verbose=verbose)

    # Analyze failure patterns
    tasks = create_diverse_test_suite()
    analyze_failure_patterns(results, tasks)

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
