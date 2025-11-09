"""
Comprehensive tests for repair loops and landscape analytics.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.arc_loader import ARCTask
from solver_enhanced import EnhancedARCGraphPendulumSolver


def create_placement_error_task() -> ARCTask:
    """
    Create a task where output is input shifted by (1, 1).
    """
    train_examples = []

    # Example 1
    input1 = np.array([
        [1, 2, 3, 0],
        [4, 5, 6, 0],
        [7, 8, 9, 0],
        [0, 0, 0, 0]
    ])
    # Shift down-right by (1, 1)
    output1 = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 4, 5, 6],
        [0, 7, 8, 9]
    ])
    train_examples.append((input1, output1))

    # Example 2
    input2 = np.array([
        [5, 6, 0, 0],
        [7, 8, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    output2 = np.array([
        [0, 0, 0, 0],
        [0, 5, 6, 0],
        [0, 7, 8, 0],
        [0, 0, 0, 0]
    ])
    train_examples.append((input2, output2))

    # Test
    test_input = np.array([
        [2, 3, 4, 0],
        [5, 6, 7, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    test_output = np.array([
        [0, 0, 0, 0],
        [0, 2, 3, 4],
        [0, 5, 6, 7],
        [0, 0, 0, 0]
    ])

    return ARCTask(
        task_id="test_placement_error",
        train=train_examples,
        test=[(test_input, test_output)]
    )


def create_color_remap_task() -> ARCTask:
    """
    Create a task where colors are remapped (1->2, 2->3, etc).
    """
    train_examples = []

    color_map = {1: 2, 2: 3, 3: 4, 4: 5}

    # Example 1
    input1 = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 1]
    ])
    output1 = input1.copy()
    for old, new in color_map.items():
        output1[input1 == old] = new
    train_examples.append((input1, output1))

    # Example 2
    input2 = np.array([
        [1, 1, 2],
        [3, 3, 4],
        [2, 2, 1]
    ])
    output2 = input2.copy()
    for old, new in color_map.items():
        output2[input2 == old] = new
    train_examples.append((input2, output2))

    # Test
    test_input = np.array([
        [4, 3, 2],
        [1, 2, 3]
    ])
    test_output = test_input.copy()
    for old, new in color_map.items():
        test_output[test_input == old] = new

    return ARCTask(
        task_id="test_color_remap",
        train=train_examples,
        test=[(test_input, test_output)]
    )


def create_scale_task() -> ARCTask:
    """
    Create a task where output is 2x scaled version of input.
    """
    train_examples = []

    # Example 1
    input1 = np.array([
        [1, 2],
        [3, 4]
    ])
    output1 = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ])
    train_examples.append((input1, output1))

    # Example 2
    input2 = np.array([
        [5, 6],
        [7, 8]
    ])
    output2 = np.array([
        [5, 5, 6, 6],
        [5, 5, 6, 6],
        [7, 7, 8, 8],
        [7, 7, 8, 8]
    ])
    train_examples.append((input2, output2))

    # Test
    test_input = np.array([
        [2, 3],
        [4, 5]
    ])
    test_output = np.array([
        [2, 2, 3, 3],
        [2, 2, 3, 3],
        [4, 4, 5, 5],
        [4, 4, 5, 5]
    ])

    return ARCTask(
        task_id="test_scale_2x",
        train=train_examples,
        test=[(test_input, test_output)]
    )


def test_repair_loops():
    """Test repair loop functionality."""
    print("="*60)
    print("TEST 1: Repair Loops")
    print("="*60)

    solver = EnhancedARCGraphPendulumSolver(
        beam_width=3,
        use_repair_loops=True,
        use_landscape_analytics=False
    )

    tests = [
        ("Placement Error", create_placement_error_task()),
        ("Color Remap", create_color_remap_task()),
        ("Scale 2x", create_scale_task()),
    ]

    results = []

    for name, task in tests:
        print(f"\n--- Testing {name} ---")
        result = solver.evaluate_on_task(task, verbose=True)
        results.append((name, result))

    # Summary
    print("\n" + "="*60)
    print("REPAIR LOOPS TEST SUMMARY")
    print("="*60)

    for name, result in results:
        status = "✓ PASSED" if result['solved'] else "✗ FAILED"
        print(f"{name}: {status} (score: {result['avg_score']:.3f})")

    passed = sum(1 for _, r in results if r['avg_score'] >= 0.8)
    print(f"\nOverall: {passed}/{len(results)} tests passed (>0.80 IoU)")

    return results


def test_landscape_analytics():
    """Test landscape analytics functionality."""
    print("\n" + "="*60)
    print("TEST 2: Landscape Analytics")
    print("="*60)

    solver = EnhancedARCGraphPendulumSolver(
        beam_width=3,
        use_repair_loops=False,
        use_landscape_analytics=True
    )

    # Create several synthetic tasks to build landscape
    tasks = [
        create_placement_error_task(),
        create_color_remap_task(),
        create_scale_task(),
    ]

    # Solve multiple tasks to build trajectory landscape
    print(f"\nSolving {len(tasks)} tasks to build landscape...")
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}/{len(tasks)}: {task.task_id}")
        solver.solve_task(task, verbose=False)

    # Analyze landscape
    print("\nAnalyzing trajectory landscape...")
    landscape_results = solver.analyze_landscape(verbose=True)

    if landscape_results:
        print("\n✓ Landscape analysis completed successfully")

        # Try visualization
        try:
            print("\nGenerating visualization...")
            solver.visualize_landscape("test_landscape.png")
            print("✓ Visualization saved to test_landscape.png")
        except Exception as e:
            print(f"⚠ Visualization failed (matplotlib may not be available): {e}")

        # Save analysis
        try:
            solver.save_landscape_analysis("test_landscape_analysis.json")
            print("✓ Analysis saved to test_landscape_analysis.json")
        except Exception as e:
            print(f"⚠ Saving analysis failed: {e}")

        return True
    else:
        print("✗ Landscape analysis failed")
        return False


def test_integrated_system():
    """Test integrated system with both repair loops and landscape analytics."""
    print("\n" + "="*60)
    print("TEST 3: Integrated System (Repairs + Landscape)")
    print("="*60)

    solver = EnhancedARCGraphPendulumSolver(
        beam_width=3,
        use_repair_loops=True,
        use_landscape_analytics=True
    )

    # Solve multiple tasks
    tasks = [
        create_placement_error_task(),
        create_color_remap_task(),
        create_scale_task(),
    ]

    results = []
    print(f"\nSolving {len(tasks)} tasks with full system...")

    for task in tasks:
        print(f"\n--- Task: {task.task_id} ---")
        result = solver.evaluate_on_task(task, verbose=True)
        results.append(result)

    # Landscape analysis
    print("\n" + "-"*60)
    landscape_results = solver.analyze_landscape(verbose=True)

    # Summary
    print("\n" + "="*60)
    print("INTEGRATED SYSTEM TEST SUMMARY")
    print("="*60)

    solved = sum(1 for r in results if r['solved'])
    avg_score = np.mean([r['avg_score'] for r in results])

    print(f"Tasks solved: {solved}/{len(results)}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Landscape analysis: {'✓ SUCCESS' if landscape_results else '✗ FAILED'}")

    for result in results:
        status = "✓ SOLVED" if result['solved'] else "✗ FAILED"
        print(f"  {result['task_id']}: {status} (score: {result['avg_score']:.3f})")

    return results, landscape_results


def main():
    """Run all tests."""
    print("="*60)
    print("COMPREHENSIVE ENHANCEMENT TESTS")
    print("="*60)
    print()

    all_passed = True

    # Test 1: Repair Loops
    try:
        repair_results = test_repair_loops()
        repair_passed = sum(1 for _, r in repair_results if r['avg_score'] >= 0.8)
        if repair_passed < 2:  # At least 2/3 should pass
            all_passed = False
    except Exception as e:
        print(f"\n✗ Repair loops test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: Landscape Analytics
    try:
        landscape_passed = test_landscape_analytics()
        if not landscape_passed:
            all_passed = False
    except Exception as e:
        print(f"\n✗ Landscape analytics test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: Integrated System
    try:
        integrated_results, landscape = test_integrated_system()
        integrated_passed = (
            sum(1 for r in integrated_results if r['avg_score'] >= 0.8) >= 2
            and landscape is not None
        )
        if not integrated_passed:
            all_passed = False
    except Exception as e:
        print(f"\n✗ Integrated system test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Final Summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nEnhancements are working correctly:")
        print("  ✓ Repair loops functional")
        print("  ✓ Landscape analytics functional")
        print("  ✓ Integrated system functional")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the errors above")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
