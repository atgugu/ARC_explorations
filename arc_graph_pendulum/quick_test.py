"""
Quick test of the ARC Graph Pendulum Solver on a simple synthetic task.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.arc_loader import ARCTask
from solver import ARCGraphPendulumSolver


def create_simple_test_task() -> ARCTask:
    """
    Create a simple synthetic task for testing.
    The task: flip the grid horizontally.
    """
    # Training examples
    train_examples = []

    # Example 1
    input1 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    output1 = np.fliplr(input1)
    train_examples.append((input1, output1))

    # Example 2
    input2 = np.array([
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0]
    ])
    output2 = np.fliplr(input2)
    train_examples.append((input2, output2))

    # Test example
    test_input = np.array([
        [5, 6, 7],
        [2, 3, 4],
        [8, 9, 1]
    ])
    test_output = np.fliplr(test_input)
    test_examples = [(test_input, test_output)]

    return ARCTask(
        task_id="test_flip_horizontal",
        train=train_examples,
        test=test_examples
    )


def create_identity_task() -> ARCTask:
    """
    Create an identity task (output = input).
    """
    train_examples = []

    # Example 1
    grid1 = np.array([
        [1, 2],
        [3, 4]
    ])
    train_examples.append((grid1, grid1.copy()))

    # Example 2
    grid2 = np.array([
        [5, 6, 7],
        [8, 9, 1]
    ])
    train_examples.append((grid2, grid2.copy()))

    # Test
    test_grid = np.array([
        [2, 3],
        [4, 5]
    ])
    test_examples = [(test_grid, test_grid.copy())]

    return ARCTask(
        task_id="test_identity",
        train=train_examples,
        test=test_examples
    )


def main():
    """Run quick tests."""
    print("=== Quick Test of ARC Graph Pendulum Solver ===\n")

    # Create solver
    solver = ARCGraphPendulumSolver(beam_width=3, use_stability=True)

    # Test 1: Identity task
    print("\n" + "="*60)
    print("TEST 1: Identity Task")
    print("="*60)

    identity_task = create_identity_task()
    result1 = solver.evaluate_on_task(identity_task, verbose=True)

    # Test 2: Horizontal flip task
    print("\n" + "="*60)
    print("TEST 2: Horizontal Flip Task")
    print("="*60)

    flip_task = create_simple_test_task()
    result2 = solver.evaluate_on_task(flip_task, verbose=True)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    tests = [
        ("Identity", result1),
        ("Horizontal Flip", result2),
    ]

    for name, result in tests:
        status = "✓ PASSED" if result['solved'] else "✗ FAILED"
        print(f"{name}: {status} (score: {result['avg_score']:.3f})")

    # Overall
    passed = sum(1 for _, r in tests if r['solved'])
    print(f"\nOverall: {passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
