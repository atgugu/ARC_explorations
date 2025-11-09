"""
Test DSL primitives on real ARC-AGI tasks

This script loads real ARC tasks and attempts to solve them using DSL primitives.
"""

import sys
import os
import json
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from dsl.core_primitives import *


def load_arc_task(filepath):
    """Load an ARC task from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def grid_from_list(grid_list):
    """Convert list of lists to numpy array"""
    return np.array(grid_list, dtype=int)


def grids_equal(grid1, grid2):
    """Check if two grids are equal"""
    if grid1.shape != grid2.shape:
        return False
    return np.array_equal(grid1, grid2)


def print_grid(grid, title=""):
    """Pretty print a grid"""
    if title:
        print(f"\n{title}:")
    print("-" * (grid.shape[1] * 2 + 1))
    for row in grid:
        print("|" + "".join(str(cell) if cell != 0 else "." for cell in row) + "|")
    print("-" * (grid.shape[1] * 2 + 1))


# ============================================================================
# TASK-SPECIFIC SOLVERS
# ============================================================================

def solve_rotation_task(train_pairs):
    """
    Attempt to solve tasks involving rotation.

    Strategy: Check if output is rotated version of input
    """
    # Analyze first training pair
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    # Try different rotations
    for angle in [90, 180, 270]:
        # Get all objects
        for color in range(1, 10):
            objects = select_by_color(input_grid, color)
            if not objects:
                continue

            # Rotate and check
            rotated = rotate(objects[0], angle)
            result = object_to_grid([rotated], color, input_grid.shape)

            if grids_equal(result, output_grid):
                return lambda inp: solve_with_rotation(inp, color, angle)

    return None


def solve_with_rotation(input_grid, color, angle):
    """Apply rotation solution"""
    objects = select_by_color(input_grid, color)
    if not objects:
        return input_grid

    rotated = rotate(objects[0], angle)
    return object_to_grid([rotated], color, input_grid.shape)


def solve_fill_holes_task(train_pairs):
    """
    Attempt to solve tasks involving filling holes.

    Strategy: Check if output has holes filled
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    # Try filling holes for each color
    for color in range(1, 10):
        objects = select_by_color(input_grid, color)
        if not objects:
            continue

        filled = [fill_holes(obj, input_grid.shape) for obj in objects]
        result = object_to_grid(filled, [color] * len(filled), input_grid.shape)

        if grids_equal(result, output_grid):
            return lambda inp: solve_with_fill_holes(inp, color)

    return None


def solve_with_fill_holes(input_grid, color):
    """Apply fill holes solution"""
    objects = select_by_color(input_grid, color)
    if not objects:
        return input_grid

    filled = [fill_holes(obj, input_grid.shape) for obj in objects]
    return object_to_grid(filled, [color] * len(filled), input_grid.shape)


def solve_color_by_size_task(train_pairs):
    """
    Attempt to solve tasks involving recoloring by size.

    Strategy: Check if objects are recolored by size rank
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    # Get all objects (try different colors as base)
    for base_color in range(1, 10):
        objects = select_by_color(input_grid, base_color)
        if len(objects) < 2:
            continue

        # Try recoloring by size
        result = recolor_by_rule(objects, "size_ascending", input_grid)

        if grids_equal(result, output_grid):
            return lambda inp: solve_with_color_by_size(inp, base_color)

    return None


def solve_with_color_by_size(input_grid, base_color):
    """Apply color by size solution"""
    objects = select_by_color(input_grid, base_color)
    if not objects:
        return input_grid

    return recolor_by_rule(objects, "size_ascending", input_grid)


def solve_gravity_task(train_pairs):
    """
    Attempt to solve tasks involving gravity.

    Strategy: Check if objects fall to bottom
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    for color in range(1, 10):
        objects = select_by_color(input_grid, color)
        if not objects:
            continue

        # Try gravity
        fallen = gravity(objects, Direction.DOWN, 'edge', input_grid.shape)
        result = object_to_grid(fallen, [color] * len(fallen), input_grid.shape)

        if grids_equal(result, output_grid):
            return lambda inp: solve_with_gravity(inp, color)

    return None


def solve_with_gravity(input_grid, color):
    """Apply gravity solution"""
    objects = select_by_color(input_grid, color)
    if not objects:
        return input_grid

    fallen = gravity(objects, Direction.DOWN, 'edge', input_grid.shape)
    return object_to_grid(fallen, [color] * len(fallen), input_grid.shape)


def solve_largest_object_task(train_pairs):
    """
    Attempt to solve tasks involving selecting largest object.

    Strategy: Check if output is just the largest object
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    for color in range(1, 10):
        objects = select_by_color(input_grid, color)
        if not objects:
            continue

        largest = select_largest(objects, k=1)
        result = object_to_grid(largest, color, input_grid.shape)

        if grids_equal(result, output_grid):
            return lambda inp: solve_with_largest(inp, color)

    return None


def solve_with_largest(input_grid, color):
    """Apply largest object solution"""
    objects = select_by_color(input_grid, color)
    if not objects:
        return input_grid

    largest = select_largest(objects, k=1)
    return object_to_grid(largest, color, input_grid.shape)


# ============================================================================
# MAIN SOLVER
# ============================================================================

def attempt_solve(task_data):
    """
    Attempt to solve an ARC task using DSL primitives.

    Returns:
        (success, solver_function) tuple
    """
    train_pairs = []
    for pair in task_data['train']:
        train_pairs.append({
            'input': grid_from_list(pair['input']),
            'output': grid_from_list(pair['output'])
        })

    # Try different solution strategies
    strategies = [
        ("Rotation", solve_rotation_task),
        ("Fill Holes", solve_fill_holes_task),
        ("Color by Size", solve_color_by_size_task),
        ("Gravity", solve_gravity_task),
        ("Largest Object", solve_largest_object_task),
    ]

    for strategy_name, strategy_fn in strategies:
        try:
            solver = strategy_fn(train_pairs)
            if solver is not None:
                # Validate on all training pairs
                all_correct = True
                for pair in train_pairs:
                    predicted = solver(pair['input'])
                    if not grids_equal(predicted, pair['output']):
                        all_correct = False
                        break

                if all_correct:
                    return True, solver, strategy_name
        except Exception as e:
            # Strategy failed, try next one
            continue

    return False, None, None


def test_on_arc_tasks(num_tasks=20):
    """Test solver on real ARC tasks"""
    print("=" * 70)
    print("TESTING DSL PRIMITIVES ON REAL ARC-AGI TASKS")
    print("=" * 70)

    # Get training tasks
    training_dir = "data/ARC-AGI/data/training"
    if not os.path.exists(training_dir):
        print(f"\n✗ Error: ARC data not found at {training_dir}")
        print("Please download ARC dataset first:")
        print("  cd data && git clone https://github.com/fchollet/ARC-AGI.git")
        return

    task_files = sorted(glob.glob(f"{training_dir}/*.json"))[:num_tasks]

    if not task_files:
        print(f"\n✗ No task files found in {training_dir}")
        return

    print(f"\nTesting on {len(task_files)} tasks...\n")

    solved = 0
    failed = 0
    results = []

    for i, task_file in enumerate(task_files):
        task_id = os.path.basename(task_file).replace('.json', '')
        print(f"\n[{i+1}/{len(task_files)}] Task: {task_id}")

        try:
            task_data = load_arc_task(task_file)

            success, solver, strategy = attempt_solve(task_data)

            if success:
                print(f"  ✓ SOLVED using {strategy} strategy")
                solved += 1
                results.append((task_id, "SOLVED", strategy))

                # Test on test input if available
                if 'test' in task_data and len(task_data['test']) > 0:
                    test_input = grid_from_list(task_data['test'][0]['input'])
                    prediction = solver(test_input)
                    print(f"  → Generated prediction for test input")

            else:
                print(f"  ✗ Not solved")
                failed += 1
                results.append((task_id, "FAILED", None))

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
            results.append((task_id, "ERROR", str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTested: {len(task_files)} tasks")
    print(f"Solved: {solved} ({solved/len(task_files)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(task_files)*100:.1f}%)")

    if solved > 0:
        print(f"\n✓ Successfully solved {solved} tasks!")
        print("\nSolved tasks:")
        for task_id, status, strategy in results:
            if status == "SOLVED":
                print(f"  • {task_id} - {strategy}")

    print("\nNote: This is a simple proof-of-concept solver.")
    print("With a full hypothesis proposer, many more tasks would be solved!")

    return solved, failed


if __name__ == "__main__":
    import sys

    # Default to 20 tasks
    num_tasks = 20
    if len(sys.argv) > 1:
        num_tasks = int(sys.argv[1])

    test_on_arc_tasks(num_tasks)
