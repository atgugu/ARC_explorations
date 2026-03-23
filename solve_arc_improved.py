"""
Improved ARC solver using DSL primitives with better pattern detection
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


def solve_tiling_task(train_pairs):
    """
    Detect if output is a tiled version of input.
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    ih, iw = input_grid.shape
    oh, ow = output_grid.shape

    # Check if output is a multiple of input
    if oh % ih == 0 and ow % iw == 0:
        rows_factor = oh // ih
        cols_factor = ow // iw

        # Try tiling
        tiled = np.tile(input_grid, (rows_factor, cols_factor))

        if grids_equal(tiled, output_grid):
            # Validate on all pairs
            all_match = True
            for pair in train_pairs:
                inp = pair['input']
                out = pair['output']
                test_tiled = np.tile(inp, (rows_factor, cols_factor))
                if not grids_equal(test_tiled, out):
                    all_match = False
                    break

            if all_match:
                return lambda inp: np.tile(inp, (rows_factor, cols_factor)), f"Tile {rows_factor}x{cols_factor}"

    return None, None


def solve_crop_task(train_pairs):
    """
    Detect if output is a cropped version of input.
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    ih, iw = input_grid.shape
    oh, ow = output_grid.shape

    # Try to find output within input
    for r in range(ih - oh + 1):
        for c in range(iw - ow + 1):
            cropped = input_grid[r:r+oh, c:c+ow]
            if grids_equal(cropped, output_grid):
                # Validate
                all_match = True
                for pair in train_pairs:
                    inp = pair['input']
                    out = pair['output']
                    test_crop = inp[r:r+oh, c:c+ow]
                    if not grids_equal(test_crop, out):
                        all_match = False
                        break

                if all_match:
                    return lambda inp: inp[r:r+oh, c:c+ow], f"Crop at ({r},{c})"

    return None, None


def solve_scale_task(train_pairs):
    """
    Detect if output is a scaled version of input.
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    ih, iw = input_grid.shape
    oh, ow = output_grid.shape

    # Check if it's exact scaling
    if oh % ih == 0 and ow % iw == 0:
        factor_h = oh // ih
        factor_w = ow // iw

        if factor_h == factor_w:
            factor = factor_h

            # Try scaling each object
            scaled_grid = np.zeros((oh, ow), dtype=int)

            for color in range(1, 10):
                objects = select_by_color(input_grid, color)
                if not objects:
                    continue

                for obj in objects:
                    scaled_obj = scale(obj, factor)
                    for r, c in scaled_obj:
                        if 0 <= r < oh and 0 <= c < ow:
                            scaled_grid[r, c] = color

            if grids_equal(scaled_grid, output_grid):
                # Validate
                all_match = True
                for pair in train_pairs:
                    inp = pair['input']
                    out = pair['output']
                    test_scaled = np.zeros((out.shape[0], out.shape[1]), dtype=int)

                    for color in range(1, 10):
                        objs = select_by_color(inp, color)
                        if not objs:
                            continue
                        for obj in objs:
                            sc_obj = scale(obj, factor)
                            for r, c in sc_obj:
                                if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                                    test_scaled[r, c] = color

                    if not grids_equal(test_scaled, out):
                        all_match = False
                        break

                if all_match:
                    return lambda inp: scale_grid(inp, factor), f"Scale {factor}x"

    return None, None


def scale_grid(grid, factor):
    """Scale entire grid"""
    h, w = grid.shape
    result = np.zeros((h * factor, w * factor), dtype=int)

    for color in range(1, 10):
        objects = select_by_color(grid, color)
        if not objects:
            continue

        for obj in objects:
            scaled_obj = scale(obj, factor)
            for r, c in scaled_obj:
                if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                    result[r, c] = color

    return result


def solve_transpose_task(train_pairs):
    """
    Detect if output is transposed input.
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    transposed = input_grid.T

    if grids_equal(transposed, output_grid):
        # Validate
        all_match = True
        for pair in train_pairs:
            if not grids_equal(pair['input'].T, pair['output']):
                all_match = False
                break

        if all_match:
            return lambda inp: inp.T, "Transpose"

    return None, None


def solve_flip_task(train_pairs):
    """
    Detect if output is flipped input.
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    # Try vertical flip
    flipped_v = np.flipud(input_grid)
    if grids_equal(flipped_v, output_grid):
        all_match = True
        for pair in train_pairs:
            if not grids_equal(np.flipud(pair['input']), pair['output']):
                all_match = False
                break

        if all_match:
            return lambda inp: np.flipud(inp), "Flip vertical"

    # Try horizontal flip
    flipped_h = np.fliplr(input_grid)
    if grids_equal(flipped_h, output_grid):
        all_match = True
        for pair in train_pairs:
            if not grids_equal(np.fliplr(pair['input']), pair['output']):
                all_match = False
                break

        if all_match:
            return lambda inp: np.fliplr(inp), "Flip horizontal"

    return None, None


def solve_color_swap_task(train_pairs):
    """
    Detect if output is color-swapped input.
    """
    input_grid = train_pairs[0]['input']
    output_grid = train_pairs[0]['output']

    # Try all pairs of colors
    input_colors = set(input_grid.flatten()) - {0}
    output_colors = set(output_grid.flatten()) - {0}

    if len(input_colors) == 2 and len(output_colors) == 2:
        colors_in = list(input_colors)
        colors_out = list(output_colors)

        if set(colors_in) == set(colors_out):
            # Try swapping
            swapped = swap_colors(input_grid, colors_in[0], colors_in[1])

            if grids_equal(swapped, output_grid):
                all_match = True
                for pair in train_pairs:
                    test_swap = swap_colors(pair['input'], colors_in[0], colors_in[1])
                    if not grids_equal(test_swap, pair['output']):
                        all_match = False
                        break

                if all_match:
                    c1, c2 = colors_in[0], colors_in[1]
                    return lambda inp: swap_colors(inp, c1, c2), f"Swap colors {c1}↔{c2}"

    return None, None


def attempt_solve(task_data):
    """
    Attempt to solve an ARC task.
    """
    train_pairs = []
    for pair in task_data['train']:
        train_pairs.append({
            'input': grid_from_list(pair['input']),
            'output': grid_from_list(pair['output'])
        })

    # Try different strategies
    strategies = [
        ("Tiling", solve_tiling_task),
        ("Cropping", solve_crop_task),
        ("Scaling", solve_scale_task),
        ("Transpose", solve_transpose_task),
        ("Flip", solve_flip_task),
        ("Color Swap", solve_color_swap_task),
    ]

    for strategy_name, strategy_fn in strategies:
        try:
            solver, description = strategy_fn(train_pairs)
            if solver is not None:
                return True, solver, description
        except Exception as e:
            continue

    return False, None, None


def test_on_arc_tasks(num_tasks=50):
    """Test solver on real ARC tasks"""
    print("=" * 70)
    print("IMPROVED ARC SOLVER - Testing on Real Tasks")
    print("=" * 70)

    training_dir = "data/ARC-AGI/data/training"
    task_files = sorted(glob.glob(f"{training_dir}/*.json"))[:num_tasks]

    print(f"\nTesting on {len(task_files)} tasks...\n")

    solved = 0
    results = []

    for i, task_file in enumerate(task_files):
        task_id = os.path.basename(task_file).replace('.json', '')

        try:
            task_data = load_arc_task(task_file)
            success, solver, description = attempt_solve(task_data)

            if success:
                print(f"[{i+1}/{len(task_files)}] {task_id}: ✓ SOLVED - {description}")
                solved += 1
                results.append((task_id, "SOLVED", description))
            else:
                print(f"[{i+1}/{len(task_files)}] {task_id}: ✗ Not solved")
                results.append((task_id, "FAILED", None))

        except Exception as e:
            print(f"[{i+1}/{len(task_files)}] {task_id}: ✗ Error - {e}")
            results.append((task_id, "ERROR", str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTested: {len(task_files)} tasks")
    print(f"Solved: {solved} ({solved/len(task_files)*100:.1f}%)")

    if solved > 0:
        print(f"\n✓ Successfully solved {solved} tasks!")
        print("\nSolved tasks:")
        for task_id, status, description in results:
            if status == "SOLVED":
                print(f"  • {task_id}: {description}")

    return solved


if __name__ == "__main__":
    import sys
    num_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    test_on_arc_tasks(num_tasks)
