"""
Deep Failure Analysis

Analyzes specific failure cases to identify what patterns the solver misses.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

sys.path.insert(0, '/home/user/ARC_explorations')

from arc_curiosity_solver.solver import ARCCuriositySolver


def visualize_comparison(input_grid, expected, pred1, pred2, title=""):
    """Visualize input, expected, and predictions side by side."""
    print(f"\n{title}")
    print("="*80)

    # Convert to strings for alignment
    def grid_to_str(grid):
        lines = []
        for row in grid:
            lines.append(" ".join(str(int(x)) for x in row))
        return lines

    input_lines = grid_to_str(input_grid)
    expected_lines = grid_to_str(expected)
    pred1_lines = grid_to_str(pred1) if pred1.shape == expected.shape else [f"Shape: {pred1.shape}"]
    pred2_lines = grid_to_str(pred2) if pred2.shape == expected.shape else [f"Shape: {pred2.shape}"]

    # Pad to same height
    max_height = max(len(input_lines), len(expected_lines), len(pred1_lines), len(pred2_lines))
    for lines in [input_lines, expected_lines, pred1_lines, pred2_lines]:
        while len(lines) < max_height:
            lines.append("")

    # Print side by side
    print(f"{'INPUT':<20} {'EXPECTED':<20} {'PRED1':<20} {'PRED2':<20}")
    print("-"*80)
    for i in range(max_height):
        print(f"{input_lines[i]:<20} {expected_lines[i]:<20} {pred1_lines[i]:<20} {pred2_lines[i]:<20}")


def analyze_transformation_pattern(train_pairs):
    """Analyze what kind of transformation is happening."""
    patterns = {
        'size_change': [],
        'color_mapping': {},
        'object_count_change': [],
        'grid_operations': []
    }

    for inp, out in train_pairs:
        # Size change
        if inp.shape != out.shape:
            ratio = (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
            patterns['size_change'].append(ratio)

        # Color mapping
        inp_colors = set(inp.flatten())
        out_colors = set(out.flatten())
        if inp_colors != out_colors:
            patterns['color_mapping'][frozenset(inp_colors)] = frozenset(out_colors)

        # Object count (rough estimate)
        from scipy.ndimage import label
        inp_objects = label(inp != 0)[1]
        out_objects = label(out != 0)[1]
        if inp_objects != out_objects:
            patterns['object_count_change'].append((inp_objects, out_objects))

    return patterns


def analyze_specific_failure(task_file: str, solver: ARCCuriositySolver):
    """Deep dive into a specific failure case."""
    with open(task_file, 'r') as f:
        data = json.load(f)

    task_id = Path(task_file).stem

    # Load train and test
    train_pairs = []
    for example in data['train']:
        inp = np.array(example['input'])
        out = np.array(example['output'])
        train_pairs.append((inp, out))

    test_input = np.array(data['test'][0]['input'])
    test_output = np.array(data['test'][0]['output'])

    print(f"\n{'='*80}")
    print(f"Deep Analysis: Task {task_id}")
    print(f"{'='*80}")

    # Analyze transformation patterns
    patterns = analyze_transformation_pattern(train_pairs)

    print(f"\n📊 Transformation Pattern Analysis:")
    if patterns['size_change']:
        avg_ratio = np.mean(patterns['size_change'], axis=0)
        print(f"   Size changes: {patterns['size_change']}")
        print(f"   Average ratio: ({avg_ratio[0]:.2f}x, {avg_ratio[1]:.2f}x)")
    else:
        print(f"   No size changes (shape preserved)")

    if patterns['color_mapping']:
        print(f"   Color mappings detected: {len(patterns['color_mapping'])}")

    if patterns['object_count_change']:
        print(f"   Object count changes: {patterns['object_count_change']}")

    # Show training examples
    print(f"\n📚 Training Examples:")
    for i, (inp, out) in enumerate(train_pairs[:2]):  # Show first 2
        print(f"\nExample {i+1}:")
        print(f"  Input shape: {inp.shape}, Output shape: {out.shape}")
        visualize_comparison(inp, out, out, out, f"Training Example {i+1}")

    # Test solver
    print(f"\n🔍 Solver Predictions:")
    pred1, pred2 = solver.solve(train_pairs, test_input, verbose=False)

    acc1 = np.sum(pred1 == test_output) / test_output.size if pred1.shape == test_output.shape else 0.0
    acc2 = np.sum(pred2 == test_output) / test_output.size if pred2.shape == test_output.shape else 0.0

    print(f"  Prediction 1 accuracy: {acc1*100:.1f}%")
    print(f"  Prediction 2 accuracy: {acc2*100:.1f}%")

    visualize_comparison(test_input, test_output, pred1, pred2, "Test Case Comparison")

    # Analyze what went wrong
    print(f"\n❌ What Went Wrong:")

    if pred1.shape != test_output.shape:
        print(f"   • Wrong output shape: got {pred1.shape}, expected {test_output.shape}")
        if patterns['size_change']:
            print(f"     → Needs size transformation ({patterns['size_change'][0]})")
    else:
        # Pixel-level differences
        diff = pred1 != test_output
        n_errors = np.sum(diff)
        print(f"   • {n_errors} pixels wrong out of {test_output.size} ({n_errors/test_output.size*100:.1f}%)")

        # Where are the errors?
        error_positions = np.argwhere(diff)
        if len(error_positions) > 0:
            print(f"   • Error positions (first 5): {error_positions[:5].tolist()}")

            # What colors are wrong?
            wrong_predictions = pred1[diff]
            expected_colors = test_output[diff]
            print(f"   • Predicted colors: {set(wrong_predictions.flatten())}")
            print(f"   • Expected colors: {set(expected_colors.flatten())}")

    # Check if predictions are different
    if pred1.shape == pred2.shape and np.array_equal(pred1, pred2):
        print(f"   • ⚠️  Both predictions identical (no diversity)")
    else:
        print(f"   • ✓ Predictions are different (good diversity)")

    return {
        'task_id': task_id,
        'patterns': patterns,
        'accuracy': (acc1, acc2),
        'shapes': {
            'input': test_input.shape,
            'expected': test_output.shape,
            'pred1': pred1.shape,
            'pred2': pred2.shape
        }
    }


def main():
    """Run detailed failure analysis on interesting cases."""

    print("="*80)
    print("DETAILED FAILURE ANALYSIS")
    print("="*80)

    # Create solver
    solver = ARCCuriositySolver(
        workspace_capacity=7,
        learning_rate=0.1,
        exploration_bonus=1.0,
        n_hypotheses_to_explore=40
    )

    # Select interesting failure cases
    training_dir = Path("/home/user/ARC_explorations/ARC-AGI/data/training")

    # Cases to analyze in detail:
    # 1. Close but not perfect (high accuracy but not 100%)
    # 2. Wrong size (needs size transformation)
    # 3. Wrong transformation (low accuracy)

    interesting_cases = [
        "007bbfb7",  # Partial match (77% accuracy) - scaling task
        "025d127b",  # Close but not perfect (98% accuracy)
        "0b148d64",  # Wrong output shape (size change)
        "05269061",  # Wrong transformation (22% accuracy)
        "11852cab",  # Very close (97% accuracy)
    ]

    results = []
    for task_id in interesting_cases:
        task_file = training_dir / f"{task_id}.json"
        if task_file.exists():
            try:
                result = analyze_specific_failure(str(task_file), solver)
                results.append(result)
            except Exception as e:
                print(f"\nERROR analyzing {task_id}: {e}")
                import traceback
                traceback.print_exc()

    # Summary of findings
    print(f"\n{'='*80}")
    print("SUMMARY OF FINDINGS")
    print(f"{'='*80}")

    print(f"\n🔍 Pattern Analysis Across Failed Tasks:")

    size_change_tasks = sum(1 for r in results if r['patterns']['size_change'])
    color_change_tasks = sum(1 for r in results if r['patterns']['color_mapping'])
    object_change_tasks = sum(1 for r in results if r['patterns']['object_count_change'])

    print(f"   Tasks with size changes: {size_change_tasks}/{len(results)}")
    print(f"   Tasks with color changes: {color_change_tasks}/{len(results)}")
    print(f"   Tasks with object count changes: {object_change_tasks}/{len(results)}")

    print(f"\n💡 Key Insights:")
    print(f"   1. Solver gets shapes RIGHT in {sum(1 for r in results if r['shapes']['pred1'] == r['shapes']['expected'])}/{len(results)} cases")
    print(f"   2. When shapes are right, average accuracy is {np.mean([max(r['accuracy']) for r in results if r['shapes']['pred1'] == r['shapes']['expected']])*100:.1f}%")
    print(f"   3. This suggests the solver identifies APPROXIMATELY correct transformations")
    print(f"   4. Missing: Fine-grained details, exact patterns, object-level operations")


if __name__ == "__main__":
    main()
