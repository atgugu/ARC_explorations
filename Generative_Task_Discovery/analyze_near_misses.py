#!/usr/bin/env python3
"""
Analyze near-miss tasks (70-95% accuracy) to identify common failure patterns.
Goal: Understand what small refinements could convert near-misses to successes.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Import solvers
from inferred_solver import InferredCompositionalSolver

def load_arc_task(task_path: str) -> Dict:
    """Load ARC task from JSON file"""
    with open(task_path, 'r') as f:
        data = json.load(f)
    return {
        'train': data['train'],
        'test': data['test']
    }

def load_near_misses(report_path: str, min_accuracy: float = 0.70, max_accuracy: float = 0.95, top_n: int = 20) -> List[Dict]:
    """Load tasks with accuracy between min and max"""
    with open(report_path, 'r') as f:
        data = json.load(f)

    near_misses = []
    for result in data['results']:
        acc = result['best_accuracy']
        if min_accuracy <= acc <= max_accuracy:
            near_misses.append(result)

    # Sort by accuracy (highest first)
    near_misses.sort(key=lambda x: x['best_accuracy'], reverse=True)

    return near_misses[:top_n]

def visualize_grid(grid: np.ndarray, label: str = "") -> str:
    """Convert grid to visual string representation"""
    if len(grid.shape) != 2:
        return f"{label}: Invalid grid shape {grid.shape}"

    # Color mapping for visualization
    color_chars = {
        0: '·',  # background
        1: '█',  # blue
        2: '▓',  # red
        3: '▒',  # green
        4: '░',  # yellow
        5: '▄',  # gray
        6: '▀',  # magenta
        7: '▌',  # orange
        8: '▐',  # cyan
        9: '▆',  # brown
    }

    lines = []
    if label:
        lines.append(f"\n{label}:")
    lines.append(f"  Shape: {grid.shape[0]}×{grid.shape[1]}")

    for row in grid:
        line = "  " + "".join(color_chars.get(int(cell), '?') for cell in row)
        lines.append(line)

    return "\n".join(lines)

def compare_grids(predicted: np.ndarray, expected: np.ndarray) -> Dict:
    """Compare two grids and identify differences"""
    if predicted.shape != expected.shape:
        return {
            'shape_match': False,
            'predicted_shape': predicted.shape,
            'expected_shape': expected.shape,
            'accuracy': 0.0,
            'differences': []
        }

    # Calculate accuracy
    total_cells = predicted.size
    matching_cells = np.sum(predicted == expected)
    accuracy = matching_cells / total_cells

    # Find differences
    differences = []
    diff_mask = predicted != expected
    diff_positions = np.argwhere(diff_mask)

    for pos in diff_positions[:10]:  # Limit to first 10 differences
        i, j = pos
        differences.append({
            'position': (int(i), int(j)),
            'predicted': int(predicted[i, j]),
            'expected': int(expected[i, j])
        })

    return {
        'shape_match': True,
        'accuracy': accuracy,
        'total_differences': int(np.sum(diff_mask)),
        'differences': differences,
        'diff_positions': diff_positions
    }

def analyze_difference_patterns(task_id: str, task: Dict, prediction: np.ndarray, expected: np.ndarray) -> Dict:
    """Analyze what type of error occurred"""
    comparison = compare_grids(prediction, expected)

    if not comparison['shape_match']:
        return {
            'error_type': 'shape_mismatch',
            'description': f"Shape mismatch: predicted {comparison['predicted_shape']} vs expected {comparison['expected_shape']}",
            'comparison': comparison
        }

    # Analyze error patterns
    diff_count = comparison['total_differences']
    total_cells = prediction.size
    diff_ratio = diff_count / total_cells

    # Check if it's a color mapping issue
    pred_colors = set(prediction.flatten())
    exp_colors = set(expected.flatten())

    color_overlap = len(pred_colors & exp_colors) / max(len(pred_colors | exp_colors), 1)

    # Check if differences are spatially localized
    if diff_count > 0 and comparison.get('diff_positions') is not None:
        diff_positions = comparison['diff_positions']

        # Check if errors are in edges/corners
        h, w = prediction.shape
        edge_diffs = sum(1 for pos in diff_positions
                        if pos[0] == 0 or pos[0] == h-1 or pos[1] == 0 or pos[1] == w-1)
        edge_ratio = edge_diffs / diff_count if diff_count > 0 else 0

        # Check if errors form clusters
        if len(diff_positions) > 1:
            mean_pos = np.mean(diff_positions, axis=0)
            std_pos = np.std(diff_positions, axis=0)
            is_clustered = np.mean(std_pos) < min(h, w) / 4
        else:
            is_clustered = False
    else:
        edge_ratio = 0
        is_clustered = False

    # Categorize error type
    if diff_ratio < 0.05:
        error_type = 'minor_details'
        description = f"Very close ({comparison['accuracy']:.1%}), only {diff_count} cells wrong"
    elif diff_ratio < 0.10:
        error_type = 'small_details'
        description = f"Small details wrong ({comparison['accuracy']:.1%}), {diff_count} cells differ"
    elif color_overlap < 0.5:
        error_type = 'wrong_colors'
        description = f"Color mapping issue: predicted {pred_colors} vs expected {exp_colors}"
    elif edge_ratio > 0.7:
        error_type = 'edge_errors'
        description = f"Errors concentrated at edges ({edge_ratio:.0%} of errors)"
    elif is_clustered:
        error_type = 'localized_errors'
        description = f"Errors clustered in specific region"
    elif diff_ratio < 0.20:
        error_type = 'pattern_incomplete'
        description = f"Pattern mostly correct but incomplete ({comparison['accuracy']:.1%})"
    else:
        error_type = 'structural_mismatch'
        description = f"Significant structural differences ({comparison['accuracy']:.1%})"

    return {
        'error_type': error_type,
        'description': description,
        'accuracy': comparison['accuracy'],
        'diff_count': diff_count,
        'diff_ratio': diff_ratio,
        'color_overlap': color_overlap,
        'edge_ratio': edge_ratio,
        'is_clustered': is_clustered,
        'comparison': comparison
    }

def analyze_single_task(task_id: str, arc_data_dir: str = "arc_data") -> Dict:
    """Deep dive analysis of a single near-miss task"""
    print(f"\n{'='*70}")
    print(f"ANALYZING TASK: {task_id}")
    print(f"{'='*70}")

    # Load task
    task_path = Path(arc_data_dir) / "evaluation" / f"{task_id}.json"
    if not task_path.exists():
        print(f"ERROR: Task file not found: {task_path}")
        return None

    task = load_arc_task(str(task_path))

    # Print task info
    print(f"\nTask ID: {task_id}")
    print(f"Training examples: {len(task['train'])}")
    print(f"Test examples: {len(task['test'])}")

    # Show training examples
    print("\n--- TRAINING EXAMPLES ---")
    for i, example in enumerate(task['train'][:2]):  # Show first 2
        inp = np.array(example['input'])
        out = np.array(example['output'])
        print(f"\nTraining {i+1}:")
        print(visualize_grid(inp, "Input"))
        print(visualize_grid(out, "Output"))

    # Solve task
    print("\n--- SOLVING TASK ---")
    solver = InferredCompositionalSolver(
        max_candidates=150,
        beam_width=20,
        active_inference_steps=3,
        max_depth=2,
        composition_beam_width=10
    )

    result = solver.solve(task)

    # Analyze test case
    test_input = np.array(task['test'][0]['input'])
    test_output = np.array(task['test'][0]['output'])

    print(f"\n--- TEST CASE ---")
    print(visualize_grid(test_input, "Test Input"))
    print(visualize_grid(test_output, "Expected Output"))

    # solver.solve() returns (pred1, pred2, metadata)
    if result and len(result) == 3:
        pred1, pred2, metadata = result
        # Use the first prediction
        if pred1 is not None and len(pred1.shape) == 2:
            prediction = pred1
            print(visualize_grid(prediction, "Predicted Output"))

            # Program used
            program = metadata.get('top_programs', [None])[0] if metadata else None
            if program:
                print(f"\nProgram used: {program}")

            # Analyze differences
            analysis = analyze_difference_patterns(task_id, task, prediction, test_output)

            print(f"\n--- ERROR ANALYSIS ---")
            print(f"Error type: {analysis['error_type']}")
            print(f"Description: {analysis['description']}")
            print(f"Accuracy: {analysis['accuracy']:.2%}")

            if analysis.get('comparison', {}).get('differences'):
                print(f"\nFirst few differences:")
                for diff in analysis['comparison']['differences'][:5]:
                    pos = diff['position']
                    pred_color = diff['predicted']
                    exp_color = diff['expected']
                    print(f"  Position {pos}: predicted {pred_color}, expected {exp_color}")

            return {
                'task_id': task_id,
                'task': task,
                'prediction': prediction,
                'expected': test_output,
                'analysis': analysis,
                'program': str(program) if program else 'unknown'
            }
        else:
            print("\nERROR: Solver failed to produce valid output")
            return {
                'task_id': task_id,
                'task': task,
                'analysis': {'error_type': 'solver_failed', 'description': 'No valid output produced'}
            }
    else:
        print(f"\nERROR: Solver returned unexpected result format: {type(result)}, length {len(result) if result else 0}")
        return {
            'task_id': task_id,
            'task': task,
            'analysis': {'error_type': 'solver_failed', 'description': 'Invalid result format'}
        }

def main():
    """Main analysis function"""
    print("="*70)
    print("NEAR-MISS ANALYSIS")
    print("="*70)
    print("\nGoal: Understand what prevents 70-95% accurate solutions from succeeding")

    # Load near-misses
    report_path = "real_arc_evaluation_report.json"
    near_misses = load_near_misses(report_path, min_accuracy=0.90, max_accuracy=0.99, top_n=10)

    print(f"\nFound {len(near_misses)} tasks with 90-99% accuracy")
    print("\nTop 10 near-misses:")
    for i, nm in enumerate(near_misses, 1):
        print(f"  {i:2d}. {nm['id']:12s} {nm['best_accuracy']:5.1%}  {nm['top_program']}")

    # Analyze each task
    results = []
    for i, nm in enumerate(near_misses[:5], 1):  # Analyze top 5
        print(f"\n\n{'#'*70}")
        print(f"# TASK {i}/5")
        print(f"{'#'*70}")

        result = analyze_single_task(nm['id'])
        if result:
            results.append(result)

    # Summarize findings
    print(f"\n\n{'='*70}")
    print("SUMMARY OF FINDINGS")
    print(f"{'='*70}")

    error_types = {}
    for result in results:
        error_type = result['analysis'].get('error_type', 'unknown')
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(result['task_id'])

    print("\nError type distribution:")
    for error_type, tasks in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {error_type:20s}: {len(tasks)} tasks - {', '.join(tasks)}")

    return results

if __name__ == "__main__":
    results = main()
