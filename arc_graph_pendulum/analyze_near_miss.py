"""
Analyze near-miss tasks to identify specific execution errors.
"""

import numpy as np
import json
from solver_v6 import ARCGraphPendulumSolverV6
from utils.arc_loader import ARCLoader
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_pixel_differences(predicted, actual):
    """Analyze where predictions differ from ground truth."""
    if predicted.shape != actual.shape:
        return {
            'type': 'shape_mismatch',
            'predicted_shape': predicted.shape,
            'actual_shape': actual.shape,
            'description': f"Shape mismatch: predicted {predicted.shape} vs actual {actual.shape}"
        }

    diff_mask = (predicted != actual)
    num_errors = np.sum(diff_mask)
    total_pixels = predicted.size

    # Find error locations
    error_positions = np.argwhere(diff_mask)

    # Analyze error patterns
    errors_by_color = {}
    for pos in error_positions:
        pred_color = predicted[tuple(pos)]
        actual_color = actual[tuple(pos)]
        key = f"{actual_color}→{pred_color}"
        if key not in errors_by_color:
            errors_by_color[key] = []
        errors_by_color[key].append(tuple(pos))

    # Check if errors are clustered
    if len(error_positions) > 0:
        error_rows = error_positions[:, 0]
        error_cols = error_positions[:, 1]

        is_edge = (
            np.any(error_rows == 0) or
            np.any(error_rows == predicted.shape[0] - 1) or
            np.any(error_cols == 0) or
            np.any(error_cols == predicted.shape[1] - 1)
        )
    else:
        is_edge = False

    return {
        'type': 'pixel_errors',
        'num_errors': int(num_errors),
        'total_pixels': int(total_pixels),
        'error_rate': float(num_errors / total_pixels),
        'errors_by_color': {k: len(v) for k, v in errors_by_color.items()},
        'is_edge_error': bool(is_edge),
        'error_positions': error_positions.tolist()[:10]  # First 10 for inspection
    }

def analyze_task(task_id, task, solver):
    """Analyze a single task in detail."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {task_id}")
    print(f"{'='*80}")

    # Run solver
    result = solver.evaluate_on_task(task, verbose=False)

    print(f"IoU Score: {result['avg_score']:.4f}")
    print(f"Num training examples: {len(task.train)}")
    print(f"Num test examples: {len(task.test)}")

    # Analyze each test example
    analyses = []

    for i, (test_input, test_output) in enumerate(task.test):
        print(f"\nTest example {i+1}:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {test_output.shape}")

        # Get prediction
        predictions = solver.solve_task(task, verbose=False)

        if predictions and len(predictions) > i:
            predicted = predictions[i]

            # Compare
            diff_analysis = analyze_pixel_differences(predicted, test_output)

            print(f"  Error analysis:")
            if diff_analysis['type'] == 'shape_mismatch':
                print(f"    ✗ {diff_analysis['description']}")
            else:
                print(f"    • Errors: {diff_analysis['num_errors']}/{diff_analysis['total_pixels']} pixels ({diff_analysis['error_rate']*100:.2f}%)")
                print(f"    • Edge errors: {diff_analysis['is_edge_error']}")
                if diff_analysis['errors_by_color']:
                    print(f"    • Color substitutions:")
                    for color_change, count in diff_analysis['errors_by_color'].items():
                        print(f"      - {color_change}: {count} pixels")

            analyses.append({
                'test_idx': i,
                'diff_analysis': diff_analysis,
                'predicted_shape': predicted.shape,
                'actual_shape': test_output.shape
            })
        else:
            print(f"  ✗ No prediction generated")
            analyses.append({
                'test_idx': i,
                'error': 'no_prediction'
            })

    return {
        'task_id': task_id,
        'iou': result['avg_score'],
        'analyses': analyses
    }

def main():
    print("="*80)
    print("NEAR-MISS TASK ANALYSIS - EXECUTION ERROR DIAGNOSIS")
    print("="*80)

    # Load evaluation results to find near-miss tasks
    with open('v6_evaluation_results.json', 'r') as f:
        eval_results = json.load(f)

    # Find tasks with IoU >= 0.95 but not solved
    near_miss_tasks = [
        r for r in eval_results['results']
        if r['avg_score'] >= 0.95 and not r['solved']
    ]

    near_miss_tasks.sort(key=lambda x: x['avg_score'], reverse=True)

    print(f"\nFound {len(near_miss_tasks)} near-miss tasks (IoU >= 0.95):")
    for r in near_miss_tasks:
        print(f"  {r['task_id']}: IoU {r['avg_score']:.4f}")

    # Load tasks
    loader = ARCLoader(cache_dir="./arc_data")

    # Create solver
    solver = ARCGraphPendulumSolverV6(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False
    )

    # Analyze each near-miss task
    all_analyses = []

    for task_result in near_miss_tasks:
        task_id = task_result['task_id']

        # Load task
        task_file = Path(f"./arc_data/evaluation/{task_id}.json")
        if not task_file.exists():
            print(f"\n✗ Task file not found: {task_id}")
            continue

        task = loader.load_task(str(task_file))

        # Analyze
        analysis = analyze_task(task_id, task, solver)
        all_analyses.append(analysis)

    # Summary of error patterns
    print(f"\n{'='*80}")
    print("ERROR PATTERN SUMMARY")
    print(f"{'='*80}")

    shape_mismatches = sum(1 for a in all_analyses
                          for ta in a['analyses']
                          if ta.get('diff_analysis', {}).get('type') == 'shape_mismatch')

    edge_errors = sum(1 for a in all_analyses
                     for ta in a['analyses']
                     if ta.get('diff_analysis', {}).get('is_edge_error', False))

    print(f"\nShape mismatches: {shape_mismatches}")
    print(f"Edge errors: {edge_errors}")

    # Collect all color substitution errors
    all_color_errors = {}
    for a in all_analyses:
        for ta in a['analyses']:
            if 'diff_analysis' in ta and 'errors_by_color' in ta['diff_analysis']:
                for color_change, count in ta['diff_analysis']['errors_by_color'].items():
                    if color_change not in all_color_errors:
                        all_color_errors[color_change] = 0
                    all_color_errors[color_change] += count

    if all_color_errors:
        print(f"\nMost common color substitution errors:")
        sorted_errors = sorted(all_color_errors.items(), key=lambda x: x[1], reverse=True)
        for color_change, count in sorted_errors[:10]:
            print(f"  {color_change}: {count} pixels")

    # Save detailed analysis
    output_file = "near_miss_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'num_tasks': len(all_analyses),
                'shape_mismatches': shape_mismatches,
                'edge_errors': edge_errors,
                'color_substitution_errors': all_color_errors
            },
            'analyses': all_analyses
        }, f, indent=2)

    print(f"\n✓ Detailed analysis saved to {output_file}")

if __name__ == "__main__":
    main()
