"""
Test V3 solver on the same 10 tasks used for V1 and V2 comparison.
"""

import numpy as np
import json
from typing import List, Dict, Any

from solver_v3 import ARCGraphPendulumSolverV3
from utils.arc_loader import ARCLoader


def test_v3_solver(num_tasks: int = 10):
    """Test V3 solver on first N tasks."""

    print("="*80)
    print("TESTING V3: EXAMPLE-DRIVEN RULE INFERENCE")
    print("="*80)
    print()

    # Load tasks
    loader = ARCLoader(cache_dir="./arc_data")
    print("Loading ARC tasks...")
    tasks = loader.load_all_tasks("training")

    if not tasks:
        print("ERROR: No tasks loaded")
        return

    print(f"Loaded {len(tasks)} total tasks")
    print(f"Testing on first {num_tasks} tasks\n")

    # Create V3 solver
    print("Initializing V3 solver...")
    solver = ARCGraphPendulumSolverV3(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False
    )
    print()

    # Test on first N tasks
    task_list = list(tasks.values())[:num_tasks]
    results = []

    for i, task in enumerate(task_list):
        print("="*80)
        print(f"TASK {i+1}/{num_tasks}: {task.task_id}")
        print("="*80)

        result = solver.evaluate_on_task(task, verbose=True)
        results.append(result)

        print()

    # Compute summary statistics
    print("\n" + "="*80)
    print("V3 RESULTS SUMMARY")
    print("="*80)

    solved_count = sum(1 for r in results if r['solved'])
    high_quality_count = sum(1 for r in results if r['avg_score'] >= 0.8)
    medium_quality_count = sum(1 for r in results if 0.5 <= r['avg_score'] < 0.8)
    low_quality_count = sum(1 for r in results if r['avg_score'] < 0.5)

    avg_iou = np.mean([r['avg_score'] for r in results])

    print(f"\nPerformance Metrics:")
    print(f"  Tasks tested: {len(results)}")
    print(f"  Perfect solves (100%): {solved_count} ({solved_count/len(results)*100:.1f}%)")
    print(f"  High quality (≥0.80): {high_quality_count} ({high_quality_count/len(results)*100:.1f}%)")
    print(f"  Medium quality (0.50-0.79): {medium_quality_count} ({medium_quality_count/len(results)*100:.1f}%)")
    print(f"  Low quality (<0.50): {low_quality_count} ({low_quality_count/len(results)*100:.1f}%)")
    print(f"  Average IoU: {avg_iou:.3f}")

    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'#':<4} {'Task ID':<13} {'Status':<10} {'Quality':<8} {'Avg IoU':<8}")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        status = "✓ SOLVED" if result['solved'] else "✗ FAILED"
        if result['avg_score'] >= 0.8:
            quality = "HIGH"
        elif result['avg_score'] >= 0.5:
            quality = "MEDIUM"
        else:
            quality = "LOW"

        print(f"{i:<4} {result['task_id']:<13} {status:<10} {quality:<8} {result['avg_score']:.3f}")

    print("-" * 80)

    # Save results to JSON (clean up non-serializable data)
    output_file = "v3_results.json"

    # Create clean results without functions or complex objects
    clean_results = []
    for r in results:
        clean_r = {
            'task_id': str(r['task_id']),
            'solved': bool(r['solved']),
            'avg_score': float(r['avg_score']),
            'scores': [float(s) for s in r['scores']]
        }
        clean_results.append(clean_r)

    results_data = {
        'version': 'v3',
        'approach': 'example-driven rule inference',
        'num_tasks': int(len(results)),
        'solved_count': int(solved_count),
        'high_quality_count': int(high_quality_count),
        'medium_quality_count': int(medium_quality_count),
        'low_quality_count': int(low_quality_count),
        'avg_iou': float(avg_iou),
        'results': clean_results
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    return results


def compare_versions():
    """Compare V1, V2, and V3 results."""

    print("\n" + "="*80)
    print("VERSION COMPARISON: V1 vs V2 vs V3")
    print("="*80)

    # Load results files
    try:
        with open('comprehensive_evaluation_results.json', 'r') as f:
            v1_data = json.load(f)
    except FileNotFoundError:
        print("WARNING: V1 results not found (comprehensive_evaluation_results.json)")
        v1_data = None

    try:
        with open('v2_results.json', 'r') as f:
            v2_data = json.load(f)
    except FileNotFoundError:
        print("WARNING: V2 results not found (v2_results.json)")
        v2_data = None

    try:
        with open('v3_results.json', 'r') as f:
            v3_data = json.load(f)
    except FileNotFoundError:
        print("ERROR: V3 results not found (v3_results.json)")
        return

    print("\nComparative Summary:")
    print("-" * 80)
    print(f"{'Metric':<30} {'V1':<15} {'V2':<15} {'V3':<15}")
    print("-" * 80)

    # Extract metrics
    if v1_data:
        # V1 has attempt-based results
        v1_solved = v1_data.get('tasks_solved_count', 0)
        v1_total = v1_data.get('total_tasks', 10)
        v1_solve_rate = v1_solved / v1_total if v1_total > 0 else 0

        # Get high quality count from task results
        v1_high_quality = 0
        for task_result in v1_data.get('task_results', []):
            # A task is high quality if either attempt achieved ≥0.8
            attempt1_iou = task_result.get('attempt1', {}).get('avg_iou', 0)
            attempt2_iou = task_result.get('attempt2', {}).get('avg_iou', 0)
            best_iou = max(attempt1_iou, attempt2_iou)
            if best_iou >= 0.8:
                v1_high_quality += 1

        v1_avg_iou = 0
        count = 0
        for task_result in v1_data.get('task_results', []):
            attempt1_iou = task_result.get('attempt1', {}).get('avg_iou', 0)
            attempt2_iou = task_result.get('attempt2', {}).get('avg_iou', 0)
            v1_avg_iou += max(attempt1_iou, attempt2_iou)
            count += 1
        v1_avg_iou = v1_avg_iou / count if count > 0 else 0

    else:
        v1_solve_rate = 0
        v1_high_quality = 0
        v1_avg_iou = 0
        v1_total = 0

    if v2_data:
        v2_solved = v2_data.get('solved_count', 0)
        v2_total = v2_data.get('num_tasks', 10)
        v2_solve_rate = v2_solved / v2_total if v2_total > 0 else 0
        v2_high_quality = v2_data.get('high_quality_count', 0)
        v2_avg_iou = v2_data.get('avg_iou', 0)
    else:
        v2_solve_rate = 0
        v2_high_quality = 0
        v2_avg_iou = 0
        v2_total = 0

    v3_solved = v3_data.get('solved_count', 0)
    v3_total = v3_data.get('num_tasks', 10)
    v3_solve_rate = v3_solved / v3_total if v3_total > 0 else 0
    v3_high_quality = v3_data.get('high_quality_count', 0)
    v3_avg_iou = v3_data.get('avg_iou', 0)

    # Print comparison
    print(f"{'Perfect Solves (100%)':<30} {v1_solved}/{v1_total} ({v1_solve_rate*100:.0f}%){'':<6} "
          f"{v2_solved}/{v2_total} ({v2_solve_rate*100:.0f}%){'':<6} "
          f"{v3_solved}/{v3_total} ({v3_solve_rate*100:.0f}%)")

    print(f"{'High Quality (≥0.80)':<30} {v1_high_quality}/{v1_total} ({v1_high_quality/v1_total*100 if v1_total > 0 else 0:.0f}%){'':<6} "
          f"{v2_high_quality}/{v2_total} ({v2_high_quality/v2_total*100 if v2_total > 0 else 0:.0f}%){'':<6} "
          f"{v3_high_quality}/{v3_total} ({v3_high_quality/v3_total*100:.0f}%)")

    print(f"{'Average IoU':<30} {v1_avg_iou:.3f}{'':<11} {v2_avg_iou:.3f}{'':<11} {v3_avg_iou:.3f}")

    print("-" * 80)

    # Compute improvements
    if v1_avg_iou > 0:
        v3_improvement = ((v3_avg_iou - v1_avg_iou) / v1_avg_iou) * 100
        print(f"\nV3 vs V1 Improvement:")
        print(f"  Perfect solves: {v3_solved - v1_solved:+d} ({(v3_solve_rate - v1_solve_rate)*100:+.1f}%)")
        print(f"  High quality: {v3_high_quality - v1_high_quality:+d} ({(v3_high_quality/v3_total - v1_high_quality/v1_total)*100:+.1f}%)")
        print(f"  Average IoU: {v3_avg_iou - v1_avg_iou:+.3f} ({v3_improvement:+.1f}% relative)")

    if v2_avg_iou > 0:
        v3_v2_improvement = ((v3_avg_iou - v2_avg_iou) / v2_avg_iou) * 100
        print(f"\nV3 vs V2 Improvement:")
        print(f"  Perfect solves: {v3_solved - v2_solved:+d} ({(v3_solve_rate - v2_solve_rate)*100:+.1f}%)")
        print(f"  High quality: {v3_high_quality - v2_high_quality:+d}")
        print(f"  Average IoU: {v3_avg_iou - v2_avg_iou:+.3f} ({v3_v2_improvement:+.1f}% relative)")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Test V3
    test_v3_solver(num_tasks=10)

    # Compare versions
    print("\n")
    compare_versions()
