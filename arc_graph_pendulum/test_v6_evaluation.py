"""
Test V6 solver on ARC-AGI evaluation dataset.
"""

import numpy as np
import json
import time
from solver_v6 import ARCGraphPendulumSolverV6
from utils.arc_loader import ARCLoader

def main():
    print("="*80)
    print("V6 SOLVER - EVALUATION ON ARC-AGI EVALUATION DATASET")
    print("="*80)

    # Load evaluation dataset
    loader = ARCLoader(cache_dir="./arc_data")
    tasks = loader.load_all_tasks("evaluation")

    print(f"\nTesting on {len(tasks)} evaluation tasks\n")

    # Create V6 solver
    solver = ARCGraphPendulumSolverV6(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False
    )

    results = []
    start_time = time.time()

    for i, (task_id, task) in enumerate(sorted(tasks.items()), 1):
        print(f"[{i}/{len(tasks)}] {task_id}...", end=" ", flush=True)

        try:
            result = solver.evaluate_on_task(task, verbose=False)

            results.append({
                'task_id': task_id,
                'solved': bool(result['solved']),
                'avg_score': float(result['avg_score']),
                'test_examples': result.get('test_examples', 1)
            })

            status = "✓ SOLVED" if result['solved'] else f"IoU {result['avg_score']:.3f}"
            print(status)

        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            results.append({
                'task_id': task_id,
                'solved': False,
                'avg_score': 0.0,
                'error': str(e)
            })

    elapsed = time.time() - start_time

    # Calculate statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    solved_count = sum(1 for r in results if r['solved'])
    solve_rate = (solved_count / len(results)) * 100

    # IoU distribution
    scores = [r['avg_score'] for r in results]
    avg_iou = np.mean(scores)

    high_quality = sum(1 for s in scores if s >= 0.80)
    medium_quality = sum(1 for s in scores if 0.50 <= s < 0.80)
    low_quality = sum(1 for s in scores if 0.20 <= s < 0.50)
    failures = sum(1 for s in scores if s < 0.20)

    print(f"\nTotal tasks: {len(results)}")
    print(f"Perfect solves (IoU ≥ 0.99): {solved_count}/{len(results)} ({solve_rate:.1f}%)")
    print(f"Average IoU: {avg_iou:.3f}")
    print(f"\nQuality distribution:")
    print(f"  High (0.80-0.99):   {high_quality:3d} ({high_quality/len(results)*100:.1f}%)")
    print(f"  Medium (0.50-0.79): {medium_quality:3d} ({medium_quality/len(results)*100:.1f}%)")
    print(f"  Low (0.20-0.49):    {low_quality:3d} ({low_quality/len(results)*100:.1f}%)")
    print(f"  Failure (< 0.20):   {failures:3d} ({failures/len(results)*100:.1f}%)")
    print(f"\nTime elapsed: {elapsed:.1f}s ({elapsed/len(results):.1f}s per task)")

    # List solved tasks
    if solved_count > 0:
        print(f"\n{'='*80}")
        print(f"SOLVED TASKS ({solved_count}):")
        print(f"{'='*80}")
        solved_tasks = [r for r in results if r['solved']]
        for r in solved_tasks:
            print(f"  {r['task_id']}")

    # Save results
    output_file = "v6_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tasks': len(results),
                'solved': solved_count,
                'solve_rate': solve_rate,
                'avg_iou': avg_iou,
                'high_quality': high_quality,
                'medium_quality': medium_quality,
                'low_quality': low_quality,
                'failures': failures,
                'elapsed_seconds': elapsed
            },
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
