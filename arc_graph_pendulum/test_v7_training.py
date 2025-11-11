"""
Test V7 solver on training set and compare with V6.
"""

import numpy as np
import json
from solver_v7 import ARCGraphPendulumSolverV7
from utils.arc_loader import ARCLoader
import time

def main():
    print("="*80)
    print("V7 SOLVER - TRAINING SET TEST (46 TASKS)")
    print("="*80)

    # Load training dataset
    loader = ARCLoader(cache_dir="./arc_data")
    tasks = loader.load_all_tasks("training")

    print(f"\nTesting on {len(tasks)} training tasks\n")

    # Create V7 solver
    solver = ARCGraphPendulumSolverV7(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False,
        enable_refinement=True
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
    print("V7 TRAINING RESULTS")
    print("="*80)

    solved_count = sum(1 for r in results if r['solved'])
    solve_rate = (solved_count / len(results)) * 100

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

    # Load V6 results for comparison
    try:
        with open('v6_comprehensive_results.json', 'r') as f:
            v6_results = json.load(f)

        v6_solved = v6_results['summary']['solved']
        v6_avg_iou = v6_results['summary']['avg_iou']

        print(f"\n{'='*80}")
        print("COMPARISON: V7 vs V6")
        print(f"{'='*80}")
        print(f"\nSolve rate:")
        print(f"  V6: {v6_solved}/46 ({v6_solved/46*100:.1f}%)")
        print(f"  V7: {solved_count}/46 ({solve_rate:.1f}%)")
        print(f"  Change: {solved_count - v6_solved:+d} ({(solve_rate - v6_solved/46*100):+.1f}%)")

        print(f"\nAverage IoU:")
        print(f"  V6: {v6_avg_iou:.3f}")
        print(f"  V7: {avg_iou:.3f}")
        print(f"  Change: {avg_iou - v6_avg_iou:+.3f}")

        # Find tasks with improved scores
        v6_results_dict = {r['task_id']: r['avg_score'] for r in v6_results['results']}
        v7_results_dict = {r['task_id']: r['avg_score'] for r in results}

        improved = []
        degraded = []

        for task_id in v7_results_dict:
            if task_id in v6_results_dict:
                v6_score = v6_results_dict[task_id]
                v7_score = v7_results_dict[task_id]
                diff = v7_score - v6_score

                if diff > 0.01:  # Improvement
                    improved.append((task_id, v6_score, v7_score, diff))
                elif diff < -0.01:  # Degradation
                    degraded.append((task_id, v6_score, v7_score, diff))

        improved.sort(key=lambda x: x[3], reverse=True)
        degraded.sort(key=lambda x: x[3])

        if improved:
            print(f"\n{'='*80}")
            print(f"IMPROVED TASKS ({len(improved)}):")
            print(f"{'='*80}")
            for task_id, v6_score, v7_score, diff in improved[:10]:
                v6_status = "SOLVED" if v6_score >= 0.99 else f"{v6_score:.3f}"
                v7_status = "SOLVED" if v7_score >= 0.99 else f"{v7_score:.3f}"
                print(f"  {task_id}: {v6_status} → {v7_status} ({diff:+.3f})")

        if degraded:
            print(f"\n{'='*80}")
            print(f"DEGRADED TASKS ({len(degraded)}):")
            print(f"{'='*80}")
            for task_id, v6_score, v7_score, diff in degraded[:10]:
                v6_status = "SOLVED" if v6_score >= 0.99 else f"{v6_score:.3f}"
                v7_status = "SOLVED" if v7_score >= 0.99 else f"{v7_score:.3f}"
                print(f"  {task_id}: {v6_status} → {v7_status} ({diff:+.3f})")

    except Exception as e:
        print(f"\n✗ Could not compare with V6: {e}")

    # Save results
    output_file = "v7_training_results.json"
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
