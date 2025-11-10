"""
Test V4 solver on all 46 diverse tasks.
"""

import numpy as np
import json
from solver_v4 import ARCGraphPendulumSolverV4
from utils.arc_loader import ARCLoader

def main():
    print("="*70)
    print("V4 SOLVER - COMPREHENSIVE TEST ON ALL 46 TASKS")
    print("="*70)

    # Load dataset
    loader = ARCLoader(cache_dir="./arc_data")
    tasks = loader.load_all_tasks("training")

    # Create V4 solver
    solver = ARCGraphPendulumSolverV4(
        beam_width=5,
        use_stability=True,
        use_landscape_analytics=False
    )

    results = []

    for i, (task_id, task) in enumerate(tasks.items(), 1):
        print(f"\n[{i}/46] Testing: {task_id}...", end=" ", flush=True)

        result = solver.evaluate_on_task(task, verbose=False)

        results.append({
            'task_id': task_id,
            'solved': result['solved'],
            'avg_score': result['avg_score']
        })

        status = "✓ SOLVED" if result['solved'] else f"{result['avg_score']:.3f}"
        print(status)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    solved_count = sum(1 for r in results if r['solved'])
    high_quality = sum(1 for r in results if r['avg_score'] >= 0.80)
    medium_quality = sum(1 for r in results if 0.50 <= r['avg_score'] < 0.80)
    failures = sum(1 for r in results if r['avg_score'] < 0.20)
    avg_iou = np.mean([r['avg_score'] for r in results])

    print(f"\nPerfect solves (≥0.99):  {solved_count}/46 ({solved_count/46*100:.1f}%)")
    print(f"High quality (≥0.80):    {high_quality}/46 ({high_quality/46*100:.1f}%)")
    print(f"Medium quality (0.50-0.79): {medium_quality}/46 ({medium_quality/46*100:.1f}%)")
    print(f"Failures (<0.20):        {failures}/46 ({failures/46*100:.1f}%)")
    print(f"Average IoU:             {avg_iou:.3f}")

    print("\nComparison to V3+:")
    print("  V3+ Perfect solves: 5/46 (10.9%), avg IoU: 0.611")
    print(f"  V4  Perfect solves: {solved_count}/46 ({solved_count/46*100:.1f}%), avg IoU: {avg_iou:.3f}")

    improvement = solved_count - 5
    print(f"\n  Improvement: {improvement:+d} tasks solved ({improvement/46*100:+.1f}%)")
    print(f"  Avg IoU improvement: {avg_iou - 0.611:+.3f}")

    # Save results
    with open('v4_comprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: v4_comprehensive_results.json")

    # Show solved tasks
    print("\n" + "="*70)
    print("SOLVED TASKS")
    print("="*70)
    solved_tasks = [r for r in results if r['solved']]
    for r in solved_tasks:
        print(f"  ✓ {r['task_id']}")

    # Show new solves (compared to V3+)
    v3plus_solved = ['007bbfb7', '25ff71a9', '1e0a9b12', '0d3d703e', '017c7c7b']
    new_solves = [r['task_id'] for r in results if r['solved'] and r['task_id'] not in v3plus_solved]

    if new_solves:
        print("\n" + "="*70)
        print("NEW SOLVES (compared to V3+)")
        print("="*70)
        for task_id in new_solves:
            print(f"  ★ {task_id}")


if __name__ == "__main__":
    main()
