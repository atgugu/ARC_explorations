"""
Comprehensive test of V4 solver on all 13 previously failing tasks.
"""

import numpy as np
from solver_v4 import ARCGraphPendulumSolverV4
from utils.arc_loader import ARCLoader

# All 13 tasks that failed in comprehensive analysis (all shape-changing)
FAILING_TASK_IDS = [
    '10fcaaa3',  # 2x4 → 4x8 (upsampling)
    '0b148d64',  # 21x21 → 10x10
    '1fad071e',  # 9x9 → 1x5
    '1b2d62fb',  # 5x7 → 5x3
    '23b5c85d',  # 10x10 → 3x3
    '137eaa0f',  # 11x11 → 3x3
    '234bbc79',  # 3x9 → 3x7
    '1cf80156',  # 10x12 → 4x4
    '1f85a75f',  # 30x30 → 5x3
    '1c786137',  # 23x21 → 6x8
    '1190e5a7',  # 15x15 → 2x4
    '239be575',  # 5x5 → 1x1
    '2013d3e2',  # 10x10 → 3x3
]

def main():
    print("="*70)
    print("V4 COMPREHENSIVE TEST ON 13 FAILING TASKS")
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

    for task_id in FAILING_TASK_IDS:
        if task_id not in tasks:
            print(f"\nTask {task_id}: NOT FOUND")
            continue

        task = tasks[task_id]

        print(f"\n{'='*70}")
        print(f"Testing: {task_id}")
        print(f"{'='*70}")

        result = solver.evaluate_on_task(task, verbose=False)

        results.append({
            'task_id': task_id,
            'solved': result['solved'],
            'avg_score': result['avg_score']
        })

        status = "✓ SOLVED" if result['solved'] else f"  {result['avg_score']:.3f}"
        print(f"Result: {status}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    solved_count = sum(1 for r in results if r['solved'])
    total_count = len(results)
    avg_iou = np.mean([r['avg_score'] for r in results])

    print(f"\nSolved: {solved_count}/{total_count} ({solved_count/total_count*100:.1f}%)")
    print(f"Average IoU: {avg_iou:.3f}")

    print("\nDetailed results:")
    for r in results:
        status = "✓ SOLVED" if r['solved'] else f"  {r['avg_score']:.3f}"
        print(f"  {r['task_id']}: {status}")

    print("\nComparison to V3+:")
    print("  V3+ on these 13 tasks: 0/13 (0.0%), avg IoU: 0.000")
    print(f"  V4  on these 13 tasks: {solved_count}/{total_count} ({solved_count/total_count*100:.1f}%), avg IoU: {avg_iou:.3f}")

    improvement = solved_count - 0
    print(f"\n  Improvement: +{improvement} tasks solved (+{improvement/total_count*100:.1f}%)")


if __name__ == "__main__":
    main()
