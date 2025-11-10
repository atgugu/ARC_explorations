"""
Test V5 solver on all 46 diverse tasks and compare with V4.
"""

import numpy as np
import json
from solver_v5 import ARCGraphPendulumSolverV5
from utils.arc_loader import ARCLoader

def main():
    print("="*70)
    print("V5 SOLVER - COMPREHENSIVE TEST ON ALL 46 TASKS")
    print("="*70)

    # Load dataset
    loader = ARCLoader(cache_dir="./arc_data")
    tasks = loader.load_all_tasks("training")

    # Create V5 solver
    solver = ARCGraphPendulumSolverV5(
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
            'solved': bool(result['solved']),
            'avg_score': float(result['avg_score'])
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

    print("\nComparison:")
    print("  V4  Perfect solves: 8/46 (17.4%), avg IoU: 0.676")
    print(f"  V5  Perfect solves: {solved_count}/46 ({solved_count/46*100:.1f}%), avg IoU: {avg_iou:.3f}")

    improvement = solved_count - 8
    iou_improvement = avg_iou - 0.676
    print(f"\n  Improvement: {improvement:+d} tasks solved ({improvement/46*100:+.1f}%)")
    print(f"  Avg IoU improvement: {iou_improvement:+.3f}")

    # Save results
    with open('v5_comprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: v5_comprehensive_results.json")

    # Show solved tasks
    print("\n" + "="*70)
    print("SOLVED TASKS")
    print("="*70)
    solved_tasks = [r for r in results if r['solved']]
    for r in solved_tasks:
        print(f"  ✓ {r['task_id']}")

    # Show new solves (compared to V4)
    v4_solved = ['007bbfb7', '25ff71a9', '0d3d703e', '6150a2bd', '3c9b0459',
                 '0b148d64', '23b5c85d', '1cf80156']
    new_solves = [r['task_id'] for r in results if r['solved'] and r['task_id'] not in v4_solved]

    if new_solves:
        print("\n" + "="*70)
        print("NEW SOLVES (compared to V4)")
        print("="*70)
        for task_id in new_solves:
            print(f"  ★ {task_id}")
    else:
        print("\nNo new solves compared to V4")

    # Show improvements (non-solved tasks with better IoU)
    print("\n" + "="*70)
    print("LOADING V4 RESULTS FOR COMPARISON")
    print("="*70)

    try:
        with open('v4_comprehensive_results.json', 'r') as f:
            v4_results = json.load(f)

        v4_scores = {r['task_id']: r['avg_score'] for r in v4_results}

        improvements = []
        regressions = []

        for r in results:
            task_id = r['task_id']
            v5_score = r['avg_score']
            v4_score = v4_scores.get(task_id, 0.0)

            diff = v5_score - v4_score

            if abs(diff) >= 0.05:  # Significant change
                if diff > 0:
                    improvements.append((task_id, v4_score, v5_score, diff))
                else:
                    regressions.append((task_id, v4_score, v5_score, diff))

        if improvements:
            print("\nSIGNIFICANT IMPROVEMENTS (≥0.05 IoU increase):")
            improvements.sort(key=lambda x: x[3], reverse=True)
            for task_id, v4_score, v5_score, diff in improvements[:10]:
                print(f"  {task_id}: {v4_score:.3f} → {v5_score:.3f} (+{diff:.3f})")

        if regressions:
            print("\nREGRESSIONS (≥0.05 IoU decrease):")
            regressions.sort(key=lambda x: x[3])
            for task_id, v4_score, v5_score, diff in regressions[:10]:
                print(f"  {task_id}: {v4_score:.3f} → {v5_score:.3f} ({diff:.3f})")

    except FileNotFoundError:
        print("V4 results not found for detailed comparison")


if __name__ == "__main__":
    main()
