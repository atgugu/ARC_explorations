"""
Demo of enhanced solver on real ARC tasks.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_enhanced import EnhancedARCGraphPendulumSolver
from utils.arc_loader import ARCLoader


def main():
    """Demo the enhanced solver on real ARC tasks."""
    print("="*70)
    print("ENHANCED ARC GRAPH PENDULUM SOLVER - REAL ARC TASKS DEMO")
    print("="*70)
    print()

    # Load ARC dataset
    loader = ARCLoader(cache_dir="./arc_data")
    print("Loading ARC dataset...")
    tasks = loader.load_all_tasks("training")

    if not tasks:
        print("No tasks loaded.")
        return

    print(f"Loaded {len(tasks)} tasks\n")

    # Create enhanced solver
    solver = EnhancedARCGraphPendulumSolver(
        beam_width=5,
        use_stability=True,
        use_repair_loops=True,
        use_landscape_analytics=True
    )

    # Solve tasks
    print("\n" + "="*70)
    print("SOLVING REAL ARC TASKS")
    print("="*70)

    results = []
    num_to_solve = min(10, len(tasks))

    task_list = list(tasks.values())[:num_to_solve]

    for i, task in enumerate(task_list):
        print(f"\n{'='*70}")
        print(f"TASK {i+1}/{num_to_solve}")
        print(f"{'='*70}")

        result = solver.evaluate_on_task(task, verbose=True)
        results.append(result)

    # Landscape analysis
    print("\n" + "="*70)
    print("LANDSCAPE ANALYTICS")
    print("="*70)

    landscape_results = solver.analyze_landscape(verbose=True)

    if landscape_results:
        # Try to visualize
        try:
            print("\nGenerating landscape visualization...")
            solver.visualize_landscape("arc_landscape_enhanced.png")
            print("✓ Visualization saved to arc_landscape_enhanced.png")
        except Exception as e:
            print(f"⚠ Visualization skipped: {e}")

        # Save analysis
        try:
            solver.save_landscape_analysis("arc_landscape_enhanced.json")
            print("✓ Analysis saved to arc_landscape_enhanced.json")
        except Exception as e:
            print(f"⚠ Save failed: {e}")

    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    solved_count = sum(1 for r in results if r['solved'])
    avg_score = np.mean([r['avg_score'] for r in results])
    high_score_count = sum(1 for r in results if r['avg_score'] >= 0.8)

    print(f"\nPerformance Metrics:")
    print(f"  Tasks solved (>0.99 IoU): {solved_count}/{len(results)} ({solved_count/len(results)*100:.1f}%)")
    print(f"  High quality (>0.80 IoU): {high_score_count}/{len(results)} ({high_score_count/len(results)*100:.1f}%)")
    print(f"  Average IoU score: {avg_score:.3f}")

    # Show which tasks improved with repairs
    repair_improvements = []
    for r in results:
        # Check if task has repair improvement info (we'll need to track this)
        if 'repair_improved' in r:
            repair_improvements.append(r)

    print(f"\nEnhancement Features:")
    print(f"  Repair loops: ENABLED")
    print(f"  Landscape analytics: ENABLED")
    print(f"  Trajectories analyzed: {len(solver.landscape_analyzer.points)}")

    if landscape_results:
        stats = landscape_results['statistics']
        num_basins = len(stats)
        best_basins = solver.landscape_analyzer.find_best_basins(top_k=3)

        print(f"  Basins discovered: {num_basins}")
        print(f"  Best basins (by quality): {best_basins}")

    print("\nDetailed Results:")
    print("-" * 70)

    for i, result in enumerate(results):
        status = "✓ SOLVED" if result['solved'] else "✗ FAILED"
        quality = "HIGH" if result['avg_score'] >= 0.8 else "MED" if result['avg_score'] >= 0.5 else "LOW"

        print(f"{i+1:2d}. {result['task_id']:12s} {status} {quality:4s} (IoU: {result['avg_score']:.3f})")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
