"""
Evaluate solver on real ARC evaluation dataset

Tests on the official 400 ARC evaluation tasks
"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict
from advanced_solver import AdvancedARCSolver
from compositional_solver import CompositionalARCSolver
from inferred_solver import InferredCompositionalSolver
from ensemble_solver import EnsembleCompositionalSolver
from arc_generative_solver import evaluate_predictions


def load_real_arc_tasks(limit=None, shuffle=False) -> List[Dict[str, Any]]:
    """Load real ARC tasks from arc_data directory"""

    print("="*70)
    print("LOADING REAL ARC DATASET")
    print("="*70)

    arc_dir = Path("arc_data/evaluation")

    if not arc_dir.exists():
        print("✗ ARC data directory not found")
        print("Run download_real_arc.py first")
        return []

    json_files = sorted(arc_dir.glob("*.json"))

    if shuffle:
        import random
        random.shuffle(json_files)

    if limit:
        json_files = json_files[:limit]

    print(f"\nLoading {len(json_files)} tasks...")

    tasks = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            task = {
                "id": json_file.stem,
                "name": json_file.stem,
                "task": data
            }
            tasks.append(task)

        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")

    print(f"✓ Loaded {len(tasks)} real ARC tasks")

    return tasks


def run_real_arc_evaluation(tasks: List[Dict[str, Any]],
                             max_time_per_task: float = 30.0,
                             use_compositional: bool = True,
                             use_inference: bool = True) -> Dict[str, Any]:
    """
    Run evaluation on real ARC tasks

    Args:
        tasks: List of real ARC tasks
        max_time_per_task: Max time per task (real ARC is harder)

    Returns:
        Dictionary with detailed results
    """

    print("\n" + "="*70)
    print("RUNNING REAL ARC EVALUATION")
    print("="*70)
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Max time per task: {max_time_per_task}s")
    print(f"Estimated total time: {len(tasks) * max_time_per_task / 60:.1f} minutes")

    if use_compositional and use_inference:
        solver = InferredCompositionalSolver(
            max_candidates=150,  # Reverted from 200 after depth=3 analysis
            beam_width=20,
            active_inference_steps=3,  # Reduced for compositional
            max_depth=2,  # Allow 1-2 step compositions (REVERTED from 3)
            composition_beam_width=10  # Reverted from 15
        )
        print(f"Using InferredCompositionalSolver (max_depth={solver.max_depth}, with parameter inference)")
    elif use_compositional:
        solver = CompositionalARCSolver(
            max_candidates=150,  # Reverted from 200 after depth=3 analysis
            beam_width=20,
            active_inference_steps=3,  # Reduced for compositional
            max_depth=2,  # Allow 1-2 step compositions (REVERTED from 3)
            composition_beam_width=10  # Reverted from 15
        )
        print(f"Using CompositionalARCSolver (max_depth={solver.max_depth})")
    else:
        solver = AdvancedARCSolver(
            max_candidates=150,
            beam_width=20,
            active_inference_steps=5
        )
        print("Using AdvancedARCSolver (single-step)")

    print(f"Solver type: {type(solver).__name__}")

    results = []
    timing_stats = []

    # Track complexity
    complexity_stats = {
        "small": {"success": 0, "total": 0},   # ≤10x10
        "medium": {"success": 0, "total": 0},  # 10x10 to 20x20
        "large": {"success": 0, "total": 0}    # >20x20
    }

    start_time = time.time()

    for i, task_data in enumerate(tasks, 1):
        task = task_data["task"]
        task_id = task_data.get("id", f"task_{i}")
        name = task_data.get("name", task_id)

        print(f"[{i:3d}/{len(tasks)}] {name:<30} ", end="", flush=True)

        task_start = time.time()

        try:
            # Get task complexity
            test_input = np.array(task["test"][0]["input"])
            h, w = test_input.shape
            size = max(h, w)

            if size <= 10:
                complexity = "small"
            elif size <= 20:
                complexity = "medium"
            else:
                complexity = "large"

            # Run solver with timeout
            pred1, pred2, metadata = solver.solve(task)
            target = np.array(task["test"][0]["output"])

            task_time = time.time() - task_start
            timing_stats.append(task_time)

            # Evaluate
            eval_res = evaluate_predictions(pred1, pred2, target)

            success = eval_res['any_correct']
            best_acc = max(eval_res['pixel_accuracy_1'], eval_res['pixel_accuracy_2'])

            # Store result
            result = {
                "id": task_id,
                "name": name,
                "success": success,
                "accuracy_1": eval_res['pixel_accuracy_1'],
                "accuracy_2": eval_res['pixel_accuracy_2'],
                "best_accuracy": best_acc,
                "time": task_time,
                "complexity": complexity,
                "grid_size": f"{h}x{w}",
                "train_examples": len(task.get("train", []))
            }

            # Get top program if available
            if metadata.get('top_programs'):
                result["top_program"] = metadata['top_programs'][0].get('schema', 'unknown')

            results.append(result)

            # Update complexity stats
            complexity_stats[complexity]["total"] += 1
            if success:
                complexity_stats[complexity]["success"] += 1

            # Print status
            status = "✓" if success else "✗"
            print(f"{status} ({best_acc:5.1%}) [{task_time:5.2f}s] ({complexity})")

            # Progress update every 20 tasks
            if i % 20 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(tasks) - i) * avg_time
                success_so_far = sum(1 for r in results if r['success'])
                print(f"  Progress: {i}/{len(tasks)} | Success: {success_so_far/i:.1%} | "
                      f"Avg time: {avg_time:.2f}s | ETA: {remaining/60:.1f}m")

        except Exception as e:
            task_time = time.time() - task_start
            print(f"✗ ERROR [{task_time:.2f}s]: {str(e)[:40]}")

            results.append({
                "id": task_id,
                "name": name,
                "success": False,
                "error": str(e),
                "time": task_time,
                "best_accuracy": 0.0
            })

    total_time = time.time() - start_time

    # Compute overall statistics
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get('success', False))
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0

    avg_accuracy = np.mean([r.get('best_accuracy', 0) for r in results])
    median_accuracy = np.median([r.get('best_accuracy', 0) for r in results])

    avg_time = np.mean(timing_stats) if timing_stats else 0
    median_time = np.median(timing_stats) if timing_stats else 0

    return {
        "results": results,
        "summary": {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "avg_accuracy": avg_accuracy,
            "median_accuracy": median_accuracy,
            "avg_time": avg_time,
            "median_time": median_time,
            "total_time": total_time
        },
        "complexity_stats": complexity_stats,
        "timing_stats": timing_stats
    }


def analyze_real_arc_results(eval_data: Dict[str, Any]) -> None:
    """Analyze and print detailed results"""

    print("\n" + "="*70)
    print("REAL ARC EVALUATION RESULTS")
    print("="*70)

    summary = eval_data['summary']

    print("\n### OVERALL PERFORMANCE ###")
    print(f"  Total tasks:      {summary['total_tasks']}")
    print(f"  Successful:       {summary['successful_tasks']} ({summary['success_rate']:.1%})")
    print(f"  Average accuracy: {summary['avg_accuracy']:.1%}")
    print(f"  Median accuracy:  {summary['median_accuracy']:.1%}")

    print("\n### TIMING ###")
    print(f"  Total time:       {summary['total_time']/60:.1f} minutes")
    print(f"  Average per task: {summary['avg_time']:.2f}s")
    print(f"  Median per task:  {summary['median_time']:.2f}s")

    # Complexity breakdown
    print("\n### PERFORMANCE BY COMPLEXITY ###")
    complexity_stats = eval_data['complexity_stats']

    print(f"{'Complexity':<15} {'Success Rate':<15} {'Tasks':<10}")
    print("-"*50)

    for complexity in ["small", "medium", "large"]:
        stats = complexity_stats.get(complexity, {"success": 0, "total": 0})
        if stats["total"] > 0:
            rate = stats['success'] / stats['total']
            print(f"{complexity.capitalize():<15} {rate:<15.1%} {stats['success']}/{stats['total']}")

    # Accuracy distribution
    print("\n### ACCURACY DISTRIBUTION ###")
    accuracies = [r.get('best_accuracy', 0) for r in eval_data['results']]

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.0)]
    print(f"{'Range':<15} {'Count':<10} {'Percentage'}")
    print("-"*40)

    for low, high in bins:
        count = sum(1 for a in accuracies if low <= a < high or (high == 1.0 and a == 1.0))
        pct = count / len(accuracies) if accuracies else 0
        bar = "█" * int(pct * 30)
        range_str = f"{low:.0%}-{high:.0%}"
        print(f"{range_str:<15} {count:<10} {pct:.1%} {bar}")

    # Near misses
    print("\n### NEAR MISSES (70-95% accuracy) ###")
    near_misses = [r for r in eval_data['results']
                   if not r.get('success', False) and r.get('best_accuracy', 0) >= 0.7]

    if near_misses:
        print(f"Total near misses: {len(near_misses)}")
        print("\nTop near misses:")
        sorted_misses = sorted(near_misses, key=lambda x: x.get('best_accuracy', 0), reverse=True)
        for r in sorted_misses[:10]:
            print(f"  {r['name']:<30} {r.get('best_accuracy', 0):.1%}")

    # Fastest successes
    print("\n### FASTEST SOLUTIONS ###")
    successes = [r for r in eval_data['results'] if r.get('success', False)]
    if successes:
        fastest = sorted(successes, key=lambda x: x.get('time', float('inf')))[:5]
        for s in fastest:
            print(f"  {s['name']:<30} {s.get('time', 0):.3f}s")

    # Slowest successes
    print("\n### SLOWEST SOLUTIONS ###")
    if successes:
        slowest = sorted(successes, key=lambda x: x.get('time', 0), reverse=True)[:5]
        for s in slowest:
            print(f"  {s['name']:<30} {s.get('time', 0):.3f}s")


def save_real_arc_report(eval_data: Dict[str, Any], filename: str = "real_arc_evaluation_report.json"):
    """Save detailed results to JSON file"""

    with open(filename, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print(f"\n✓ Detailed report saved to {filename}")


def main():
    """Main evaluation pipeline for real ARC"""

    print("="*70)
    print("ARC SOLVER - REAL ARC EVALUATION")
    print("="*70)

    # Load tasks (start with 100 for testing, then do all 400)
    print("\nSelect evaluation mode:")
    print("  1. Quick test (50 tasks)")
    print("  2. Medium test (100 tasks)")
    print("  3. Full evaluation (400 tasks)")

    # Default to medium test for now
    n_tasks = 100
    print(f"\nRunning medium test ({n_tasks} tasks)...")

    tasks = load_real_arc_tasks(limit=n_tasks, shuffle=True)

    if not tasks:
        print("\n✗ No tasks loaded. Please run download_real_arc.py first.")
        return

    # Run evaluation with compositional reasoning + parameter inference
    eval_data = run_real_arc_evaluation(
        tasks,
        max_time_per_task=30.0,
        use_compositional=True,  # Enable compositional reasoning
        use_inference=True  # Enable parameter inference
    )

    # Analyze results
    analyze_real_arc_results(eval_data)

    # Save report
    save_real_arc_report(eval_data)

    print("\n" + "="*70)
    print("✓ REAL ARC EVALUATION COMPLETE")
    print("="*70)

    return eval_data


if __name__ == "__main__":
    results = main()
