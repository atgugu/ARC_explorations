"""
Evaluate solver on 200 ARC tasks

This script tests the AdvancedARCSolver on 200 evaluation tasks and
provides comprehensive analysis of success/failure patterns.
"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict
from advanced_solver import AdvancedARCSolver
from arc_generative_solver import evaluate_predictions


def load_arc_tasks(limit=200) -> List[Dict[str, Any]]:
    """Load ARC tasks from various sources"""

    tasks = []

    # Try to load from arc_data directory (official dataset)
    arc_data_dir = Path("arc_data")

    if arc_data_dir.exists():
        print("Loading tasks from arc_data directory...")

        # Try evaluation folder
        eval_dir = arc_data_dir / "evaluation"
        if eval_dir.exists():
            json_files = list(eval_dir.glob("*.json"))[:limit]

            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    task = {
                        "id": json_file.stem,
                        "name": json_file.stem,
                        "category": "arc_official",
                        "task": data
                    }
                    tasks.append(task)

                except Exception as e:
                    print(f"  Error loading {json_file}: {e}")

        if tasks:
            print(f"✓ Loaded {len(tasks)} tasks from official ARC dataset")
            return tasks[:limit]

    # Try synthetic dataset
    synthetic_file = Path("synthetic_evaluation_200.json")
    if synthetic_file.exists():
        print("Loading tasks from synthetic_evaluation_200.json...")
        with open(synthetic_file, 'r') as f:
            tasks = json.load(f)
        print(f"✓ Loaded {len(tasks)} synthetic tasks")
        return tasks[:limit]

    print("No task files found. Generate them first with download_arc_data.py")
    return []


def run_evaluation(tasks: List[Dict[str, Any]],
                   max_time_per_task: float = 10.0) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on all tasks

    Args:
        tasks: List of task dictionaries
        max_time_per_task: Maximum time (seconds) to spend on each task

    Returns:
        Dictionary with detailed results
    """

    print("="*70)
    print("RUNNING 200-TASK EVALUATION")
    print("="*70)
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Max time per task: {max_time_per_task}s")
    print(f"Estimated total time: {len(tasks) * max_time_per_task / 60:.1f} minutes")

    solver = AdvancedARCSolver(
        max_candidates=150,
        beam_width=20,
        active_inference_steps=5
    )

    results = []
    category_stats = defaultdict(lambda: {"success": 0, "total": 0, "accuracies": []})
    timing_stats = []

    start_time = time.time()

    for i, task_data in enumerate(tasks, 1):
        task = task_data["task"]
        task_id = task_data.get("id", f"task_{i}")
        name = task_data.get("name", task_id)
        category = task_data.get("category", "unknown")

        print(f"[{i:3d}/{len(tasks)}] {name:<30} ", end="", flush=True)

        task_start = time.time()

        try:
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
                "category": category,
                "success": success,
                "accuracy_1": eval_res['pixel_accuracy_1'],
                "accuracy_2": eval_res['pixel_accuracy_2'],
                "best_accuracy": best_acc,
                "time": task_time,
                "input_shape": task["test"][0]["input"].__class__.__name__,
                "output_shape": task["test"][0]["output"].__class__.__name__,
            }

            # Get top program if available
            if metadata.get('top_programs'):
                result["top_program"] = metadata['top_programs'][0].get('schema', 'unknown')

            results.append(result)

            # Update category stats
            category_stats[category]["total"] += 1
            category_stats[category]["accuracies"].append(best_acc)
            if success:
                category_stats[category]["success"] += 1

            # Print status
            status = "✓" if success else "✗"
            print(f"{status} ({best_acc:5.1%}) [{task_time:5.2f}s]")

            # Progress update every 20 tasks
            if i % 20 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(tasks) - i) * avg_time
                success_so_far = sum(1 for r in results if r['success'])
                print(f"  Progress: {i}/{len(tasks)} | Success rate: {success_so_far/i:.1%} | "
                      f"Avg time: {avg_time:.2f}s | ETA: {remaining/60:.1f}m")

        except Exception as e:
            task_time = time.time() - task_start
            print(f"✗ ERROR [{task_time:.2f}s]: {str(e)[:40]}")

            results.append({
                "id": task_id,
                "name": name,
                "category": category,
                "success": False,
                "error": str(e),
                "time": task_time,
                "best_accuracy": 0.0
            })

            category_stats[category]["total"] += 1
            category_stats[category]["accuracies"].append(0.0)

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
        "category_stats": dict(category_stats),
        "timing_stats": timing_stats
    }


def analyze_results(eval_data: Dict[str, Any]) -> None:
    """Analyze and print detailed results"""

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
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

    # Category breakdown
    print("\n### PERFORMANCE BY CATEGORY ###")
    category_stats = eval_data['category_stats']

    print(f"{'Category':<20} {'Success Rate':<15} {'Tasks':<10} {'Avg Accuracy'}")
    print("-"*70)

    for category, stats in sorted(category_stats.items(),
                                  key=lambda x: x[1]['success']/max(x[1]['total'], 1),
                                  reverse=True):
        rate = stats['success'] / stats['total']
        avg_acc = np.mean(stats['accuracies'])
        print(f"{category:<20} {rate:<15.1%} {stats['success']}/{stats['total']:<8} {avg_acc:.1%}")

    # Accuracy distribution
    print("\n### ACCURACY DISTRIBUTION ###")
    accuracies = [r.get('best_accuracy', 0) for r in eval_data['results']]

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.0)]
    print(f"{'Range':<15} {'Count':<10} {'Percentage'}")
    print("-"*40)

    for low, high in bins:
        count = sum(1 for a in accuracies if low <= a < high or (high == 1.0 and a == 1.0))
        pct = count / len(accuracies)
        bar = "█" * int(pct * 30)
        range_str = f"{low:.0%}-{high:.0%}"
        print(f"{range_str:<15} {count:<10} {pct:.1%} {bar}")

    # Most common failures
    print("\n### FAILURE ANALYSIS ###")
    failures = [r for r in eval_data['results'] if not r.get('success', False)]

    if failures:
        print(f"Total failures: {len(failures)}")

        # Group by accuracy range
        near_misses = [f for f in failures if f.get('best_accuracy', 0) >= 0.7]
        partial = [f for f in failures if 0.3 <= f.get('best_accuracy', 0) < 0.7]
        total_fails = [f for f in failures if f.get('best_accuracy', 0) < 0.3]

        print(f"  Near misses (≥70% acc):  {len(near_misses)} ({len(near_misses)/len(failures):.1%})")
        print(f"  Partial success (30-70%): {len(partial)} ({len(partial)/len(failures):.1%})")
        print(f"  Total failures (<30%):   {len(total_fails)} ({len(total_fails)/len(failures):.1%})")

        # Show some example failures
        if near_misses:
            print("\n  Example near misses:")
            for f in near_misses[:5]:
                print(f"    - {f['name']:<30} ({f.get('best_accuracy', 0):.1%}) [{f.get('category', 'unknown')}]")

    # Best successes
    print("\n### TOP SUCCESSES ###")
    successes = [r for r in eval_data['results'] if r.get('success', False)]
    fastest = sorted(successes, key=lambda x: x.get('time', float('inf')))[:5]

    print("  Fastest solutions:")
    for s in fastest:
        print(f"    - {s['name']:<30} {s.get('time', 0):.3f}s")


def save_detailed_report(eval_data: Dict[str, Any], filename: str = "evaluation_200_report.json"):
    """Save detailed results to JSON file"""

    with open(filename, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print(f"\n✓ Detailed report saved to {filename}")


def main():
    """Main evaluation pipeline"""

    print("="*70)
    print("ARC SOLVER - 200 TASK EVALUATION")
    print("="*70)

    # Load tasks
    tasks = load_arc_tasks(limit=200)

    if not tasks:
        print("\nNo tasks available. Please run:")
        print("  python download_arc_data.py")
        return

    # Run evaluation
    eval_data = run_evaluation(tasks, max_time_per_task=15.0)

    # Analyze results
    analyze_results(eval_data)

    # Save report
    save_detailed_report(eval_data)

    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)

    return eval_data


if __name__ == "__main__":
    results = main()
