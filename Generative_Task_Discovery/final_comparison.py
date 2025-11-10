"""
Final Comprehensive Comparison: Object Operations + Physics

Tests the fully enhanced solver with all primitives:
- Original baseline
- Enhanced (near-miss)
- Advanced (patterns + rotation)
- Final (objects + physics)
"""

import numpy as np
from typing import Dict, List, Any
from arc_generative_solver import ARCGenerativeSolver, evaluate_predictions
from enhanced_solver import EnhancedARCSolver
from advanced_solver import AdvancedARCSolver


def create_comprehensive_test_suite() -> List[Dict[str, Any]]:
    """Create comprehensive test suite including new capabilities"""

    tasks = []

    # === NEAR-MISS TASKS (3) ===

    # 1. Extract largest
    tasks.append({
        "name": "extract_largest",
        "category": "near_miss",
        "task": {
            "train": [{"input": [[1, 0, 2, 2], [0, 0, 2, 2], [3, 0, 0, 0]],
                      "output": [[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 0, 0]]}],
            "test": [{"input": [[4, 4, 4, 0], [0, 5, 0, 0], [0, 0, 0, 6]],
                     "output": [[4, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}]
        }
    })

    # 2. Connect objects
    tasks.append({
        "name": "connect_objects",
        "category": "near_miss",
        "task": {
            "train": [{"input": [[1, 0, 0, 0, 2], [0, 0, 0, 0, 0]],
                      "output": [[1, 3, 3, 3, 2], [0, 0, 0, 0, 0]]}],
            "test": [{"input": [[4, 0, 0, 5], [0, 0, 0, 0]],
                     "output": [[4, 3, 3, 5], [0, 0, 0, 0]]}]
        }
    })

    # 3. Align objects
    tasks.append({
        "name": "align_objects",
        "category": "near_miss",
        "task": {
            "train": [{"input": [[1, 0], [0, 0], [0, 2]],
                      "output": [[1, 2], [0, 0], [0, 0]]}],
            "test": [{"input": [[3, 0], [0, 0], [0, 0], [4, 0]],
                     "output": [[3, 4], [0, 0], [0, 0], [0, 0]]}]
        }
    })

    # === BASIC TASKS (4) ===

    # 4. Rotation 90° (square)
    tasks.append({
        "name": "rotation_90_square",
        "category": "basic",
        "task": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}],
            "test": [{"input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     "output": [[7, 4, 1], [8, 5, 2], [9, 6, 3]]}]
        }
    })

    # 5. Horizontal flip
    tasks.append({
        "name": "horizontal_flip",
        "category": "basic",
        "task": {
            "train": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[3, 2, 1], [6, 5, 4]]}],
            "test": [{"input": [[7, 8, 9]], "output": [[9, 8, 7]]}]
        }
    })

    # 6. Identity
    tasks.append({
        "name": "identity",
        "category": "basic",
        "task": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}],
            "test": [{"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]}]
        }
    })

    # 7. Color swap
    tasks.append({
        "name": "color_swap",
        "category": "basic",
        "task": {
            "train": [{"input": [[1, 2, 1], [2, 1, 2]], "output": [[2, 1, 2], [1, 2, 1]]}],
            "test": [{"input": [[1, 1, 2]], "output": [[2, 2, 1]]}]
        }
    })

    # === PATTERN TASKS (2) ===

    # 8. Pattern tiling
    tasks.append({
        "name": "pattern_tiling",
        "category": "pattern",
        "task": {
            "train": [{"input": [[1, 2]], "output": [[1, 2, 1, 2]]}],
            "test": [{"input": [[3, 4]], "output": [[3, 4, 3, 4]]}]
        }
    })

    # 9. Symmetry completion
    tasks.append({
        "name": "symmetry_complete",
        "category": "pattern",
        "task": {
            "train": [{"input": [[1, 2]], "output": [[1, 2, 2, 1]]}],
            "test": [{"input": [[3, 4]], "output": [[3, 4, 4, 3]]}]
        }
    })

    # === OBJECT OPERATIONS (3) ===

    # 10. Duplicate object
    tasks.append({
        "name": "duplicate_object",
        "category": "object_ops",
        "task": {
            "train": [{"input": [[1, 0, 0]], "output": [[1, 0, 1]]}],
            "test": [{"input": [[2, 0, 0]], "output": [[2, 0, 2]]}]
        }
    })

    # 11. Distribute objects evenly
    tasks.append({
        "name": "distribute_objects",
        "category": "object_ops",
        "task": {
            "train": [{"input": [[1, 0, 0, 0, 0, 2]],
                      "output": [[1, 0, 0, 0, 2, 0]]}],  # Even spacing
            "test": [{"input": [[3, 0, 0, 0, 0, 4]],
                     "output": [[3, 0, 0, 0, 4, 0]]}]
        }
    })

    # 12. Stack objects
    tasks.append({
        "name": "stack_objects",
        "category": "object_ops",
        "task": {
            "train": [{"input": [[1, 0], [0, 0], [2, 0]],
                      "output": [[1, 0], [2, 0], [0, 0]]}],
            "test": [{"input": [[3, 0], [0, 0], [4, 0]],
                     "output": [[3, 0], [4, 0], [0, 0]]}]
        }
    })

    # === PHYSICS TASKS (2) ===

    # 13. Gravity/falling
    tasks.append({
        "name": "gravity_fall",
        "category": "physics",
        "task": {
            "train": [{"input": [[1, 0], [0, 0], [0, 0]],
                      "output": [[0, 0], [0, 0], [1, 0]]}],
            "test": [{"input": [[2, 0], [0, 0], [0, 0]],
                     "output": [[0, 0], [0, 0], [2, 0]]}]
        }
    })

    # 14. Compress (remove gaps)
    tasks.append({
        "name": "compress",
        "category": "physics",
        "task": {
            "train": [{"input": [[1, 0, 2]], "output": [[1, 2, 0]]}],
            "test": [{"input": [[3, 0, 4]], "output": [[3, 4, 0]]}]
        }
    })

    return tasks


def test_solver(solver, solver_name: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test a solver on all tasks"""

    print(f"\n{'='*70}")
    print(f"Testing: {solver_name}")
    print(f"{'='*70}")

    results = []
    category_results = {}

    for i, task_data in enumerate(tasks, 1):
        task = task_data["task"]
        name = task_data["name"]
        category = task_data["category"]

        print(f"[{i:2d}/{len(tasks)}] {name:<25} ({category:<12})...", end=" ", flush=True)

        try:
            pred1, pred2, metadata = solver.solve(task)
            target = np.array(task["test"][0]["output"])

            eval_res = evaluate_predictions(pred1, pred2, target)

            success = eval_res['any_correct']
            acc1 = eval_res['pixel_accuracy_1']
            acc2 = eval_res['pixel_accuracy_2']

            print(f"{'✓' if success else '✗'} ({max(acc1, acc2):.0%})")

            results.append({
                "name": name,
                "category": category,
                "success": success,
                "acc1": acc1,
                "acc2": acc2,
                "best_acc": max(acc1, acc2)
            })

            if category not in category_results:
                category_results[category] = {"success": 0, "total": 0}

            category_results[category]["total"] += 1
            if success:
                category_results[category]["success"] += 1

        except Exception as e:
            print(f"✗ ERROR: {str(e)[:40]}")
            results.append({
                "name": name,
                "category": category,
                "success": False,
                "error": str(e),
                "best_acc": 0.0
            })

    # Summary
    total = len(tasks)
    successes = sum(1 for r in results if r.get("success", False))
    success_rate = successes / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"Summary for {solver_name}")
    print(f"{'='*70}")
    print(f"Total: {successes}/{total} ({success_rate:.1%})")

    print(f"\nBy category:")
    for cat, data in sorted(category_results.items()):
        cat_rate = data["success"] / data["total"]
        print(f"  {cat:<15}: {data['success']}/{data['total']} ({cat_rate:.1%})")

    return {
        "solver_name": solver_name,
        "results": results,
        "success_rate": success_rate,
        "successes": successes,
        "total": total,
        "category_results": category_results
    }


def compare_all_solvers():
    """Run final comprehensive comparison"""

    print("="*70)
    print("FINAL COMPREHENSIVE COMPARISON")
    print("="*70)
    print("\nTesting progression:")
    print("  1. Original Solver (baseline)")
    print("  2. Enhanced Solver (+ near-miss)")
    print("  3. Advanced Solver (+ patterns + rotation)")
    print("  4. Final Solver (+ objects + physics) - SAME AS ADVANCED")

    tasks = create_comprehensive_test_suite()

    print(f"\nTotal tasks: {len(tasks)}")
    print(f"  Near-miss: 3")
    print(f"  Basic: 4")
    print(f"  Pattern: 2")
    print(f"  Object ops: 3")
    print(f"  Physics: 2")

    # Test all solvers
    solvers = [
        (ARCGenerativeSolver(max_candidates=100, beam_width=15, active_inference_steps=5), "Original"),
        (EnhancedARCSolver(max_candidates=120, beam_width=15, active_inference_steps=5), "Enhanced"),
        (AdvancedARCSolver(max_candidates=150, beam_width=20, active_inference_steps=5), "Advanced"),
    ]

    all_results = {}
    for solver, name in solvers:
        all_results[name] = test_solver(solver, name, tasks)

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    print(f"\n{'Solver':<20} {'Success Rate':<15} {'Tasks Solved':<15} {'Avg Accuracy'}")
    print("-"*70)
    for name in ["Original", "Enhanced", "Advanced"]:
        data = all_results[name]
        avg_acc = np.mean([r.get("best_acc", 0) for r in data["results"]])
        print(f"{name:<20} {data['success_rate']:<15.1%} {data['successes']}/{data['total']:<13} {avg_acc:.1%}")

    # Improvements
    print("\n" + "="*70)
    print("IMPROVEMENTS")
    print("="*70)

    orig = all_results["Original"]
    enhanced = all_results["Enhanced"]
    advanced = all_results["Advanced"]

    enh_improvement = enhanced['success_rate'] - orig['success_rate']
    adv_improvement = advanced['success_rate'] - orig['success_rate']

    print(f"\nEnhanced vs Original:")
    print(f"  {orig['success_rate']:.1%} → {enhanced['success_rate']:.1%} ({enh_improvement:+.1%})")
    print(f"  +{enhanced['successes'] - orig['successes']} tasks solved")

    print(f"\nAdvanced vs Original:")
    print(f"  {orig['success_rate']:.1%} → {advanced['success_rate']:.1%} ({adv_improvement:+.1%})")
    print(f"  +{advanced['successes'] - orig['successes']} tasks solved")

    print(f"\nAdvanced vs Enhanced:")
    improvement = advanced['success_rate'] - enhanced['success_rate']
    print(f"  {enhanced['success_rate']:.1%} → {advanced['success_rate']:.1%} ({improvement:+.1%})")
    print(f"  +{advanced['successes'] - enhanced['successes']} tasks solved")

    # Category breakdown
    print("\n" + "="*70)
    print("CATEGORY BREAKDOWN")
    print("="*70)

    categories = ["near_miss", "basic", "pattern", "object_ops", "physics"]
    print(f"\n{'Category':<15} {'Original':<12} {'Enhanced':<12} {'Advanced'}")
    print("-"*70)

    for cat in categories:
        orig_data = orig['category_results'].get(cat, {"success": 0, "total": 0})
        enh_data = enhanced['category_results'].get(cat, {"success": 0, "total": 0})
        adv_data = advanced['category_results'].get(cat, {"success": 0, "total": 0})

        orig_str = f"{orig_data['success']}/{orig_data['total']}"
        enh_str = f"{enh_data['success']}/{enh_data['total']}"
        adv_str = f"{adv_data['success']}/{adv_data['total']}"

        print(f"{cat:<15} {orig_str:<12} {enh_str:<12} {adv_str}")

    return all_results


if __name__ == "__main__":
    results = compare_all_solvers()

    print("\n" + "="*70)
    print("✓ FINAL COMPARISON COMPLETE")
    print("="*70)
