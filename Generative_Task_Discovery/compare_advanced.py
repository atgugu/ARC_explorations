"""
Comprehensive Comparison: Original → Enhanced → Advanced Solver

Tests progression across three solver versions:
1. Original: Base solver (40% on 35-task suite)
2. Enhanced: + Near-miss primitives (48.6% expected)
3. Advanced: + Rotation fix + Pattern + Morphology (55-60% expected)
"""

import numpy as np
from typing import Dict, List, Any
from arc_generative_solver import ARCGenerativeSolver, evaluate_predictions
from enhanced_solver import EnhancedARCSolver
from advanced_solver import AdvancedARCSolver


def create_test_tasks() -> List[Dict[str, Any]]:
    """Create test tasks for comparison"""

    tasks = []

    # 1. Extract largest (near-miss, 83% → 100%)
    tasks.append({
        "name": "extract_largest",
        "category": "near_miss",
        "task": {
            "train": [
                {
                    "input": [[1, 0, 2, 2], [0, 0, 2, 2], [3, 0, 0, 0]],
                    "output": [[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 0, 0]]
                }
            ],
            "test": [
                {
                    "input": [[4, 4, 4, 0], [0, 5, 0, 0], [0, 0, 0, 6]],
                    "output": [[4, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                }
            ]
        }
    })

    # 2. Connect objects (near-miss, 75% → 100%)
    tasks.append({
        "name": "connect_objects",
        "category": "near_miss",
        "task": {
            "train": [
                {
                    "input": [[1, 0, 0, 0, 2], [0, 0, 0, 0, 0]],
                    "output": [[1, 3, 3, 3, 2], [0, 0, 0, 0, 0]]
                }
            ],
            "test": [
                {
                    "input": [[4, 0, 0, 5], [0, 0, 0, 0]],
                    "output": [[4, 3, 3, 5], [0, 0, 0, 0]]
                }
            ]
        }
    })

    # 3. Align objects (near-miss, 75% → 100%)
    tasks.append({
        "name": "align_objects",
        "category": "near_miss",
        "task": {
            "train": [
                {
                    "input": [[1, 0], [0, 0], [0, 2]],
                    "output": [[1, 2], [0, 0], [0, 0]]
                }
            ],
            "test": [
                {
                    "input": [[3, 0], [0, 0], [0, 0], [4, 0]],
                    "output": [[3, 4], [0, 0], [0, 0], [0, 0]]
                }
            ]
        }
    })

    # 4. Rotation 90° non-square (broken, 33% → 100%)
    tasks.append({
        "name": "rotation_90_nonsquare",
        "category": "rotation_fix",
        "task": {
            "train": [
                {
                    "input": [[1, 2, 3, 4], [5, 6, 7, 8]],
                    "output": [[5, 1], [6, 2], [7, 3], [8, 4]]
                }
            ],
            "test": [
                {
                    "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    "output": [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
                }
            ]
        }
    })

    # 5. Simple rotation (should still work)
    tasks.append({
        "name": "rotation_90_square",
        "category": "basic",
        "task": {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[3, 1], [4, 2]]
                }
            ],
            "test": [
                {
                    "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    "output": [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
                }
            ]
        }
    })

    # 6. Horizontal flip (should still work)
    tasks.append({
        "name": "horizontal_flip",
        "category": "basic",
        "task": {
            "train": [
                {
                    "input": [[1, 2, 3], [4, 5, 6]],
                    "output": [[3, 2, 1], [6, 5, 4]]
                }
            ],
            "test": [
                {
                    "input": [[7, 8, 9]],
                    "output": [[9, 8, 7]]
                }
            ]
        }
    })

    # 7. Pattern tiling (new capability)
    tasks.append({
        "name": "pattern_tiling",
        "category": "pattern",
        "task": {
            "train": [
                {
                    "input": [[1, 2]],
                    "output": [[1, 2, 1, 2]]
                }
            ],
            "test": [
                {
                    "input": [[3, 4]],
                    "output": [[3, 4, 3, 4]]
                }
            ]
        }
    })

    # 8. Symmetry completion (new capability)
    tasks.append({
        "name": "symmetry_complete",
        "category": "pattern",
        "task": {
            "train": [
                {
                    "input": [[1, 2]],
                    "output": [[1, 2, 2, 1]]
                }
            ],
            "test": [
                {
                    "input": [[3, 4]],
                    "output": [[3, 4, 4, 3]]
                }
            ]
        }
    })

    # 9. Identity (trivial, should always work)
    tasks.append({
        "name": "identity",
        "category": "basic",
        "task": {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[1, 2], [3, 4]]
                }
            ],
            "test": [
                {
                    "input": [[5, 6], [7, 8]],
                    "output": [[5, 6], [7, 8]]
                }
            ]
        }
    })

    # 10. Color swap (should still work)
    tasks.append({
        "name": "color_swap",
        "category": "basic",
        "task": {
            "train": [
                {
                    "input": [[1, 2, 1], [2, 1, 2]],
                    "output": [[2, 1, 2], [1, 2, 1]]
                }
            ],
            "test": [
                {
                    "input": [[1, 1, 2]],
                    "output": [[2, 2, 1]]
                }
            ]
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

        print(f"\n[{i}/{len(tasks)}] {name} ({category})...", end=" ", flush=True)

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
                "acc2": acc2
            })

            if category not in category_results:
                category_results[category] = {"success": 0, "total": 0}

            category_results[category]["total"] += 1
            if success:
                category_results[category]["success"] += 1

        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            results.append({
                "name": name,
                "category": category,
                "success": False,
                "error": str(e)
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
        print(f"  {cat}: {data['success']}/{data['total']} ({cat_rate:.1%})")

    return {
        "solver_name": solver_name,
        "results": results,
        "success_rate": success_rate,
        "successes": successes,
        "total": total,
        "category_results": category_results
    }


def compare_solvers():
    """Run comprehensive comparison"""

    print("="*70)
    print("COMPREHENSIVE SOLVER COMPARISON")
    print("="*70)
    print("\nTesting progression:")
    print("  1. Original Solver (baseline)")
    print("  2. Enhanced Solver (+ near-miss primitives)")
    print("  3. Advanced Solver (+ rotation fix + patterns + morphology)")

    tasks = create_test_tasks()

    print(f"\nTotal tasks: {len(tasks)}")

    # Test all three solvers
    solvers = [
        (ARCGenerativeSolver(max_candidates=100, beam_width=15, active_inference_steps=5),
         "Original"),
        (EnhancedARCSolver(max_candidates=120, beam_width=15, active_inference_steps=5),
         "Enhanced"),
        (AdvancedARCSolver(max_candidates=150, beam_width=20, active_inference_steps=5),
         "Advanced")
    ]

    all_results = {}
    for solver, name in solvers:
        all_results[name] = test_solver(solver, name, tasks)

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    print(f"\n{'Solver':<20} {'Success Rate':<15} {'Tasks Solved'}")
    print("-"*70)
    for name in ["Original", "Enhanced", "Advanced"]:
        data = all_results[name]
        print(f"{name:<20} {data['success_rate']:<15.1%} {data['successes']}/{data['total']}")

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
    print(f"  {enhanced['success_rate']:.1%} → {advanced['success_rate']:.1%} ({advanced['success_rate'] - enhanced['success_rate']:+.1%})")
    print(f"  +{advanced['successes'] - enhanced['successes']} tasks solved")

    # Task-by-task improvements
    print("\n" + "="*70)
    print("TASK-BY-TASK IMPROVEMENTS")
    print("="*70)

    print(f"\n{'Task':<25} {'Category':<15} {'Orig':<8} {'Enh':<8} {'Adv'}")
    print("-"*70)

    for i in range(len(tasks)):
        task_name = tasks[i]["name"]
        category = tasks[i]["category"]

        orig_success = "✓" if orig['results'][i].get("success", False) else "✗"
        enh_success = "✓" if enhanced['results'][i].get("success", False) else "✗"
        adv_success = "✓" if advanced['results'][i].get("success", False) else "✗"

        print(f"{task_name:<25} {category:<15} {orig_success:<8} {enh_success:<8} {adv_success}")

    return all_results


if __name__ == "__main__":
    results = compare_solvers()

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
