"""
Test Flood Fill Primitive

Tests the new flood fill operations and compares performance.
"""

import numpy as np
from typing import Dict, List, Any
from arc_generative_solver import ARCGenerativeSolver, evaluate_predictions
from enhanced_solver import EnhancedARCSolver
from advanced_solver import AdvancedARCSolver


def create_flood_fill_tasks() -> List[Dict[str, Any]]:
    """Create tasks that require flood fill"""

    tasks = []

    # 1. Fill enclosed region
    tasks.append({
        "name": "fill_enclosed_square",
        "category": "flood_fill",
        "task": {
            "train": [
                {
                    "input": [[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]],
                    "output": [[1, 1, 1],
                              [1, 2, 1],
                              [1, 1, 1]]
                }
            ],
            "test": [
                {
                    "input": [[2, 2, 2, 2],
                             [2, 0, 0, 2],
                             [2, 0, 0, 2],
                             [2, 2, 2, 2]],
                    "output": [[2, 2, 2, 2],
                              [2, 3, 3, 2],
                              [2, 3, 3, 2],
                              [2, 2, 2, 2]]
                }
            ]
        }
    })

    # 2. Fill all background
    tasks.append({
        "name": "fill_all_background",
        "category": "flood_fill",
        "task": {
            "train": [
                {
                    "input": [[0, 1, 0],
                             [0, 0, 0],
                             [0, 1, 0]],
                    "output": [[2, 1, 2],
                              [2, 2, 2],
                              [2, 1, 2]]
                }
            ],
            "test": [
                {
                    "input": [[0, 2, 0],
                             [0, 0, 0]],
                    "output": [[3, 2, 3],
                              [3, 3, 3]]
                }
            ]
        }
    })

    # 3. Fill holes in objects (already implemented in morphology)
    tasks.append({
        "name": "fill_holes",
        "category": "flood_fill",
        "task": {
            "train": [
                {
                    "input": [[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]],
                    "output": [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]
                }
            ],
            "test": [
                {
                    "input": [[2, 2, 2, 2],
                             [2, 0, 0, 2],
                             [2, 2, 2, 2]],
                    "output": [[2, 2, 2, 2],
                              [2, 2, 2, 2],
                              [2, 2, 2, 2]]
                }
            ]
        }
    })

    return tasks


def test_solver_with_flood_fill(solver, solver_name: str, tasks: List[Dict[str, Any]]):
    """Test solver on flood fill tasks"""

    print(f"\n{'='*70}")
    print(f"Testing: {solver_name}")
    print(f"{'='*70}")

    results = []

    for i, task_data in enumerate(tasks, 1):
        task = task_data["task"]
        name = task_data["name"]

        print(f"[{i}/{len(tasks)}] {name:<30}...", end=" ", flush=True)

        try:
            pred1, pred2, metadata = solver.solve(task)
            target = np.array(task["test"][0]["output"])

            eval_res = evaluate_predictions(pred1, pred2, target)

            success = eval_res['any_correct']
            acc = max(eval_res['pixel_accuracy_1'], eval_res['pixel_accuracy_2'])

            print(f"{'✓' if success else '✗'} ({acc:.0%})")

            results.append({
                "name": name,
                "success": success,
                "accuracy": acc,
                "schema1": metadata['top_programs'][0]['schema'] if metadata['top_programs'] else "none",
                "schema2": metadata['top_programs'][1]['schema'] if len(metadata['top_programs']) > 1 else "none"
            })

        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            results.append({"name": name, "success": False, "error": str(e)})

    # Summary
    successes = sum(1 for r in results if r.get("success", False))
    print(f"\nTotal: {successes}/{len(tasks)} ({successes/len(tasks):.1%})")

    return results


def compare_with_flood_fill():
    """Compare all solvers with flood fill tasks"""

    print("="*70)
    print("FLOOD FILL COMPARISON")
    print("="*70)

    tasks = create_flood_fill_tasks()
    print(f"\nTotal flood fill tasks: {len(tasks)}")

    # Test all solvers
    solvers = [
        (ARCGenerativeSolver(max_candidates=100, beam_width=15, active_inference_steps=5), "Original"),
        (EnhancedARCSolver(max_candidates=120, beam_width=15, active_inference_steps=5), "Enhanced"),
        (AdvancedARCSolver(max_candidates=150, beam_width=20, active_inference_steps=5), "Advanced (with flood fill)"),
    ]

    all_results = {}
    for solver, name in solvers:
        all_results[name] = test_solver_with_flood_fill(solver, name, tasks)

    # Final comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    for name in ["Original", "Enhanced", "Advanced (with flood fill)"]:
        results = all_results[name]
        successes = sum(1 for r in results if r.get("success", False))
        rate = successes / len(tasks) if tasks else 0
        print(f"{name:<30}: {successes}/{len(tasks)} ({rate:.1%})")

    # Show which schemas solved which tasks
    print("\n" + "="*70)
    print("SCHEMAS USED")
    print("="*70)

    advanced_results = all_results["Advanced (with flood fill)"]
    for r in advanced_results:
        if r.get("success"):
            schema = r.get("schema1", "unknown")
            print(f"  ✓ {r['name']:<30} -> {schema}")

    return all_results


if __name__ == "__main__":
    results = compare_with_flood_fill()

    print("\n" + "="*70)
    print("✓ FLOOD FILL TEST COMPLETE")
    print("="*70)
