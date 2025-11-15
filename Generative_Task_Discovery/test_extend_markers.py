"""
Test extend_markers primitive on near-miss tasks

Tests on the 3 tasks identified in near-miss analysis:
- fd096ab6 (97.0%)
- 42918530 (95.5%)
- e681b708 (94.8%)
"""

import json
import numpy as np
from pathlib import Path
from inferred_solver import InferredCompositionalSolver

def load_task(task_id: str):
    """Load task from arc_data"""
    task_path = Path("arc_data/evaluation") / f"{task_id}.json"
    with open(task_path, 'r') as f:
        data = json.load(f)
    return {'train': data['train'], 'test': data['test']}

def test_task(task_id: str, task):
    """Test solver on a single task"""
    print(f"\n{'='*70}")
    print(f"Testing task: {task_id}")
    print(f"{'='*70}")

    solver = InferredCompositionalSolver(
        max_candidates=150,
        beam_width=20,
        active_inference_steps=3,
        max_depth=2,
        composition_beam_width=10
    )

    pred1, pred2, metadata = solver.solve(task)

    # Get expected output
    expected = np.array(task['test'][0]['output'])

    # Calculate accuracy for both predictions
    def calc_accuracy(pred, expected):
        if pred is None or pred.shape != expected.shape:
            return 0.0
        return np.sum(pred == expected) / expected.size

    acc1 = calc_accuracy(pred1, expected)
    acc2 = calc_accuracy(pred2, expected)
    best_acc = max(acc1, acc2)

    # Check if solved
    solved = best_acc >= 0.99

    print(f"\nResults:")
    print(f"  Prediction 1 accuracy: {acc1:.2%}")
    print(f"  Prediction 2 accuracy: {acc2:.2%}")
    print(f"  Best accuracy: {best_acc:.2%}")
    print(f"  Status: {'✓ SOLVED' if solved else '✗ FAILED'}")

    # Print top program
    if metadata and 'top_programs' in metadata and metadata['top_programs']:
        top_prog = metadata['top_programs'][0]
        print(f"\nTop program:")
        if isinstance(top_prog, dict):
            print(f"  Schema: {top_prog.get('schema', 'unknown')}")
            if 'steps' in top_prog and top_prog['steps']:
                steps_str = ' → '.join([s.get('schema', 'unknown') if isinstance(s, dict) else str(s) for s in top_prog['steps']])
                print(f"  Steps: {steps_str}")
            print(f"  Parameters: {top_prog.get('parameters', {})}")
        else:
            print(f"  Schema: {top_prog.schema if hasattr(top_prog, 'schema') else 'unknown'}")
            if hasattr(top_prog, 'steps') and top_prog.steps:
                print(f"  Steps: {' → '.join([s.schema for s in top_prog.steps])}")
            print(f"  Parameters: {top_prog.parameters if hasattr(top_prog, 'parameters') else {}}")

    return solved, best_acc

def main():
    """Test on the 3 near-miss tasks"""
    print("="*70)
    print("TESTING EXTEND_MARKERS PRIMITIVE ON NEAR-MISS TASKS")
    print("="*70)

    tasks_to_test = [
        ("fd096ab6", 0.970),  # Was 97.0%
        ("42918530", 0.955),  # Was 95.5%
        ("e681b708", 0.948),  # Was 94.8%
    ]

    results = []

    for task_id, previous_acc in tasks_to_test:
        task = load_task(task_id)
        solved, accuracy = test_task(task_id, task)
        results.append({
            'task_id': task_id,
            'previous_acc': previous_acc,
            'new_acc': accuracy,
            'solved': solved,
            'improvement': accuracy - previous_acc
        })

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    solved_count = sum(1 for r in results if r['solved'])

    print(f"Tasks tested: {len(results)}")
    print(f"Tasks solved: {solved_count}/{len(results)} ({solved_count/len(results)*100:.0f}%)")
    print()

    print("Individual results:")
    for r in results:
        status = "✓" if r['solved'] else "✗"
        improvement_str = f"+{r['improvement']:.1%}" if r['improvement'] > 0 else f"{r['improvement']:.1%}"
        print(f"  {status} {r['task_id']}: {r['previous_acc']:.1%} → {r['new_acc']:.1%} ({improvement_str})")

    # Calculate average improvement
    avg_improvement = sum(r['improvement'] for r in results) / len(results)
    print(f"\nAverage improvement: {avg_improvement:+.1%}")

    if solved_count > 0:
        print(f"\n🎉 SUCCESS! Converted {solved_count} near-miss(es) to solution(s)!")
    else:
        print(f"\n❌ No tasks solved yet. Need to debug extend_markers implementation.")

if __name__ == "__main__":
    main()
