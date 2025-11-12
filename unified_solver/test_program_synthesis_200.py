"""
Test Program Synthesis Solver on 200 ARC-AGI Evaluation Tasks
=============================================================

Compare baseline (primitive selection) vs program synthesis.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import time

from arc_active_inference_solver import ARCTask, Grid, ARCActiveInferenceSolver
from arc_program_solver import ARCProgramSolver
from arc_loader import ARCDataLoader


def load_evaluation_tasks(data_dir: str, num_tasks: int = 200) -> Dict[str, ARCTask]:
    """Load evaluation tasks from JSON files"""
    eval_dir = Path(data_dir) / "evaluation"

    if not eval_dir.exists():
        raise ValueError(f"Evaluation directory not found: {eval_dir}")

    json_files = sorted(eval_dir.glob("*.json"))

    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {eval_dir}")

    print(f"Found {len(json_files)} evaluation tasks")
    print(f"Loading first {num_tasks} tasks...\n")

    tasks = {}
    for json_file in json_files[:num_tasks]:
        task_id = json_file.stem

        try:
            with open(json_file, 'r') as f:
                task_data = json.load(f)

            task = ARCDataLoader.load_task_from_dict(task_data)
            tasks[task_id] = task

        except Exception as e:
            print(f"Warning: Failed to load task {task_id}: {e}")
            continue

    print(f"Successfully loaded {len(tasks)} tasks\n")
    return tasks


def test_solver(solver, solver_name: str, tasks: Dict[str, ARCTask]) -> Dict:
    """Test a solver on evaluation tasks"""

    results = {
        'solver_name': solver_name,
        'total_tasks': len(tasks),
        'exact_match_attempt_1': 0,
        'exact_match_attempt_2': 0,
        'exact_match_either': 0,
        'both_wrong': 0,
        'identical_predictions': 0,
        'task_results': {},
        'failure_modes': defaultdict(int),
        'timing': [],
    }

    print(f"\n{'='*80}")
    print(f"TESTING: {solver_name}")
    print(f"{'='*80}\n")

    task_counter = 0

    for task_id, task in sorted(tasks.items()):
        task_counter += 1

        # Run solver
        start_time = time.time()
        try:
            predictions = solver.solve(task, verbose=False)
        except Exception as e:
            print(f"✗  Task {task_counter:3d} {task_id}: ERROR - {e}")
            results['task_results'][task_id] = {
                'success': False,
                'error': str(e)
            }
            results['both_wrong'] += 1
            continue

        elapsed = time.time() - start_time
        results['timing'].append(elapsed)

        # Evaluate
        if task.test_output is None:
            continue

        pred_1 = predictions[0]
        pred_2 = predictions[1]
        gt = task.test_output

        match_1 = np.array_equal(pred_1.data, gt.data)
        match_2 = np.array_equal(pred_2.data, gt.data)
        identical = np.array_equal(pred_1.data, pred_2.data)

        # Update statistics
        if match_1:
            results['exact_match_attempt_1'] += 1
            results['exact_match_either'] += 1
            success_str = "✓1"
        elif match_2:
            results['exact_match_attempt_2'] += 1
            results['exact_match_either'] += 1
            success_str = "✓2"
        else:
            results['both_wrong'] += 1
            success_str = "✗ "

            # Analyze failure mode
            if pred_1.shape != gt.shape:
                results['failure_modes']['size_mismatch'] += 1
            else:
                results['failure_modes']['wrong_transform'] += 1

        if identical:
            results['identical_predictions'] += 1

        # Record result
        results['task_results'][task_id] = {
            'success': match_1 or match_2,
            'match_1': match_1,
            'match_2': match_2,
            'identical': identical,
            'elapsed': elapsed,
        }

        # Print progress every 10 tasks
        if task_counter % 10 == 0:
            print(f"{success_str} Task {task_counter:3d}/{len(tasks)} {task_id} "
                  f"({elapsed:.2f}s)")

    return results


def print_comparison(baseline_results: Dict, synthesis_results: Dict):
    """Print side-by-side comparison"""

    total = baseline_results['total_tasks']

    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs Program Synthesis")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Baseline':>15} {'Synthesis':>15} {'Change':>15}")
    print("-" * 80)

    # Success rates
    baseline_success = baseline_results['exact_match_either']
    synthesis_success = synthesis_results['exact_match_either']
    delta_success = synthesis_success - baseline_success

    print(f"{'Exact Match (either)':<30} "
          f"{baseline_success:>4d} ({100*baseline_success/total:>5.1f}%) "
          f"{synthesis_success:>4d} ({100*synthesis_success/total:>5.1f}%) "
          f"{delta_success:>+4d} ({100*delta_success/total:>+5.1f}%)")

    # Attempt 1
    baseline_att1 = baseline_results['exact_match_attempt_1']
    synthesis_att1 = synthesis_results['exact_match_attempt_1']
    delta_att1 = synthesis_att1 - baseline_att1

    print(f"{'  Attempt 1':<30} "
          f"{baseline_att1:>4d} ({100*baseline_att1/total:>5.1f}%) "
          f"{synthesis_att1:>4d} ({100*synthesis_att1/total:>5.1f}%) "
          f"{delta_att1:>+4d} ({100*delta_att1/total:>+5.1f}%)")

    # Attempt 2
    baseline_att2 = baseline_results['exact_match_attempt_2']
    synthesis_att2 = synthesis_results['exact_match_attempt_2']
    delta_att2 = synthesis_att2 - baseline_att2

    print(f"{'  Attempt 2':<30} "
          f"{baseline_att2:>4d} ({100*baseline_att2/total:>5.1f}%) "
          f"{synthesis_att2:>4d} ({100*synthesis_att2/total:>5.1f}%) "
          f"{delta_att2:>+4d} ({100*delta_att2/total:>+5.1f}%)")

    # Diversity
    baseline_ident = baseline_results['identical_predictions']
    synthesis_ident = synthesis_results['identical_predictions']
    delta_ident = synthesis_ident - baseline_ident

    print(f"{'Identical Predictions':<30} "
          f"{baseline_ident:>4d} ({100*baseline_ident/total:>5.1f}%) "
          f"{synthesis_ident:>4d} ({100*synthesis_ident/total:>5.1f}%) "
          f"{delta_ident:>+4d} ({100*delta_ident/total:>+5.1f}%)")

    # Timing
    baseline_time = np.mean(baseline_results['timing']) if baseline_results['timing'] else 0
    synthesis_time = np.mean(synthesis_results['timing']) if synthesis_results['timing'] else 0
    delta_time = synthesis_time - baseline_time

    print(f"{'Avg Time (seconds)':<30} "
          f"{baseline_time:>14.3f}s "
          f"{synthesis_time:>14.3f}s "
          f"{delta_time:>+14.3f}s")

    # Failure modes
    print("\n" + "=" * 80)
    print("FAILURE MODE COMPARISON")
    print("=" * 80)

    baseline_wrong = baseline_results['both_wrong']
    synthesis_wrong = synthesis_results['both_wrong']

    print(f"\n{'Mode':<30} {'Baseline':>15} {'Synthesis':>15}")
    print("-" * 80)

    # Size mismatch
    baseline_size = baseline_results['failure_modes'].get('size_mismatch', 0)
    synthesis_size = synthesis_results['failure_modes'].get('size_mismatch', 0)

    if baseline_wrong > 0:
        baseline_size_pct = 100 * baseline_size / baseline_wrong
    else:
        baseline_size_pct = 0

    if synthesis_wrong > 0:
        synthesis_size_pct = 100 * synthesis_size / synthesis_wrong
    else:
        synthesis_size_pct = 0

    print(f"{'Size Mismatch':<30} "
          f"{baseline_size:>4d} ({baseline_size_pct:>5.1f}%) "
          f"{synthesis_size:>4d} ({synthesis_size_pct:>5.1f}%)")

    # Wrong transform
    baseline_wrong_tf = baseline_results['failure_modes'].get('wrong_transform', 0)
    synthesis_wrong_tf = synthesis_results['failure_modes'].get('wrong_transform', 0)

    if baseline_wrong > 0:
        baseline_wrong_pct = 100 * baseline_wrong_tf / baseline_wrong
    else:
        baseline_wrong_pct = 0

    if synthesis_wrong > 0:
        synthesis_wrong_pct = 100 * synthesis_wrong_tf / synthesis_wrong
    else:
        synthesis_wrong_pct = 0

    print(f"{'Wrong Transform':<30} "
          f"{baseline_wrong_tf:>4d} ({baseline_wrong_pct:>5.1f}%) "
          f"{synthesis_wrong_tf:>4d} ({synthesis_wrong_pct:>5.1f}%)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    improvement = synthesis_success - baseline_success
    if baseline_success > 0:
        improvement_pct = 100 * improvement / baseline_success
    else:
        improvement_pct = 100 * synthesis_success if synthesis_success > 0 else 0

    print(f"\nBaseline Success:  {baseline_success}/{total} ({100*baseline_success/total:.1f}%)")
    print(f"Synthesis Success: {synthesis_success}/{total} ({100*synthesis_success/total:.1f}%)")
    print(f"Improvement:       {improvement:+d} tasks ({improvement_pct:+.1f}%)")

    if improvement > 0:
        print(f"\n✓ Program synthesis IMPROVED performance by {improvement} tasks!")
    elif improvement == 0:
        print(f"\n⊙ Program synthesis performed the SAME as baseline")
    else:
        print(f"\n✗ Program synthesis DECREASED performance by {abs(improvement)} tasks")

    return improvement


def main():
    """Main test execution"""

    print("\n" + "=" * 80)
    print("PROGRAM SYNTHESIS vs BASELINE COMPARISON")
    print("=" * 80)

    # Load tasks
    print("\nLoading evaluation tasks...")
    data_dir = "../data"

    if not os.path.exists(data_dir):
        data_dir = "data"

    try:
        tasks = load_evaluation_tasks(data_dir, num_tasks=200)
    except Exception as e:
        print(f"\n❌ Error loading tasks: {e}")
        return

    # Test baseline solver
    print("\n[1/2] Testing BASELINE solver (fixed primitives)...")
    baseline_solver = ARCActiveInferenceSolver(
        workspace_capacity=20,
        n_perturbations=5
    )
    baseline_results = test_solver(baseline_solver, "Baseline", tasks)

    # Test program synthesis solver
    print("\n[2/2] Testing PROGRAM SYNTHESIS solver...")
    synthesis_solver = ARCProgramSolver(
        workspace_capacity=20,
        n_perturbations=5,
        max_synthesis_depth=2,
        max_programs=100,
        verbose=False
    )
    synthesis_results = test_solver(synthesis_solver, "Program Synthesis", tasks)

    # Print comparison
    print_comparison(baseline_results, synthesis_results)

    # Save results
    output = {
        'baseline': {
            'total': baseline_results['total_tasks'],
            'success': baseline_results['exact_match_either'],
            'attempt_1': baseline_results['exact_match_attempt_1'],
            'attempt_2': baseline_results['exact_match_attempt_2'],
            'identical': baseline_results['identical_predictions'],
            'avg_time': float(np.mean(baseline_results['timing'])),
        },
        'synthesis': {
            'total': synthesis_results['total_tasks'],
            'success': synthesis_results['exact_match_either'],
            'attempt_1': synthesis_results['exact_match_attempt_1'],
            'attempt_2': synthesis_results['exact_match_attempt_2'],
            'identical': synthesis_results['identical_predictions'],
            'avg_time': float(np.mean(synthesis_results['timing'])),
        },
    }

    with open('program_synthesis_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: program_synthesis_comparison.json")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
