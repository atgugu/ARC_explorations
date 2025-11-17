"""
Test Phase 2 (Conditionals + Loops + Patterns) on 200 Tasks
==========================================================

Compare Phase 1 (depth=2) vs Phase 2 (depth=3)
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict
from collections import defaultdict
import time

from arc_active_inference_solver import ARCTask, Grid
from arc_program_solver import ARCProgramSolver
from arc_loader import ARCDataLoader


def load_evaluation_tasks(data_dir: str, num_tasks: int = 200) -> Dict[str, ARCTask]:
    """Load evaluation tasks"""
    eval_dir = Path(data_dir) / "evaluation"
    json_files = sorted(eval_dir.glob("*.json"))

    print(f"Loading first {num_tasks} tasks...")

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
    """Test solver on evaluation tasks"""

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
            results['task_results'][task_id] = {'success': False, 'error': str(e)}
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


def print_comparison(phase1_results: Dict, phase2_results: Dict):
    """Print side-by-side comparison"""

    total = phase1_results['total_tasks']

    print("\n" + "=" * 80)
    print("COMPARISON: Phase 1 (depth=2) vs Phase 2 (depth=3)")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Phase 1':>15} {'Phase 2':>15} {'Change':>15}")
    print("-" * 80)

    # Success rates
    phase1_success = phase1_results['exact_match_either']
    phase2_success = phase2_results['exact_match_either']
    delta_success = phase2_success - phase1_success

    print(f"{'Exact Match (either)':<30} "
          f"{phase1_success:>4d} ({100*phase1_success/total:>5.1f}%) "
          f"{phase2_success:>4d} ({100*phase2_success/total:>5.1f}%) "
          f"{delta_success:>+4d} ({100*delta_success/total:>+5.1f}%)")

    # Attempt 1
    phase1_att1 = phase1_results['exact_match_attempt_1']
    phase2_att1 = phase2_results['exact_match_attempt_1']
    delta_att1 = phase2_att1 - phase1_att1

    print(f"{'  Attempt 1':<30} "
          f"{phase1_att1:>4d} ({100*phase1_att1/total:>5.1f}%) "
          f"{phase2_att1:>4d} ({100*phase2_att1/total:>5.1f}%) "
          f"{delta_att1:>+4d} ({100*delta_att1/total:>+5.1f}%)")

    # Attempt 2
    phase1_att2 = phase1_results['exact_match_attempt_2']
    phase2_att2 = phase2_results['exact_match_attempt_2']
    delta_att2 = phase2_att2 - phase1_att2

    print(f"{'  Attempt 2':<30} "
          f"{phase1_att2:>4d} ({100*phase1_att2/total:>5.1f}%) "
          f"{phase2_att2:>4d} ({100*phase2_att2/total:>5.1f}%) "
          f"{delta_att2:>+4d} ({100*delta_att2/total:>+5.1f}%)")

    # Diversity
    phase1_ident = phase1_results['identical_predictions']
    phase2_ident = phase2_results['identical_predictions']
    delta_ident = phase2_ident - phase1_ident

    print(f"{'Identical Predictions':<30} "
          f"{phase1_ident:>4d} ({100*phase1_ident/total:>5.1f}%) "
          f"{phase2_ident:>4d} ({100*phase2_ident/total:>5.1f}%) "
          f"{delta_ident:>+4d} ({100*delta_ident/total:>+5.1f}%)")

    # Timing
    phase1_time = np.mean(phase1_results['timing']) if phase1_results['timing'] else 0
    phase2_time = np.mean(phase2_results['timing']) if phase2_results['timing'] else 0
    delta_time = phase2_time - phase1_time

    print(f"{'Avg Time (seconds)':<30} "
          f"{phase1_time:>14.3f}s "
          f"{phase2_time:>14.3f}s "
          f"{delta_time:>+14.3f}s")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    improvement = phase2_success - phase1_success
    if phase1_success > 0:
        improvement_pct = 100 * improvement / phase1_success
    else:
        improvement_pct = 100 * phase2_success if phase2_success > 0 else 0

    print(f"\nPhase 1 Success:  {phase1_success}/{total} ({100*phase1_success/total:.1f}%)")
    print(f"Phase 2 Success:  {phase2_success}/{total} ({100*phase2_success/total:.1f}%)")
    print(f"Improvement:      {improvement:+d} tasks ({improvement_pct:+.1f}%)")

    if improvement > 0:
        print(f"\n✓ Phase 2 IMPROVED performance by {improvement} tasks!")
    elif improvement == 0:
        print(f"\n⊙ Phase 2 performed the SAME as Phase 1")
    else:
        print(f"\n✗ Phase 2 DECREASED performance by {abs(improvement)} tasks")

    return improvement


def main():
    """Main test execution"""

    print("\n" + "=" * 80)
    print("PHASE 2 EVALUATION: Conditionals + Loops + Patterns")
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

    # Test Phase 1 (depth=2)
    print("\n[1/2] Testing PHASE 1 (depth=2)...")
    phase1_solver = ARCProgramSolver(
        workspace_capacity=20,
        n_perturbations=5,
        max_synthesis_depth=2,
        max_programs=100,
        verbose=False
    )
    phase1_results = test_solver(phase1_solver, "Phase 1 (depth=2)", tasks)

    # Test Phase 2 (depth=3)
    print("\n[2/2] Testing PHASE 2 (depth=3 with conditionals/loops)...")
    phase2_solver = ARCProgramSolver(
        workspace_capacity=20,
        n_perturbations=5,
        max_synthesis_depth=3,
        max_programs=150,
        verbose=False
    )
    phase2_results = test_solver(phase2_solver, "Phase 2 (depth=3)", tasks)

    # Print comparison
    print_comparison(phase1_results, phase2_results)

    # Save results
    output = {
        'phase1': {
            'depth': 2,
            'total': phase1_results['total_tasks'],
            'success': phase1_results['exact_match_either'],
            'attempt_1': phase1_results['exact_match_attempt_1'],
            'attempt_2': phase1_results['exact_match_attempt_2'],
            'identical': phase1_results['identical_predictions'],
            'avg_time': float(np.mean(phase1_results['timing'])),
        },
        'phase2': {
            'depth': 3,
            'total': phase2_results['total_tasks'],
            'success': phase2_results['exact_match_either'],
            'attempt_1': phase2_results['exact_match_attempt_1'],
            'attempt_2': phase2_results['exact_match_attempt_2'],
            'identical': phase2_results['identical_predictions'],
            'avg_time': float(np.mean(phase2_results['timing'])),
        },
    }

    with open('phase2_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: phase2_comparison.json")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
