"""
Test ARC Active Inference Solver on 200 ARC-AGI Evaluation Tasks
================================================================

Comprehensive testing on official ARC-AGI evaluation dataset
Focus: Exact match (100% pixel accuracy) performance analysis
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import time

from arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid
from arc_loader import ARCDataLoader, ARCEvaluator


def load_evaluation_tasks(data_dir: str, num_tasks: int = 200) -> Dict[str, ARCTask]:
    """
    Load evaluation tasks from JSON files

    Args:
        data_dir: Directory containing evaluation JSON files
        num_tasks: Number of tasks to load (default 200)

    Returns:
        Dictionary mapping task IDs to ARCTask objects
    """
    eval_dir = Path(data_dir) / "evaluation"

    if not eval_dir.exists():
        raise ValueError(f"Evaluation directory not found: {eval_dir}")

    # Get all JSON files and sort them
    json_files = sorted(eval_dir.glob("*.json"))

    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {eval_dir}")

    print(f"Found {len(json_files)} evaluation tasks")
    print(f"Loading first {num_tasks} tasks...\n")

    tasks = {}
    for json_file in json_files[:num_tasks]:
        task_id = json_file.stem  # Filename without .json

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


def analyze_task_properties(tasks: Dict[str, ARCTask]) -> Dict:
    """Analyze properties of the task set"""

    properties = {
        'total': len(tasks),
        'train_sizes': [],
        'grid_sizes': [],
        'color_counts': [],
        'size_categories': defaultdict(int),
    }

    for task_id, task in tasks.items():
        # Training examples count
        properties['train_sizes'].append(len(task.train_pairs))

        # Grid sizes
        for inp, out in task.train_pairs:
            properties['grid_sizes'].append(inp.shape)
            properties['grid_sizes'].append(out.shape)

            # Categorize by size
            if inp.shape[0] <= 5 and inp.shape[1] <= 5:
                properties['size_categories']['tiny'] += 1
            elif inp.shape[0] <= 10 and inp.shape[1] <= 10:
                properties['size_categories']['small'] += 1
            elif inp.shape[0] <= 20 and inp.shape[1] <= 20:
                properties['size_categories']['medium'] += 1
            else:
                properties['size_categories']['large'] += 1

            # Color count
            unique_colors = len(np.unique(inp.data))
            properties['color_counts'].append(unique_colors)

    return properties


def test_solver_on_evaluation(
    solver: ARCActiveInferenceSolver,
    tasks: Dict[str, ARCTask],
    verbose: bool = False
) -> Dict:
    """
    Test solver on evaluation tasks

    Args:
        solver: The solver instance
        tasks: Dictionary of tasks to test
        verbose: Print detailed output for each task

    Returns:
        Dictionary with detailed results
    """
    results = {
        'total_tasks': len(tasks),
        'exact_match_attempt_1': 0,
        'exact_match_attempt_2': 0,
        'exact_match_either': 0,
        'both_wrong': 0,
        'identical_predictions': 0,
        'task_results': {},
        'failure_modes': defaultdict(int),
        'success_train_sizes': [],
        'failure_train_sizes': [],
        'timing': [],
    }

    print("=" * 80)
    print("TESTING ARC-AGI SOLVER ON 200 EVALUATION TASKS")
    print("=" * 80)
    print(f"\nTesting {len(tasks)} tasks with exact match criteria")
    print("Each task gets 2 attempts (as in competition)\n")

    task_counter = 0

    for task_id, task in sorted(tasks.items()):
        task_counter += 1

        # Run solver
        start_time = time.time()
        try:
            predictions = solver.solve(task, verbose=False)
        except Exception as e:
            print(f"✗  Task {task_counter:3d} {task_id}: SOLVER ERROR - {e}")
            results['task_results'][task_id] = {
                'success': False,
                'error': str(e)
            }
            results['both_wrong'] += 1
            continue

        elapsed = time.time() - start_time
        results['timing'].append(elapsed)

        # Evaluate predictions
        if task.test_output is None:
            print(f"⊘  Task {task_counter:3d} {task_id}: No ground truth")
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
            results['success_train_sizes'].append(len(task.train_pairs))
            success_str = "✓1"
        elif match_2:
            results['exact_match_attempt_2'] += 1
            results['exact_match_either'] += 1
            results['success_train_sizes'].append(len(task.train_pairs))
            success_str = "✓2"
        else:
            results['both_wrong'] += 1
            results['failure_train_sizes'].append(len(task.train_pairs))
            success_str = "✗ "

            # Analyze failure mode
            if pred_1.shape != gt.shape:
                results['failure_modes']['size_mismatch'] += 1
            else:
                results['failure_modes']['wrong_transform'] += 1

        if identical:
            results['identical_predictions'] += 1

        # Record task result
        results['task_results'][task_id] = {
            'success': match_1 or match_2,
            'match_1': match_1,
            'match_2': match_2,
            'identical': identical,
            'train_size': len(task.train_pairs),
            'test_shape': gt.shape,
            'pred_1_shape': pred_1.shape,
            'pred_2_shape': pred_2.shape,
            'elapsed': elapsed,
        }

        # Print progress
        if verbose or task_counter % 10 == 0:
            print(f"{success_str} Task {task_counter:3d}/{len(tasks)} {task_id} "
                  f"(train={len(task.train_pairs)}, {elapsed:.2f}s)")

    return results


def print_results_summary(results: Dict, properties: Dict):
    """Print comprehensive results summary"""

    total = results['total_tasks']
    match_1 = results['exact_match_attempt_1']
    match_2 = results['exact_match_attempt_2']
    match_either = results['exact_match_either']
    both_wrong = results['both_wrong']
    identical = results['identical_predictions']

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nTotal Tasks:              {total}")
    print(f"Exact Match (either):     {match_either:3d} ({100*match_either/total:.1f}%)")
    print(f"  - Attempt 1 correct:    {match_1:3d} ({100*match_1/total:.1f}%)")
    print(f"  - Attempt 2 correct:    {match_2:3d} ({100*match_2/total:.1f}%)")
    print(f"Both Wrong:               {both_wrong:3d} ({100*both_wrong/total:.1f}%)")
    print(f"Predictions Same:         {identical:3d} ({100*identical/total:.1f}%)")

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    # Success rate by attempt
    print(f"\n✓ Success Rate:")
    print(f"  Overall (either):  {100*match_either/total:.1f}%")
    print(f"  Attempt 1 only:    {100*match_1/total:.1f}%")
    print(f"  Attempt 2 only:    {100*match_2/total:.1f}%")
    print(f"  Attempt 2 value:   {100*match_2/total:.1f}% (additional tasks solved by 2nd attempt)")

    # Timing
    if len(results['timing']) > 0:
        avg_time = np.mean(results['timing'])
        median_time = np.median(results['timing'])
        max_time = np.max(results['timing'])
        print(f"\n⏱ Timing:")
        print(f"  Average:  {avg_time:.2f}s per task")
        print(f"  Median:   {median_time:.2f}s per task")
        print(f"  Max:      {max_time:.2f}s per task")
        print(f"  Total:    {sum(results['timing'])/60:.1f} minutes")

    # Diversity
    print(f"\n🎲 Prediction Diversity:")
    diverse = total - identical
    print(f"  Different predictions:  {diverse}/{total} ({100*diverse/total:.1f}%)")
    print(f"  Identical predictions:  {identical}/{total} ({100*identical/total:.1f}%)")

    if identical == 0:
        print(f"  ✓ Perfect diversity achieved!")

    # Failure analysis
    if len(results['failure_modes']) > 0:
        print("\n" + "=" * 80)
        print("FAILURE ANALYSIS")
        print("=" * 80)

        print(f"\n❌ Failure Modes (out of {both_wrong} failed tasks):")
        for mode, count in sorted(results['failure_modes'].items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"  {mode:20s}: {count:3d} ({100*count/both_wrong:.1f}%)")

    # Training size analysis
    if len(results['success_train_sizes']) > 0 and len(results['failure_train_sizes']) > 0:
        print("\n" + "=" * 80)
        print("TRAINING SIZE ANALYSIS")
        print("=" * 80)

        avg_success = np.mean(results['success_train_sizes'])
        avg_failure = np.mean(results['failure_train_sizes'])

        print(f"\n📊 Average training examples:")
        print(f"  Successful tasks: {avg_success:.1f}")
        print(f"  Failed tasks:     {avg_failure:.1f}")
        print(f"  Difference:       {avg_success - avg_failure:+.1f}")

    # Dataset properties
    print("\n" + "=" * 80)
    print("DATASET PROPERTIES")
    print("=" * 80)

    print(f"\nTask Set Size: {properties['total']}")

    if len(properties['train_sizes']) > 0:
        print(f"\nTraining Examples per Task:")
        print(f"  Min:     {np.min(properties['train_sizes'])}")
        print(f"  Max:     {np.max(properties['train_sizes'])}")
        print(f"  Average: {np.mean(properties['train_sizes']):.1f}")
        print(f"  Median:  {np.median(properties['train_sizes']):.0f}")

    if len(properties['size_categories']) > 0:
        print(f"\nGrid Size Distribution:")
        for category in ['tiny', 'small', 'medium', 'large']:
            count = properties['size_categories'].get(category, 0)
            print(f"  {category:10s}: {count}")


def save_detailed_results(results: Dict, output_file: str):
    """Save detailed results to JSON"""

    # Convert numpy types to native Python for JSON serialization
    serializable_results = {}

    for task_id, task_result in results['task_results'].items():
        serializable_results[task_id] = {
            k: (int(v) if isinstance(v, (np.integer, np.bool_)) else
                float(v) if isinstance(v, np.floating) else
                tuple(v) if isinstance(v, tuple) else v)
            for k, v in task_result.items()
        }

    output_data = {
        'summary': {
            'total_tasks': results['total_tasks'],
            'exact_match_attempt_1': results['exact_match_attempt_1'],
            'exact_match_attempt_2': results['exact_match_attempt_2'],
            'exact_match_either': results['exact_match_either'],
            'both_wrong': results['both_wrong'],
            'identical_predictions': results['identical_predictions'],
            'success_rate': 100 * results['exact_match_either'] / results['total_tasks'],
            'diversity_rate': 100 * (results['total_tasks'] - results['identical_predictions']) / results['total_tasks'],
            'average_time': float(np.mean(results['timing'])) if results['timing'] else 0.0,
        },
        'failure_modes': dict(results['failure_modes']),
        'task_results': serializable_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")


def main():
    """Main test execution"""

    print("\n" + "=" * 80)
    print("ARC-AGI EVALUATION: 200 TASKS")
    print("=" * 80)
    print("\nInitializing solver...")

    # Initialize solver
    solver = ARCActiveInferenceSolver(
        workspace_capacity=20,
        n_perturbations=5
    )

    print("✓ Solver initialized")
    print(f"  Workspace capacity: {solver.workspace.capacity}")
    print(f"  Perturbations: {solver.stability_filter.n_perturbations}")

    # Load tasks
    print("\nLoading evaluation tasks...")
    data_dir = "../data"  # Relative to unified_solver/

    if not os.path.exists(data_dir):
        data_dir = "data"  # Try from repo root

    try:
        tasks = load_evaluation_tasks(data_dir, num_tasks=200)
    except Exception as e:
        print(f"\n❌ Error loading tasks: {e}")
        print("\nPlease ensure ARC-AGI evaluation data is in: data/evaluation/")
        return

    # Analyze dataset
    print("Analyzing dataset properties...")
    properties = analyze_task_properties(tasks)

    # Run tests
    print("\nStarting evaluation...\n")
    results = test_solver_on_evaluation(solver, tasks, verbose=False)

    # Print summary
    print_results_summary(results, properties)

    # Save detailed results
    output_file = "evaluation_200_results.json"
    save_detailed_results(results, output_file)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    # Return success rate for scripting
    success_rate = 100 * results['exact_match_either'] / results['total_tasks']
    print(f"\n✓ Final Success Rate: {success_rate:.1f}%\n")

    return results


if __name__ == "__main__":
    results = main()
