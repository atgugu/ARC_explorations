"""
Direct comparison test using proper solver API

This test directly compares the diverse and conditional solvers
on a sample of tasks to measure actual breakthrough.
"""

import json
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver
from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver


def test_solver_on_task(solver, task_data):
    """Test a solver on a task, returning results."""
    try:
        # Get test case
        test_input = np.array(task_data['test'][0]['input'])
        expected_output = np.array(task_data['test'][0]['output'])

        # Build training pairs
        train_pairs = []
        for example in task_data.get('train', []):
            inp = np.array(example['input'])
            out = np.array(example['output'])
            train_pairs.append((inp, out))

        # Generate hypotheses
        solver.verbose = False
        hypotheses = solver._generate_hypotheses(train_pairs, test_input)

        if not hypotheses:
            return {
                'solved': False,
                'accuracy': 0.0,
                'hypothesis_count': 0,
                'error': None
            }

        # Evaluate top 2 hypotheses
        best_accuracy = 0.0
        solved = False

        for h in hypotheses[:2]:
            try:
                # Apply transformation
                prediction = h.program.function(test_input.copy())

                # Check if solved
                if np.array_equal(prediction, expected_output):
                    solved = True
                    best_accuracy = 1.0
                    break

                # Calculate accuracy
                if prediction.shape == expected_output.shape:
                    accuracy = (prediction == expected_output).mean()
                    best_accuracy = max(best_accuracy, accuracy)
            except Exception as e:
                continue

        return {
            'solved': solved,
            'accuracy': best_accuracy,
            'hypothesis_count': len(hypotheses),
            'error': None
        }

    except Exception as e:
        return {
            'solved': False,
            'accuracy': 0.0,
            'hypothesis_count': 0,
            'error': str(e)
        }


def run_comparison_test(num_tasks=50):
    """Run comparison test on first N tasks."""

    print("\n" + "="*70)
    print("CONDITIONAL SOLVER: DIRECT COMPARISON TEST")
    print("="*70)
    print(f"\nTesting {num_tasks} tasks from training set")
    print("Measuring actual solve rate improvement\n")

    # Load tasks
    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:num_tasks]

    print(f"Loaded {len(task_files)} tasks\n")

    # Initialize solvers
    diverse_solver = DiverseARCCuriositySolver()
    conditional_solver = ConditionalARCCuriositySolver()

    # Test each task
    diverse_results = []
    conditional_results = []

    diverse_solved = []
    conditional_solved = []
    newly_solved = []

    print("Testing tasks...")
    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Test diverse solver
        result_div = test_solver_on_task(diverse_solver, task_data)
        result_div['task_id'] = task_id
        diverse_results.append(result_div)

        # Test conditional solver
        result_cond = test_solver_on_task(conditional_solver, task_data)
        result_cond['task_id'] = task_id
        conditional_results.append(result_cond)

        # Track solved tasks
        if result_div['solved']:
            diverse_solved.append(task_id)

        if result_cond['solved']:
            conditional_solved.append(task_id)
            if task_id not in diverse_solved:
                newly_solved.append(task_id)

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(task_files)}")

    print("\nTesting complete!\n")

    # === ANALYSIS ===
    print("="*70)
    print("RESULTS")
    print("="*70)

    # Solve rates
    div_solve_rate = len(diverse_solved) / len(task_files) * 100
    cond_solve_rate = len(conditional_solved) / len(task_files) * 100

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Diverse Solver:     {len(diverse_solved)}/{len(task_files)} ({div_solve_rate:.1f}%)")
    print(f"  Conditional Solver: {len(conditional_solved)}/{len(task_files)} ({cond_solve_rate:.1f}%)")
    print(f"  Change: {len(conditional_solved) - len(diverse_solved):+d} tasks ({cond_solve_rate - div_solve_rate:+.1f}%)")

    # Average accuracy
    div_avg_acc = np.mean([r['accuracy'] for r in diverse_results]) * 100
    cond_avg_acc = np.mean([r['accuracy'] for r in conditional_results]) * 100

    print(f"\n📊 AVERAGE ACCURACY:")
    print(f"  Diverse:     {div_avg_acc:.1f}%")
    print(f"  Conditional: {cond_avg_acc:.1f}%")
    print(f"  Change: {cond_avg_acc - div_avg_acc:+.1f}%")

    # Hypothesis generation
    div_avg_hyps = np.mean([r['hypothesis_count'] for r in diverse_results])
    cond_avg_hyps = np.mean([r['hypothesis_count'] for r in conditional_results])

    print(f"\n🔍 HYPOTHESIS GENERATION:")
    print(f"  Diverse:     {div_avg_hyps:.1f} hypotheses/task")
    print(f"  Conditional: {cond_avg_hyps:.1f} hypotheses/task")

    # Improvements
    improved = sum(1 for i in range(len(task_files))
                  if conditional_results[i]['accuracy'] > diverse_results[i]['accuracy'])

    regressed = sum(1 for i in range(len(task_files))
                   if conditional_results[i]['accuracy'] < diverse_results[i]['accuracy'])

    print(f"\n📈 TASK-LEVEL CHANGES:")
    print(f"  Improved:    {improved}/{len(task_files)} ({improved/len(task_files)*100:.1f}%)")
    print(f"  Regressed:   {regressed}/{len(task_files)} ({regressed/len(task_files)*100:.1f}%)")
    print(f"  Unchanged:   {len(task_files) - improved - regressed}/{len(task_files)}")

    # Solved tasks
    if diverse_solved:
        print(f"\n✅ DIVERSE SOLVER SOLVED:")
        for task_id in diverse_solved[:10]:
            print(f"  - {task_id}")
        if len(diverse_solved) > 10:
            print(f"  ... and {len(diverse_solved) - 10} more")

    if conditional_solved:
        print(f"\n✅ CONDITIONAL SOLVER SOLVED:")
        for task_id in conditional_solved[:10]:
            marker = "🎉 NEW" if task_id in newly_solved else "   "
            print(f"  {marker} {task_id}")
        if len(conditional_solved) > 10:
            print(f"  ... and {len(conditional_solved) - 10} more")

    if newly_solved:
        print(f"\n🎉 NEWLY SOLVED BY CONDITIONAL SOLVER:")
        for task_id in newly_solved:
            div_acc = next(r['accuracy'] for r in diverse_results if r['task_id'] == task_id)
            print(f"  - {task_id}: {div_acc*100:.1f}% → 100% ✓")
    else:
        print(f"\n⚠️  No new tasks solved by conditional solver")

    # Error analysis
    div_errors = sum(1 for r in diverse_results if r['error'])
    cond_errors = sum(1 for r in conditional_results if r['error'])

    if div_errors or cond_errors:
        print(f"\n⚠️  ERRORS:")
        print(f"  Diverse:     {div_errors}/{len(task_files)}")
        print(f"  Conditional: {cond_errors}/{len(task_files)}")

    # Detailed breakdown of top improvements
    improvements = []
    for i in range(len(task_files)):
        diff = conditional_results[i]['accuracy'] - diverse_results[i]['accuracy']
        if diff > 0.05:  # More than 5% improvement
            improvements.append({
                'task_id': conditional_results[i]['task_id'],
                'before': diverse_results[i]['accuracy'] * 100,
                'after': conditional_results[i]['accuracy'] * 100,
                'improvement': diff * 100
            })

    if improvements:
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        print(f"\n📈 TOP IMPROVEMENTS (>5%):")
        for imp in improvements[:10]:
            print(f"  {imp['task_id']}: {imp['before']:.1f}% → {imp['after']:.1f}% (+{imp['improvement']:.1f}%)")

    # Overall assessment
    print(f"\n{'='*70}")
    print("ASSESSMENT")
    print("="*70)

    if cond_solve_rate > div_solve_rate + 5:
        print(f"✅ BREAKTHROUGH ACHIEVED!")
        print(f"   Conditional solver significantly outperforms diverse solver")
        print(f"   +{len(newly_solved)} tasks solved ({cond_solve_rate - div_solve_rate:+.1f}% improvement)")
    elif cond_solve_rate > div_solve_rate:
        print(f"✅ IMPROVEMENT CONFIRMED")
        print(f"   Conditional solver shows modest improvement")
        print(f"   +{len(newly_solved)} tasks solved")
    elif cond_avg_acc > div_avg_acc + 3:
        print(f"📊 ACCURACY IMPROVEMENT")
        print(f"   No new exact solves, but average accuracy improved")
        print(f"   Conditional logic helps partial solutions")
    else:
        print(f"⚠️  LIMITED IMPACT")
        print(f"   Conditional solver shows minimal improvement")
        print(f"   May need more sophisticated pattern detection")

    print(f"\n{'='*70}")

    return diverse_results, conditional_results


if __name__ == '__main__':
    # Run on first 50 tasks
    run_comparison_test(num_tasks=50)
