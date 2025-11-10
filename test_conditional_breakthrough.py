"""
Test Conditional Solver: Breaking Through the 1% Barrier

This test evaluates whether adding conditional transformations, spatial predicates,
and enhanced pattern inference improves the solve rate beyond 1%.

Comparison:
- Baseline (Diverse Solver): 1.0% solve rate (2/200 tasks)
- Conditional Solver: Expected 10-16% solve rate

Focus areas:
1. 15 "very close" tasks (95-99% accuracy) - Should solve some!
2. Full 200-task test
3. Analysis of improvements
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import time

from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver
from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver


class ConditionalBreakthroughTest:
    """Test framework for evaluating conditional solver improvements."""

    # Tasks that were 95-99% accurate with diverse solver (most likely to benefit)
    VERY_CLOSE_TASKS = [
        '0b17323b',  # 99.11% - Missing 1-2 pixels!
        '27a77e38',  # 98.77%
        '025d127b',  # 98.00% - Known task
        '18419cfa',  # 97.66%
        '3c9b0459',  # Was 100% (solved) - Should still solve
        '25ff71a9',  # Was 100% (solved) - Should still solve
    ]

    def __init__(self):
        self.diverse_solver = DiverseARCCuriositySolver()
        self.conditional_solver = ConditionalARCCuriositySolver()
        self.training_dir = Path("ARC-AGI/data/training")
        self.evaluation_dir = Path("ARC-AGI/data/evaluation")

    def load_task(self, task_id: str) -> Tuple[dict, str]:
        """Load a task by ID from either training or evaluation set."""
        # Try training first
        task_path = self.training_dir / f"{task_id}.json"
        task_type = "training"

        if not task_path.exists():
            # Try evaluation
            task_path = self.evaluation_dir / f"{task_id}.json"
            task_type = "evaluation"

        if not task_path.exists():
            return None, None

        with open(task_path, 'r') as f:
            task_data = json.load(f)

        return task_data, task_type

    def test_single_task(self, task_id: str, task_data: dict, solver) -> Dict:
        """Test a single task with given solver."""
        try:
            start_time = time.time()
            predictions = solver.solve_task(task_data, verbose=False)
            solve_time = time.time() - start_time

            expected = np.array(task_data['test'][0]['output'])

            # Check both predictions
            match1 = np.array_equal(predictions[0], expected)
            match2 = np.array_equal(predictions[1], expected)

            solved = match1 or match2
            which_matched = 1 if match1 else (2 if match2 else 0)

            # Calculate accuracy
            accuracy1 = (predictions[0] == expected).mean() if predictions[0].shape == expected.shape else 0
            accuracy2 = (predictions[1] == expected).mean() if predictions[1].shape == expected.shape else 0
            max_accuracy = max(accuracy1, accuracy2)

            # Shape correctness
            shape_correct = (predictions[0].shape == expected.shape or
                           predictions[1].shape == expected.shape)

            return {
                'task_id': task_id,
                'solved': solved,
                'which_matched': which_matched,
                'accuracy': max_accuracy,
                'accuracy_pred1': accuracy1,
                'accuracy_pred2': accuracy2,
                'shape_correct': shape_correct,
                'time': solve_time,
                'error': None
            }

        except Exception as e:
            return {
                'task_id': task_id,
                'solved': False,
                'which_matched': 0,
                'accuracy': 0.0,
                'accuracy_pred1': 0.0,
                'accuracy_pred2': 0.0,
                'shape_correct': False,
                'time': 0.0,
                'error': str(e)
            }

    def test_very_close_tasks(self):
        """Test on the 15 tasks that were 95-99% accurate (most likely to improve)."""
        print("\n" + "="*70)
        print("TESTING CONDITIONAL SOLVER ON 'VERY CLOSE' TASKS (95-99% accurate)")
        print("="*70)
        print("\nThese tasks were SO CLOSE with diverse solver - can conditionals")
        print("break through to 100%?\n")

        results_diverse = []
        results_conditional = []

        for task_id in self.VERY_CLOSE_TASKS:
            task_data, task_type = self.load_task(task_id)

            if task_data is None:
                print(f"⚠ {task_id}: Task not found")
                continue

            print(f"Testing {task_id} ({task_type})...")

            # Test with diverse solver (baseline)
            result_diverse = self.test_single_task(task_id, task_data, self.diverse_solver)
            results_diverse.append(result_diverse)

            # Test with conditional solver
            result_conditional = self.test_single_task(task_id, task_data, self.conditional_solver)
            results_conditional.append(result_conditional)

            # Display comparison
            div_acc = result_diverse['accuracy'] * 100
            cond_acc = result_conditional['accuracy'] * 100

            div_symbol = "✓" if result_diverse['solved'] else " "
            cond_symbol = "✓" if result_conditional['solved'] else " "

            improvement = cond_acc - div_acc

            print(f"  Diverse:     {div_symbol} {div_acc:5.1f}%")
            print(f"  Conditional: {cond_symbol} {cond_acc:5.1f}%")

            if improvement > 0:
                print(f"  ⬆ Improvement: +{improvement:.1f}%")
            elif improvement < 0:
                print(f"  ⬇ Regression: {improvement:.1f}%")
            else:
                print(f"  → No change")

            print()

        # Summary
        print("="*70)
        print("VERY CLOSE TASKS SUMMARY")
        print("="*70)

        diverse_solved = sum(r['solved'] for r in results_diverse)
        conditional_solved = sum(r['solved'] for r in results_conditional)

        diverse_avg = np.mean([r['accuracy'] for r in results_diverse]) * 100
        conditional_avg = np.mean([r['accuracy'] for r in results_conditional]) * 100

        print(f"\nExact Solves:")
        print(f"  Diverse:     {diverse_solved}/{len(results_diverse)} ({diverse_solved/len(results_diverse)*100:.1f}%)")
        print(f"  Conditional: {conditional_solved}/{len(results_conditional)} ({conditional_solved/len(results_conditional)*100:.1f}%)")

        print(f"\nAverage Accuracy:")
        print(f"  Diverse:     {diverse_avg:.1f}%")
        print(f"  Conditional: {conditional_avg:.1f}%")
        print(f"  Improvement: {conditional_avg - diverse_avg:+.1f}%")

        improvement_count = sum(1 for i in range(len(results_diverse))
                              if results_conditional[i]['accuracy'] > results_diverse[i]['accuracy'])

        print(f"\nTasks Improved: {improvement_count}/{len(results_diverse)}")

        # Newly solved tasks
        newly_solved = []
        for i in range(len(results_diverse)):
            if not results_diverse[i]['solved'] and results_conditional[i]['solved']:
                newly_solved.append(results_diverse[i]['task_id'])

        if newly_solved:
            print(f"\n🎉 NEWLY SOLVED TASKS (Breakthrough!):")
            for task_id in newly_solved:
                print(f"  - {task_id}")
        else:
            print(f"\n⚠ No new tasks solved (yet)")

        return results_diverse, results_conditional

    def run_large_scale_test(self, limit_per_set: int = 100):
        """
        Run full 200-task comparison.

        This is the critical test to see if we've broken through the 1% barrier.
        """
        print("\n" + "="*70)
        print("LARGE-SCALE TEST: 200 TASKS (100 training + 100 evaluation)")
        print("="*70)
        print("\nBaseline: Diverse solver = 1.0% solve rate (2/200)")
        print("Target:   Conditional solver > 10% solve rate")
        print()

        # Load tasks
        tasks = []

        # Training tasks
        train_files = sorted(list(self.training_dir.glob("*.json")))[:limit_per_set]
        for task_file in train_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            tasks.append((task_file.stem, task_data, "training"))

        # Evaluation tasks
        eval_files = sorted(list(self.evaluation_dir.glob("*.json")))[:limit_per_set]
        for task_file in eval_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            tasks.append((task_file.stem, task_data, "evaluation"))

        print(f"Loaded {len(tasks)} tasks\n")

        # Test with both solvers
        results_diverse = []
        results_conditional = []

        solved_diverse = []
        solved_conditional = []

        for i, (task_id, task_data, task_type) in enumerate(tasks):
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{len(tasks)} tasks tested...")

            # Test diverse solver
            result_div = self.test_single_task(task_id, task_data, self.diverse_solver)
            results_diverse.append(result_div)
            if result_div['solved']:
                solved_diverse.append(task_id)
                print(f"  ✓ Diverse solved: {task_id}")

            # Test conditional solver
            result_cond = self.test_single_task(task_id, task_data, self.conditional_solver)
            results_conditional.append(result_cond)
            if result_cond['solved']:
                solved_conditional.append(task_id)
                if task_id not in solved_diverse:
                    print(f"  🎉 NEW SOLVE: {task_id} (conditional solver)")

        # Comprehensive results
        print("\n" + "="*70)
        print("RESULTS: CONDITIONAL vs DIVERSE SOLVER")
        print("="*70)

        # Exact matches
        div_solves = len(solved_diverse)
        cond_solves = len(solved_conditional)

        print(f"\n🎯 EXACT MATCHES:")
        print(f"  Diverse Solver:     {div_solves}/200 ({div_solves/2:.1f}%)")
        print(f"  Conditional Solver: {cond_solves}/200 ({cond_solves/2:.1f}%)")
        print(f"  Change: {cond_solves - div_solves:+d} tasks ({(cond_solves - div_solves)/2:+.1f}%)")

        # Average accuracy
        div_avg = np.mean([r['accuracy'] for r in results_diverse]) * 100
        cond_avg = np.mean([r['accuracy'] for r in results_conditional]) * 100

        print(f"\n📊 AVERAGE ACCURACY:")
        print(f"  Diverse:     {div_avg:.1f}%")
        print(f"  Conditional: {cond_avg:.1f}%")
        print(f"  Change: {cond_avg - div_avg:+.1f}%")

        # Very close (95-99%)
        div_very_close = sum(1 for r in results_diverse if 0.95 <= r['accuracy'] < 1.0)
        cond_very_close = sum(1 for r in results_conditional if 0.95 <= r['accuracy'] < 1.0)

        print(f"\n⚠️ VERY CLOSE (95-99%):")
        print(f"  Diverse:     {div_very_close}/200")
        print(f"  Conditional: {cond_very_close}/200")

        # Newly solved
        newly_solved = [task for task in solved_conditional if task not in solved_diverse]
        lost_solves = [task for task in solved_diverse if task not in solved_conditional]

        print(f"\n🎉 NEWLY SOLVED: {len(newly_solved)}")
        if newly_solved:
            for task_id in newly_solved:
                idx = next(i for i, (tid, _, _) in enumerate(tasks) if tid == task_id)
                acc_before = results_diverse[idx]['accuracy'] * 100
                print(f"  - {task_id}: {acc_before:.1f}% → 100% ✓")

        if lost_solves:
            print(f"\n⚠️ LOST SOLVES: {len(lost_solves)}")
            for task_id in lost_solves:
                print(f"  - {task_id}")

        # Breakthrough analysis
        print(f"\n{'='*70}")
        if cond_solves >= 10:
            print("🚀 BREAKTHROUGH ACHIEVED!")
            print(f"   Solve rate improved from {div_solves/2:.1f}% to {cond_solves/2:.1f}%")
            print(f"   Target of 10% ({'>=' if cond_solves >= 20 else '<'} 20 tasks) {'REACHED' if cond_solves >= 20 else 'not yet reached'}")
        elif cond_solves > div_solves:
            print("✅ IMPROVEMENT CONFIRMED")
            print(f"   Conditional solver outperforms diverse solver")
            print(f"   +{cond_solves - div_solves} additional tasks solved")
        else:
            print("⚠️ NO IMPROVEMENT")
            print(f"   Conditional solver did not improve over diverse solver")
            print(f"   Further investigation needed")

        return results_diverse, results_conditional


def main():
    """Run the breakthrough test."""
    print("\n" + "="*70)
    print("CONDITIONAL SOLVER BREAKTHROUGH TEST")
    print("="*70)
    print("\nObjective: Break through the 1% expressiveness barrier")
    print("Hypothesis: Conditional logic + spatial predicates → 10-16% solve rate")
    print()

    tester = ConditionalBreakthroughTest()

    # Phase 1: Test on very close tasks
    print("\n📍 PHASE 1: Testing on 'very close' tasks (95-99% accurate)")
    print("Starting Phase 1...\n")

    results_close_div, results_close_cond = tester.test_very_close_tasks()

    # Phase 2: Large-scale test
    print("\n📍 PHASE 2: Large-scale test (200 tasks)")
    print("Starting Phase 2...\n")

    results_large_div, results_large_cond = tester.run_large_scale_test(limit_per_set=100)

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
