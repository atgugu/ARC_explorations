"""
Test Enhanced Solver with Object Reasoning

Compares the enhanced solver (with object reasoning) against the baseline.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import time

sys.path.insert(0, '/home/user/ARC_explorations')

from arc_curiosity_solver.solver import ARCCuriositySolver
from arc_curiosity_solver.solver_enhanced import EnhancedARCCuriositySolver


class SolverComparison:
    """Compare baseline and enhanced solvers."""

    def __init__(self):
        self.baseline_results = []
        self.enhanced_results = []
        self.improvements = []

    def load_task(self, task_file: str) -> Dict[str, Any]:
        """Load ARC task from JSON file."""
        with open(task_file, 'r') as f:
            data = json.load(f)

        task = {
            'id': Path(task_file).stem,
            'train_pairs': [],
            'test_pairs': []
        }

        for example in data['train']:
            inp = np.array(example['input'])
            out = np.array(example['output'])
            task['train_pairs'].append((inp, out))

        for example in data['test']:
            inp = np.array(example['input'])
            out = np.array(example['output']) if 'output' in example else None
            task['test_pairs'].append((inp, out))

        return task

    def test_both_solvers(self, task: Dict, verbose: bool = False) -> Dict[str, Any]:
        """Test both solvers on a task and compare."""
        task_id = task['id']

        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing: {task_id}")
            print(f"{'='*70}")

        results = {
            'task_id': task_id,
            'baseline': {},
            'enhanced': {},
            'improvement': {}
        }

        # Test baseline solver
        if verbose:
            print("\n[Baseline Solver]")

        baseline_solver = ARCCuriositySolver(
            workspace_capacity=7,
            learning_rate=0.1,
            exploration_bonus=1.0,
            n_hypotheses_to_explore=40
        )

        for test_idx, (test_input, test_output) in enumerate(task['test_pairs']):
            if test_output is None:
                continue

            start_time = time.time()
            try:
                pred1, pred2 = baseline_solver.solve(
                    task['train_pairs'],
                    test_input,
                    verbose=False
                )

                acc1 = np.sum(pred1 == test_output) / test_output.size if pred1.shape == test_output.shape else 0.0
                acc2 = np.sum(pred2 == test_output) / test_output.size if pred2.shape == test_output.shape else 0.0

                solved = (acc1 == 1.0) or (acc2 == 1.0)
                best_acc = max(acc1, acc2)

                results['baseline'] = {
                    'solved': solved,
                    'best_accuracy': best_acc,
                    'acc1': acc1,
                    'acc2': acc2,
                    'time': time.time() - start_time
                }

                if verbose:
                    print(f"  Accuracy: {best_acc*100:.1f}% | Solved: {'✓' if solved else '✗'}")

            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                results['baseline'] = {'error': str(e), 'solved': False, 'best_accuracy': 0.0}

        # Test enhanced solver
        if verbose:
            print("\n[Enhanced Solver with Object Reasoning]")

        enhanced_solver = EnhancedARCCuriositySolver(
            workspace_capacity=7,
            learning_rate=0.1,
            exploration_bonus=1.0,
            n_hypotheses_to_explore=40
        )

        for test_idx, (test_input, test_output) in enumerate(task['test_pairs']):
            if test_output is None:
                continue

            start_time = time.time()
            try:
                pred1, pred2 = enhanced_solver.solve(
                    task['train_pairs'],
                    test_input,
                    verbose=False
                )

                acc1 = np.sum(pred1 == test_output) / test_output.size if pred1.shape == test_output.shape else 0.0
                acc2 = np.sum(pred2 == test_output) / test_output.size if pred2.shape == test_output.shape else 0.0

                solved = (acc1 == 1.0) or (acc2 == 1.0)
                best_acc = max(acc1, acc2)

                results['enhanced'] = {
                    'solved': solved,
                    'best_accuracy': best_acc,
                    'acc1': acc1,
                    'acc2': acc2,
                    'used_object_reasoning': enhanced_solver.used_object_reasoning,
                    'time': time.time() - start_time
                }

                if verbose:
                    print(f"  Accuracy: {best_acc*100:.1f}% | Solved: {'✓' if solved else '✗'}")
                    if enhanced_solver.used_object_reasoning:
                        print(f"  Used object reasoning: ✓")

            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                results['enhanced'] = {'error': str(e), 'solved': False, 'best_accuracy': 0.0}

        # Compute improvement
        if 'error' not in results['baseline'] and 'error' not in results['enhanced']:
            baseline_acc = results['baseline']['best_accuracy']
            enhanced_acc = results['enhanced']['best_accuracy']

            improvement = enhanced_acc - baseline_acc

            results['improvement'] = {
                'accuracy_gain': improvement,
                'percentage_gain': (improvement / baseline_acc * 100) if baseline_acc > 0 else 0,
                'newly_solved': results['enhanced']['solved'] and not results['baseline']['solved']
            }

            if verbose:
                print(f"\n[Comparison]")
                print(f"  Improvement: {improvement*100:+.1f}% ({baseline_acc*100:.1f}% → {enhanced_acc*100:.1f}%)")
                if results['improvement']['newly_solved']:
                    print(f"  ✓ NEWLY SOLVED!")

        return results

    def run_comparison(self, task_files: List[str], max_tasks: int = None, verbose: bool = True):
        """Run comparison on multiple tasks."""

        if max_tasks:
            task_files = task_files[:max_tasks]

        print(f"\n{'='*70}")
        print(f"SOLVER COMPARISON: Baseline vs Enhanced (with Object Reasoning)")
        print(f"Total tasks: {len(task_files)}")
        print(f"{'='*70}\n")

        for i, task_file in enumerate(task_files):
            try:
                task = self.load_task(task_file)
                result = self.test_both_solvers(task, verbose=verbose)

                if 'improvement' in result:
                    self.improvements.append(result)

                self.baseline_results.append(result.get('baseline', {}))
                self.enhanced_results.append(result.get('enhanced', {}))

                if (i + 1) % 10 == 0:
                    print(f"\nProgress: {i+1}/{len(task_files)} tasks tested...")

            except Exception as e:
                print(f"ERROR loading {task_file}: {e}")
                continue

    def print_summary(self):
        """Print comprehensive comparison summary."""

        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")

        # Overall statistics
        total = len(self.improvements)

        if total == 0:
            print("No tasks tested.")
            return

        # Baseline stats
        baseline_solved = sum(1 for r in self.baseline_results if r.get('solved', False))
        baseline_avg_acc = np.mean([r.get('best_accuracy', 0.0) for r in self.baseline_results])

        # Enhanced stats
        enhanced_solved = sum(1 for r in self.enhanced_results if r.get('solved', False))
        enhanced_avg_acc = np.mean([r.get('best_accuracy', 0.0) for r in self.enhanced_results])

        # Improvements
        newly_solved = sum(1 for imp in self.improvements if imp['improvement'].get('newly_solved', False))
        avg_improvement = np.mean([imp['improvement']['accuracy_gain'] for imp in self.improvements])

        used_object_reasoning = sum(1 for r in self.enhanced_results
                                    if r.get('used_object_reasoning', False))

        print(f"\n📊 Overall Statistics:")
        print(f"   Total tasks: {total}")
        print(f"\n   Baseline Solver:")
        print(f"     Perfect solves: {baseline_solved}/{total} ({baseline_solved/total*100:.1f}%)")
        print(f"     Average accuracy: {baseline_avg_acc*100:.1f}%")
        print(f"\n   Enhanced Solver:")
        print(f"     Perfect solves: {enhanced_solved}/{total} ({enhanced_solved/total*100:.1f}%)")
        print(f"     Average accuracy: {enhanced_avg_acc*100:.1f}%")
        print(f"     Used object reasoning: {used_object_reasoning}/{total} tasks")

        print(f"\n📈 Improvements:")
        print(f"   Newly solved: {newly_solved} tasks")
        print(f"   Average accuracy gain: {avg_improvement*100:+.1f}%")

        # Categorize improvements
        significant_improvements = [imp for imp in self.improvements
                                   if imp['improvement']['accuracy_gain'] > 0.1]
        moderate_improvements = [imp for imp in self.improvements
                                if 0.01 < imp['improvement']['accuracy_gain'] <= 0.1]
        no_change = [imp for imp in self.improvements
                    if abs(imp['improvement']['accuracy_gain']) <= 0.01]
        regressions = [imp for imp in self.improvements
                      if imp['improvement']['accuracy_gain'] < -0.01]

        print(f"\n   Significant improvements (>10%): {len(significant_improvements)}")
        print(f"   Moderate improvements (1-10%): {len(moderate_improvements)}")
        print(f"   No change: {len(no_change)}")
        print(f"   Regressions: {len(regressions)}")

        # Detailed results
        print(f"\n📋 Detailed Comparison:")
        print(f"{'Task ID':<15} {'Baseline':<12} {'Enhanced':<12} {'Change':<12} {'Status'}")
        print("-" * 70)

        for imp in self.improvements:
            task_id = imp['task_id']
            baseline_acc = imp['baseline']['best_accuracy']
            enhanced_acc = imp['enhanced']['best_accuracy']
            change = imp['improvement']['accuracy_gain']

            baseline_str = f"{baseline_acc*100:.1f}%"
            enhanced_str = f"{enhanced_acc*100:.1f}%"
            change_str = f"{change*100:+.1f}%"

            if imp['improvement']['newly_solved']:
                status = "✓ SOLVED"
            elif change > 0.1:
                status = "⬆ Better"
            elif change > 0.01:
                status = "↗ Improved"
            elif change < -0.01:
                status = "↘ Worse"
            else:
                status = "= Same"

            print(f"{task_id:<15} {baseline_str:<12} {enhanced_str:<12} {change_str:<12} {status}")

        # Highlight newly solved tasks
        if newly_solved > 0:
            print(f"\n🎉 Newly Solved Tasks:")
            for imp in self.improvements:
                if imp['improvement']['newly_solved']:
                    task_id = imp['task_id']
                    baseline = imp['baseline']['best_accuracy']
                    enhanced = imp['enhanced']['best_accuracy']
                    print(f"   {task_id}: {baseline*100:.1f}% → {enhanced*100:.1f}% ✓")

        # Show biggest improvements (not yet solved)
        improvements_list = [(imp['task_id'], imp['improvement']['accuracy_gain'])
                            for imp in self.improvements
                            if not imp['improvement']['newly_solved'] and imp['improvement']['accuracy_gain'] > 0]

        if improvements_list:
            improvements_list.sort(key=lambda x: x[1], reverse=True)
            print(f"\n⬆️  Biggest Improvements (not yet solved):")
            for task_id, gain in improvements_list[:5]:
                print(f"   {task_id}: {gain*100:+.1f}%")


def main():
    """Run comprehensive comparison."""

    # Get task files
    training_dir = Path("/home/user/ARC_explorations/ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))

    print(f"Found {len(task_files)} training tasks")

    # Create comparison
    comparison = SolverComparison()

    # Run on same 30 tasks as baseline evaluation
    comparison.run_comparison(task_files, max_tasks=30, verbose=True)

    # Print summary
    comparison.print_summary()

    print(f"\n{'='*70}")
    print("Comparison complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
