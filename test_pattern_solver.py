"""
Test Pattern-Based Solver

Compares three solver versions:
1. Baseline: Original curiosity-driven solver
2. Enhanced: With object reasoning
3. Pattern-Based: With pattern inference (NEW)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time

from arc_curiosity_solver.solver import ARCCuriositySolver
from arc_curiosity_solver.solver_enhanced import EnhancedARCCuriositySolver
from arc_curiosity_solver.solver_pattern_based import PatternBasedARCCuriositySolver


class ThreeWaySolverComparison:
    """Compare baseline, enhanced, and pattern-based solvers."""

    def __init__(self, data_dir: str = "ARC-AGI/data/training"):
        self.data_dir = Path(data_dir)
        self.baseline_solver = ARCCuriositySolver()
        self.enhanced_solver = EnhancedARCCuriositySolver()
        self.pattern_solver = PatternBasedARCCuriositySolver()

    def load_tasks(self, limit: int = 30) -> List[Tuple[str, dict]]:
        """Load ARC tasks."""
        tasks = []
        task_files = sorted(self.data_dir.glob("*.json"))[:limit]

        for task_file in task_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                tasks.append((task_file.stem, task_data))

        return tasks

    def test_task(self, task_id: str, task_data: dict, solver, solver_name: str, verbose: bool = False) -> dict:
        """Test a single task with a solver."""
        train_pairs = [
            (np.array(pair['input']), np.array(pair['output']))
            for pair in task_data['train']
        ]
        test_input = np.array(task_data['test'][0]['input'])
        test_output = np.array(task_data['test'][0]['output'])

        try:
            start_time = time.time()

            if verbose:
                print(f"\n{'='*60}")
                print(f"Task: {task_id} ({solver_name})")
                print(f"{'='*60}")

            pred1, pred2 = solver.solve(train_pairs, test_input, verbose=verbose)
            elapsed = time.time() - start_time

            # Check accuracy
            match1 = np.array_equal(pred1, test_output)
            match2 = np.array_equal(pred2, test_output)
            solved = match1 or match2

            # Calculate pixel accuracy for best prediction
            if pred1.shape == test_output.shape:
                acc1 = np.mean(pred1 == test_output)
            else:
                acc1 = 0.0

            if pred2.shape == test_output.shape:
                acc2 = np.mean(pred2 == test_output)
            else:
                acc2 = 0.0

            best_accuracy = max(acc1, acc2)

            # Shape correctness
            shape_correct = (pred1.shape == test_output.shape) or (pred2.shape == test_output.shape)

            # Prediction diversity
            if pred1.shape == pred2.shape:
                diversity = 1.0 - np.mean(pred1 == pred2)
            else:
                diversity = 1.0

            # Get pattern info if available
            pattern_info = {}
            if hasattr(solver, 'get_pattern_info'):
                pattern_info = solver.get_pattern_info()

            result = {
                'task_id': task_id,
                'solver': solver_name,
                'solved': solved,
                'accuracy': float(best_accuracy),
                'shape_correct': shape_correct,
                'diversity': float(diversity),
                'time': elapsed,
                'pattern_info': pattern_info
            }

            if verbose:
                status = "✅ SOLVED" if solved else f"❌ {best_accuracy*100:.1f}% accurate"
                print(f"\n{status}")
                if pattern_info:
                    print(f"Patterns detected: {pattern_info.get('num_patterns', 0)}")
                    if pattern_info.get('patterns'):
                        for p in pattern_info['patterns'][:3]:
                            print(f"  - {p['description']} (conf={p['confidence']:.2f})")

            return result

        except Exception as e:
            print(f"Error on task {task_id} ({solver_name}): {e}")
            import traceback
            traceback.print_exc()
            return {
                'task_id': task_id,
                'solver': solver_name,
                'solved': False,
                'accuracy': 0.0,
                'shape_correct': False,
                'diversity': 0.0,
                'time': 0.0,
                'error': str(e),
                'pattern_info': {}
            }

    def run_comparison(self, num_tasks: int = 30, verbose: bool = False, show_examples: int = 3):
        """Run three-way comparison."""
        print("="*80)
        print("THREE-WAY SOLVER COMPARISON")
        print("="*80)
        print("\nTesting:")
        print("  1. Baseline Solver (original)")
        print("  2. Enhanced Solver (+ object reasoning)")
        print("  3. Pattern-Based Solver (+ pattern inference)")
        print(f"\nRunning on {num_tasks} tasks...\n")

        tasks = self.load_tasks(limit=num_tasks)

        baseline_results = []
        enhanced_results = []
        pattern_results = []

        # Test first few tasks with verbose output
        for i, (task_id, task_data) in enumerate(tasks):
            is_verbose = verbose and i < show_examples

            if is_verbose:
                print("\n" + "="*80)
                print(f"TASK {i+1}/{len(tasks)}: {task_id}")
                print("="*80)

            # Test all three solvers
            print(f"\rProgress: {i+1}/{len(tasks)}", end='', flush=True)

            baseline_res = self.test_task(task_id, task_data, self.baseline_solver, "Baseline", is_verbose)
            enhanced_res = self.test_task(task_id, task_data, self.enhanced_solver, "Enhanced", is_verbose)
            pattern_res = self.test_task(task_id, task_data, self.pattern_solver, "Pattern-Based", is_verbose)

            baseline_results.append(baseline_res)
            enhanced_results.append(enhanced_res)
            pattern_results.append(pattern_res)

        print("\n")

        # Compute statistics
        self._print_comparison_report(baseline_results, enhanced_results, pattern_results)

        return baseline_results, enhanced_results, pattern_results

    def _print_comparison_report(self, baseline_results, enhanced_results, pattern_results):
        """Print comprehensive comparison report."""
        print("\n" + "="*80)
        print("COMPARISON REPORT")
        print("="*80)

        # Overall statistics
        print("\n📊 Overall Statistics:")
        print(f"\n{'Metric':<30} | {'Baseline':<12} | {'Enhanced':<12} | {'Pattern-Based':<12}")
        print("-" * 80)

        # Solve rate
        baseline_solved = sum(r['solved'] for r in baseline_results)
        enhanced_solved = sum(r['solved'] for r in enhanced_results)
        pattern_solved = sum(r['solved'] for r in pattern_results)

        print(f"{'Perfect solves (100%)':<30} | {baseline_solved:>2}/{len(baseline_results):<8} | "
              f"{enhanced_solved:>2}/{len(enhanced_results):<8} | {pattern_solved:>2}/{len(pattern_results):<8}")

        # Average accuracy
        baseline_avg = np.mean([r['accuracy'] for r in baseline_results])
        enhanced_avg = np.mean([r['accuracy'] for r in enhanced_results])
        pattern_avg = np.mean([r['accuracy'] for r in pattern_results])

        print(f"{'Average accuracy':<30} | {baseline_avg:>11.1%} | "
              f"{enhanced_avg:>11.1%} | {pattern_avg:>11.1%}")

        # Shape correctness
        baseline_shape = sum(r['shape_correct'] for r in baseline_results)
        enhanced_shape = sum(r['shape_correct'] for r in enhanced_results)
        pattern_shape = sum(r['shape_correct'] for r in pattern_results)

        print(f"{'Correct output shape':<30} | {baseline_shape:>2}/{len(baseline_results):<8} | "
              f"{enhanced_shape:>2}/{len(enhanced_results):<8} | {pattern_shape:>2}/{len(pattern_results):<8}")

        # Average time
        baseline_time = np.mean([r['time'] for r in baseline_results])
        enhanced_time = np.mean([r['time'] for r in enhanced_results])
        pattern_time = np.mean([r['time'] for r in pattern_results])

        print(f"{'Average time (seconds)':<30} | {baseline_time:>11.2f} | "
              f"{enhanced_time:>11.2f} | {pattern_time:>11.2f}")

        # Pattern inference statistics
        pattern_used = sum(1 for r in pattern_results if r.get('pattern_info', {}).get('used_pattern_inference', False))
        avg_patterns = np.mean([r.get('pattern_info', {}).get('num_patterns', 0) for r in pattern_results])

        print(f"\n{'Pattern-Based Solver Stats:':<30}")
        print(f"  Used pattern inference: {pattern_used}/{len(pattern_results)} tasks")
        print(f"  Average patterns detected: {avg_patterns:.1f}")

        # Improvement analysis
        print("\n📈 Improvement Analysis:")

        # New solves
        pattern_new_solves = []
        for i, (b, e, p) in enumerate(zip(baseline_results, enhanced_results, pattern_results)):
            if p['solved'] and not b['solved']:
                pattern_new_solves.append(p['task_id'])

        print(f"\n  Pattern-Based newly solved: {len(pattern_new_solves)} tasks")
        if pattern_new_solves:
            for task_id in pattern_new_solves[:5]:
                print(f"    - {task_id}")

        # Accuracy improvements
        improvements = []
        for b, p in zip(baseline_results, pattern_results):
            diff = p['accuracy'] - b['accuracy']
            if diff > 0.05:  # Significant improvement
                improvements.append((p['task_id'], diff, p['accuracy']))

        improvements.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Significant accuracy improvements: {len(improvements)} tasks")
        if improvements:
            for task_id, diff, final_acc in improvements[:5]:
                print(f"    - {task_id}: +{diff*100:.1f}% (→ {final_acc*100:.1f}%)")

        # Regressions
        regressions = []
        for b, p in zip(baseline_results, pattern_results):
            diff = p['accuracy'] - b['accuracy']
            if diff < -0.05:  # Significant regression
                regressions.append((p['task_id'], diff, p['accuracy']))

        regressions.sort(key=lambda x: x[1])

        if regressions:
            print(f"\n  ⚠️  Regressions: {len(regressions)} tasks")
            for task_id, diff, final_acc in regressions[:5]:
                print(f"    - {task_id}: {diff*100:.1f}% (→ {final_acc*100:.1f}%)")

        # Close but not perfect
        close = [(r['task_id'], r['accuracy']) for r in pattern_results
                 if not r['solved'] and r['accuracy'] >= 0.8]
        close.sort(key=lambda x: x[1], reverse=True)

        if close:
            print(f"\n  🎯 Close but not perfect (≥80% accurate): {len(close)} tasks")
            for task_id, acc in close[:5]:
                print(f"    - {task_id}: {acc*100:.1f}%")

        print("\n" + "="*80)


if __name__ == "__main__":
    print("Initializing solvers...")
    comparison = ThreeWaySolverComparison()

    print("\nRunning comparison on 30 ARC tasks...")
    print("(First 3 tasks will show detailed output)\n")

    baseline_results, enhanced_results, pattern_results = comparison.run_comparison(
        num_tasks=30,
        verbose=True,
        show_examples=3
    )

    # Save results
    import json
    with open('pattern_solver_comparison.json', 'w') as f:
        json.dump({
            'baseline': baseline_results,
            'enhanced': enhanced_results,
            'pattern_based': pattern_results
        }, f, indent=2)

    print("\n✅ Results saved to pattern_solver_comparison.json")
