"""
Test Diverse Pattern-Based Solver

Focus: EXACT PIXEL MATCH (100% accuracy)

Compares four solver versions:
1. Baseline: Original curiosity-driven solver
2. Enhanced: + object reasoning
3. Pattern-Based: + pattern inference
4. Diverse: + parameter variations & combinations (NEW)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time

from arc_curiosity_solver.solver import ARCCuriositySolver
from arc_curiosity_solver.solver_enhanced import EnhancedARCCuriositySolver
from arc_curiosity_solver.solver_pattern_based import PatternBasedARCCuriositySolver
from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver


class FourWaySolverComparison:
    """Compare all four solver versions with focus on exact matches."""

    def __init__(self, data_dir: str = "ARC-AGI/data/training"):
        self.data_dir = Path(data_dir)
        self.baseline_solver = ARCCuriositySolver()
        self.enhanced_solver = EnhancedARCCuriositySolver()
        self.pattern_solver = PatternBasedARCCuriositySolver()
        self.diverse_solver = DiverseARCCuriositySolver()

    def load_tasks(self, limit: int = 30) -> List[Tuple[str, dict]]:
        """Load ARC tasks."""
        tasks = []
        task_files = sorted(self.data_dir.glob("*.json"))[:limit]

        for task_file in task_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                tasks.append((task_file.stem, task_data))

        return tasks

    def test_task(self, task_id: str, task_data: dict, solver, solver_name: str,
                  verbose: bool = False) -> dict:
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

            # Check EXACT match (most important!)
            exact_match1 = np.array_equal(pred1, test_output)
            exact_match2 = np.array_equal(pred2, test_output)
            solved = exact_match1 or exact_match2

            # Also track which prediction matched
            which_matched = None
            if exact_match1:
                which_matched = 1
            elif exact_match2:
                which_matched = 2

            # Calculate pixel accuracy for both predictions
            if pred1.shape == test_output.shape:
                acc1 = np.mean(pred1 == test_output)
            else:
                acc1 = 0.0

            if pred2.shape == test_output.shape:
                acc2 = np.mean(pred2 == test_output)
            else:
                acc2 = 0.0

            best_accuracy = max(acc1, acc2)

            # How close to perfect?
            if not solved and best_accuracy >= 0.95:
                closeness = "VERY CLOSE (95-99%)"
            elif not solved and best_accuracy >= 0.90:
                closeness = "CLOSE (90-95%)"
            elif not solved and best_accuracy >= 0.80:
                closeness = "NEAR (80-90%)"
            else:
                closeness = None

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
                'which_matched': which_matched,
                'accuracy': float(best_accuracy),
                'acc1': float(acc1),
                'acc2': float(acc2),
                'closeness': closeness,
                'shape_correct': shape_correct,
                'diversity': float(diversity),
                'time': elapsed,
                'pattern_info': pattern_info
            }

            if verbose:
                if solved:
                    print(f"\n✅ SOLVED (prediction {which_matched})")
                else:
                    status_msg = f"❌ {best_accuracy*100:.1f}% accurate"
                    if closeness:
                        status_msg += f" ({closeness})"
                    print(f"\n{status_msg}")

                if pattern_info:
                    print(f"Patterns: {pattern_info.get('num_patterns', 0)}")

            return result

        except Exception as e:
            print(f"Error on task {task_id} ({solver_name}): {e}")
            import traceback
            traceback.print_exc()
            return {
                'task_id': task_id,
                'solver': solver_name,
                'solved': False,
                'which_matched': None,
                'accuracy': 0.0,
                'acc1': 0.0,
                'acc2': 0.0,
                'closeness': None,
                'shape_correct': False,
                'diversity': 0.0,
                'time': 0.0,
                'error': str(e),
                'pattern_info': {}
            }

    def run_comparison(self, num_tasks: int = 30, verbose: bool = False, show_examples: int = 2):
        """Run four-way comparison focused on exact matches."""
        print("="*80)
        print("FOUR-WAY SOLVER COMPARISON - EXACT MATCH FOCUS")
        print("="*80)
        print("\nTesting:")
        print("  1. Baseline Solver (original)")
        print("  2. Enhanced Solver (+ object reasoning)")
        print("  3. Pattern-Based Solver (+ pattern inference)")
        print("  4. Diverse Solver (+ parameter variations & combinations)")
        print(f"\nRunning on {num_tasks} tasks...\n")
        print("🎯 PRIMARY METRIC: Exact pixel match (100% accuracy)\n")

        tasks = self.load_tasks(limit=num_tasks)

        baseline_results = []
        enhanced_results = []
        pattern_results = []
        diverse_results = []

        # Test all solvers
        for i, (task_id, task_data) in enumerate(tasks):
            is_verbose = verbose and i < show_examples

            if is_verbose:
                print("\n" + "="*80)
                print(f"TASK {i+1}/{len(tasks)}: {task_id}")
                print("="*80)

            print(f"\rProgress: {i+1}/{len(tasks)}", end='', flush=True)

            baseline_res = self.test_task(task_id, task_data, self.baseline_solver, "Baseline", is_verbose)
            enhanced_res = self.test_task(task_id, task_data, self.enhanced_solver, "Enhanced", is_verbose)
            pattern_res = self.test_task(task_id, task_data, self.pattern_solver, "Pattern-Based", is_verbose)
            diverse_res = self.test_task(task_id, task_data, self.diverse_solver, "Diverse", is_verbose)

            baseline_results.append(baseline_res)
            enhanced_results.append(enhanced_res)
            pattern_results.append(pattern_res)
            diverse_results.append(diverse_res)

        print("\n")

        # Compute statistics
        self._print_comparison_report(baseline_results, enhanced_results,
                                     pattern_results, diverse_results)

        return baseline_results, enhanced_results, pattern_results, diverse_results

    def _print_comparison_report(self, baseline_results, enhanced_results,
                                pattern_results, diverse_results):
        """Print comprehensive comparison report focused on exact matches."""
        print("\n" + "="*80)
        print("COMPARISON REPORT - EXACT MATCH FOCUS")
        print("="*80)

        # EXACT MATCH STATISTICS (MOST IMPORTANT!)
        print("\n🎯 EXACT MATCH RESULTS (100% accuracy):")
        print(f"\n{'Solver':<20} | {'Solved':<10} | {'Pred 1':<8} | {'Pred 2':<8} | {'Rate':<6}")
        print("-" * 80)

        for name, results in [
            ("Baseline", baseline_results),
            ("Enhanced", enhanced_results),
            ("Pattern-Based", pattern_results),
            ("Diverse", diverse_results)
        ]:
            solved = sum(r['solved'] for r in results)
            pred1_matches = sum(1 for r in results if r['which_matched'] == 1)
            pred2_matches = sum(1 for r in results if r['which_matched'] == 2)
            rate = (solved / len(results)) * 100 if results else 0

            print(f"{name:<20} | {solved:>2}/{len(results):<6} | {pred1_matches:>7} | {pred2_matches:>7} | {rate:>5.1f}%")

        # NEW SOLVES
        print("\n✨ NEWLY SOLVED TASKS:")

        diverse_solved = set(r['task_id'] for r in diverse_results if r['solved'])
        pattern_solved = set(r['task_id'] for r in pattern_results if r['solved'])
        baseline_solved = set(r['task_id'] for r in baseline_results if r['solved'])

        diverse_new = diverse_solved - baseline_solved
        pattern_new = pattern_solved - baseline_solved

        if diverse_new:
            print(f"\n  Diverse solver (NEW): {len(diverse_new)} tasks")
            for task_id in sorted(diverse_new):
                # Find which prediction matched
                res = next(r for r in diverse_results if r['task_id'] == task_id)
                print(f"    ✅ {task_id} (prediction {res['which_matched']})")
        else:
            print("\n  Diverse solver: No new tasks solved")

        if pattern_new - diverse_new:
            print(f"\n  Pattern-Based solver (NEW): {len(pattern_new - diverse_new)} tasks")
            for task_id in sorted(pattern_new - diverse_new):
                res = next(r for r in pattern_results if r['task_id'] == task_id)
                print(f"    ✅ {task_id} (prediction {res['which_matched']})")

        # CLOSE CALLS (Very close but not exact)
        print("\n🎯 VERY CLOSE (95-99% accurate):")

        for name, results in [("Diverse", diverse_results)]:
            very_close = [(r['task_id'], r['accuracy']) for r in results
                         if not r['solved'] and r['accuracy'] >= 0.95]
            very_close.sort(key=lambda x: x[1], reverse=True)

            if very_close:
                print(f"\n  {name}: {len(very_close)} tasks")
                for task_id, acc in very_close[:10]:
                    print(f"    - {task_id}: {acc*100:.1f}%")
            else:
                print(f"\n  {name}: None")

        # OVERALL STATISTICS
        print("\n📊 Overall Statistics:")
        print(f"\n{'Metric':<30} | {'Baseline':<12} | {'Enhanced':<12} | {'Pattern':<12} | {'Diverse':<12}")
        print("-" * 100)

        # Solve rate
        baseline_solved = sum(r['solved'] for r in baseline_results)
        enhanced_solved = sum(r['solved'] for r in enhanced_results)
        pattern_solved = sum(r['solved'] for r in pattern_results)
        diverse_solved = sum(r['solved'] for r in diverse_results)

        print(f"{'Perfect solves (100%)':<30} | {baseline_solved:>2}/{len(baseline_results):<8} | "
              f"{enhanced_solved:>2}/{len(enhanced_results):<8} | {pattern_solved:>2}/{len(pattern_results):<8} | "
              f"{diverse_solved:>2}/{len(diverse_results):<8}")

        # Average accuracy
        baseline_avg = np.mean([r['accuracy'] for r in baseline_results])
        enhanced_avg = np.mean([r['accuracy'] for r in enhanced_results])
        pattern_avg = np.mean([r['accuracy'] for r in pattern_results])
        diverse_avg = np.mean([r['accuracy'] for r in diverse_results])

        print(f"{'Average accuracy':<30} | {baseline_avg:>11.1%} | "
              f"{enhanced_avg:>11.1%} | {pattern_avg:>11.1%} | {diverse_avg:>11.1%}")

        # Shape correctness
        baseline_shape = sum(r['shape_correct'] for r in baseline_results)
        enhanced_shape = sum(r['shape_correct'] for r in enhanced_results)
        pattern_shape = sum(r['shape_correct'] for r in pattern_results)
        diverse_shape = sum(r['shape_correct'] for r in diverse_results)

        print(f"{'Correct output shape':<30} | {baseline_shape:>2}/{len(baseline_results):<8} | "
              f"{enhanced_shape:>2}/{len(enhanced_results):<8} | {pattern_shape:>2}/{len(pattern_results):<8} | "
              f"{diverse_shape:>2}/{len(diverse_results):<8}")

        # Average diversity
        diverse_diversity = np.mean([r['diversity'] for r in diverse_results])
        pattern_diversity = np.mean([r['diversity'] for r in pattern_results])

        print(f"{'Prediction diversity':<30} | {'N/A':<12} | {'N/A':<12} | "
              f"{pattern_diversity:>11.1%} | {diverse_diversity:>11.1%}")

        # Pattern usage
        diverse_pattern_use = sum(1 for r in diverse_results if r.get('pattern_info', {}).get('used_pattern_inference', False))
        diverse_avg_patterns = np.mean([r.get('pattern_info', {}).get('num_patterns', 0) for r in diverse_results])

        print(f"\n{'Diverse Solver Stats:':<30}")
        print(f"  Used pattern inference: {diverse_pattern_use}/{len(diverse_results)} tasks")
        print(f"  Average patterns detected: {diverse_avg_patterns:.1f}")

        # Improvement analysis
        print("\n📈 Improvement vs Baseline:")

        improvements = []
        for b, d in zip(baseline_results, diverse_results):
            diff = d['accuracy'] - b['accuracy']
            if diff > 0.05:
                improvements.append((d['task_id'], diff, d['accuracy'], b['accuracy']))

        improvements.sort(key=lambda x: x[1], reverse=True)

        if improvements:
            print(f"\n  Significant improvements: {len(improvements)} tasks")
            for task_id, diff, final_acc, base_acc in improvements[:10]:
                print(f"    - {task_id}: {base_acc*100:.1f}% → {final_acc*100:.1f}% (+{diff*100:.1f}%)")
        else:
            print("\n  No significant improvements (>5%)")

        # Regressions
        regressions = []
        for b, d in zip(baseline_results, diverse_results):
            diff = d['accuracy'] - b['accuracy']
            if diff < -0.05:
                regressions.append((d['task_id'], diff, d['accuracy'], b['accuracy']))

        regressions.sort(key=lambda x: x[1])

        if regressions:
            print(f"\n  ⚠️  Regressions: {len(regressions)} tasks")
            for task_id, diff, final_acc, base_acc in regressions[:5]:
                print(f"    - {task_id}: {base_acc*100:.1f}% → {final_acc*100:.1f}% ({diff*100:.1f}%)")

        print("\n" + "="*80)


if __name__ == "__main__":
    print("Initializing all four solvers...")
    comparison = FourWaySolverComparison()

    print("\nRunning comparison on 30 ARC tasks...")
    print("(First 2 tasks will show detailed output)\n")

    baseline, enhanced, pattern, diverse = comparison.run_comparison(
        num_tasks=30,
        verbose=True,
        show_examples=2
    )

    # Save results
    import json
    with open('diverse_solver_comparison.json', 'w') as f:
        json.dump({
            'baseline': baseline,
            'enhanced': enhanced,
            'pattern_based': pattern,
            'diverse': diverse
        }, f, indent=2)

    print("\n✅ Results saved to diverse_solver_comparison.json")
