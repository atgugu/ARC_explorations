"""
Compare 3x Diversity vs 10x Diversity

Tests whether increasing parameter variations from 3x to 10x improves solve rate.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import time

from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver
from arc_curiosity_solver.solver_high_diversity import HighDiversityARCCuriositySolver


class DiversityComparison:
    """Compare standard diversity (3x) vs high diversity (10x)."""

    def __init__(self):
        self.diverse_solver = DiverseARCCuriositySolver()  # 3x variations
        self.high_diversity_solver = HighDiversityARCCuriositySolver()  # 10x variations
        self.training_dir = Path("ARC-AGI/data/training")
        self.evaluation_dir = Path("ARC-AGI/data/evaluation")

    def load_tasks(self, limit_per_set: int = 100) -> List[Tuple[str, dict, str]]:
        """Load tasks from both sets."""
        tasks = []

        # Training
        for task_file in sorted(self.training_dir.glob("*.json"))[:limit_per_set]:
            with open(task_file, 'r') as f:
                tasks.append((task_file.stem, json.load(f), 'training'))

        # Evaluation
        for task_file in sorted(self.evaluation_dir.glob("*.json"))[:limit_per_set]:
            with open(task_file, 'r') as f:
                tasks.append((task_file.stem, json.load(f), 'evaluation'))

        return tasks

    def test_task(self, task_id: str, task_data: dict, solver, solver_name: str) -> dict:
        """Test single task with a solver."""
        train_pairs = [
            (np.array(pair['input']), np.array(pair['output']))
            for pair in task_data['train']
        ]
        test_input = np.array(task_data['test'][0]['input'])
        test_output = np.array(task_data['test'][0]['output'])

        try:
            start_time = time.time()
            pred1, pred2 = solver.solve(train_pairs, test_input, verbose=False)
            elapsed = time.time() - start_time

            # Check exact match
            exact_match1 = np.array_equal(pred1, test_output)
            exact_match2 = np.array_equal(pred2, test_output)
            solved = exact_match1 or exact_match2

            which_matched = 1 if exact_match1 else (2 if exact_match2 else None)

            # Accuracies
            if pred1.shape == test_output.shape:
                acc1 = np.mean(pred1 == test_output)
            else:
                acc1 = 0.0

            if pred2.shape == test_output.shape:
                acc2 = np.mean(pred2 == test_output)
            else:
                acc2 = 0.0

            best_accuracy = max(acc1, acc2)

            # Pattern info
            pattern_info = solver.get_pattern_info() if hasattr(solver, 'get_pattern_info') else {}

            return {
                'task_id': task_id,
                'solver': solver_name,
                'solved': solved,
                'which_matched': which_matched,
                'accuracy': float(best_accuracy),
                'time': elapsed,
                'patterns_detected': pattern_info.get('num_patterns', 0),
                'used_patterns': pattern_info.get('used_pattern_inference', False)
            }

        except Exception as e:
            return {
                'task_id': task_id,
                'solver': solver_name,
                'solved': False,
                'which_matched': None,
                'accuracy': 0.0,
                'time': 0.0,
                'patterns_detected': 0,
                'used_patterns': False,
                'error': str(e)
            }

    def run_comparison(self, limit_per_set: int = 100):
        """Run comparison between 3x and 10x diversity."""
        print("="*80)
        print("DIVERSITY COMPARISON: 3x vs 10x Variations")
        print("="*80)
        print(f"\nTesting {limit_per_set} tasks from each set (200 total)")
        print("  3x Diversity: ~3-5 variations per pattern")
        print("  10x Diversity: ~10-20 variations per pattern")
        print("\n🎯 GOAL: Test if more variations → more exact matches\n")

        # Load tasks
        print("Loading tasks...")
        tasks = self.load_tasks(limit_per_set=limit_per_set)
        print(f"Loaded {len(tasks)} total tasks\n")

        # Test both solvers
        diverse_results = []
        high_div_results = []
        diverse_solved = []
        high_div_solved = []

        for i, (task_id, task_data, task_type) in enumerate(tasks):
            print(f"\rProgress: {i+1}/{len(tasks)} | 3x solved: {len(diverse_solved)} | 10x solved: {len(high_div_solved)}", end='', flush=True)

            # Test 3x diversity
            diverse_res = self.test_task(task_id, task_data, self.diverse_solver, "3x Diversity")
            diverse_results.append(diverse_res)
            if diverse_res['solved']:
                diverse_solved.append(diverse_res)
                print(f"\n🎯 3x SOLVED: {task_id} (prediction {diverse_res['which_matched']})")

            # Test 10x diversity
            high_div_res = self.test_task(task_id, task_data, self.high_diversity_solver, "10x Diversity")
            high_div_results.append(high_div_res)
            if high_div_res['solved']:
                high_div_solved.append(high_div_res)
                print(f"\n🎉 10x SOLVED: {task_id} (prediction {high_div_res['which_matched']})")

        print("\n")

        # Print report
        self._print_report(diverse_results, high_div_results, diverse_solved, high_div_solved)

        # Save results
        with open('diversity_comparison_results.json', 'w') as f:
            json.dump({
                'diverse_3x': diverse_results,
                'high_diversity_10x': high_div_results,
                'diverse_3x_solved': diverse_solved,
                'high_diversity_10x_solved': high_div_solved
            }, f, indent=2)

        return diverse_results, high_div_results

    def _print_report(self, diverse_results, high_div_results, diverse_solved, high_div_solved):
        """Print comparison report."""
        print("\n" + "="*80)
        print("DIVERSITY COMPARISON RESULTS")
        print("="*80)

        total = len(diverse_results)

        # EXACT MATCHES
        print(f"\n🎯 EXACT MATCHES (100% accuracy):")
        print(f"\n  3x Diversity:  {len(diverse_solved)}/{total} ({len(diverse_solved)/total*100:.1f}%)")
        print(f"  10x Diversity: {len(high_div_solved)}/{total} ({len(high_div_solved)/total*100:.1f}%)")

        # Check for NEW solves with 10x
        diverse_solved_ids = set(r['task_id'] for r in diverse_solved)
        high_div_solved_ids = set(r['task_id'] for r in high_div_solved)

        new_with_10x = high_div_solved_ids - diverse_solved_ids
        lost_with_10x = diverse_solved_ids - high_div_solved_ids

        if new_with_10x:
            print(f"\n  ✨ NEWLY SOLVED with 10x diversity: {len(new_with_10x)}")
            for task_id in sorted(new_with_10x):
                res = next(r for r in high_div_solved if r['task_id'] == task_id)
                print(f"    - {task_id} (prediction {res['which_matched']}, {res['patterns_detected']} patterns)")

        if lost_with_10x:
            print(f"\n  ⚠️  LOST with 10x diversity: {len(lost_with_10x)}")
            for task_id in sorted(lost_with_10x):
                print(f"    - {task_id}")

        # Average accuracy
        diverse_avg = np.mean([r['accuracy'] for r in diverse_results])
        high_div_avg = np.mean([r['accuracy'] for r in high_div_results])

        print(f"\n📊 Average Accuracy:")
        print(f"  3x Diversity:  {diverse_avg*100:.1f}%")
        print(f"  10x Diversity: {high_div_avg*100:.1f}%")
        print(f"  Difference: {(high_div_avg - diverse_avg)*100:+.1f}%")

        # Very close tasks (95-99%)
        diverse_very_close = [r for r in diverse_results if not r['solved'] and r['accuracy'] >= 0.95]
        high_div_very_close = [r for r in high_div_results if not r['solved'] and r['accuracy'] >= 0.95]

        print(f"\n🎯 Very Close (95-99%):")
        print(f"  3x Diversity:  {len(diverse_very_close)} tasks")
        print(f"  10x Diversity: {len(high_div_very_close)} tasks")

        # Performance
        diverse_time = np.mean([r['time'] for r in diverse_results])
        high_div_time = np.mean([r['time'] for r in high_div_results])

        print(f"\n⏱️  Performance:")
        print(f"  3x Diversity:  {diverse_time:.3f}s per task")
        print(f"  10x Diversity: {high_div_time:.3f}s per task")
        print(f"  Slowdown: {high_div_time/diverse_time:.1f}x")

        # Pattern usage
        diverse_patterns = sum(1 for r in diverse_results if r['used_patterns'])
        high_div_patterns = sum(1 for r in high_div_results if r['used_patterns'])

        print(f"\n🔍 Pattern Inference Usage:")
        print(f"  3x Diversity:  {diverse_patterns}/{total} tasks")
        print(f"  10x Diversity: {high_div_patterns}/{total} tasks")

        # CONCLUSION
        print(f"\n{'='*80}")
        print("CONCLUSION:")
        print(f"{'='*80}")

        improvement = len(high_div_solved) - len(diverse_solved)
        if improvement > 0:
            print(f"\n✅ 10x diversity IMPROVED solve rate by {improvement} tasks!")
            print(f"   {len(diverse_solved)}/{total} → {len(high_div_solved)}/{total}")
            print(f"   {len(diverse_solved)/total*100:.1f}% → {len(high_div_solved)/total*100:.1f}%")
        elif improvement < 0:
            print(f"\n⚠️  10x diversity DECREASED solve rate by {abs(improvement)} tasks")
            print(f"   {len(diverse_solved)}/{total} → {len(high_div_solved)}/{total}")
        else:
            print(f"\n➡️  10x diversity had NO CHANGE in solve rate")
            print(f"   Both: {len(diverse_solved)}/{total} ({len(diverse_solved)/total*100:.1f}%)")

        print(f"\n{'='*80}")


if __name__ == "__main__":
    print("Initializing diversity comparison...")
    comparison = DiversityComparison()

    print("\nRunning comparison on 200 tasks...\n")

    diverse_results, high_div_results = comparison.run_comparison(limit_per_set=100)

    print("\n✅ Results saved to diversity_comparison_results.json")
