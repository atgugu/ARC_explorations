"""
Large-Scale Test: Diverse Solver on 100+ ARC Tasks

Tests on a diverse sample from both training and evaluation sets.
Primary goal: Find ANY exact matches (100% accuracy).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
from collections import defaultdict

from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver


class LargeScaleTest:
    """Test diverse solver on large, diverse task set."""

    def __init__(self):
        self.solver = DiverseARCCuriositySolver()
        self.training_dir = Path("ARC-AGI/data/training")
        self.evaluation_dir = Path("ARC-AGI/data/evaluation")

    def load_all_tasks(self, limit_per_set: int = 100) -> List[Tuple[str, dict, str]]:
        """Load tasks from both training and evaluation sets."""
        tasks = []

        # Load training tasks
        training_files = sorted(self.training_dir.glob("*.json"))[:limit_per_set]
        for task_file in training_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                tasks.append((task_file.stem, task_data, 'training'))

        # Load evaluation tasks
        eval_files = sorted(self.evaluation_dir.glob("*.json"))[:limit_per_set]
        for task_file in eval_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                tasks.append((task_file.stem, task_data, 'evaluation'))

        return tasks

    def test_task(self, task_id: str, task_data: dict, task_type: str) -> dict:
        """Test a single task."""
        train_pairs = [
            (np.array(pair['input']), np.array(pair['output']))
            for pair in task_data['train']
        ]
        test_input = np.array(task_data['test'][0]['input'])
        test_output = np.array(task_data['test'][0]['output'])

        try:
            start_time = time.time()
            pred1, pred2 = self.solver.solve(train_pairs, test_input, verbose=False)
            elapsed = time.time() - start_time

            # Check exact match
            exact_match1 = np.array_equal(pred1, test_output)
            exact_match2 = np.array_equal(pred2, test_output)
            solved = exact_match1 or exact_match2

            which_matched = None
            if exact_match1:
                which_matched = 1
            elif exact_match2:
                which_matched = 2

            # Calculate accuracies
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
            pattern_info = self.solver.get_pattern_info() if hasattr(self.solver, 'get_pattern_info') else {}

            return {
                'task_id': task_id,
                'task_type': task_type,
                'solved': solved,
                'which_matched': which_matched,
                'accuracy': float(best_accuracy),
                'acc1': float(acc1),
                'acc2': float(acc2),
                'shape_correct': (pred1.shape == test_output.shape) or (pred2.shape == test_output.shape),
                'time': elapsed,
                'patterns_detected': pattern_info.get('num_patterns', 0),
                'used_patterns': pattern_info.get('used_pattern_inference', False)
            }

        except Exception as e:
            return {
                'task_id': task_id,
                'task_type': task_type,
                'solved': False,
                'which_matched': None,
                'accuracy': 0.0,
                'acc1': 0.0,
                'acc2': 0.0,
                'shape_correct': False,
                'time': 0.0,
                'patterns_detected': 0,
                'used_patterns': False,
                'error': str(e)
            }

    def run_large_scale_test(self, limit_per_set: int = 100):
        """Run large-scale test and report results."""
        print("="*80)
        print("LARGE-SCALE DIVERSE SOLVER TEST")
        print("="*80)
        print(f"\nTesting up to {limit_per_set} tasks from EACH set:")
        print(f"  - Training set: {limit_per_set} tasks")
        print(f"  - Evaluation set: {limit_per_set} tasks")
        print(f"\n🎯 PRIMARY GOAL: Find ANY exact matches (100% accuracy)\n")

        # Load tasks
        print("Loading tasks...")
        tasks = self.load_all_tasks(limit_per_set=limit_per_set)
        print(f"Loaded {len(tasks)} total tasks\n")

        # Test all tasks
        results = []
        solved_tasks = []
        very_close_tasks = []  # 95-99%
        close_tasks = []  # 90-95%

        for i, (task_id, task_data, task_type) in enumerate(tasks):
            print(f"\rProgress: {i+1}/{len(tasks)} | Solved: {len(solved_tasks)} | Very close: {len(very_close_tasks)}", end='', flush=True)

            result = self.test_task(task_id, task_data, task_type)
            results.append(result)

            if result['solved']:
                solved_tasks.append(result)
                print(f"\n🎉 SOLVED: {task_id} ({task_type}) - Prediction {result['which_matched']}")

            elif result['accuracy'] >= 0.95:
                very_close_tasks.append(result)

            elif result['accuracy'] >= 0.90:
                close_tasks.append(result)

        print("\n")

        # Print comprehensive report
        self._print_report(results, solved_tasks, very_close_tasks, close_tasks)

        # Save results
        with open('large_scale_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tasks': len(results),
                    'solved': len(solved_tasks),
                    'very_close': len(very_close_tasks),
                    'close': len(close_tasks)
                },
                'results': results,
                'solved_tasks': solved_tasks,
                'very_close_tasks': very_close_tasks,
                'close_tasks': close_tasks
            }, f, indent=2)

        return results, solved_tasks, very_close_tasks, close_tasks

    def _print_report(self, results, solved_tasks, very_close_tasks, close_tasks):
        """Print comprehensive report."""
        print("\n" + "="*80)
        print("LARGE-SCALE TEST RESULTS")
        print("="*80)

        # Overall statistics
        total = len(results)
        training_results = [r for r in results if r['task_type'] == 'training']
        eval_results = [r for r in results if r['task_type'] == 'evaluation']

        print(f"\n📊 Overall Statistics:")
        print(f"  Total tasks tested: {total}")
        print(f"  Training tasks: {len(training_results)}")
        print(f"  Evaluation tasks: {len(eval_results)}")

        # EXACT MATCHES (most important!)
        print(f"\n🎯 EXACT MATCHES (100% accuracy):")
        print(f"  Total solved: {len(solved_tasks)}/{total} ({len(solved_tasks)/total*100:.1f}%)")

        if solved_tasks:
            print(f"\n  ✅ SOLVED TASKS:")
            training_solved = [t for t in solved_tasks if t['task_type'] == 'training']
            eval_solved = [t for t in solved_tasks if t['task_type'] == 'evaluation']

            print(f"    Training: {len(training_solved)}")
            for task in training_solved:
                print(f"      - {task['task_id']} (prediction {task['which_matched']}, {task['patterns_detected']} patterns)")

            print(f"    Evaluation: {len(eval_solved)}")
            for task in eval_solved:
                print(f"      - {task['task_id']} (prediction {task['which_matched']}, {task['patterns_detected']} patterns)")
        else:
            print(f"  ❌ No exact matches found")

        # Very close (95-99%)
        print(f"\n🎯 VERY CLOSE (95-99% accurate):")
        print(f"  Count: {len(very_close_tasks)}/{total} ({len(very_close_tasks)/total*100:.1f}%)")

        if very_close_tasks:
            # Sort by accuracy
            very_close_sorted = sorted(very_close_tasks, key=lambda x: x['accuracy'], reverse=True)
            print(f"\n  Top 10:")
            for task in very_close_sorted[:10]:
                patterns = task['patterns_detected']
                print(f"    - {task['task_id']} ({task['task_type']}): {task['accuracy']*100:.2f}% ({patterns} patterns)")

        # Close (90-95%)
        print(f"\n🎯 CLOSE (90-95% accurate):")
        print(f"  Count: {len(close_tasks)}/{total} ({len(close_tasks)/total*100:.1f}%)")

        # Average accuracy by set
        train_acc = np.mean([r['accuracy'] for r in training_results])
        eval_acc = np.mean([r['accuracy'] for r in eval_results])

        print(f"\n📈 Average Accuracy:")
        print(f"  Training set: {train_acc*100:.1f}%")
        print(f"  Evaluation set: {eval_acc*100:.1f}%")
        print(f"  Overall: {np.mean([r['accuracy'] for r in results])*100:.1f}%")

        # Pattern inference usage
        used_patterns = sum(1 for r in results if r['used_patterns'])
        avg_patterns = np.mean([r['patterns_detected'] for r in results])

        print(f"\n🔍 Pattern Inference:")
        print(f"  Used on: {used_patterns}/{total} tasks ({used_patterns/total*100:.1f}%)")
        print(f"  Average patterns detected: {avg_patterns:.1f}")

        # Shape correctness
        shape_correct = sum(1 for r in results if r['shape_correct'])
        print(f"\n📐 Shape Correctness:")
        print(f"  Correct output shape: {shape_correct}/{total} ({shape_correct/total*100:.1f}%)")

        # Performance
        avg_time = np.mean([r['time'] for r in results])
        print(f"\n⏱️  Performance:")
        print(f"  Average solve time: {avg_time:.3f}s")
        print(f"  Total time: {sum(r['time'] for r in results):.1f}s")

        print("\n" + "="*80)


if __name__ == "__main__":
    print("Initializing large-scale test...")
    tester = LargeScaleTest()

    print("\nRunning test on 100 tasks from each set (200 total)...\n")

    results, solved, very_close, close = tester.run_large_scale_test(limit_per_set=100)

    print("\n✅ Results saved to large_scale_results.json")

    if solved:
        print(f"\n🎉 SUCCESS: Found {len(solved)} exact match(es)!")
    else:
        print(f"\n⚠️  No exact matches found in {len(results)} tasks")
        if very_close:
            print(f"   But {len(very_close)} tasks were very close (95-99%)")
