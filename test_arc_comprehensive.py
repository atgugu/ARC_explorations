"""
Comprehensive ARC-AGI Task Testing Framework

Tests the curiosity-driven solver on real ARC tasks and analyzes:
- Success rate (100% pixel match)
- Prediction diversity (are the 2 attempts different?)
- Failure modes and weaknesses
- Strengths by task type
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


class ARCTaskTester:
    """Test framework for evaluating solver on ARC tasks."""

    def __init__(self, solver: ARCCuriositySolver):
        self.solver = solver
        self.results = []
        self.statistics = {
            'total_tasks': 0,
            'solved_first_attempt': 0,
            'solved_second_attempt': 0,
            'solved_either_attempt': 0,
            'predictions_identical': 0,
            'predictions_different': 0,
            'failure_modes': defaultdict(int),
            'task_types': defaultdict(lambda: {'total': 0, 'solved': 0})
        }

    def load_task(self, task_file: str) -> Dict[str, Any]:
        """Load ARC task from JSON file."""
        with open(task_file, 'r') as f:
            data = json.load(f)

        task = {
            'id': Path(task_file).stem,
            'train_pairs': [],
            'test_pairs': []
        }

        # Load training pairs
        for example in data['train']:
            inp = np.array(example['input'])
            out = np.array(example['output'])
            task['train_pairs'].append((inp, out))

        # Load test pairs
        for example in data['test']:
            inp = np.array(example['input'])
            out = np.array(example['output']) if 'output' in example else None
            task['test_pairs'].append((inp, out))

        return task

    def predictions_are_different(self, pred1: np.ndarray, pred2: np.ndarray) -> bool:
        """Check if two predictions are different."""
        if pred1.shape != pred2.shape:
            return True
        return not np.array_equal(pred1, pred2)

    def compute_accuracy(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute pixel-level accuracy."""
        if prediction.shape != ground_truth.shape:
            return 0.0

        matches = np.sum(prediction == ground_truth)
        total = ground_truth.size
        return matches / total

    def analyze_task_properties(self, task: Dict) -> Dict[str, Any]:
        """Analyze task properties to categorize it."""
        properties = {
            'size_changes': False,
            'color_changes': False,
            'object_count_changes': False,
            'complexity': 'unknown'
        }

        if not task['train_pairs']:
            return properties

        # Check for size changes
        for inp, out in task['train_pairs']:
            if inp.shape != out.shape:
                properties['size_changes'] = True
                break

        # Check for color changes
        for inp, out in task['train_pairs']:
            inp_colors = set(inp.flatten())
            out_colors = set(out.flatten())
            if inp_colors != out_colors:
                properties['color_changes'] = True
                break

        # Estimate complexity
        avg_input_size = np.mean([inp.size for inp, _ in task['train_pairs']])
        n_train = len(task['train_pairs'])

        if avg_input_size < 50 and n_train >= 3:
            properties['complexity'] = 'simple'
        elif avg_input_size < 200:
            properties['complexity'] = 'medium'
        else:
            properties['complexity'] = 'complex'

        return properties

    def classify_failure_mode(self,
                             pred1: np.ndarray,
                             pred2: np.ndarray,
                             ground_truth: np.ndarray) -> str:
        """Classify why the solver failed."""

        # Shape mismatch
        if pred1.shape != ground_truth.shape and pred2.shape != ground_truth.shape:
            return "wrong_output_shape"

        # Got shape right but wrong transformation
        acc1 = self.compute_accuracy(pred1, ground_truth)
        acc2 = self.compute_accuracy(pred2, ground_truth)

        max_acc = max(acc1, acc2)

        if max_acc > 0.8:
            return "close_but_not_perfect"
        elif max_acc > 0.5:
            return "partial_match"
        elif max_acc > 0.2:
            return "wrong_transformation"
        else:
            return "completely_wrong"

    def test_task(self, task: Dict, verbose: bool = False) -> Dict[str, Any]:
        """Test solver on a single task."""
        task_id = task['id']

        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing Task: {task_id}")
            print(f"{'='*70}")

        result = {
            'task_id': task_id,
            'n_train': len(task['train_pairs']),
            'n_test': len(task['test_pairs']),
            'properties': self.analyze_task_properties(task),
            'test_results': []
        }

        # Test on each test case
        for test_idx, (test_input, test_output) in enumerate(task['test_pairs']):
            if test_output is None:
                continue  # Skip if no ground truth

            if verbose:
                print(f"\nTest case {test_idx + 1}:")
                print(f"  Input shape: {test_input.shape}")
                print(f"  Expected output shape: {test_output.shape}")

            # Solve
            start_time = time.time()
            try:
                pred1, pred2 = self.solver.solve(
                    task['train_pairs'],
                    test_input,
                    verbose=False
                )
                solve_time = time.time() - start_time

                # Check accuracy
                acc1 = self.compute_accuracy(pred1, test_output)
                acc2 = self.compute_accuracy(pred2, test_output)

                # Check if predictions are different
                predictions_different = self.predictions_are_different(pred1, pred2)

                # Check for perfect match (100%)
                solved_first = (acc1 == 1.0)
                solved_second = (acc2 == 1.0)
                solved = solved_first or solved_second

                test_result = {
                    'test_idx': test_idx,
                    'pred1_accuracy': acc1,
                    'pred2_accuracy': acc2,
                    'pred1_shape': pred1.shape,
                    'pred2_shape': pred2.shape,
                    'expected_shape': test_output.shape,
                    'solved_first_attempt': solved_first,
                    'solved_second_attempt': solved_second,
                    'solved': solved,
                    'predictions_different': predictions_different,
                    'solve_time': solve_time,
                    'failure_mode': None if solved else self.classify_failure_mode(pred1, pred2, test_output)
                }

                if verbose:
                    print(f"  Prediction 1 accuracy: {acc1*100:.1f}%")
                    print(f"  Prediction 2 accuracy: {acc2*100:.1f}%")
                    print(f"  Solved: {'✓' if solved else '✗'}")
                    print(f"  Predictions different: {'✓' if predictions_different else '✗'}")
                    print(f"  Time: {solve_time:.2f}s")
                    if not solved:
                        print(f"  Failure mode: {test_result['failure_mode']}")

            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")

                test_result = {
                    'test_idx': test_idx,
                    'error': str(e),
                    'solved': False,
                    'predictions_different': False,
                    'failure_mode': 'exception'
                }

            result['test_results'].append(test_result)

        return result

    def run_tests(self, task_files: List[str], max_tasks: int = None, verbose: bool = False):
        """Run tests on multiple tasks."""

        if max_tasks:
            task_files = task_files[:max_tasks]

        print(f"\n{'='*70}")
        print(f"Running Comprehensive ARC-AGI Tests")
        print(f"Total tasks: {len(task_files)}")
        print(f"{'='*70}\n")

        for i, task_file in enumerate(task_files):
            try:
                task = self.load_task(task_file)
                result = self.test_task(task, verbose=verbose)
                self.results.append(result)

                # Update statistics
                self._update_statistics(result)

                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{len(task_files)} tasks tested...")

            except Exception as e:
                print(f"ERROR loading {task_file}: {e}")
                continue

    def _update_statistics(self, result: Dict):
        """Update running statistics."""
        self.statistics['total_tasks'] += 1

        # Get task properties
        props = result['properties']
        complexity = props['complexity']

        # Process test results
        for test_result in result['test_results']:
            if 'error' in test_result:
                self.statistics['failure_modes']['exception'] += 1
                continue

            if test_result['solved_first_attempt']:
                self.statistics['solved_first_attempt'] += 1

            if test_result['solved_second_attempt']:
                self.statistics['solved_second_attempt'] += 1

            if test_result['solved']:
                self.statistics['solved_either_attempt'] += 1
                self.statistics['task_types'][complexity]['solved'] += 1

            if test_result['predictions_different']:
                self.statistics['predictions_different'] += 1
            else:
                self.statistics['predictions_identical'] += 1

            if not test_result['solved'] and test_result['failure_mode']:
                self.statistics['failure_modes'][test_result['failure_mode']] += 1

            self.statistics['task_types'][complexity]['total'] += 1

    def print_summary(self):
        """Print comprehensive summary of results."""

        stats = self.statistics
        total = stats['total_tasks']

        if total == 0:
            print("No tasks tested.")
            return

        print("\n" + "="*70)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*70)

        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"   Total tasks tested: {total}")

        if stats['solved_first_attempt'] > 0:
            print(f"   Solved (1st attempt): {stats['solved_first_attempt']} ({stats['solved_first_attempt']/total*100:.1f}%)")

        if stats['solved_second_attempt'] > 0:
            print(f"   Solved (2nd attempt): {stats['solved_second_attempt']} ({stats['solved_second_attempt']/total*100:.1f}%)")

        print(f"   Solved (either attempt): {stats['solved_either_attempt']} ({stats['solved_either_attempt']/total*100:.1f}%)")

        # Prediction diversity
        print(f"\n🔄 Prediction Diversity:")
        total_predictions = stats['predictions_different'] + stats['predictions_identical']
        if total_predictions > 0:
            print(f"   Different predictions: {stats['predictions_different']} ({stats['predictions_different']/total_predictions*100:.1f}%)")
            print(f"   Identical predictions: {stats['predictions_identical']} ({stats['predictions_identical']/total_predictions*100:.1f}%)")

        # By task complexity
        print(f"\n📈 Performance by Complexity:")
        for complexity, data in sorted(stats['task_types'].items()):
            if data['total'] > 0:
                success_rate = data['solved'] / data['total'] * 100
                print(f"   {complexity.capitalize()}: {data['solved']}/{data['total']} ({success_rate:.1f}%)")

        # Failure modes
        if stats['failure_modes']:
            print(f"\n❌ Failure Mode Analysis:")
            total_failures = sum(stats['failure_modes'].values())
            for mode, count in sorted(stats['failure_modes'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {mode}: {count} ({count/total_failures*100:.1f}%)")

        # Detailed results
        print(f"\n📋 Detailed Task Results:")
        print(f"{'Task ID':<15} {'Solved':<8} {'Acc1':<8} {'Acc2':<8} {'Diverse':<8} {'Failure Mode'}")
        print("-" * 70)

        for result in self.results:
            for test_result in result['test_results']:
                if 'error' in test_result:
                    continue

                task_id = result['task_id']
                solved = '✓' if test_result['solved'] else '✗'
                acc1 = f"{test_result['pred1_accuracy']*100:.0f}%"
                acc2 = f"{test_result['pred2_accuracy']*100:.0f}%"
                diverse = '✓' if test_result['predictions_different'] else '✗'
                failure = test_result.get('failure_mode', '-')

                print(f"{task_id:<15} {solved:<8} {acc1:<8} {acc2:<8} {diverse:<8} {failure}")

        # Strengths and weaknesses
        print(f"\n💪 Strengths:")
        self._analyze_strengths()

        print(f"\n⚠️  Weaknesses:")
        self._analyze_weaknesses()

    def _analyze_strengths(self):
        """Analyze what the solver is good at."""
        strengths = []

        # Check if predictions are diverse
        if self.statistics['predictions_different'] > self.statistics['predictions_identical']:
            strengths.append("✓ Generates diverse predictions (good exploration)")

        # Check performance on simple tasks
        simple_stats = self.statistics['task_types'].get('simple', {'total': 0, 'solved': 0})
        if simple_stats['total'] > 0 and simple_stats['solved'] / simple_stats['total'] > 0.3:
            strengths.append(f"✓ Handles simple tasks well ({simple_stats['solved']}/{simple_stats['total']})")

        # Check if getting shapes right
        shape_failures = self.statistics['failure_modes'].get('wrong_output_shape', 0)
        total_failures = sum(self.statistics['failure_modes'].values())
        if total_failures > 0 and shape_failures / total_failures < 0.3:
            strengths.append("✓ Good at predicting output shapes")

        if strengths:
            for strength in strengths:
                print(f"   {strength}")
        else:
            print("   (Limited strengths detected on current task set)")

    def _analyze_weaknesses(self):
        """Analyze what the solver struggles with."""
        weaknesses = []

        # Check solve rate
        if self.statistics['total_tasks'] > 0:
            solve_rate = self.statistics['solved_either_attempt'] / self.statistics['total_tasks']
            if solve_rate < 0.3:
                weaknesses.append(f"✗ Low overall solve rate ({solve_rate*100:.1f}%)")

        # Check failure modes
        failure_modes = self.statistics['failure_modes']
        total_failures = sum(failure_modes.values())

        if total_failures > 0:
            if failure_modes.get('wrong_output_shape', 0) / total_failures > 0.3:
                weaknesses.append("✗ Struggles with determining correct output size")

            if failure_modes.get('completely_wrong', 0) / total_failures > 0.3:
                weaknesses.append("✗ Often produces completely wrong transformations")

            if failure_modes.get('wrong_transformation', 0) / total_failures > 0.3:
                weaknesses.append("✗ Identifies wrong transformation patterns")

        # Check prediction diversity
        total_predictions = self.statistics['predictions_different'] + self.statistics['predictions_identical']
        if total_predictions > 0 and self.statistics['predictions_identical'] / total_predictions > 0.5:
            weaknesses.append("✗ Predictions often identical (limited exploration)")

        # Check complex task performance
        complex_stats = self.statistics['task_types'].get('complex', {'total': 0, 'solved': 0})
        if complex_stats['total'] > 0 and complex_stats['solved'] == 0:
            weaknesses.append("✗ Cannot handle complex tasks")

        if weaknesses:
            for weakness in weaknesses:
                print(f"   {weakness}")
        else:
            print("   (No major weaknesses detected)")


def main():
    """Run comprehensive tests."""

    # Create solver
    print("Initializing solver...")
    solver = ARCCuriositySolver(
        workspace_capacity=7,
        learning_rate=0.1,
        exploration_bonus=1.0,
        n_hypotheses_to_explore=40
    )

    # Create tester
    tester = ARCTaskTester(solver)

    # Get task files (training set)
    training_dir = Path("/home/user/ARC_explorations/ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))

    print(f"Found {len(task_files)} training tasks")

    # Run tests on subset (30 tasks for reasonable runtime)
    tester.run_tests(task_files, max_tasks=30, verbose=True)

    # Print summary
    tester.print_summary()

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)


if __name__ == "__main__":
    main()
