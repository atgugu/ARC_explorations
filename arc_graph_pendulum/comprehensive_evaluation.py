"""
Comprehensive evaluation on ARC tasks with competition-style 2 attempts per task.
"""

import numpy as np
import sys
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_enhanced import EnhancedARCGraphPendulumSolver
from utils.arc_loader import ARCLoader, ARCTask
from utils.grid_utils import compute_iou


class CompetitionEvaluator:
    """
    Evaluator for competition-style testing with 2 attempts per task.
    """

    def __init__(self):
        self.results = []
        self.task_categories = defaultdict(list)

    def is_perfect_match(self, pred: np.ndarray, target: np.ndarray) -> bool:
        """
        Check if prediction is 100% pixel-perfect match.

        Args:
            pred: Predicted grid
            target: Target grid

        Returns:
            True if perfect match
        """
        if pred.shape != target.shape:
            return False
        return np.array_equal(pred, target)

    def evaluate_predictions(
        self,
        predictions: List[np.ndarray],
        task: ARCTask
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of predicted grids
            task: ARCTask with ground truth

        Returns:
            Dictionary with evaluation metrics
        """
        test_results = []
        all_perfect = True

        for i, (pred, (_, target)) in enumerate(zip(predictions, task.test)):
            perfect = self.is_perfect_match(pred, target)
            iou = compute_iou(pred, target)

            test_results.append({
                'test_idx': i,
                'perfect_match': perfect,
                'iou': iou,
                'shape_match': pred.shape == target.shape,
            })

            if not perfect:
                all_perfect = False

        avg_iou = np.mean([r['iou'] for r in test_results])

        return {
            'test_results': test_results,
            'avg_iou': avg_iou,
            'all_perfect': all_perfect,
            'any_perfect': any(r['perfect_match'] for r in test_results),
            'num_perfect': sum(r['perfect_match'] for r in test_results),
            'num_tests': len(test_results),
        }

    def solve_with_attempt(
        self,
        task: ARCTask,
        attempt_num: int,
        solver_config: Dict[str, Any],
        verbose: bool = False
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Solve a task with specific configuration.

        Args:
            task: ARCTask to solve
            attempt_num: Attempt number (1 or 2)
            solver_config: Solver configuration
            verbose: Whether to print progress

        Returns:
            (predictions, metadata)
        """
        # Create solver with specified config
        solver = EnhancedARCGraphPendulumSolver(**solver_config)

        # Solve task
        predictions = solver.solve_task(task, verbose=verbose)

        # Get metadata from last trajectory
        metadata = {
            'attempt': attempt_num,
            'config': solver_config,
            'num_trajectories': len(solver.trajectory_batch.trajectories),
        }

        if solver.trajectory_batch.trajectories:
            last_traj = solver.trajectory_batch.trajectories[-1]
            metadata.update({
                'final_score': last_traj.final_score,
                'nodes_used': last_traj.nodes,
                'num_nodes': len(last_traj.nodes),
            })

        return predictions, metadata

    def evaluate_task_with_attempts(
        self,
        task: ARCTask,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a task with 2 different attempts.

        Args:
            task: ARCTask to evaluate
            verbose: Whether to print progress

        Returns:
            Dictionary with results from both attempts
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Task: {task.task_id}")
            print(f"Train: {task.num_train}, Test: {task.num_test}")
            print(f"{'='*70}")

        # Attempt 1: Standard enhanced solver
        attempt1_config = {
            'beam_width': 5,
            'use_stability': True,
            'use_repair_loops': True,
            'use_landscape_analytics': False,  # Disable for speed
        }

        if verbose:
            print("\n--- ATTEMPT 1: Standard Enhanced ---")

        pred1, meta1 = self.solve_with_attempt(
            task, 1, attempt1_config, verbose=verbose
        )
        eval1 = self.evaluate_predictions(pred1, task)

        # Attempt 2: Different strategy - wider beam, no repairs
        attempt2_config = {
            'beam_width': 8,  # Wider search
            'use_stability': True,
            'use_repair_loops': False,  # Try without repairs
            'use_landscape_analytics': False,
        }

        if verbose:
            print("\n--- ATTEMPT 2: Wider Beam, No Repairs ---")

        pred2, meta2 = self.solve_with_attempt(
            task, 2, attempt2_config, verbose=verbose
        )
        eval2 = self.evaluate_predictions(pred2, task)

        # Determine which attempt was better
        better_attempt = 1 if eval1['avg_iou'] >= eval2['avg_iou'] else 2
        best_eval = eval1 if better_attempt == 1 else eval2
        best_meta = meta1 if better_attempt == 1 else meta2

        # Check if either attempt solved it
        solved = eval1['all_perfect'] or eval2['all_perfect']

        result = {
            'task_id': task.task_id,
            'num_train': task.num_train,
            'num_test': task.num_test,
            'attempt1': {
                'eval': eval1,
                'meta': meta1,
            },
            'attempt2': {
                'eval': eval2,
                'meta': meta2,
            },
            'better_attempt': better_attempt,
            'best_avg_iou': best_eval['avg_iou'],
            'solved': solved,
            'any_perfect': eval1['any_perfect'] or eval2['any_perfect'],
        }

        if verbose:
            print(f"\n--- RESULTS ---")
            print(f"Attempt 1: IoU={eval1['avg_iou']:.3f}, Perfect={eval1['num_perfect']}/{eval1['num_tests']}")
            print(f"Attempt 2: IoU={eval2['avg_iou']:.3f}, Perfect={eval2['num_perfect']}/{eval2['num_tests']}")
            print(f"Best: Attempt {better_attempt}")
            print(f"Solved: {'✓ YES' if solved else '✗ NO'}")

        return result

    def categorize_task(self, task: ARCTask, result: Dict[str, Any]) -> List[str]:
        """
        Categorize task based on characteristics and results.

        Args:
            task: ARCTask
            result: Evaluation result

        Returns:
            List of category labels
        """
        categories = []

        # Size categories
        if task.num_train <= 2:
            categories.append('few_shot')
        elif task.num_train >= 5:
            categories.append('many_shot')
        else:
            categories.append('medium_shot')

        # Test set size
        if task.num_test > 1:
            categories.append('multi_test')
        else:
            categories.append('single_test')

        # Performance categories
        if result['solved']:
            categories.append('solved')
        elif result['best_avg_iou'] >= 0.8:
            categories.append('high_quality')
        elif result['best_avg_iou'] >= 0.5:
            categories.append('medium_quality')
        else:
            categories.append('low_quality')

        # Check if repairs helped
        if 'attempt1' in result and 'attempt2' in result:
            iou1 = result['attempt1']['eval']['avg_iou']
            iou2 = result['attempt2']['eval']['avg_iou']

            if iou1 > iou2 + 0.05:  # Attempt 1 (with repairs) significantly better
                categories.append('repairs_helped')
            elif iou2 > iou1 + 0.05:  # Attempt 2 (wider beam) better
                categories.append('search_helped')

        return categories

    def evaluate_dataset(
        self,
        tasks: Dict[str, ARCTask],
        max_tasks: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate multiple tasks.

        Args:
            tasks: Dictionary of tasks
            max_tasks: Maximum number of tasks to evaluate
            verbose: Whether to print progress

        Returns:
            Comprehensive evaluation results
        """
        task_list = list(tasks.values())
        if max_tasks:
            task_list = task_list[:max_tasks]

        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE EVALUATION")
        print(f"{'='*70}")
        print(f"Total tasks: {len(task_list)}")
        print(f"Attempts per task: 2")
        print(f"{'='*70}\n")

        self.results = []

        for i, task in enumerate(task_list):
            if verbose and i % 5 == 0:
                print(f"\nProgress: {i}/{len(task_list)} tasks completed")

            result = self.evaluate_task_with_attempts(task, verbose=verbose)

            # Categorize
            categories = self.categorize_task(task, result)
            result['categories'] = categories

            for cat in categories:
                self.task_categories[cat].append(task.task_id)

            self.results.append(result)

        return self.analyze_results()

    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze all results and identify strengths/weaknesses.

        Returns:
            Analysis dictionary
        """
        if not self.results:
            return {}

        # Overall statistics
        total_tasks = len(self.results)
        solved_count = sum(1 for r in self.results if r['solved'])
        any_perfect_count = sum(1 for r in self.results if r['any_perfect'])

        # Average IoU
        avg_ious = [r['best_avg_iou'] for r in self.results]
        overall_avg_iou = np.mean(avg_ious)

        # High quality (>0.8 IoU)
        high_quality = sum(1 for iou in avg_ious if iou >= 0.8)

        # Attempt comparison
        attempt1_better = sum(1 for r in self.results if r['better_attempt'] == 1)
        attempt2_better = sum(1 for r in self.results if r['better_attempt'] == 2)

        # Category analysis
        category_stats = {}
        for cat, task_ids in self.task_categories.items():
            cat_results = [r for r in self.results if r['task_id'] in task_ids]
            cat_stats = {
                'count': len(cat_results),
                'solved': sum(1 for r in cat_results if r['solved']),
                'avg_iou': np.mean([r['best_avg_iou'] for r in cat_results]),
            }
            category_stats[cat] = cat_stats

        # Find best and worst tasks
        sorted_results = sorted(self.results, key=lambda r: r['best_avg_iou'], reverse=True)
        best_tasks = sorted_results[:5]
        worst_tasks = sorted_results[-5:]

        return {
            'total_tasks': total_tasks,
            'solved_count': solved_count,
            'solved_rate': solved_count / total_tasks if total_tasks > 0 else 0,
            'any_perfect_count': any_perfect_count,
            'high_quality_count': high_quality,
            'high_quality_rate': high_quality / total_tasks if total_tasks > 0 else 0,
            'overall_avg_iou': overall_avg_iou,
            'attempt1_better_count': attempt1_better,
            'attempt2_better_count': attempt2_better,
            'category_stats': category_stats,
            'best_tasks': [(t['task_id'], t['best_avg_iou']) for t in best_tasks],
            'worst_tasks': [(t['task_id'], t['best_avg_iou']) for t in worst_tasks],
        }

    def print_analysis(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis."""
        print("\n" + "="*70)
        print("COMPREHENSIVE ANALYSIS")
        print("="*70)

        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total tasks evaluated: {analysis['total_tasks']}")
        print(f"  Perfect solves (100% match): {analysis['solved_count']} ({analysis['solved_rate']*100:.1f}%)")
        print(f"  Any test perfect: {analysis['any_perfect_count']}")
        print(f"  High quality (>0.80 IoU): {analysis['high_quality_count']} ({analysis['high_quality_rate']*100:.1f}%)")
        print(f"  Average IoU: {analysis['overall_avg_iou']:.3f}")

        print(f"\nATTEMPT COMPARISON:")
        print(f"  Attempt 1 better (repairs enabled): {analysis['attempt1_better_count']}")
        print(f"  Attempt 2 better (wider beam): {analysis['attempt2_better_count']}")

        print(f"\nCATEGORY BREAKDOWN:")
        for cat, stats in sorted(analysis['category_stats'].items()):
            if stats['count'] > 0:
                print(f"  {cat}:")
                print(f"    Count: {stats['count']}")
                print(f"    Solved: {stats['solved']}/{stats['count']} ({stats['solved']/stats['count']*100:.1f}%)")
                print(f"    Avg IoU: {stats['avg_iou']:.3f}")

        print(f"\nBEST PERFORMING TASKS:")
        for i, (task_id, iou) in enumerate(analysis['best_tasks'], 1):
            print(f"  {i}. {task_id}: {iou:.3f}")

        print(f"\nWORST PERFORMING TASKS:")
        for i, (task_id, iou) in enumerate(analysis['worst_tasks'], 1):
            print(f"  {i}. {task_id}: {iou:.3f}")

        print("\n" + "="*70)

    def save_results(self, output_path: str):
        """Save detailed results to JSON."""
        output = {
            'results': self.results,
            'analysis': self.analyze_results(),
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nDetailed results saved to {output_path}")


def main():
    """Run comprehensive evaluation."""
    # Load ARC dataset
    loader = ARCLoader(cache_dir="./arc_data")
    print("Loading ARC dataset...")
    tasks = loader.load_all_tasks("training")

    if not tasks:
        print("No tasks loaded.")
        return

    print(f"Loaded {len(tasks)} tasks")

    # Create evaluator
    evaluator = CompetitionEvaluator()

    # Run evaluation (use all tasks, or limit for testing)
    max_tasks = len(tasks)  # Evaluate all available tasks

    analysis = evaluator.evaluate_dataset(
        tasks,
        max_tasks=max_tasks,
        verbose=True  # Set to False for less output
    )

    # Print analysis
    evaluator.print_analysis(analysis)

    # Save results
    evaluator.save_results("comprehensive_evaluation_results.json")

    # Print strengths and weaknesses
    print("\n" + "="*70)
    print("STRENGTHS & WEAKNESSES ANALYSIS")
    print("="*70)

    cat_stats = analysis['category_stats']

    print("\n✓ STRENGTHS:")
    if 'repairs_helped' in cat_stats and cat_stats['repairs_helped']['count'] > 0:
        print(f"  - Repair loops effective on {cat_stats['repairs_helped']['count']} tasks")

    if 'solved' in cat_stats:
        print(f"  - Can achieve perfect solutions ({cat_stats['solved']['count']} tasks)")

    if 'high_quality' in cat_stats and cat_stats['high_quality']['count'] > 0:
        print(f"  - Strong performance (>0.8 IoU) on {cat_stats['high_quality']['count']} tasks")

    # Check performance by training set size
    if 'few_shot' in cat_stats and cat_stats['few_shot']['avg_iou'] > 0.5:
        print(f"  - Handles few-shot tasks (avg IoU: {cat_stats['few_shot']['avg_iou']:.3f})")

    print("\n✗ WEAKNESSES:")
    if 'low_quality' in cat_stats and cat_stats['low_quality']['count'] > 0:
        print(f"  - Poor performance on {cat_stats['low_quality']['count']} tasks (<0.5 IoU)")

    unsolved_rate = 1.0 - analysis['solved_rate']
    print(f"  - {unsolved_rate*100:.1f}% of tasks not perfectly solved")

    # Identify specific weaknesses
    if 'few_shot' in cat_stats and cat_stats['few_shot']['avg_iou'] < 0.3:
        print(f"  - Struggles with few-shot learning (avg IoU: {cat_stats['few_shot']['avg_iou']:.3f})")

    if 'many_shot' in cat_stats and cat_stats['many_shot']['solved'] == 0:
        print(f"  - No perfect solutions on many-shot tasks")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
