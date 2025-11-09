"""
ARC Dataset Loader and Evaluation Utilities
============================================

Load ARC tasks from JSON format and evaluate solver performance.
"""

import json
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from arc_active_inference_solver import ARCTask, Grid, ARCActiveInferenceSolver


class ARCDataLoader:
    """Load ARC tasks from JSON files"""

    @staticmethod
    def load_task_from_json(json_path: str) -> Dict[str, ARCTask]:
        """
        Load ARC tasks from a JSON file

        Args:
            json_path: Path to JSON file containing ARC tasks

        Returns:
            Dictionary mapping task IDs to ARCTask objects
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        tasks = {}
        for task_id, task_data in data.items():
            tasks[task_id] = ARCDataLoader._parse_task(task_data)

        return tasks

    @staticmethod
    def load_task_from_dict(task_data: Dict) -> ARCTask:
        """Load a single task from dictionary"""
        return ARCDataLoader._parse_task(task_data)

    @staticmethod
    def _parse_task(task_data: Dict) -> ARCTask:
        """
        Parse task data into ARCTask object

        Expected format:
        {
            "train": [
                {"input": [[...]], "output": [[...]]}
            ],
            "test": [
                {"input": [[...]], "output": [[...]]}  # output optional
            ]
        }
        """
        # Parse training pairs
        train_pairs = []
        for pair in task_data.get('train', []):
            input_grid = Grid(pair['input'])
            output_grid = Grid(pair['output'])
            train_pairs.append((input_grid, output_grid))

        # Parse test input (use first test case)
        test_data = task_data.get('test', [])
        if len(test_data) == 0:
            raise ValueError("Task must have at least one test case")

        test_input = Grid(test_data[0]['input'])

        # Store ground truth if available (for evaluation)
        task = ARCTask(train_pairs, test_input)

        if 'output' in test_data[0]:
            task.test_output = Grid(test_data[0]['output'])
        else:
            task.test_output = None

        return task

    @staticmethod
    def create_simple_task(
        input_arrays: List[np.ndarray],
        output_arrays: List[np.ndarray],
        test_input: np.ndarray
    ) -> ARCTask:
        """
        Create a task from numpy arrays

        Args:
            input_arrays: List of input grids
            output_arrays: List of output grids
            test_input: Test input grid

        Returns:
            ARCTask object
        """
        train_pairs = []
        for inp, out in zip(input_arrays, output_arrays):
            train_pairs.append((Grid(inp), Grid(out)))

        return ARCTask(train_pairs, Grid(test_input))


class ARCEvaluator:
    """Evaluate solver performance on ARC tasks"""

    @staticmethod
    def evaluate_prediction(prediction: Grid, ground_truth: Grid) -> Dict[str, float]:
        """
        Evaluate a single prediction against ground truth

        Returns:
            Dictionary with metrics:
            - pixel_accuracy: fraction of correct pixels
            - exact_match: 1.0 if perfect match, 0.0 otherwise
            - iou: Intersection over Union for non-zero pixels
        """
        if prediction.shape != ground_truth.shape:
            return {
                'pixel_accuracy': 0.0,
                'exact_match': 0.0,
                'iou': 0.0,
            }

        # Pixel accuracy
        matches = np.sum(prediction.data == ground_truth.data)
        total = prediction.data.size
        pixel_accuracy = matches / total

        # Exact match
        exact_match = 1.0 if np.array_equal(prediction.data, ground_truth.data) else 0.0

        # IoU for non-zero pixels
        pred_nonzero = prediction.data != 0
        gt_nonzero = ground_truth.data != 0

        intersection = np.sum(pred_nonzero & gt_nonzero)
        union = np.sum(pred_nonzero | gt_nonzero)

        iou = intersection / union if union > 0 else 0.0

        return {
            'pixel_accuracy': pixel_accuracy,
            'exact_match': exact_match,
            'iou': iou,
        }

    @staticmethod
    def evaluate_task(
        solver: ARCActiveInferenceSolver,
        task: ARCTask,
        verbose: bool = False
    ) -> Dict[str, any]:
        """
        Evaluate solver on a single task

        Args:
            solver: The solver to evaluate
            task: The task to solve
            verbose: Print detailed output

        Returns:
            Dictionary with:
            - predictions: List of 2 prediction grids
            - metrics: Evaluation metrics (if ground truth available)
            - solved: Whether any prediction matches exactly
        """
        # Solve task
        predictions = solver.solve(task, verbose=verbose)

        result = {
            'predictions': predictions,
            'metrics': None,
            'solved': False,
        }

        # Evaluate if ground truth available
        if hasattr(task, 'test_output') and task.test_output is not None:
            # Evaluate both predictions
            metrics_1 = ARCEvaluator.evaluate_prediction(predictions[0], task.test_output)
            metrics_2 = ARCEvaluator.evaluate_prediction(predictions[1], task.test_output)

            # Task is solved if either prediction is exact match
            solved = (metrics_1['exact_match'] == 1.0) or (metrics_2['exact_match'] == 1.0)

            result['metrics'] = {
                'prediction_1': metrics_1,
                'prediction_2': metrics_2,
                'best_pixel_accuracy': max(metrics_1['pixel_accuracy'], metrics_2['pixel_accuracy']),
                'best_iou': max(metrics_1['iou'], metrics_2['iou']),
            }
            result['solved'] = solved

        return result

    @staticmethod
    def evaluate_dataset(
        solver: ARCActiveInferenceSolver,
        tasks: Dict[str, ARCTask],
        verbose: bool = False,
        max_tasks: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Evaluate solver on a dataset of tasks

        Args:
            solver: The solver to evaluate
            tasks: Dictionary mapping task IDs to ARCTask objects
            verbose: Print progress
            max_tasks: Maximum number of tasks to evaluate (for testing)

        Returns:
            Dictionary with:
            - results: Dict mapping task ID to evaluation results
            - summary: Overall statistics
        """
        results = {}
        task_ids = list(tasks.keys())

        if max_tasks is not None:
            task_ids = task_ids[:max_tasks]

        for i, task_id in enumerate(task_ids):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Task {i+1}/{len(task_ids)}: {task_id}")
                print(f"{'='*60}")

            task = tasks[task_id]
            result = ARCEvaluator.evaluate_task(solver, task, verbose=verbose)
            results[task_id] = result

            if verbose and result['metrics'] is not None:
                print(f"\nSolved: {result['solved']}")
                print(f"Best Pixel Accuracy: {result['metrics']['best_pixel_accuracy']:.3f}")
                print(f"Best IoU: {result['metrics']['best_iou']:.3f}")

        # Compute summary statistics
        summary = ARCEvaluator._compute_summary(results)

        if verbose:
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Total tasks: {summary['total_tasks']}")
            print(f"Solved: {summary['solved_count']} ({summary['solve_rate']:.1%})")
            print(f"Average pixel accuracy: {summary['avg_pixel_accuracy']:.3f}")
            print(f"Average IoU: {summary['avg_iou']:.3f}")

        return {
            'results': results,
            'summary': summary
        }

    @staticmethod
    def _compute_summary(results: Dict[str, Dict]) -> Dict[str, float]:
        """Compute summary statistics from results"""
        total_tasks = len(results)
        solved_count = sum(1 for r in results.values() if r['solved'])

        # Collect metrics
        pixel_accuracies = []
        ious = []

        for result in results.values():
            if result['metrics'] is not None:
                pixel_accuracies.append(result['metrics']['best_pixel_accuracy'])
                ious.append(result['metrics']['best_iou'])

        return {
            'total_tasks': total_tasks,
            'solved_count': solved_count,
            'solve_rate': solved_count / total_tasks if total_tasks > 0 else 0.0,
            'avg_pixel_accuracy': np.mean(pixel_accuracies) if pixel_accuracies else 0.0,
            'avg_iou': np.mean(ious) if ious else 0.0,
        }


def create_example_tasks() -> Dict[str, ARCTask]:
    """Create a few example tasks for testing"""
    tasks = {}

    # Task 1: Flip vertical
    tasks['flip_vertical'] = ARCTask(
        train_pairs=[
            (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
            (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
        ],
        test_input=Grid([[1, 0], [2, 3]])
    )
    tasks['flip_vertical'].test_output = Grid([[0, 1], [3, 2]])

    # Task 2: Rotate 90
    tasks['rotate_90'] = ARCTask(
        train_pairs=[
            (Grid([[1, 2], [3, 4]]), Grid([[3, 1], [4, 2]])),
            (Grid([[5, 6], [7, 8]]), Grid([[7, 5], [8, 6]])),
        ],
        test_input=Grid([[1, 2], [3, 4]])
    )
    tasks['rotate_90'].test_output = Grid([[3, 1], [4, 2]])

    # Task 3: Identity
    tasks['identity'] = ARCTask(
        train_pairs=[
            (Grid([[1, 2], [3, 4]]), Grid([[1, 2], [3, 4]])),
            (Grid([[5, 6], [7, 8]]), Grid([[5, 6], [7, 8]])),
        ],
        test_input=Grid([[9, 0], [1, 2]])
    )
    tasks['identity'].test_output = Grid([[9, 0], [1, 2]])

    # Task 4: Replace color 1 with 5
    tasks['replace_color'] = ARCTask(
        train_pairs=[
            (Grid([[1, 2, 1], [1, 3, 1]]), Grid([[5, 2, 5], [5, 3, 5]])),
            (Grid([[1, 0, 0], [1, 1, 2]]), Grid([[5, 0, 0], [5, 5, 2]])),
        ],
        test_input=Grid([[1, 1, 2], [0, 1, 0]])
    )
    tasks['replace_color'].test_output = Grid([[5, 5, 2], [0, 5, 0]])

    # Task 5: Zoom 2x
    tasks['zoom_2x'] = ARCTask(
        train_pairs=[
            (Grid([[1, 2]]), Grid([[1, 1, 2, 2]])),
            (Grid([[3]]), Grid([[3, 3]])),
        ],
        test_input=Grid([[1, 2]])
    )
    tasks['zoom_2x'].test_output = Grid([[1, 1, 2, 2]])

    return tasks


if __name__ == "__main__":
    # Example usage
    print("ARC Data Loader - Example\n")

    # Create example tasks
    tasks = create_example_tasks()
    print(f"Created {len(tasks)} example tasks")

    # Create solver
    solver = ARCActiveInferenceSolver(workspace_capacity=20)

    # Evaluate on example tasks
    print("\nEvaluating solver on example tasks...\n")
    evaluation = ARCEvaluator.evaluate_dataset(
        solver,
        tasks,
        verbose=True,
        max_tasks=5
    )

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    summary = evaluation['summary']
    print(f"Solve Rate: {summary['solve_rate']:.1%}")
    print(f"Average Pixel Accuracy: {summary['avg_pixel_accuracy']:.3f}")
    print(f"Average IoU: {summary['avg_iou']:.3f}")
