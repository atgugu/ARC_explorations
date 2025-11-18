"""
Test parameter inference on a single task with verbose output
"""

import json
from pathlib import Path
from arc_active_inference_solver import ARCTask
from arc_program_solver import ARCProgramSolver
from arc_loader import ARCDataLoader


def main():
    # Load a task
    data_dir = "../data"
    eval_dir = Path(data_dir) / "evaluation"

    # Test on task 60c09cac (one of the two tasks we solve)
    task_file = eval_dir / "60c09cac.json"

    print("Loading task 60c09cac...")
    with open(task_file, 'r') as f:
        task_data = json.load(f)
    task = ARCDataLoader.load_task_from_dict(task_data)

    print(f"\nTask has {len(task.train_pairs)} training examples")
    print(f"Input shape: {task.train_pairs[0][0].shape}")
    print(f"Output shape: {task.train_pairs[0][1].shape}")

    # Create solver with parameter inference
    solver = ARCProgramSolver(
        workspace_capacity=20,
        n_perturbations=5,
        max_synthesis_depth=3,
        max_programs=150,
        verbose=True  # Enable verbose output
    )

    print("\n" + "="*80)
    print("SOLVING WITH PARAMETER INFERENCE")
    print("="*80)

    predictions = solver.solve(task, verbose=True)

    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)
    print(f"\nPrediction 1 shape: {predictions[0].shape}")
    print(f"Prediction 2 shape: {predictions[1].shape}")
    print(f"Ground truth shape: {task.test_output.shape}")

    import numpy as np
    match_1 = np.array_equal(predictions[0].data, task.test_output.data)
    match_2 = np.array_equal(predictions[1].data, task.test_output.data)

    print(f"\nPrediction 1 matches: {match_1}")
    print(f"Prediction 2 matches: {match_2}")


if __name__ == "__main__":
    main()
