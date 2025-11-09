"""
ARC Active Inference Solver - Examples
======================================

Demonstration of the solver on various ARC-style tasks.
"""

import numpy as np
from arc_active_inference_solver import (
    ARCActiveInferenceSolver,
    ARCTask,
    Grid,
    visualize_grid
)
from arc_loader import ARCDataLoader, ARCEvaluator


def example_1_flip_vertical():
    """Example 1: Flip vertical transformation"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Flip Vertical")
    print("="*60)

    task = ARCTask(
        train_pairs=[
            (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
            (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
        ],
        test_input=Grid([[1, 0], [2, 3]])
    )

    solver = ARCActiveInferenceSolver()
    predictions = solver.solve(task, verbose=True)

    print("\nRESULTS:")
    visualize_grid(task.test_input, "Test Input:")
    visualize_grid(predictions[0], "Prediction 1:")
    visualize_grid(predictions[1], "Prediction 2:")
    visualize_grid(Grid([[0, 1], [3, 2]]), "Expected Output:")


def example_2_rotate_90():
    """Example 2: Rotate 90 degrees"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Rotate 90 Degrees")
    print("="*60)

    task = ARCTask(
        train_pairs=[
            (Grid([[1, 2], [3, 4]]), Grid([[3, 1], [4, 2]])),
            (Grid([[5, 6], [7, 8]]), Grid([[7, 5], [8, 6]])),
        ],
        test_input=Grid([[9, 0], [1, 2]])
    )

    solver = ARCActiveInferenceSolver()
    predictions = solver.solve(task, verbose=True)

    print("\nRESULTS:")
    visualize_grid(task.test_input, "Test Input:")
    visualize_grid(predictions[0], "Prediction 1:")
    visualize_grid(predictions[1], "Prediction 2:")


def example_3_color_replacement():
    """Example 3: Color replacement"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Replace Color 1 with 5")
    print("="*60)

    task = ARCTask(
        train_pairs=[
            (Grid([[1, 2, 1], [1, 3, 1]]), Grid([[5, 2, 5], [5, 3, 5]])),
            (Grid([[1, 0, 0], [1, 1, 2]]), Grid([[5, 0, 0], [5, 5, 2]])),
        ],
        test_input=Grid([[1, 1, 2], [0, 1, 0]])
    )

    solver = ARCActiveInferenceSolver()
    predictions = solver.solve(task, verbose=True)

    print("\nRESULTS:")
    visualize_grid(task.test_input, "Test Input:")
    visualize_grid(predictions[0], "Prediction 1:")
    visualize_grid(predictions[1], "Prediction 2:")
    visualize_grid(Grid([[5, 5, 2], [0, 5, 0]]), "Expected Output:")


def example_4_zoom():
    """Example 4: Zoom/scale transformation"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Zoom 2x")
    print("="*60)

    task = ARCTask(
        train_pairs=[
            (Grid([[1, 2]]), Grid([[1, 1, 2, 2]])),
            (Grid([[3, 4]]), Grid([[3, 3, 4, 4]])),
        ],
        test_input=Grid([[5, 6]])
    )

    solver = ARCActiveInferenceSolver()
    predictions = solver.solve(task, verbose=True)

    print("\nRESULTS:")
    visualize_grid(task.test_input, "Test Input:")
    visualize_grid(predictions[0], "Prediction 1:")
    visualize_grid(predictions[1], "Prediction 2:")
    visualize_grid(Grid([[5, 5, 6, 6]]), "Expected Output:")


def example_5_complex_pattern():
    """Example 5: More complex pattern"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Complex Pattern - Largest Object")
    print("="*60)

    # Task: Keep only the largest connected component
    task = ARCTask(
        train_pairs=[
            (
                Grid([
                    [1, 1, 0, 2],
                    [1, 1, 0, 0],
                    [0, 0, 0, 3],
                ]),
                Grid([
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                ])
            ),
            (
                Grid([
                    [5, 0, 7, 7],
                    [0, 0, 7, 7],
                ]),
                Grid([
                    [0, 0, 7, 7],
                    [0, 0, 7, 7],
                ])
            ),
        ],
        test_input=Grid([
            [1, 0, 2, 2],
            [0, 0, 2, 2],
            [3, 3, 0, 0],
        ])
    )

    solver = ARCActiveInferenceSolver()
    predictions = solver.solve(task, verbose=True)

    print("\nRESULTS:")
    visualize_grid(task.test_input, "Test Input:")
    visualize_grid(predictions[0], "Prediction 1:")
    visualize_grid(predictions[1], "Prediction 2:")


def example_6_symmetry():
    """Example 6: Symmetry detection"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Flip Horizontal (Symmetry)")
    print("="*60)

    task = ARCTask(
        train_pairs=[
            (Grid([[1, 2, 3]]), Grid([[1, 2, 3]][::-1])),
            (Grid([[4], [5], [6]]), Grid([[4], [5], [6]][::-1])),
        ],
        test_input=Grid([[7], [8], [9]])
    )

    solver = ARCActiveInferenceSolver()
    predictions = solver.solve(task, verbose=True)

    print("\nRESULTS:")
    visualize_grid(task.test_input, "Test Input:")
    visualize_grid(predictions[0], "Prediction 1:")
    visualize_grid(predictions[1], "Prediction 2:")


def run_comprehensive_evaluation():
    """Run evaluation on all example tasks"""
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)

    from arc_loader import create_example_tasks

    tasks = create_example_tasks()
    solver = ARCActiveInferenceSolver()

    evaluation = ARCEvaluator.evaluate_dataset(
        solver,
        tasks,
        verbose=False
    )

    print("\nRESULTS BY TASK:")
    print("-" * 60)
    for task_id, result in evaluation['results'].items():
        if result['metrics']:
            solved = "✓" if result['solved'] else "✗"
            acc = result['metrics']['best_pixel_accuracy']
            print(f"{solved} {task_id:20s} - Accuracy: {acc:.3f}")

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    summary = evaluation['summary']
    print(f"Total Tasks:       {summary['total_tasks']}")
    print(f"Solved:            {summary['solved_count']} ({summary['solve_rate']:.1%})")
    print(f"Avg Pixel Acc:     {summary['avg_pixel_accuracy']:.3f}")
    print(f"Avg IoU:           {summary['avg_iou']:.3f}")


def demonstrate_active_inference():
    """Demonstrate active inference learning during inference"""
    print("\n" + "="*60)
    print("DEMONSTRATION: Active Inference Learning")
    print("="*60)
    print("\nThis example shows how the solver learns during inference")
    print("by updating its beliefs with each training example.\n")

    # Task with 3 training examples to show learning progression
    task = ARCTask(
        train_pairs=[
            (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
            (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
            (Grid([[0, 1], [2, 0]]), Grid([[1, 0], [0, 2]])),
        ],
        test_input=Grid([[9, 8], [7, 6]])
    )

    print("Task: Flip vertical")
    print(f"Number of training examples: {len(task.train_pairs)}")
    print("\nWatch how entropy decreases as we observe more examples:")

    solver = ARCActiveInferenceSolver()
    predictions = solver.solve(task, verbose=True)

    print("\n" + "="*60)
    print("EXPLANATION:")
    print("="*60)
    print("- Initial entropy is high (many possible hypotheses)")
    print("- After each observation, entropy decreases (fewer plausible hypotheses)")
    print("- Learning progress shows how much we learned from each example")
    print("- Stable hypotheses have consistent behavior across examples")
    print("- Top-2 predictions come from most probable × most stable hypotheses")


def demonstrate_curiosity():
    """Demonstrate curiosity-driven exploration"""
    print("\n" + "="*60)
    print("DEMONSTRATION: Curiosity-Driven Exploration")
    print("="*60)
    print("\nThis example shows how curiosity signals guide hypothesis selection.\n")

    task = ARCTask(
        train_pairs=[
            (Grid([[1, 0], [0, 1]]), Grid([[0, 1], [1, 0]])),
            (Grid([[2, 0], [0, 2]]), Grid([[0, 2], [2, 0]])),
        ],
        test_input=Grid([[3, 0], [0, 3]])
    )

    print("Task: Diagonal flip / transpose")
    print("\nThe solver uses curiosity signals to explore hypotheses:")
    print("- Information Gain: How much does this hypothesis reduce uncertainty?")
    print("- Epistemic Uncertainty: How uncertain are we about this hypothesis?")
    print("- Learning Progress: How much have we learned?")

    solver = ARCActiveInferenceSolver()
    predictions = solver.solve(task, verbose=True)

    visualize_grid(task.test_input, "\nTest Input:")
    visualize_grid(predictions[0], "Prediction 1:")
    visualize_grid(predictions[1], "Prediction 2:")


def main():
    """Run all examples"""
    import sys

    print("\n" + "="*60)
    print("ARC ACTIVE INFERENCE SOLVER - EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates the unified solver on various tasks.")
    print("Each example shows how the solver:")
    print("  1. Perceives patterns in training examples")
    print("  2. Generates transformation hypotheses")
    print("  3. Updates beliefs using Active Inference")
    print("  4. Computes curiosity signals")
    print("  5. Filters unstable hypotheses")
    print("  6. Selects top-2 predictions")
    print("\n" + "="*60)

    examples = [
        ("1", "Flip Vertical", example_1_flip_vertical),
        ("2", "Rotate 90°", example_2_rotate_90),
        ("3", "Color Replacement", example_3_color_replacement),
        ("4", "Zoom 2x", example_4_zoom),
        ("5", "Complex Pattern", example_5_complex_pattern),
        ("6", "Symmetry", example_6_symmetry),
        ("7", "Active Inference Demo", demonstrate_active_inference),
        ("8", "Curiosity Demo", demonstrate_curiosity),
        ("9", "Comprehensive Evaluation", run_comprehensive_evaluation),
        ("a", "All Examples", None),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nAvailable examples:")
        for num, name, _ in examples:
            print(f"  {num}. {name}")
        print("\nUsage: python examples.py [example_number]")
        print("       python examples.py a    # Run all")
        choice = input("\nEnter example number (or 'a' for all): ").strip()

    if choice == 'a':
        # Run all examples
        for num, name, func in examples:
            if func is not None:
                try:
                    func()
                except Exception as e:
                    print(f"\nError in example {num}: {e}")
                print("\n" + "="*60)
                input("Press Enter to continue to next example...")
    else:
        # Run specific example
        for num, name, func in examples:
            if num == choice and func is not None:
                func()
                break
        else:
            print(f"Invalid example number: {choice}")


if __name__ == "__main__":
    main()
