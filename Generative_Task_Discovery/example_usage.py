"""
Example Usage of ARC Generative Solver with Active Inference

Demonstrates:
- Loading ARC tasks (from JSON or created programmatically)
- Solving with active inference
- Getting dual predictions
- Evaluating results
"""

import numpy as np
import json
from pathlib import Path
from arc_generative_solver import (
    ARCGenerativeSolver,
    evaluate_predictions,
    TRGPrimitives
)


def create_example_tasks():
    """Create diverse example ARC tasks for testing"""

    # Task 1: Horizontal reflection
    task1 = {
        "name": "horizontal_reflection",
        "train": [
            {
                "input": [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ],
                "output": [
                    [3, 2, 1],
                    [6, 5, 4],
                    [9, 8, 7]
                ]
            },
            {
                "input": [
                    [1, 1, 0],
                    [0, 1, 1]
                ],
                "output": [
                    [0, 1, 1],
                    [1, 1, 0]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [2, 4, 6],
                    [1, 3, 5]
                ],
                "output": [
                    [6, 4, 2],
                    [5, 3, 1]
                ]
            }
        ]
    }

    # Task 2: Vertical reflection
    task2 = {
        "name": "vertical_reflection",
        "train": [
            {
                "input": [
                    [1, 2, 3],
                    [4, 5, 6]
                ],
                "output": [
                    [4, 5, 6],
                    [1, 2, 3]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [7, 8],
                    [9, 0]
                ],
                "output": [
                    [9, 0],
                    [7, 8]
                ]
            }
        ]
    }

    # Task 3: 90-degree rotation
    task3 = {
        "name": "rotation_90",
        "train": [
            {
                "input": [
                    [1, 2],
                    [3, 4]
                ],
                "output": [
                    [3, 1],
                    [4, 2]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [5, 6],
                    [7, 8]
                ],
                "output": [
                    [7, 5],
                    [8, 6]
                ]
            }
        ]
    }

    # Task 4: Color remapping
    task4 = {
        "name": "color_remap",
        "train": [
            {
                "input": [
                    [1, 1, 2],
                    [1, 2, 2],
                    [2, 2, 1]
                ],
                "output": [
                    [3, 3, 4],
                    [3, 4, 4],
                    [4, 4, 3]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [1, 2, 1],
                    [2, 1, 2]
                ],
                "output": [
                    [3, 4, 3],
                    [4, 3, 4]
                ]
            }
        ]
    }

    # Task 5: Translation
    task5 = {
        "name": "translation",
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ],
                "output": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 2, 2, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ],
                "output": [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 2, 2],
                    [0, 0, 0, 0]
                ]
            }
        ]
    }

    # Task 6: Composite (rotation + reflection)
    task6 = {
        "name": "composite_rot_reflect",
        "train": [
            {
                "input": [
                    [1, 2],
                    [3, 4]
                ],
                "output": [
                    [2, 4],
                    [1, 3]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [5, 6],
                    [7, 8]
                ],
                "output": [
                    [6, 8],
                    [5, 7]
                ]
            }
        ]
    }

    return [task1, task2, task3, task4, task5, task6]


def load_arc_task_from_json(filepath: str):
    """Load ARC task from JSON file (standard ARC format)"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def visualize_grid(grid: np.ndarray, title: str = ""):
    """Simple text visualization of grid"""
    if title:
        print(f"\n{title}:")
    print("-" * (grid.shape[1] * 2 + 1))
    for row in grid:
        print("|" + " ".join(str(c) for c in row) + "|")
    print("-" * (grid.shape[1] * 2 + 1))


def solve_and_evaluate_task(solver: ARCGenerativeSolver, task: dict,
                            verbose: bool = True):
    """Solve a single task and evaluate results"""

    if verbose:
        print("\n" + "=" * 70)
        print(f"Task: {task.get('name', 'Unknown')}")
        print("=" * 70)

    # Solve
    pred1, pred2, metadata = solver.solve(task)

    # Get target
    test_input = np.array(task["test"][0]["input"])
    test_output = np.array(task["test"][0]["output"])

    # Evaluate
    results = evaluate_predictions(pred1, pred2, test_output)

    if verbose:
        # Show test input
        visualize_grid(test_input, "Test Input")

        # Show predictions
        visualize_grid(pred1, "Prediction 1")
        visualize_grid(pred2, "Prediction 2")

        # Show target
        visualize_grid(test_output, "Target Output")

        # Show top programs
        print("\nTop Programs:")
        for i, prog_info in enumerate(metadata["top_programs"], 1):
            print(f"\n  {i}. Schema: {prog_info['schema']}")
            print(f"     Parameters: {prog_info['parameters']}")
            print(f"     Probability: {prog_info['probability']:.4f}")
            print(f"     Complexity: {prog_info['complexity']:.2f}")

        # Show active inference metrics
        print("\nActive Inference Metrics:")
        print(f"  Free Energy: {metadata['free_energy']:.4f}")
        print(f"  Belief Entropy: {metadata['entropy']:.4f}")
        print(f"  Candidates Evaluated: {metadata['n_candidates']}")
        print(f"  Valid Programs: {metadata['n_valid']}")

        # Show evaluation results
        print("\nEvaluation Results:")
        print(f"  Prediction 1 - Exact Match: {results['exact_match_1']}")
        print(f"  Prediction 1 - Pixel Accuracy: {results['pixel_accuracy_1']:.2%}")
        print(f"  Prediction 2 - Exact Match: {results['exact_match_2']}")
        print(f"  Prediction 2 - Pixel Accuracy: {results['pixel_accuracy_2']:.2%}")
        print(f"  Any Correct: {bool(results['any_correct'])}")

        if results['any_correct']:
            print("\n  ✓ SOLVED!")
        else:
            print("\n  ✗ Not solved")

    return results, metadata


def run_benchmark(tasks: list, solver: ARCGenerativeSolver):
    """Run solver on multiple tasks and aggregate results"""

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    all_results = []

    for task in tasks:
        results, metadata = solve_and_evaluate_task(
            solver, task, verbose=False
        )
        all_results.append({
            "name": task.get("name", "Unknown"),
            "results": results,
            "metadata": metadata
        })

    # Aggregate statistics
    n_tasks = len(all_results)
    n_solved_pred1 = sum(r["results"]["exact_match_1"] for r in all_results)
    n_solved_pred2 = sum(r["results"]["exact_match_2"] for r in all_results)
    n_solved_any = sum(r["results"]["any_correct"] for r in all_results)

    avg_pixel_acc_1 = np.mean([r["results"]["pixel_accuracy_1"]
                                for r in all_results])
    avg_pixel_acc_2 = np.mean([r["results"]["pixel_accuracy_2"]
                                for r in all_results])

    avg_free_energy = np.mean([r["metadata"]["free_energy"]
                               for r in all_results])
    avg_entropy = np.mean([r["metadata"]["entropy"]
                          for r in all_results])

    # Print summary
    print(f"\nTotal Tasks: {n_tasks}")
    print(f"\nPrediction 1 Accuracy: {n_solved_pred1}/{n_tasks} "
          f"({n_solved_pred1/n_tasks:.1%})")
    print(f"Prediction 2 Accuracy: {n_solved_pred2}/{n_tasks} "
          f"({n_solved_pred2/n_tasks:.1%})")
    print(f"Any Prediction Correct: {n_solved_any}/{n_tasks} "
          f"({n_solved_any/n_tasks:.1%})")

    print(f"\nAverage Pixel Accuracy (Pred 1): {avg_pixel_acc_1:.2%}")
    print(f"Average Pixel Accuracy (Pred 2): {avg_pixel_acc_2:.2%}")

    print(f"\nAverage Free Energy: {avg_free_energy:.4f}")
    print(f"Average Belief Entropy: {avg_entropy:.4f}")

    # Per-task breakdown
    print("\nPer-Task Results:")
    print("-" * 70)
    for r in all_results:
        status = "✓" if r["results"]["any_correct"] else "✗"
        print(f"{status} {r['name']:30s} - "
              f"P1: {r['results']['pixel_accuracy_1']:5.1%}, "
              f"P2: {r['results']['pixel_accuracy_2']:5.1%}")

    return all_results


def main():
    """Main demonstration"""

    print("=" * 70)
    print("ARC Generative Solver with Active Inference")
    print("Demonstration and Testing")
    print("=" * 70)

    # Create solver
    solver = ARCGenerativeSolver(
        max_candidates=100,
        beam_width=15,
        active_inference_steps=5
    )

    # Create example tasks
    tasks = create_example_tasks()

    # Option 1: Solve single task with detailed output
    print("\n" + "=" * 70)
    print("MODE 1: Single Task Demonstration")
    print("=" * 70)

    solve_and_evaluate_task(solver, tasks[0], verbose=True)

    # Option 2: Run benchmark on all tasks
    print("\n" + "=" * 70)
    print("MODE 2: Benchmark on Multiple Tasks")
    print("=" * 70)

    run_benchmark(tasks, solver)

    # Option 3: Demonstrate active inference dynamics
    print("\n" + "=" * 70)
    print("MODE 3: Active Inference Dynamics")
    print("=" * 70)

    # Show how beliefs evolve
    task = tasks[0]
    print(f"\nAnalyzing task: {task['name']}")

    # Create a new solver to see fresh dynamics
    solver_demo = ARCGenerativeSolver(
        max_candidates=50,
        beam_width=10,
        active_inference_steps=5
    )

    # Generate candidates
    candidates = solver_demo.generator.generate_candidates(task, 50)
    print(f"\nGenerated {len(candidates)} candidate programs")

    # Initialize beliefs
    solver_demo.active_inference.initialize_beliefs(candidates)
    initial_entropy = solver_demo.active_inference.beliefs.entropy()
    print(f"Initial belief entropy: {initial_entropy:.4f}")

    # Evaluate and update beliefs step by step
    for step in range(5):
        likelihoods = []
        valid_programs = []

        for program in candidates:
            likelihood = solver_demo._evaluate_program(program, task)
            if likelihood > 0:
                likelihoods.append(likelihood)
                valid_programs.append(program)

        if not valid_programs:
            break

        solver_demo.active_inference.update_beliefs(valid_programs, likelihoods)

        beliefs = solver_demo.active_inference.beliefs
        top_prog = solver_demo.active_inference.get_top_programs(k=1)

        print(f"\nStep {step + 1}:")
        print(f"  Valid programs: {len(valid_programs)}")
        print(f"  Free energy: {beliefs.free_energy:.4f}")
        print(f"  Entropy: {beliefs.entropy():.4f}")
        if top_prog:
            print(f"  Top program: {top_prog[0][0].schema} "
                  f"(p={top_prog[0][1]:.4f})")

        # Sample for next iteration
        if step < 4:
            candidates = solver_demo.active_inference.sample_programs(n=10)

    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
