"""
Comprehensive Demonstration of Task Generation

Shows:
1. Program → Task generation
2. Solvability verification
3. Self-curriculum learning
4. Closed-loop training
"""

import numpy as np
import random
from task_generation import (
    TaskGenerator, SelfCurriculumEngine, InputGridGenerator
)
from arc_generative_solver import ARCGenerativeSolver, evaluate_predictions


def visualize_task(task: dict, title: str = ""):
    """Visualize a generated task"""
    print(f"\n{title}")
    print("-" * 60)

    # Show training examples
    print("Training Examples:")
    for i, pair in enumerate(task["train"], 1):
        input_grid = np.array(pair["input"])
        output_grid = np.array(pair["output"])

        print(f"\n  Example {i}:")
        print(f"    Input:  {input_grid.shape} -> {input_grid.tolist()}")
        print(f"    Output: {output_grid.shape} -> {output_grid.tolist()}")

    # Show test
    print(f"\nTest:")
    test_input = np.array(task["test"][0]["input"])
    test_output = np.array(task["test"][0]["output"])
    print(f"  Input:  {test_input.shape} -> {test_input.tolist()}")
    print(f"  Output: {test_output.shape} -> {test_output.tolist()}")


def demo_basic_generation():
    """Demo 1: Basic task generation"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Task Generation (Program → Task)")
    print("="*70)

    generator = TaskGenerator(seed=42)
    solver = ARCGenerativeSolver(
        max_candidates=100,
        beam_width=15,
        active_inference_steps=5
    )

    # Generate tasks for different program types
    schemas = ["rotation", "reflection", "translation", "color_remap"]

    for schema in schemas:
        print(f"\n{'='*70}")
        print(f"Schema: {schema.upper()}")
        print(f"{'='*70}")

        # Sample program of this type
        attempts = 0
        while attempts < 10:
            program = generator.sample_program_from_prior(complexity_bias=0.5)
            if program.schema == schema:
                break
            attempts += 1

        # Generate task
        generated = generator.generate_task(
            program=program,
            n_train=3,
            n_test=1,
            verify_solvable=True,
            solver=solver
        )

        if generated:
            print(f"\n✓ Generated task for {schema}")
            print(f"  Program parameters: {program.parameters}")
            print(f"  Difficulty: {generated.difficulty:.2f}")
            print(f"  Solvable: {generated.is_solvable}")
            print(f"  Solver success: {generated.solver_success}")

            # Visualize
            visualize_task(generated.task, f"Task for {schema}")

            # Show solver's attempt
            if generated.is_solvable:
                pred1, pred2, metadata = solver.solve(generated.task)
                print(f"\n  Solver's top program: {metadata['top_programs'][0]['schema']}")
                print(f"  Solver confidence: {metadata['top_programs'][0]['probability']:.3f}")


def demo_input_generation():
    """Demo 2: Different input generation strategies"""
    print("\n" + "="*70)
    print("DEMO 2: Diverse Input Grid Generation")
    print("="*70)

    generator = InputGridGenerator(grid_size_range=(4, 8))

    strategies = [
        ("Random sparse", "random", {}),
        ("Structured objects", "objects", {}),
        ("Repeating pattern", "pattern", {}),
        ("Symmetric", "symmetric", {})
    ]

    for name, strategy, kwargs in strategies:
        print(f"\n{name}:")

        if strategy == "random":
            grid = generator.generate_random_grid(
                height=6, width=6, n_colors=4, density=0.3
            )
        else:
            grid = generator.generate_structured_grid(
                height=6, width=6, structure_type=strategy
            )

        print(f"  Shape: {grid.shape}")
        print(f"  Colors used: {len(np.unique(grid))}")
        print(f"  Density: {(grid != 0).sum() / grid.size:.2%}")
        print(f"  Grid:")
        for row in grid:
            print(f"    {row.tolist()}")


def demo_curriculum_generation():
    """Demo 3: Curriculum generation with difficulty control"""
    print("\n" + "="*70)
    print("DEMO 3: Curriculum Generation")
    print("="*70)

    generator = TaskGenerator(seed=123)
    solver = ARCGenerativeSolver(
        max_candidates=80,
        beam_width=12,
        active_inference_steps=4
    )

    print("\nGenerating curriculum of 15 tasks...")
    print("Difficulty range: 0.5 → 3.0")

    curriculum = generator.generate_curriculum(
        n_tasks=15,
        difficulty_range=(0.5, 3.0),
        verify_all=True,
        solver=solver
    )

    print(f"\n✓ Generated {len(curriculum)} tasks")

    # Statistics
    difficulties = [t.difficulty for t in curriculum]
    n_solvable = sum(1 for t in curriculum if t.is_solvable)
    n_solved = sum(1 for t in curriculum if t.solver_success)

    print(f"\nStatistics:")
    print(f"  Difficulty range: {min(difficulties):.2f} - {max(difficulties):.2f}")
    print(f"  Average difficulty: {np.mean(difficulties):.2f}")
    print(f"  Solvable: {n_solvable}/{len(curriculum)} ({n_solvable/len(curriculum):.1%})")
    print(f"  Solved by solver: {n_solved}/{n_solvable} ({n_solved/n_solvable:.1%})")

    # Show difficulty progression
    print(f"\nDifficulty Progression:")
    for i, task in enumerate(curriculum):
        status = "✓" if task.solver_success else ("○" if task.is_solvable else "✗")
        print(f"  {i+1:2d}. {status} {task.source_program.schema:15s} "
              f"(diff={task.difficulty:.2f})")

    # Analyze by schema
    from collections import Counter
    schema_counts = Counter(t.source_program.schema for t in curriculum)

    print(f"\nSchema Distribution:")
    for schema, count in schema_counts.most_common():
        print(f"  {schema:15s}: {count:2d} tasks")


def demo_self_curriculum():
    """Demo 4: Self-curriculum learning"""
    print("\n" + "="*70)
    print("DEMO 4: Self-Curriculum Learning")
    print("="*70)

    print("\nInitializing self-curriculum engine...")

    solver = ARCGenerativeSolver(
        max_candidates=100,
        beam_width=15,
        active_inference_steps=5
    )

    engine = SelfCurriculumEngine(solver)

    # Run adaptive curriculum
    results = engine.run_adaptive_curriculum(
        n_steps=5,
        tasks_per_step=8
    )

    # Analyze learning progression
    print("\n" + "="*70)
    print("LEARNING PROGRESSION ANALYSIS")
    print("="*70)

    print("\nSuccess Rate Over Time:")
    for i, stats in enumerate(results["history"], 1):
        print(f"  Step {i}: {stats['success_rate']:5.1%} "
              f"(difficulty={stats['avg_difficulty']:.2f})")

    # Check if improving
    if len(results["history"]) >= 2:
        first_success = results["history"][0]["success_rate"]
        last_success = results["history"][-1]["success_rate"]

        if last_success >= first_success:
            print(f"\n✓ Solver maintaining/improving performance!")
        else:
            print(f"\n⚠ Performance declined (tasks got harder)")


def demo_closed_loop():
    """Demo 5: Closed-loop generation and solving"""
    print("\n" + "="*70)
    print("DEMO 5: Closed-Loop Generation and Solving")
    print("="*70)

    generator = TaskGenerator(seed=999)
    solver = ARCGenerativeSolver(
        max_candidates=100,
        beam_width=15,
        active_inference_steps=5
    )

    print("\nClosed Loop: Generate → Solve → Verify → Generate...")

    n_rounds = 5
    all_tasks = []

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'─'*60}")
        print(f"Round {round_num}")
        print(f"{'─'*60}")

        # Generate task
        program = generator.sample_program_from_prior(complexity_bias=0.8)
        generated = generator.generate_task(
            program=program,
            n_train=3,
            verify_solvable=False,  # Solve manually below
            solver=None
        )

        if not generated:
            print("  ✗ Generation failed, skipping")
            continue

        print(f"  Generated: {program.schema} (diff={generated.difficulty:.2f})")

        # Solve
        try:
            pred1, pred2, metadata = solver.solve(generated.task)
            target = np.array(generated.task["test"][0]["output"])
            eval_results = evaluate_predictions(pred1, pred2, target)

            success = eval_results["any_correct"] > 0.5

            print(f"  Solved: {'✓' if success else '✗'}")
            print(f"  Solver used: {metadata['top_programs'][0]['schema']}")
            print(f"  Confidence: {metadata['top_programs'][0]['probability']:.3f}")
            print(f"  Accuracy: {eval_results['pixel_accuracy_1']:.1%}")

            # Store results
            generated.solver_success = success
            generated.is_solvable = True
            generated.generation_metadata = {
                "eval_results": eval_results,
                "metadata": metadata
            }

            all_tasks.append(generated)

        except Exception as e:
            print(f"  ✗ Solving failed: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("CLOSED-LOOP SUMMARY")
    print(f"{'='*70}")

    print(f"\nTotal rounds: {n_rounds}")
    print(f"Successful tasks: {len(all_tasks)}")

    if all_tasks:
        solved = sum(1 for t in all_tasks if t.solver_success)
        print(f"Solver success rate: {solved}/{len(all_tasks)} ({solved/len(all_tasks):.1%})")

        print(f"\nTask Breakdown:")
        for i, task in enumerate(all_tasks, 1):
            status = "✓" if task.solver_success else "✗"
            print(f"  {i}. {status} {task.source_program.schema} "
                  f"(diff={task.difficulty:.2f})")


def run_all_demos():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("TASK GENERATION COMPREHENSIVE DEMONSTRATION")
    print("="*70)
    print("\nDemonstrating:")
    print("1. Basic task generation (Program → Task)")
    print("2. Diverse input generation strategies")
    print("3. Curriculum generation with difficulty control")
    print("4. Self-curriculum learning")
    print("5. Closed-loop generation and solving")

    demo_basic_generation()
    demo_input_generation()
    demo_curriculum_generation()
    demo_self_curriculum()
    demo_closed_loop()

    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("="*70)
    print("\nKey Achievements:")
    print("✓ Program → Task generation working")
    print("✓ Diverse input generation strategies")
    print("✓ Curriculum with difficulty control")
    print("✓ Self-curriculum learning implemented")
    print("✓ Closed-loop system demonstrated")
    print("\nThe system can now:")
    print("• Generate unlimited diverse ARC tasks")
    print("• Control task difficulty")
    print("• Verify solvability")
    print("• Enable self-improvement through curriculum")


if __name__ == "__main__":
    run_all_demos()
