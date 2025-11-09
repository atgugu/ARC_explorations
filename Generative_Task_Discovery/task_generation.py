"""
Task Generation: Reverse Inference (Program → Task)

Implements the generative part of the framework:
1. Sample programs from TRG prior
2. Generate diverse input grids
3. Execute programs to create outputs
4. Verify solvability
5. Use for self-curriculum learning
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import random
from arc_generative_solver import (
    Program, TRGPrimitives, ARCGenerativeSolver,
    Executor, evaluate_predictions
)


@dataclass
class GeneratedTask:
    """A generated ARC task with metadata"""
    task: Dict[str, Any]
    source_program: Program
    difficulty: float
    is_solvable: bool
    solver_success: bool = False
    generation_metadata: Dict[str, Any] = None


class InputGridGenerator:
    """Generate diverse input grids for task creation"""

    def __init__(self, grid_size_range: Tuple[int, int] = (3, 10)):
        self.min_size, self.max_size = grid_size_range
        self.primitives = TRGPrimitives()

    def generate_random_grid(self,
                           height: Optional[int] = None,
                           width: Optional[int] = None,
                           n_colors: int = 3,
                           density: float = 0.3) -> np.ndarray:
        """Generate random grid with specified properties"""
        h = height or random.randint(self.min_size, self.max_size)
        w = width or random.randint(self.min_size, self.max_size)

        # Create sparse grid
        grid = np.zeros((h, w), dtype=int)

        # Add random colored pixels
        n_pixels = int(h * w * density)
        for _ in range(n_pixels):
            i = random.randint(0, h - 1)
            j = random.randint(0, w - 1)
            color = random.randint(1, min(n_colors, 9))
            grid[i, j] = color

        return grid

    def generate_structured_grid(self,
                                height: Optional[int] = None,
                                width: Optional[int] = None,
                                structure_type: str = "objects") -> np.ndarray:
        """Generate grid with structured content"""
        h = height or random.randint(self.min_size, self.max_size)
        w = width or random.randint(self.min_size, self.max_size)

        grid = np.zeros((h, w), dtype=int)

        if structure_type == "objects":
            # Add simple geometric objects
            n_objects = random.randint(1, 4)

            for _ in range(n_objects):
                obj_size = random.randint(1, min(3, h//2, w//2))
                color = random.randint(1, 5)

                # Random position
                y = random.randint(0, h - obj_size)
                x = random.randint(0, w - obj_size)

                # Random shape
                shape_type = random.choice(['square', 'line_h', 'line_v', 'single'])

                if shape_type == 'square':
                    grid[y:y+obj_size, x:x+obj_size] = color
                elif shape_type == 'line_h':
                    grid[y, x:x+obj_size] = color
                elif shape_type == 'line_v':
                    grid[y:y+obj_size, x] = color
                elif shape_type == 'single':
                    grid[y, x] = color

        elif structure_type == "pattern":
            # Create repeating pattern
            pattern_size = random.randint(1, min(3, h//2, w//2))
            pattern = np.random.randint(0, 4, (pattern_size, pattern_size))

            # Tile the pattern
            for i in range(0, h, pattern_size):
                for j in range(0, w, pattern_size):
                    end_i = min(i + pattern_size, h)
                    end_j = min(j + pattern_size, w)
                    grid[i:end_i, j:end_j] = pattern[:end_i-i, :end_j-j]

        elif structure_type == "symmetric":
            # Create half, then mirror
            half_w = w // 2
            left_half = np.random.randint(0, 4, (h, half_w))
            grid[:, :half_w] = left_half

            # Mirror to right
            if w % 2 == 0:
                grid[:, half_w:] = np.fliplr(left_half)
            else:
                grid[:, half_w+1:] = np.fliplr(left_half)

        return grid

    def generate_grid_for_schema(self, schema: str) -> np.ndarray:
        """Generate grid appropriate for a specific schema"""

        if schema in ["rotation", "reflection"]:
            # Asymmetric grid works well
            return self.generate_structured_grid(
                height=random.randint(3, 6),
                width=random.randint(3, 6),
                structure_type="objects"
            )

        elif schema == "translation":
            # Grid with object that can be moved
            h, w = random.randint(5, 8), random.randint(5, 8)
            grid = np.zeros((h, w), dtype=int)

            # Add small object in corner
            obj_size = random.randint(1, 2)
            color = random.randint(1, 5)
            grid[:obj_size, :obj_size] = color

            return grid

        elif schema == "color_remap":
            # Grid with multiple colors
            return self.generate_random_grid(
                n_colors=5,
                density=0.4
            )

        elif schema == "composite":
            # Complex grid
            return self.generate_structured_grid(
                structure_type=random.choice(["objects", "pattern"])
            )

        else:
            # Default
            return self.generate_random_grid()

    def generate_diverse_batch(self, n: int = 5) -> List[np.ndarray]:
        """Generate batch of diverse grids"""
        grids = []

        # Mix of different generation strategies
        for i in range(n):
            strategy = random.choice([
                "random",
                "objects",
                "pattern",
                "symmetric"
            ])

            if strategy == "random":
                grid = self.generate_random_grid()
            else:
                grid = self.generate_structured_grid(structure_type=strategy)

            grids.append(grid)

        return grids


class TaskGenerator:
    """Generate ARC tasks from programs"""

    def __init__(self, seed: Optional[int] = None):
        self.input_generator = InputGridGenerator()
        self.executor = Executor()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def sample_program_from_prior(self,
                                  complexity_bias: float = 1.0) -> Program:
        """Sample a program from the prior distribution"""

        # Schema probabilities (complexity-biased)
        schemas = [
            ("identity", 0.1, 0.1),
            ("rotation", 1.0, 3.0),
            ("reflection", 1.0, 3.0),
            ("translation", 2.0, 2.0),
            ("color_remap", 2.0, 1.5),
            ("composite", 3.0, 1.0)
        ]

        # Weight by complexity bias
        weights = [prob * np.exp(-complexity_bias * complexity)
                  for _, complexity, prob in schemas]
        weights = np.array(weights) / sum(weights)

        # Sample schema
        schema_idx = np.random.choice(len(schemas), p=weights)
        schema_name, complexity, _ = schemas[schema_idx]

        # Generate parameters based on schema
        if schema_name == "identity":
            return Program(
                schema="identity",
                primitives=[],
                parameters={},
                selectors={},
                complexity=0.1
            )

        elif schema_name == "rotation":
            k = random.choice([1, 2, 3])
            return Program(
                schema="rotation",
                primitives=["rotate"],
                parameters={"k": k},
                selectors={},
                complexity=1.0
            )

        elif schema_name == "reflection":
            axis = random.choice(["h", "v"])
            return Program(
                schema="reflection",
                primitives=["reflect"],
                parameters={"axis": axis},
                selectors={},
                complexity=1.0
            )

        elif schema_name == "translation":
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            # Avoid identity translation
            if dx == 0 and dy == 0:
                dx = 1
            return Program(
                schema="translation",
                primitives=["translate"],
                parameters={"dx": dx, "dy": dy},
                selectors={},
                complexity=2.0
            )

        elif schema_name == "color_remap":
            # Generate random color mapping
            n_colors = random.randint(2, 5)
            src_colors = random.sample(range(1, 10), n_colors)
            dst_colors = random.sample(range(1, 10), n_colors)
            mapping = dict(zip(src_colors, dst_colors))

            return Program(
                schema="color_remap",
                primitives=["remap_color"],
                parameters={"mapping": mapping},
                selectors={},
                complexity=2.0
            )

        elif schema_name == "composite":
            # Rotation + reflection
            k = random.choice([1, 2, 3])
            axis = random.choice(["h", "v"])
            return Program(
                schema="composite",
                primitives=["rotate", "reflect"],
                parameters={"k": k, "axis": axis},
                selectors={},
                complexity=3.0
            )

        # Default fallback
        return Program(
            schema="identity",
            primitives=[],
            parameters={},
            selectors={},
            complexity=0.1
        )

    def generate_task_from_program(self,
                                   program: Program,
                                   n_train: int = 3,
                                   n_test: int = 1) -> Dict[str, Any]:
        """
        Generate a task by:
        1. Creating input grids
        2. Executing program to get outputs
        3. Formatting as ARC task
        """

        # Generate input grids
        inputs = []
        for _ in range(n_train + n_test):
            grid = self.input_generator.generate_grid_for_schema(program.schema)
            inputs.append(grid)

        # Execute program on all inputs
        outputs = []
        for inp in inputs:
            try:
                out = self.executor.execute(program, inp)
                outputs.append(out)
            except Exception as e:
                # Execution failed, return None
                return None

        # Split into train/test
        train_pairs = [
            {"input": inputs[i].tolist(), "output": outputs[i].tolist()}
            for i in range(n_train)
        ]

        test_pairs = [
            {"input": inputs[n_train + i].tolist(),
             "output": outputs[n_train + i].tolist()}
            for i in range(n_test)
        ]

        task = {
            "train": train_pairs,
            "test": test_pairs
        }

        return task

    def verify_task_solvable(self,
                            task: Dict[str, Any],
                            solver: ARCGenerativeSolver,
                            threshold: float = 0.9) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that a generated task is solvable by the solver
        """

        try:
            pred1, pred2, metadata = solver.solve(task)

            # Check against ground truth
            target = np.array(task["test"][0]["output"])
            eval_results = evaluate_predictions(pred1, pred2, target)

            # Task is solvable if accuracy is high
            is_solvable = eval_results["pixel_accuracy_1"] >= threshold

            return is_solvable, {
                "eval_results": eval_results,
                "metadata": metadata
            }

        except Exception as e:
            return False, {"error": str(e)}

    def estimate_difficulty(self, program: Program, task: Dict[str, Any]) -> float:
        """
        Estimate task difficulty based on:
        - Program complexity
        - Grid size
        - Number of objects
        - Ambiguity
        """

        difficulty = program.complexity

        # Grid size factor
        if task["train"]:
            input_grid = np.array(task["train"][0]["input"])
            size_factor = (input_grid.shape[0] * input_grid.shape[1]) / 100.0
            difficulty += size_factor * 0.5

        # Number of training examples (fewer = harder)
        n_train = len(task["train"])
        if n_train < 3:
            difficulty += (3 - n_train) * 0.5

        return difficulty

    def generate_task(self,
                     program: Optional[Program] = None,
                     n_train: int = 3,
                     n_test: int = 1,
                     verify_solvable: bool = False,
                     solver: Optional[ARCGenerativeSolver] = None) -> Optional[GeneratedTask]:
        """
        Generate a complete task with metadata
        """

        # Sample program if not provided
        if program is None:
            program = self.sample_program_from_prior()

        # Generate task
        task = self.generate_task_from_program(program, n_train, n_test)

        if task is None:
            return None

        # Estimate difficulty
        difficulty = self.estimate_difficulty(program, task)

        # Verify solvability if requested
        is_solvable = True
        solver_success = False
        verification_metadata = {}

        if verify_solvable and solver is not None:
            is_solvable, verification_metadata = self.verify_task_solvable(
                task, solver
            )
            if is_solvable:
                solver_success = verification_metadata["eval_results"]["any_correct"] > 0.5

        return GeneratedTask(
            task=task,
            source_program=program,
            difficulty=difficulty,
            is_solvable=is_solvable,
            solver_success=solver_success,
            generation_metadata=verification_metadata
        )

    def generate_curriculum(self,
                           n_tasks: int = 20,
                           difficulty_range: Tuple[float, float] = (0.5, 3.0),
                           verify_all: bool = True,
                           solver: Optional[ARCGenerativeSolver] = None) -> List[GeneratedTask]:
        """
        Generate a curriculum of tasks with increasing difficulty
        """

        tasks = []
        min_diff, max_diff = difficulty_range

        for i in range(n_tasks):
            # Target difficulty increases over curriculum
            target_difficulty = min_diff + (max_diff - min_diff) * (i / n_tasks)

            # Sample programs until we get appropriate difficulty
            attempts = 0
            max_attempts = 20

            while attempts < max_attempts:
                # Bias toward target difficulty
                complexity_bias = 1.0 / (target_difficulty + 0.1)
                program = self.sample_program_from_prior(complexity_bias)

                # Generate task
                generated = self.generate_task(
                    program=program,
                    n_train=random.randint(2, 4),
                    verify_solvable=verify_all,
                    solver=solver
                )

                if generated is not None:
                    # Check difficulty match
                    if abs(generated.difficulty - target_difficulty) < 1.0:
                        tasks.append(generated)
                        break

                attempts += 1

            # If failed to generate, add any valid task
            if len(tasks) <= i:
                generated = self.generate_task(
                    verify_solvable=verify_all,
                    solver=solver
                )
                if generated is not None:
                    tasks.append(generated)

        return tasks


# ============================================================================
# Self-Curriculum Learning
# ============================================================================

class SelfCurriculumEngine:
    """
    Self-curriculum learning:
    1. Generate tasks
    2. Train solver
    3. Generate harder tasks
    4. Repeat
    """

    def __init__(self, solver: ARCGenerativeSolver):
        self.solver = solver
        self.task_generator = TaskGenerator()
        self.generated_tasks: List[GeneratedTask] = []
        self.performance_history = []

    def run_curriculum_step(self,
                           n_tasks: int = 10,
                           difficulty_bias: float = 1.0) -> Dict[str, Any]:
        """
        Run one step of self-curriculum:
        1. Generate tasks at current difficulty
        2. Evaluate solver on them
        3. Return statistics
        """

        print(f"\n{'='*70}")
        print(f"Self-Curriculum Step (difficulty_bias={difficulty_bias:.2f})")
        print(f"{'='*70}")

        # Generate tasks
        print(f"\nGenerating {n_tasks} tasks...")
        tasks = []

        for i in range(n_tasks):
            program = self.task_generator.sample_program_from_prior(
                complexity_bias=1.0/difficulty_bias
            )

            generated = self.task_generator.generate_task(
                program=program,
                n_train=random.randint(2, 4),
                verify_solvable=True,
                solver=self.solver
            )

            if generated is not None:
                tasks.append(generated)
                self.generated_tasks.append(generated)

        # Statistics
        n_generated = len(tasks)
        n_solvable = sum(1 for t in tasks if t.is_solvable)
        n_solved = sum(1 for t in tasks if t.solver_success)

        avg_difficulty = np.mean([t.difficulty for t in tasks]) if tasks else 0

        stats = {
            "n_generated": n_generated,
            "n_solvable": n_solvable,
            "n_solved": n_solved,
            "solvability_rate": n_solvable / n_generated if n_generated > 0 else 0,
            "success_rate": n_solved / n_solvable if n_solvable > 0 else 0,
            "avg_difficulty": avg_difficulty,
            "tasks": tasks
        }

        self.performance_history.append(stats)

        # Print results
        print(f"\nResults:")
        print(f"  Generated: {n_generated}")
        print(f"  Solvable: {n_solvable} ({stats['solvability_rate']:.1%})")
        print(f"  Solved: {n_solved} ({stats['success_rate']:.1%})")
        print(f"  Avg Difficulty: {avg_difficulty:.2f}")

        return stats

    def run_adaptive_curriculum(self,
                              n_steps: int = 5,
                              tasks_per_step: int = 10) -> Dict[str, Any]:
        """
        Run adaptive self-curriculum:
        - Start with easy tasks
        - Increase difficulty as solver improves
        """

        print(f"\n{'='*70}")
        print(f"ADAPTIVE SELF-CURRICULUM")
        print(f"{'='*70}")
        print(f"\n{n_steps} steps, {tasks_per_step} tasks per step")

        for step in range(n_steps):
            # Adapt difficulty based on performance
            if step == 0:
                difficulty = 0.5  # Start easy
            else:
                # Increase difficulty if doing well
                prev_success = self.performance_history[-1]["success_rate"]
                if prev_success > 0.7:
                    difficulty = min(3.0, difficulty * 1.5)
                elif prev_success < 0.3:
                    difficulty = max(0.5, difficulty * 0.8)

            # Run step
            stats = self.run_curriculum_step(
                n_tasks=tasks_per_step,
                difficulty_bias=difficulty
            )

        # Final summary
        print(f"\n{'='*70}")
        print(f"CURRICULUM COMPLETE")
        print(f"{'='*70}")

        total_generated = sum(s["n_generated"] for s in self.performance_history)
        total_solved = sum(s["n_solved"] for s in self.performance_history)

        print(f"\nTotal Tasks Generated: {total_generated}")
        print(f"Total Tasks Solved: {total_solved}")
        print(f"Overall Success Rate: {total_solved/total_generated:.1%}")

        return {
            "history": self.performance_history,
            "total_generated": total_generated,
            "total_solved": total_solved
        }


if __name__ == "__main__":
    print("="*70)
    print("TASK GENERATION: Reverse Inference (Program → Task)")
    print("="*70)

    # Create generator and solver
    generator = TaskGenerator(seed=42)
    solver = ARCGenerativeSolver(
        max_candidates=100,
        beam_width=15,
        active_inference_steps=5
    )

    # Test 1: Generate single tasks
    print("\n" + "="*70)
    print("TEST 1: Generate Tasks from Sampled Programs")
    print("="*70)

    for i in range(5):
        program = generator.sample_program_from_prior()
        generated = generator.generate_task(
            program=program,
            verify_solvable=True,
            solver=solver
        )

        if generated:
            print(f"\nTask {i+1}:")
            print(f"  Program: {program.schema}")
            print(f"  Parameters: {program.parameters}")
            print(f"  Difficulty: {generated.difficulty:.2f}")
            print(f"  Solvable: {generated.is_solvable}")
            print(f"  Solver Success: {generated.solver_success}")

    # Test 2: Generate curriculum
    print("\n" + "="*70)
    print("TEST 2: Generate Curriculum")
    print("="*70)

    curriculum = generator.generate_curriculum(
        n_tasks=10,
        difficulty_range=(0.5, 2.5),
        verify_all=True,
        solver=solver
    )

    print(f"\nGenerated {len(curriculum)} tasks")
    print(f"Difficulty range: {min(t.difficulty for t in curriculum):.2f} - {max(t.difficulty for t in curriculum):.2f}")
    print(f"Solvable: {sum(1 for t in curriculum if t.is_solvable)}/{len(curriculum)}")
    print(f"Solved: {sum(1 for t in curriculum if t.solver_success)}/{len(curriculum)}")

    # Show examples
    print("\nSample tasks:")
    for i, task in enumerate(curriculum[:3]):
        print(f"\n  Task {i+1}: {task.source_program.schema} (diff={task.difficulty:.2f}, solved={task.solver_success})")
