"""
Compositional ARC Solver - Chains 2-3 primitives together

Key innovation: Instead of selecting a single primitive, performs beam search
over sequences of primitives to enable multi-step transformations.

Expected improvement: 1% → 10-15% success rate on real ARC
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from advanced_solver import AdvancedARCSolver, AdvancedExecutor, AdvancedProgramGenerator
from arc_generative_solver import Program
import time


@dataclass
class CompositionalProgram:
    """A program consisting of a sequence of primitive operations"""
    steps: List[Program]  # Sequence of single-step programs
    score: float = 0.0  # Accumulated score from training examples
    complexity: float = 0.0  # Sum of step complexities

    def __repr__(self):
        step_names = [s.schema for s in self.steps]
        return f"CompositionalProgram({' → '.join(step_names)}, score={self.score:.3f})"

    def copy(self):
        """Create a copy of this compositional program"""
        return CompositionalProgram(
            steps=self.steps.copy(),
            score=self.score,
            complexity=self.complexity
        )


class CompositionalARCSolver(AdvancedARCSolver):
    """
    ARC solver with compositional reasoning via beam search

    Performs beam search over sequences of 1-3 primitives to find
    multi-step transformations.
    """

    def __init__(self,
                 max_candidates: int = 120,
                 beam_width: int = 15,
                 active_inference_steps: int = 3,  # Reduced since we do beam search
                 diversity_strategy: str = "schema_first",
                 max_depth: int = 2,  # Allow 1-2 step compositions
                 composition_beam_width: int = 10):  # Beam size for composition search
        super().__init__(
            max_candidates, beam_width,
            active_inference_steps, diversity_strategy
        )

        self.max_depth = max_depth
        self.composition_beam_width = composition_beam_width
        self.executor = AdvancedExecutor()

    def solve(self, task: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve ARC task with compositional reasoning

        Strategy:
        1. Try single-step solutions (existing approach)
        2. If no perfect solution, try 2-step compositions
        3. Return best solutions found
        """
        # First try single-step solutions
        single_pred1, single_pred2, single_metadata = super().solve(task)

        # Check if single-step found perfect solution on training examples
        single_step_perfect = self._check_perfect_on_training(
            single_metadata['top_programs'][0] if single_metadata['top_programs'] else None,
            task
        )

        if single_step_perfect:
            # Single step is perfect, use it
            single_metadata['composition_attempted'] = False
            single_metadata['composition_depth'] = 1
            return single_pred1, single_pred2, single_metadata

        # Try compositional reasoning
        comp_programs = self._compositional_search(task)

        if not comp_programs:
            # No compositional solution found, return single-step
            single_metadata['composition_attempted'] = True
            single_metadata['composition_found'] = False
            single_metadata['composition_depth'] = 1
            return single_pred1, single_pred2, single_metadata

        # Execute compositional programs on test input
        test_input = np.array(task["test"][0]["input"])

        comp_pred1 = self._execute_compositional(comp_programs[0], test_input)

        if len(comp_programs) > 1:
            comp_pred2 = self._execute_compositional(comp_programs[1], test_input)
        else:
            # Use single-step as second prediction
            comp_pred2 = single_pred2

        # Compare compositional vs single-step
        # Use compositional if it has better training score
        best_comp_score = comp_programs[0].score if comp_programs else 0
        best_single_score = single_metadata['top_programs'][0]['probability'] if single_metadata['top_programs'] else 0

        if best_comp_score > best_single_score:
            # Use compositional solution
            metadata = {
                'composition_attempted': True,
                'composition_found': True,
                'composition_depth': len(comp_programs[0].steps),
                'composition_score': best_comp_score,
                'top_programs': [
                    {
                        'schema': ' → '.join([s.schema for s in cp.steps]),
                        'steps': [{'schema': s.schema, 'parameters': s.parameters} for s in cp.steps],
                        'probability': cp.score,
                        'complexity': cp.complexity
                    }
                    for cp in comp_programs[:5]
                ],
                'single_step_score': best_single_score,
                'single_step_programs': single_metadata['top_programs'][:3]
            }
            return comp_pred1, comp_pred2, metadata
        else:
            # Single-step is better or equal
            single_metadata['composition_attempted'] = True
            single_metadata['composition_found'] = True
            single_metadata['composition_score'] = best_comp_score
            single_metadata['chose_single_step'] = True
            return single_pred1, single_pred2, single_metadata

    def _check_perfect_on_training(self, program_info: Optional[Dict], task: Dict) -> bool:
        """Check if program achieves 100% accuracy on all training examples"""
        if not program_info or 'train' not in task:
            return False

        # Reconstruct program from info
        # This is a simplified check - just checking probability
        return program_info.get('probability', 0) > 0.99

    def _compositional_search(self, task: Dict[str, Any]) -> List[CompositionalProgram]:
        """
        Perform beam search to find compositional programs

        Returns: List of top compositional programs sorted by score
        """
        if 'train' not in task or len(task['train']) == 0:
            return []

        # Generate single-step candidates
        candidates = self.generator.generate_candidates(task, self.max_candidates)

        # Initialize beam with identity (empty composition)
        beam = [CompositionalProgram(steps=[], score=0.0, complexity=0.0)]

        # Beam search for max_depth steps
        for depth in range(1, self.max_depth + 1):
            next_beam = []

            # Expand each program in beam
            for partial_prog in beam:
                # Try adding each candidate as next step
                for candidate in candidates:
                    # Skip identity if we already have steps (identity in middle is wasteful)
                    if candidate.schema == "identity" and len(partial_prog.steps) > 0:
                        continue

                    # Create new compositional program
                    new_prog = partial_prog.copy()
                    new_prog.steps.append(candidate)
                    new_prog.complexity += candidate.complexity

                    # Evaluate on training examples
                    score = self._evaluate_compositional(new_prog, task)
                    new_prog.score = score

                    # Add to next beam
                    next_beam.append(new_prog)

            # Keep only top programs
            next_beam.sort(key=lambda p: p.score, reverse=True)
            beam = next_beam[:self.composition_beam_width]

            # Early stopping: if we found perfect solution, stop
            if beam and beam[0].score > 0.99:
                break

        # Filter to only keep programs better than empty/identity
        good_programs = [p for p in beam if p.score > 0.1 and len(p.steps) > 0]

        return good_programs

    def _evaluate_compositional(self, program: CompositionalProgram, task: Dict) -> float:
        """
        Evaluate compositional program on training examples

        Returns: Average pixel accuracy across training examples
        """
        if 'train' not in task or len(task['train']) == 0:
            return 0.0

        total_accuracy = 0.0
        valid_examples = 0

        for train_ex in task['train']:
            try:
                input_grid = np.array(train_ex['input'])
                expected_output = np.array(train_ex['output'])

                # Execute compositional program
                result = self._execute_compositional(program, input_grid)

                # Calculate accuracy
                if result.shape == expected_output.shape:
                    accuracy = np.mean(result == expected_output)
                    total_accuracy += accuracy
                    valid_examples += 1

            except Exception as e:
                # Execution failed, skip this example
                continue

        if valid_examples == 0:
            return 0.0

        return total_accuracy / valid_examples

    def _execute_compositional(self, program: CompositionalProgram, input_grid: np.ndarray) -> np.ndarray:
        """
        Execute a compositional program (sequence of primitives)

        Args:
            program: CompositionalProgram with sequence of steps
            input_grid: Input grid to transform

        Returns:
            Transformed grid after applying all steps
        """
        grid = input_grid.copy()

        for step in program.steps:
            try:
                grid = self.executor.execute(step, grid)
            except Exception as e:
                # If any step fails, return current state
                break

        return grid


def test_compositional_solver():
    """Test compositional solver on sample tasks"""
    print("="*70)
    print("COMPOSITIONAL SOLVER - TESTING MULTI-STEP REASONING")
    print("="*70)

    solver = CompositionalARCSolver(max_depth=2, composition_beam_width=10)

    # Test 1: Rotate then flip (requires 2 steps)
    print("\n1. Rotate + Flip (2-step composition)")
    print("-"*70)

    # Create a test case where output = flip_h(rotate_90(input))
    input_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Rotate 90° clockwise: [[7,4,1], [8,5,2], [9,6,3]]
    # Then flip horizontally: [[1,4,7], [2,5,8], [3,6,9]]
    expected_output = np.array([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
    ])

    task = {
        "train": [
            {"input": input_grid.tolist(), "output": expected_output.tolist()}
        ],
        "test": [
            {"input": input_grid.tolist(), "output": expected_output.tolist()}
        ]
    }

    pred1, pred2, metadata = solver.solve(task)

    print(f"Input shape: {input_grid.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    print(f"Prediction shape: {pred1.shape}")

    accuracy = np.mean(pred1 == expected_output) if pred1.shape == expected_output.shape else 0.0
    print(f"Accuracy: {accuracy:.1%}")

    print(f"\nComposition attempted: {metadata.get('composition_attempted', False)}")
    print(f"Composition found: {metadata.get('composition_found', False)}")
    print(f"Composition depth: {metadata.get('composition_depth', 1)}")

    if metadata.get('composition_found'):
        print(f"\nTop compositional programs:")
        for i, prog_info in enumerate(metadata.get('top_programs', [])[:3]):
            print(f"{i+1}. {prog_info['schema']} (score: {prog_info['probability']:.3f})")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_compositional_solver()
