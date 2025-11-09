"""
Iterative Refiner: Refines a program through iterative repair loops.
V3 philosophy: "Refine 1 program 10 times" > "Try 100 programs once"
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Callable
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput
from utils.grid_utils import compute_iou


class IterativeRefiner:
    """
    Iteratively refines a program using repair loops.
    Implements the V1 approach that actually worked.
    """

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations

    def refine(
        self,
        program: Dict[str, Any],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Refine a program iteratively.

        Args:
            program: Initial program to refine
            train_examples: List of (input, output) pairs

        Returns:
            (refined_program, score)
        """
        best_program = program
        best_score = self._evaluate(program, train_examples)

        if best_score >= 0.99:
            return best_program, best_score  # Already excellent

        # Try refinement iterations
        for iteration in range(self.max_iterations):
            improved = False

            # Try placement repair
            refined, score = self._try_placement_repair(best_program, train_examples)
            if score > best_score:
                best_program = refined
                best_score = score
                improved = True

                if best_score >= 0.99:
                    return best_program, best_score

            # Try color repair
            refined, score = self._try_color_repair(best_program, train_examples)
            if score > best_score:
                best_program = refined
                best_score = score
                improved = True

                if best_score >= 0.99:
                    return best_program, best_score

            # Try scale repair
            refined, score = self._try_scale_repair(best_program, train_examples)
            if score > best_score:
                best_program = refined
                best_score = score
                improved = True

                if best_score >= 0.99:
                    return best_program, best_score

            # If no improvement this round, stop
            if not improved:
                break

        return best_program, best_score

    def _evaluate(self, program: Dict[str, Any], train_examples: List) -> float:
        """Evaluate program on training examples."""
        try:
            ious = []
            for input_grid, output_grid in train_examples:
                pred = program['function'](input_grid)
                iou = compute_iou(pred, output_grid)
                ious.append(iou)
            return np.mean(ious) if ious else 0.0
        except:
            return 0.0

    def _try_placement_repair(
        self,
        program: Dict[str, Any],
        train_examples: List
    ) -> Tuple[Dict[str, Any], float]:
        """Try placement repair."""
        # Get initial predictions
        predictions = []
        targets = []

        try:
            for input_grid, output_grid in train_examples:
                pred = program['function'](input_grid)
                predictions.append(pred)
                targets.append(output_grid)
        except:
            return program, 0.0

        # Learn offset
        best_offset = None
        best_iou = 0.0

        for dy in range(-5, 6):
            for dx in range(-5, 6):
                ious = []
                for pred, target in zip(predictions, targets):
                    translated = self._translate_grid(pred, dy, dx, target.shape)
                    iou = compute_iou(translated, target)
                    ious.append(iou)

                avg_iou = np.mean(ious)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    best_offset = (dy, dx)

        # If improvement found, create refined program
        if best_offset and best_iou > self._evaluate(program, train_examples):
            dy, dx = best_offset

            # Create wrapper function
            base_func = program['function']

            def refined_func(grid: np.ndarray) -> np.ndarray:
                result = base_func(grid)
                # Apply learned offset
                return self._translate_grid(result, dy, dx, result.shape)

            refined_program = program.copy()
            refined_program['function'] = refined_func
            refined_program['description'] = f"{program['description']} + translate({dy}, {dx})"
            refined_program['operations'] = program['operations'] + [f'translate({dy},{dx})']

            return refined_program, best_iou

        return program, self._evaluate(program, train_examples)

    def _try_color_repair(
        self,
        program: Dict[str, Any],
        train_examples: List
    ) -> Tuple[Dict[str, Any], float]:
        """Try color repair."""
        # Get predictions
        predictions = []
        targets = []

        try:
            for input_grid, output_grid in train_examples:
                pred = program['function'](input_grid)
                predictions.append(pred)
                targets.append(output_grid)
        except:
            return program, 0.0

        # Learn color mapping
        color_votes = {}

        for pred, target in zip(predictions, targets):
            if pred.shape != target.shape:
                continue

            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    pred_color = int(pred[i, j])
                    target_color = int(target[i, j])
                    key = (pred_color, target_color)
                    color_votes[key] = color_votes.get(key, 0) + 1

        # Build mapping by majority vote
        mapping = {}
        pred_colors = set(pc for pc, _ in color_votes.keys())

        for pred_color in pred_colors:
            votes = [(tc, count) for (pc, tc), count in color_votes.items() if pc == pred_color]
            if votes:
                best_target_color = max(votes, key=lambda x: x[1])[0]
                mapping[pred_color] = best_target_color

        # Check if mapping is non-trivial
        is_identity = all(k == v for k, v in mapping.items())
        if is_identity or not mapping:
            return program, self._evaluate(program, train_examples)

        # Create refined program
        base_func = program['function']

        def refined_func(grid: np.ndarray) -> np.ndarray:
            result = base_func(grid)
            # Apply color mapping
            remapped = result.copy()
            for from_color, to_color in mapping.items():
                remapped[result == from_color] = to_color
            return remapped

        refined_program = program.copy()
        refined_program['function'] = refined_func
        refined_program['description'] = f"{program['description']} + color_remap({mapping})"
        refined_program['operations'] = program['operations'] + ['color_remap']

        refined_score = self._evaluate(refined_program, train_examples)
        current_score = self._evaluate(program, train_examples)

        if refined_score > current_score:
            return refined_program, refined_score

        return program, current_score

    def _try_scale_repair(
        self,
        program: Dict[str, Any],
        train_examples: List
    ) -> Tuple[Dict[str, Any], float]:
        """Try scale repair (if shape mismatch)."""
        # Get predictions
        try:
            pred = program['function'](train_examples[0][0])
            target = train_examples[0][1]
        except:
            return program, 0.0

        if pred.shape == target.shape:
            return program, self._evaluate(program, train_examples)

        # Try scale factors
        best_score = 0.0
        best_scale_func = None

        for factor in [2, 3, 0.5]:
            if factor < 1:
                # Downsample
                def scale_func(grid, f=factor):
                    factor_int = int(1 / f)
                    return grid[::factor_int, ::factor_int]
            else:
                # Upsample
                def scale_func(grid, f=factor):
                    return np.repeat(np.repeat(grid, int(f), axis=0), int(f), axis=1)

            # Test
            base_func = program['function']

            def test_func(grid, sf=scale_func):
                result = base_func(grid)
                return sf(result)

            score = 0.0
            try:
                ious = []
                for input_grid, output_grid in train_examples:
                    pred = test_func(input_grid)
                    if pred.shape == output_grid.shape:
                        iou = compute_iou(pred, output_grid)
                        ious.append(iou)
                if ious:
                    score = np.mean(ious)
            except:
                score = 0.0

            if score > best_score:
                best_score = score
                best_scale_func = scale_func

        current_score = self._evaluate(program, train_examples)

        if best_score > current_score and best_scale_func:
            base_func = program['function']

            def refined_func(grid, sf=best_scale_func):
                result = base_func(grid)
                return sf(result)

            refined_program = program.copy()
            refined_program['function'] = refined_func
            refined_program['description'] = f"{program['description']} + scale_repair"
            refined_program['operations'] = program['operations'] + ['scale_repair']

            return refined_program, best_score

        return program, current_score

    @staticmethod
    def _translate_grid(
        grid: np.ndarray,
        dy: int,
        dx: int,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Translate grid with bounds checking."""
        result = np.zeros(target_shape, dtype=grid.dtype)

        # Source region
        src_y_start = max(0, -dy)
        src_y_end = min(grid.shape[0], grid.shape[0] - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(grid.shape[1], grid.shape[1] - dx)

        # Destination region
        dst_y_start = max(0, dy)
        dst_y_end = min(target_shape[0], dst_y_start + (src_y_end - src_y_start))
        dst_x_start = max(0, dx)
        dst_x_end = min(target_shape[1], dst_x_start + (src_x_end - src_x_start))

        # Adjust source to match destination
        src_y_end = src_y_start + (dst_y_end - dst_y_start)
        src_x_end = src_x_start + (dst_x_end - dst_x_start)

        if src_y_end > src_y_start and src_x_end > src_x_start:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                grid[src_y_start:src_y_end, src_x_start:src_x_end]

        return result


def iterative_refiner_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Iterative refinement function for node.

    Args:
        data: Dictionary with:
            - program: Program to refine
            - train_examples: Training examples

    Returns:
        NodeOutput with refined program and score
    """
    artifacts = {}
    telemetry = {'node_type': 'repairer', 'subtype': 'iterative_refiner'}

    program = data.get('program', None)
    train_examples = data.get('train_examples', [])

    if program is None or not train_examples:
        telemetry['success'] = False
        return NodeOutput(
            result={'program': program, 'score': 0.0},
            artifacts=artifacts,
            telemetry=telemetry
        )

    refiner = IterativeRefiner(max_iterations=10)
    refined_program, score = refiner.refine(program, train_examples)

    result = {
        'program': refined_program,
        'score': score
    }

    artifacts['refined_program'] = refined_program
    artifacts['refined_score'] = score
    telemetry['score'] = score
    telemetry['success'] = True

    return NodeOutput(result=result, artifacts=artifacts, telemetry=telemetry)


def create_iterative_refiner_node() -> Node:
    """Create an iterative refiner node."""
    return Node(
        name="iterative_refiner",
        func=iterative_refiner_func,
        input_type="program_and_examples",
        output_type="refined_program",
        deterministic=False,  # Uses search, not fully deterministic
        category="repairer"
    )
