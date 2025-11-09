"""
Enhanced ARC Solver with Object-Based Reasoning

Extends the base curiosity-driven solver with object-level understanding.

Key Design:
- Analyzes if object reasoning is needed
- Generates both pixel-level and object-level hypotheses
- Uses same belief dynamics and curiosity signals
- Smoothly integrates with existing framework
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from arc_curiosity_solver.solver import ARCCuriositySolver
from arc_curiosity_solver.core.object_reasoning import (
    ObjectDetector, ObjectTransformations, analyze_object_patterns, ArcObject
)
from arc_curiosity_solver.belief_dynamics.belief_space import Hypothesis
from arc_curiosity_solver.transformations.arc_primitives import Transform


class EnhancedARCCuriositySolver(ARCCuriositySolver):
    """
    Enhanced solver with object-based reasoning capabilities.

    Inherits from base solver and adds object-level transformations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Object reasoning components
        self.object_detector = ObjectDetector()
        self.object_transforms = ObjectTransformations()

        # Track if object reasoning was used
        self.used_object_reasoning = False

    def _generate_hypotheses(self,
                           train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                           test_input: np.ndarray) -> List[Hypothesis]:
        """
        Generate both pixel-level and object-level hypotheses.

        This extends the base implementation.
        """
        # Start with base pixel-level hypotheses
        hypotheses = super()._generate_hypotheses(train_pairs, test_input)

        # Analyze if object reasoning is needed
        patterns = analyze_object_patterns(train_pairs)

        if patterns['needs_object_reasoning']:
            self.used_object_reasoning = True

            # Generate object-based hypotheses
            object_hyps = self._generate_object_hypotheses(train_pairs, test_input, patterns)
            hypotheses.extend(object_hyps)

        return hypotheses

    def _generate_object_hypotheses(self,
                                   train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                   test_input: np.ndarray,
                                   patterns: Dict[str, Any]) -> List[Hypothesis]:
        """Generate object-based transformation hypotheses."""
        hypotheses = []

        # Strategy 1: Recolor objects by size
        if patterns.get('color_changes'):
            # Try recoloring largest object
            hyp = self._create_recolor_largest_hypothesis(train_pairs)
            if hyp:
                hypotheses.append(hyp)

            # Try recoloring smallest object
            hyp = self._create_recolor_smallest_hypothesis(train_pairs)
            if hyp:
                hypotheses.append(hyp)

        # Strategy 2: Move objects
        if patterns.get('position_changes'):
            # Try moving objects to edges
            for target in ['top', 'bottom', 'left', 'right']:
                hyp = self._create_move_to_edge_hypothesis(train_pairs, target)
                if hyp:
                    hypotheses.append(hyp)

        # Strategy 3: Scale objects
        if patterns.get('size_changes'):
            for factor in [2, 3]:
                hyp = self._create_scale_objects_hypothesis(train_pairs, factor)
                if hyp:
                    hypotheses.append(hyp)

        # Strategy 4: Object extraction/compression
        object_counts = patterns.get('object_count_changes', [])
        if object_counts:
            # Check if compression is happening
            for inp_count, out_count in object_counts:
                if inp_count > out_count:
                    # Might be extracting/compressing
                    hyp = self._create_extract_objects_hypothesis(train_pairs)
                    if hyp:
                        hypotheses.append(hyp)
                    break

        # Strategy 5: Filter objects by property
        for prop in ['size', 'color']:
            hyp = self._create_filter_objects_hypothesis(train_pairs, prop)
            if hyp:
                hypotheses.append(hyp)

        return hypotheses

    def _create_recolor_largest_hypothesis(self, train_pairs) -> Optional[Hypothesis]:
        """Create hypothesis: recolor largest object."""
        # Infer target color from training data
        target_color = self._infer_target_color_for_largest(train_pairs)

        if target_color is None:
            return None

        def transform_fn(grid):
            return self.object_transforms.recolor_objects(
                grid,
                lambda obj: obj == self.object_detector.detect_largest(grid),
                target_color
            )

        # Test on training data
        fit = self._evaluate_object_transform(transform_fn, train_pairs)

        return Hypothesis(
            program=Transform('recolor_largest', transform_fn, {'color': target_color}, 'object'),
            name=f'recolor_largest_to_{target_color}',
            parameters={'target_color': target_color},
            activation=0.0,
            evidence_count=0,
            success_count=1 if fit > 0.5 else 0
        )

    def _create_recolor_smallest_hypothesis(self, train_pairs) -> Optional[Hypothesis]:
        """Create hypothesis: recolor smallest object."""
        target_color = self._infer_target_color_for_smallest(train_pairs)

        if target_color is None:
            return None

        def transform_fn(grid):
            objects = self.object_detector.detect_objects(grid)
            if not objects:
                return grid.copy()

            smallest = min(objects, key=lambda obj: obj.size)
            return self.object_transforms.recolor_objects(
                grid,
                lambda obj: obj.position == smallest.position,
                target_color
            )

        fit = self._evaluate_object_transform(transform_fn, train_pairs)

        return Hypothesis(
            program=Transform('recolor_smallest', transform_fn, {'color': target_color}, 'object'),
            name=f'recolor_smallest_to_{target_color}',
            parameters={'target_color': target_color},
            activation=0.0,
            evidence_count=0,
            success_count=1 if fit > 0.5 else 0
        )

    def _create_move_to_edge_hypothesis(self, train_pairs, edge: str) -> Optional[Hypothesis]:
        """Create hypothesis: move all objects to edge."""

        def transform_fn(grid):
            if edge == 'top':
                target_fn = lambda obj: (0, obj.position[1])
            elif edge == 'bottom':
                target_fn = lambda obj: (grid.shape[0] - 1, obj.position[1])
            elif edge == 'left':
                target_fn = lambda obj: (obj.position[0], 0)
            else:  # right
                target_fn = lambda obj: (obj.position[0], grid.shape[1] - 1)

            return self.object_transforms.move_objects(grid, target_fn)

        fit = self._evaluate_object_transform(transform_fn, train_pairs)

        return Hypothesis(
            program=Transform(f'move_to_{edge}', transform_fn, {'edge': edge}, 'object'),
            name=f'move_to_{edge}',
            parameters={'edge': edge},
            activation=0.0,
            evidence_count=0,
            success_count=1 if fit > 0.5 else 0
        )

    def _create_scale_objects_hypothesis(self, train_pairs, factor: int) -> Optional[Hypothesis]:
        """Create hypothesis: scale all objects."""

        def transform_fn(grid):
            return self.object_transforms.scale_objects(grid, factor)

        fit = self._evaluate_object_transform(transform_fn, train_pairs)

        return Hypothesis(
            program=Transform(f'scale_objects_{factor}x', transform_fn, {'factor': factor}, 'object'),
            name=f'scale_objects_{factor}x',
            parameters={'factor': factor},
            activation=0.0,
            evidence_count=0,
            success_count=1 if fit > 0.5 else 0
        )

    def _create_extract_objects_hypothesis(self, train_pairs) -> Optional[Hypothesis]:
        """Create hypothesis: extract objects to compressed grid."""

        # Infer grid dimensions from output
        if not train_pairs:
            return None

        out_shape = train_pairs[0][1].shape

        def transform_fn(grid):
            return self.object_transforms.extract_objects_to_grid(grid, out_shape[0], out_shape[1])

        fit = self._evaluate_object_transform(transform_fn, train_pairs)

        return Hypothesis(
            program=Transform('extract_objects', transform_fn, {'shape': out_shape}, 'object'),
            name='extract_objects_to_grid',
            parameters={'shape': out_shape},
            activation=0.0,
            evidence_count=0,
            success_count=1 if fit > 0.5 else 0
        )

    def _create_filter_objects_hypothesis(self, train_pairs, property: str) -> Optional[Hypothesis]:
        """Create hypothesis: filter objects by property."""

        # Infer threshold
        if property == 'size':
            threshold = self._infer_size_threshold(train_pairs)
            if threshold is None:
                return None

            def transform_fn(grid):
                return self.object_transforms.filter_objects_by_property(
                    grid,
                    lambda obj: obj.size >= threshold
                )

            name = f'filter_size_gte_{threshold}'

        elif property == 'color':
            target_colors = self._infer_target_colors(train_pairs)
            if not target_colors:
                return None

            def transform_fn(grid):
                return self.object_transforms.filter_objects_by_property(
                    grid,
                    lambda obj: obj.dominant_color in target_colors
                )

            name = f'filter_colors_{target_colors}'

        else:
            return None

        fit = self._evaluate_object_transform(transform_fn, train_pairs)

        return Hypothesis(
            program=Transform(name, transform_fn, {}, 'object'),
            name=name,
            parameters={},
            activation=0.0,
            evidence_count=0,
            success_count=1 if fit > 0.5 else 0
        )

    # Helper methods for inference

    def _infer_target_color_for_largest(self, train_pairs) -> Optional[int]:
        """Infer what color the largest object becomes."""
        for inp, out in train_pairs:
            inp_objects = self.object_detector.detect_objects(inp)
            out_objects = self.object_detector.detect_objects(out)

            if not inp_objects or not out_objects:
                continue

            largest_inp = max(inp_objects, key=lambda obj: obj.size)
            # Find corresponding object in output (by position)
            for out_obj in out_objects:
                if np.allclose(out_obj.position, largest_inp.position, atol=2):
                    if out_obj.dominant_color != largest_inp.dominant_color:
                        return out_obj.dominant_color

        return None

    def _infer_target_color_for_smallest(self, train_pairs) -> Optional[int]:
        """Infer what color the smallest object becomes."""
        for inp, out in train_pairs:
            inp_objects = self.object_detector.detect_objects(inp)
            out_objects = self.object_detector.detect_objects(out)

            if not inp_objects or not out_objects:
                continue

            smallest_inp = min(inp_objects, key=lambda obj: obj.size)
            for out_obj in out_objects:
                if np.allclose(out_obj.position, smallest_inp.position, atol=2):
                    if out_obj.dominant_color != smallest_inp.dominant_color:
                        return out_obj.dominant_color

        return None

    def _infer_size_threshold(self, train_pairs) -> Optional[int]:
        """Infer size threshold for filtering."""
        for inp, out in train_pairs:
            inp_objects = self.object_detector.detect_objects(inp)
            out_objects = self.object_detector.detect_objects(out)

            if len(out_objects) < len(inp_objects):
                # Some objects were filtered out
                # Find threshold
                out_sizes = set(obj.size for obj in out_objects)
                filtered_sizes = [obj.size for obj in inp_objects if obj.size not in out_sizes]

                if filtered_sizes:
                    return max(filtered_sizes) + 1

        return None

    def _infer_target_colors(self, train_pairs) -> Optional[List[int]]:
        """Infer which colors to keep."""
        for inp, out in train_pairs:
            inp_colors = set(inp.flatten())
            out_colors = set(out.flatten())

            if out_colors.issubset(inp_colors) and len(out_colors) < len(inp_colors):
                return list(out_colors)

        return None

    def _evaluate_object_transform(self, transform_fn: callable,
                                  train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate how well an object transform fits training data."""
        if not train_pairs:
            return 0.0

        total_score = 0.0

        for inp, out in train_pairs:
            try:
                pred = transform_fn(inp)

                if pred.shape == out.shape:
                    match = np.sum(pred == out)
                    total = out.size
                    score = match / total
                else:
                    score = 0.1

                total_score += score

            except Exception:
                total_score += 0.0

        return total_score / len(train_pairs)

    def get_solver_statistics(self) -> Dict[str, Any]:
        """Get enhanced solver statistics."""
        stats = super().get_solver_state()
        stats['used_object_reasoning'] = self.used_object_reasoning
        return stats
