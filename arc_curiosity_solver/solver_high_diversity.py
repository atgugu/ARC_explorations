"""
High-Diversity ARC Solver

Extends diverse solver with significantly more parameter variations.
Goal: Test if 10x more variations improves solve rate beyond 1.0%.
"""

import numpy as np
from typing import List, Tuple
from .solver_diverse import DiverseARCCuriositySolver
from .core.pattern_inference import InvariantPattern
from .transformations.arc_primitives import Transform


class HighDiversityARCCuriositySolver(DiverseARCCuriositySolver):
    """
    Ultra-diverse solver with 10x parameter variations.

    Increases from 3-5 variations to 10-20 variations per pattern.
    Tests hypothesis: More parameter exploration → more exact matches.
    """

    def __init__(self):
        super().__init__()
        self.diversity_multiplier = 10  # 10x variations per pattern (up from 3x)
        self.max_combinations = 10  # More combinations

    def _vary_color_pattern(self, pattern: InvariantPattern,
                           training_colors: List[int],
                           test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """
        Generate MANY color variations.

        Strategy: Try ALL colors 0-9, not just training colors.
        Reasoning: The correct color might not appear in training.
        """
        variations = []

        # Strategy 1: All colors 0-9
        for target_color in range(10):
            def make_color_variation(pattern=pattern, color=target_color):
                def transform(grid: np.ndarray) -> np.ndarray:
                    result = grid.copy()
                    from ..core.object_reasoning import ObjectDetector
                    detector = ObjectDetector()
                    objects = detector.detect_objects(result)

                    for obj in objects:
                        if pattern.condition(obj, objects):
                            result[obj.mask] = color

                    return result
                return transform

            transform_fn = make_color_variation()
            description = f"{pattern.condition_desc} → recolor to {target_color}"

            transform_obj = Transform(
                name=f"color_var_{target_color}",
                function=transform_fn,
                parameters={},
                category='pattern_variation'
            )

            # Higher confidence for training colors
            if target_color in training_colors:
                confidence = pattern.confidence * 0.6
            else:
                confidence = pattern.confidence * 0.3

            variations.append((transform_obj, description, confidence))

        return variations  # Return all 10 color variations

    def _vary_position_pattern(self, pattern: InvariantPattern,
                               test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """
        Generate MANY position variations.

        Strategy: Comprehensive movement grid.
        """
        variations = []

        # More comprehensive movement patterns
        deltas = [
            # Cardinal directions
            (1, 0), (-1, 0), (0, 1), (0, -1),
            # Larger cardinal
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (3, 0), (-3, 0), (0, 3), (0, -3),
            # Diagonals
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (2, 2), (2, -2), (-2, 2), (-2, -2),
        ]

        for dy, dx in deltas:
            def make_position_variation(dy=dy, dx=dx):
                def transform(grid: np.ndarray) -> np.ndarray:
                    result = grid.copy()
                    from ..core.object_reasoning import ObjectDetector
                    detector = ObjectDetector()
                    objects = detector.detect_objects(result)

                    for obj in objects:
                        # Clear old position
                        result[obj.mask] = 0

                        # Calculate new position
                        y1, x1, y2, x2 = obj.bbox
                        new_y1, new_x1 = y1 + dy, x1 + dx

                        # Check bounds
                        if (0 <= new_y1 < grid.shape[0] and
                            0 <= new_x1 < grid.shape[1]):
                            # Move object
                            for y in range(y1, y2):
                                for x in range(x1, x2):
                                    if obj.mask[y, x]:
                                        new_y, new_x = y + dy, x + dx
                                        if (0 <= new_y < grid.shape[0] and
                                            0 <= new_x < grid.shape[1]):
                                            result[new_y, new_x] = obj.dominant_color

                    return result
                return transform

            transform_fn = make_position_variation()
            description = f"all objects → move by ({dy}, {dx})"

            transform_obj = Transform(
                name=f"move_{dy}_{dx}",
                function=transform_fn,
                parameters={},
                category='pattern_variation'
            )

            confidence = pattern.confidence * 0.4
            variations.append((transform_obj, description, confidence))

        return variations  # Return all ~20 position variations

    def _vary_pixel_mapping(self, pattern: InvariantPattern,
                           training_colors: List[int],
                           test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """
        Generate MANY pixel mapping variations.

        Strategy: All possible color pair mappings from training.
        """
        variations = []

        # All pairs of training colors
        for src_color in training_colors:
            for tgt_color in training_colors:
                if src_color == tgt_color:
                    continue

                def make_mapping_variation(src=src_color, tgt=tgt_color):
                    def transform(grid: np.ndarray) -> np.ndarray:
                        result = grid.copy()
                        result[result == src] = tgt
                        return result
                    return transform

                transform_fn = make_mapping_variation()
                description = f"pixels {src_color} → {tgt_color}"

                transform_obj = Transform(
                    name=f"map_{src_color}_to_{tgt_color}",
                    function=transform_fn,
                    parameters={},
                    category='pattern_variation'
                )

                confidence = pattern.confidence * 0.3
                variations.append((transform_obj, description, confidence))

        return variations  # Return all color pair mappings

    def _generate_pattern_variations(self, patterns: List[InvariantPattern],
                                    test_input: np.ndarray,
                                    train_pairs: List[Tuple]) -> List[Tuple[Transform, str, float]]:
        """
        Override to generate 10x more variations.

        Instead of limiting to 3-5 variations, generate 10-20 per pattern.
        """
        variations = []
        from ..core.object_reasoning import ObjectDetector
        object_detector = ObjectDetector()

        # Collect colors used in training data
        training_colors = set()
        for inp, out in train_pairs:
            training_colors.update(inp.flatten())
            training_colors.update(out.flatten())
        training_colors = sorted(training_colors)

        for pattern in patterns:
            if pattern.transform_type == 'color_change':
                # Generate ALL color variations (10 colors)
                color_vars = self._vary_color_pattern(pattern, training_colors, test_input)
                variations.extend(color_vars)

            elif pattern.transform_type == 'position_change':
                # Generate comprehensive position variations (~20 directions)
                pos_vars = self._vary_position_pattern(pattern, test_input)
                variations.extend(pos_vars)

            elif pattern.transform_type == 'pixel_color_map':
                # Generate all color pair mappings
                map_vars = self._vary_pixel_mapping(pattern, training_colors, test_input)
                variations.extend(map_vars)

        # Don't limit - return all variations
        # This gives us 10-20x more hypotheses to explore
        return variations[:len(patterns) * self.diversity_multiplier]
