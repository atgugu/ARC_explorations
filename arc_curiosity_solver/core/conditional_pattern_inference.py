"""
Conditional Pattern Inference

Extends pattern inference to detect IF-THEN-ELSE patterns from training examples.

Key capability: Detect when objects with different properties undergo different
transformations, then infer the conditional rule.

Example:
    Training examples show:
    - Large objects → recolor to blue
    - Small objects → recolor to red

    Inferred conditional:
    IF object.size > 3 THEN recolor to blue ELSE recolor to red
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from ..transformations.conditional_transforms import (
    ConditionalTransform, Condition, ConditionalAction,
    ConditionLibrary, ActionLibrary, ConditionalTransformBuilder
)
from ..core.object_reasoning import ObjectDetector, ArcObject
from ..transformations.arc_primitives import Transform


class ConditionalPatternAnalyzer:
    """
    Analyzes training pairs to extract conditional transformation patterns.

    This is the breakthrough component - detects when transformations depend
    on object properties or spatial context.
    """

    def __init__(self):
        self.object_detector = ObjectDetector()
        self.condition_lib = ConditionLibrary()
        self.action_lib = ActionLibrary()
        self.builder = ConditionalTransformBuilder()

    def analyze_training_pairs(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[ConditionalTransform]:
        """
        Extract conditional patterns from training pairs.

        Process:
        1. Detect objects in each input/output pair
        2. Match objects and observe transformations
        3. Look for splits: objects with different properties transform differently
        4. Infer conditional rules explaining the splits
        5. Return high-confidence conditional transforms
        """
        # Collect all object-level observations
        all_observations = []

        for inp, out in train_pairs:
            observations = self._analyze_pair_for_conditionals(inp, out)
            all_observations.extend(observations)

        if not all_observations:
            return []

        # Build conditional transforms from observations
        conditionals = self.builder.build_from_observations(all_observations)

        # Also try detecting grid-level conditional patterns
        grid_conditionals = self._detect_grid_level_conditionals(train_pairs)
        conditionals.extend(grid_conditionals)

        # Sort by confidence
        conditionals.sort(key=lambda x: x.confidence, reverse=True)

        return conditionals

    def _analyze_pair_for_conditionals(self, input_grid: np.ndarray,
                                      output_grid: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze a single training pair for conditional patterns.

        Returns observations in format expected by ConditionalTransformBuilder.
        """
        observations = []

        # Detect objects
        input_objects = self.object_detector.detect_objects(input_grid)
        output_objects = self.object_detector.detect_objects(output_grid)

        if not input_objects:
            return observations

        # Match input to output objects
        matched_pairs = self._match_objects(input_objects, output_objects, input_grid)

        for inp_obj, out_obj in matched_pairs:
            obs = {
                'obj': inp_obj,
                'obj_out': out_obj,
                'grid': input_grid,
                'grid_out': output_grid,
                'all_objs': input_objects
            }
            observations.append(obs)

        # Handle removed objects (appear in input but not output)
        matched_input_objs = {id(pair[0]) for pair in matched_pairs}
        for inp_obj in input_objects:
            if id(inp_obj) not in matched_input_objs:
                obs = {
                    'obj': inp_obj,
                    'obj_out': None,  # Removed
                    'grid': input_grid,
                    'grid_out': output_grid,
                    'all_objs': input_objects
                }
                observations.append(obs)

        return observations

    def _match_objects(self, input_objects: List[ArcObject],
                      output_objects: List[ArcObject],
                      input_grid: np.ndarray) -> List[Tuple[ArcObject, ArcObject]]:
        """Match input objects to output objects."""
        if not input_objects or not output_objects:
            return []

        matched = []
        used_output = set()

        for inp_obj in input_objects:
            best_match = None
            best_score = float('inf')

            for i, out_obj in enumerate(output_objects):
                if i in used_output:
                    continue

                # Distance between centers
                dist = np.sqrt((inp_obj.position[0] - out_obj.position[0])**2 +
                             (inp_obj.position[1] - out_obj.position[1])**2)

                # Size similarity
                size_diff = abs(inp_obj.size - out_obj.size) / max(inp_obj.size, out_obj.size, 1)

                # Combined score (lower is better)
                score = dist + size_diff * 5

                if score < best_score:
                    best_score = score
                    best_match = i

            if best_match is not None and best_score < 20:  # Reasonable threshold
                matched.append((inp_obj, output_objects[best_match]))
                used_output.add(best_match)

        return matched

    def _detect_grid_level_conditionals(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[ConditionalTransform]:
        """Detect conditional patterns at grid level (not object level)."""
        conditionals = []

        # Check for patterns like "if grid has property X, apply transform Y"
        # This is simpler than object-level conditionals

        # Example: Color mapping that depends on grid properties
        # (implement more as needed)

        return conditionals


class ConditionalHypothesisGenerator:
    """
    Generates hypotheses from conditional patterns.

    Converts ConditionalTransform objects into Transform objects
    compatible with the existing solver architecture.
    """

    def __init__(self):
        self.analyzer = ConditionalPatternAnalyzer()

    def generate_from_training(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[Transform, str, float]]:
        """
        Generate conditional transformation hypotheses from training data.

        Returns: List of (Transform, description, confidence) tuples
        """
        # Extract conditional patterns
        conditionals = self.analyzer.analyze_training_pairs(train_pairs)

        if not conditionals:
            return []

        hypotheses = []

        for cond_transform in conditionals:
            # Wrap ConditionalTransform in a Transform object
            def make_transform(ct=cond_transform):
                def transform_fn(grid: np.ndarray) -> np.ndarray:
                    return ct.apply(grid, objects=None)
                return transform_fn

            transform_fn = make_transform()

            transform_obj = Transform(
                name=f"conditional_{cond_transform.condition.name}",
                function=transform_fn,
                parameters={
                    'condition': cond_transform.condition.name,
                    'variant': 'conditional'
                },
                category='conditional'
            )

            description = cond_transform.description
            confidence = cond_transform.confidence

            hypotheses.append((transform_obj, description, confidence))

        return hypotheses


class LoopingTransform:
    """
    FOR-each loop construct for applying transformations to multiple objects/regions.

    Example:
        FOR each 3x3 region:
            extract → rotate 90° → place back
    """

    def __init__(self, iterator_type: str, operation: Any):
        """
        Args:
            iterator_type: 'objects', 'regions_3x3', 'rows', 'columns', etc.
            operation: Transformation to apply to each element
        """
        self.iterator_type = iterator_type
        self.operation = operation

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply loop transformation."""
        result = grid.copy()

        if self.iterator_type == 'objects':
            # FOR each object
            detector = ObjectDetector()
            objects = detector.detect_objects(grid)

            for obj in objects:
                # Apply operation to this object
                result = self.operation(result, obj, objects)

        elif self.iterator_type == 'regions_3x3':
            # FOR each 3x3 region
            h, w = grid.shape
            for i in range(0, h, 3):
                for j in range(0, w, 3):
                    region = grid[i:min(i+3, h), j:min(j+3, w)]
                    transformed = self.operation(region)
                    result[i:min(i+3, h), j:min(j+3, w)] = transformed

        elif self.iterator_type == 'rows':
            # FOR each row
            for i in range(grid.shape[0]):
                result[i, :] = self.operation(grid[i, :])

        elif self.iterator_type == 'columns':
            # FOR each column
            for j in range(grid.shape[1]):
                result[:, j] = self.operation(grid[:, j])

        return result


class CompositeConditionalTransform:
    """
    Multi-stage transformation composition.

    Example:
        Stage 1: Extract all blue objects
        Stage 2: FOR each → rotate 90°
        Stage 3: Tile rotated objects in grid
    """

    def __init__(self, name: str = "composite"):
        self.name = name
        self.stages: List[Any] = []

    def add_stage(self, stage: Any):
        """Add a transformation stage."""
        self.stages.append(stage)

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply all stages in sequence."""
        result = grid.copy()

        for stage in self.stages:
            try:
                if hasattr(stage, 'apply'):
                    result = stage.apply(result)
                elif callable(stage):
                    result = stage(result)
            except Exception as e:
                # Skip failed stages
                continue

        return result


class EnhancedPatternInference:
    """
    Enhanced pattern inference with conditional, looping, and compositional capabilities.

    This is the complete system for breaking through the expressiveness bottleneck.
    """

    def __init__(self):
        self.conditional_generator = ConditionalHypothesisGenerator()
        self.object_detector = ObjectDetector()

    def analyze_and_generate_hypotheses(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, List]:
        """
        Comprehensive analysis of training pairs.

        Returns:
            {
                'conditional': List of conditional hypotheses,
                'looping': List of looping hypotheses,
                'composite': List of composite hypotheses,
                'simple': List of simple (unconditional) hypotheses
            }
        """
        results = {
            'conditional': [],
            'looping': [],
            'composite': [],
            'simple': []
        }

        # Generate conditional hypotheses
        conditional_hyps = self.conditional_generator.generate_from_training(train_pairs)
        results['conditional'] = conditional_hyps

        # Generate looping hypotheses (detect repetitive patterns)
        looping_hyps = self._detect_looping_patterns(train_pairs)
        results['looping'] = looping_hyps

        # Generate composite hypotheses (multi-stage transformations)
        composite_hyps = self._detect_composite_patterns(train_pairs)
        results['composite'] = composite_hyps

        return results

    def _detect_looping_patterns(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[Transform, str, float]]:
        """Detect patterns that apply the same operation to multiple elements."""
        hypotheses = []

        # Check if each training pair shows repetitive transformations
        for inp, out in train_pairs:
            # Detect if this is a tiling pattern
            if self._is_tiling_pattern(inp, out):
                # Create tiling hypothesis
                def make_tiling():
                    def tile_fn(grid: np.ndarray) -> np.ndarray:
                        # Extract pattern and tile
                        h, w = grid.shape
                        if h >= 3 and w >= 3:
                            pattern = grid[:3, :3]
                            result = np.tile(pattern, (h//3 + 1, w//3 + 1))
                            return result[:h, :w]
                        return grid
                    return tile_fn

                transform_obj = Transform(
                    name="tile_pattern",
                    function=make_tiling(),
                    parameters={'variant': 'looping'},
                    category='looping'
                )

                hypotheses.append((transform_obj, "tile 3x3 pattern", 0.7))
                break  # Only need one tiling hypothesis

        return hypotheses

    def _detect_composite_patterns(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[Transform, str, float]]:
        """Detect multi-stage transformation patterns."""
        hypotheses = []

        # Detect common compositions like:
        # 1. Extract → Transform → Place
        # 2. Split → Transform each → Merge
        # 3. Pattern detection → Conditional application

        # (Can be expanded with more sophisticated detection)

        return hypotheses

    def _is_tiling_pattern(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output is a tiled version of input pattern."""
        if inp.shape == out.shape:
            return False

        # Check if output shows repetition
        h, w = out.shape
        if h >= 6 and w >= 6:
            # Check if 3x3 regions repeat
            region1 = out[:3, :3]
            region2 = out[3:6, :3]
            if np.array_equal(region1, region2):
                return True

        return False
