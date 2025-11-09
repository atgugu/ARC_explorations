"""
Pattern Inference: Extract invariant transformation patterns from training examples.

Safe approach:
1. Analyze each training pair to detect transformations
2. Find patterns consistent across ALL examples
3. Generate hypotheses only matching observed patterns
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from .object_reasoning import ObjectDetector, ArcObject, ObjectTransformations
from ..transformations.arc_primitives import Transform
import copy


@dataclass
class TransformationObservation:
    """Single observation of a transformation in one training example."""
    transform_type: str  # 'color_change', 'position_change', 'size_change', etc.
    source_condition: Dict[str, Any]  # What property the source object had
    target_action: Dict[str, Any]  # What happened to it
    confidence: float = 1.0

    def __hash__(self):
        # Make hashable for counting
        return hash((
            self.transform_type,
            tuple(sorted(self.source_condition.items())),
            tuple(sorted(self.target_action.items()))
        ))

    def matches(self, obj: ArcObject) -> bool:
        """Check if an object matches the source condition."""
        for key, value in self.source_condition.items():
            if key == 'relative_size':
                continue  # Handle specially
            obj_value = getattr(obj, key, None)
            if obj_value != value:
                return False
        return True


@dataclass
class InvariantPattern:
    """A pattern that holds across all (or most) training examples."""
    transform_type: str
    condition: Callable[[ArcObject, List[ArcObject]], bool]
    condition_desc: str  # Human-readable description
    action: Callable[[np.ndarray, ArcObject, List[ArcObject]], np.ndarray]
    action_desc: str
    confidence: float  # 0-1, based on consistency across examples
    support_count: int  # Number of examples supporting this pattern

    def __repr__(self):
        return f"InvariantPattern({self.condition_desc} → {self.action_desc}, conf={self.confidence:.2f})"


class PatternAnalyzer:
    """Analyzes training pairs to extract invariant transformation patterns."""

    def __init__(self):
        self.object_detector = ObjectDetector()
        self.object_transforms = ObjectTransformations()

    def analyze_training_pairs(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[InvariantPattern]:
        """
        Extract invariant patterns from training pairs.

        Safe approach:
        1. Detect objects in each input/output
        2. Match objects and detect transformations
        3. Find patterns consistent across ALL examples
        4. Return high-confidence invariants
        """
        all_observations = []

        # Step 1: Analyze each training pair
        for inp, out in train_pairs:
            observations = self._analyze_single_pair(inp, out)
            all_observations.extend(observations)

        # Step 2: Find invariant patterns (consistent across examples)
        invariants = self._find_invariants(all_observations, n_examples=len(train_pairs))

        return invariants

    def _analyze_single_pair(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[TransformationObservation]:
        """Analyze a single training pair to detect transformations."""
        observations = []

        # Detect objects
        input_objects = self.object_detector.detect_objects(input_grid)
        output_objects = self.object_detector.detect_objects(output_grid)

        # Grid-level changes
        if input_grid.shape != output_grid.shape:
            observations.append(TransformationObservation(
                transform_type='shape_change',
                source_condition={'input_shape': input_grid.shape},
                target_action={'output_shape': output_grid.shape}
            ))

        # If no objects detected, try pixel-level patterns
        if len(input_objects) == 0 or len(output_objects) == 0:
            pixel_obs = self._analyze_pixel_patterns(input_grid, output_grid)
            observations.extend(pixel_obs)
            return observations

        # Object-level changes

        # 1. Color transformations
        color_obs = self._detect_color_transformations(input_objects, output_objects, input_grid, output_grid)
        observations.extend(color_obs)

        # 2. Position transformations
        position_obs = self._detect_position_transformations(input_objects, output_objects)
        observations.extend(position_obs)

        # 3. Size transformations
        size_obs = self._detect_size_transformations(input_objects, output_objects)
        observations.extend(size_obs)

        # 4. Count changes
        if len(input_objects) != len(output_objects):
            observations.append(TransformationObservation(
                transform_type='count_change',
                source_condition={'count': len(input_objects)},
                target_action={'new_count': len(output_objects)}
            ))

        return observations

    def _detect_color_transformations(self, input_objects: List[ArcObject],
                                     output_objects: List[ArcObject],
                                     input_grid: np.ndarray,
                                     output_grid: np.ndarray) -> List[TransformationObservation]:
        """Detect color changes in objects."""
        observations = []

        # Match objects by position/size
        matched_pairs = self._match_objects(input_objects, output_objects)

        for inp_obj, out_obj in matched_pairs:
            if inp_obj.dominant_color != out_obj.dominant_color:
                # Detect what property correlates with the color change
                relative_size = self._get_relative_size_category(inp_obj, input_objects)

                observations.append(TransformationObservation(
                    transform_type='color_change',
                    source_condition={
                        'dominant_color': inp_obj.dominant_color,
                        'relative_size': relative_size,
                        'size': inp_obj.size
                    },
                    target_action={'new_color': out_obj.dominant_color}
                ))

        return observations

    def _detect_position_transformations(self, input_objects: List[ArcObject],
                                        output_objects: List[ArcObject]) -> List[TransformationObservation]:
        """Detect position changes in objects."""
        observations = []

        matched_pairs = self._match_objects(input_objects, output_objects)

        for inp_obj, out_obj in matched_pairs:
            dy = out_obj.position[0] - inp_obj.position[0]
            dx = out_obj.position[1] - inp_obj.position[1]

            if abs(dy) > 0.5 or abs(dx) > 0.5:  # Significant movement
                observations.append(TransformationObservation(
                    transform_type='position_change',
                    source_condition={'dominant_color': inp_obj.dominant_color},
                    target_action={'delta': (dy, dx)}
                ))

        return observations

    def _detect_size_transformations(self, input_objects: List[ArcObject],
                                    output_objects: List[ArcObject]) -> List[TransformationObservation]:
        """Detect size changes in objects."""
        observations = []

        matched_pairs = self._match_objects(input_objects, output_objects)

        for inp_obj, out_obj in matched_pairs:
            if abs(inp_obj.size - out_obj.size) > 2:  # Significant size change
                size_ratio = out_obj.size / max(inp_obj.size, 1)

                observations.append(TransformationObservation(
                    transform_type='size_change',
                    source_condition={'dominant_color': inp_obj.dominant_color},
                    target_action={'size_ratio': size_ratio}
                ))

        return observations

    def _analyze_pixel_patterns(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[TransformationObservation]:
        """Analyze pixel-level patterns when objects aren't detected."""
        observations = []

        # Check for simple transformations
        if input_grid.shape == output_grid.shape:
            # Check for color replacement
            unique_in = set(input_grid.flatten())
            unique_out = set(output_grid.flatten())

            # Simple color mapping
            if len(unique_in) <= 3 and len(unique_out) <= 3:
                for color_in in unique_in:
                    mask = input_grid == color_in
                    colors_out = set(output_grid[mask])
                    if len(colors_out) == 1:
                        color_out = list(colors_out)[0]
                        if color_in != color_out:
                            observations.append(TransformationObservation(
                                transform_type='pixel_color_map',
                                source_condition={'color': int(color_in)},
                                target_action={'new_color': int(color_out)}
                            ))

        return observations

    def _match_objects(self, input_objects: List[ArcObject],
                      output_objects: List[ArcObject]) -> List[Tuple[ArcObject, ArcObject]]:
        """Match input objects to output objects based on position/size similarity."""
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

                # Size difference
                size_diff = abs(inp_obj.size - out_obj.size) / max(inp_obj.size, out_obj.size, 1)

                score = dist + size_diff * 5  # Weight size similarity

                if score < best_score:
                    best_score = score
                    best_match = i

            if best_match is not None and best_score < 20:  # Reasonable threshold
                matched.append((inp_obj, output_objects[best_match]))
                used_output.add(best_match)

        return matched

    def _get_relative_size_category(self, obj: ArcObject, all_objects: List[ArcObject]) -> str:
        """Categorize object size relative to others."""
        if len(all_objects) <= 1:
            return 'only'

        sizes = [o.size for o in all_objects]
        max_size = max(sizes)
        min_size = min(sizes)

        if obj.size == max_size:
            return 'largest'
        elif obj.size == min_size:
            return 'smallest'
        else:
            return 'medium'

    def _find_invariants(self, observations: List[TransformationObservation],
                        n_examples: int) -> List[InvariantPattern]:
        """
        Find patterns that are consistent across examples.

        Safe approach: Only accept patterns with high confidence (present in most/all examples)
        """
        # Group observations by type
        by_type = defaultdict(list)
        for obs in observations:
            by_type[obs.transform_type].append(obs)

        invariants = []

        # For each transformation type, find consistent patterns
        for transform_type, obs_list in by_type.items():
            if transform_type == 'color_change':
                inv = self._extract_color_invariants(obs_list, n_examples)
                invariants.extend(inv)
            elif transform_type == 'position_change':
                inv = self._extract_position_invariants(obs_list, n_examples)
                invariants.extend(inv)
            elif transform_type == 'size_change':
                inv = self._extract_size_invariants(obs_list, n_examples)
                invariants.extend(inv)
            elif transform_type == 'pixel_color_map':
                inv = self._extract_pixel_mapping_invariants(obs_list, n_examples)
                invariants.extend(inv)

        # Sort by confidence
        invariants.sort(key=lambda x: x.confidence, reverse=True)

        return invariants

    def _extract_color_invariants(self, observations: List[TransformationObservation],
                                  n_examples: int) -> List[InvariantPattern]:
        """Extract color change invariants."""
        invariants = []

        # Count patterns: (relative_size, source_color) -> target_color
        pattern_counts = defaultdict(lambda: defaultdict(int))

        for obs in observations:
            relative_size = obs.source_condition.get('relative_size', 'any')
            source_color = obs.source_condition.get('dominant_color', -1)
            target_color = obs.target_action.get('new_color', -1)

            key = (relative_size, source_color)
            pattern_counts[key][target_color] += 1

        # Find consistent patterns
        for (relative_size, source_color), color_counts in pattern_counts.items():
            # Find most common target color
            target_color = max(color_counts.items(), key=lambda x: x[1])[0]
            support = color_counts[target_color]

            # Confidence = fraction of examples supporting this pattern
            confidence = support / n_examples

            # Only accept high-confidence patterns (>= 0.5 means majority of examples)
            if confidence >= 0.5:
                # Create the invariant
                if relative_size == 'largest':
                    condition = lambda obj, all_objs, rs=relative_size: (
                        obj.size == max(o.size for o in all_objs) if all_objs else True
                    )
                    condition_desc = f"largest object"
                elif relative_size == 'smallest':
                    condition = lambda obj, all_objs, rs=relative_size: (
                        obj.size == min(o.size for o in all_objs) if all_objs else True
                    )
                    condition_desc = f"smallest object"
                elif source_color != -1:
                    condition = lambda obj, all_objs, sc=source_color: obj.dominant_color == sc
                    condition_desc = f"objects with color {source_color}"
                else:
                    condition = lambda obj, all_objs: True
                    condition_desc = "all objects"

                action = lambda grid, obj, all_objs, tc=target_color: self._recolor_object(grid, obj, tc)
                action_desc = f"recolor to {target_color}"

                invariants.append(InvariantPattern(
                    transform_type='color_change',
                    condition=condition,
                    condition_desc=condition_desc,
                    action=action,
                    action_desc=action_desc,
                    confidence=confidence,
                    support_count=support
                ))

        return invariants

    def _extract_position_invariants(self, observations: List[TransformationObservation],
                                    n_examples: int) -> List[InvariantPattern]:
        """Extract position change invariants."""
        invariants = []

        # Count movement patterns
        deltas = [obs.target_action.get('delta', (0, 0)) for obs in observations]

        if deltas:
            # Check if all movements are the same
            delta_counts = Counter(deltas)
            most_common_delta, count = delta_counts.most_common(1)[0]
            confidence = count / n_examples

            if confidence >= 0.5:
                dy, dx = most_common_delta

                condition = lambda obj, all_objs: True
                condition_desc = "all objects"
                action = lambda grid, obj, all_objs, dy=dy, dx=dx: self._move_object(grid, obj, dy, dx)
                action_desc = f"move by ({dy:.0f}, {dx:.0f})"

                invariants.append(InvariantPattern(
                    transform_type='position_change',
                    condition=condition,
                    condition_desc=condition_desc,
                    action=action,
                    action_desc=action_desc,
                    confidence=confidence,
                    support_count=count
                ))

        return invariants

    def _extract_size_invariants(self, observations: List[TransformationObservation],
                                n_examples: int) -> List[InvariantPattern]:
        """Extract size change invariants."""
        invariants = []

        # Check for consistent scaling
        ratios = [obs.target_action.get('size_ratio', 1.0) for obs in observations]

        if ratios:
            avg_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)

            # If all ratios are similar (low variance), it's a scale pattern
            if std_ratio < 0.3 and abs(avg_ratio - 1.0) > 0.2:
                confidence = len(observations) / n_examples

                if confidence >= 0.5:
                    scale_factor = int(round(avg_ratio))

                    condition = lambda obj, all_objs: True
                    condition_desc = "all objects"
                    action = lambda grid, obj, all_objs, sf=scale_factor: self._scale_object(grid, obj, sf)
                    action_desc = f"scale by {scale_factor}x"

                    invariants.append(InvariantPattern(
                        transform_type='size_change',
                        condition=condition,
                        condition_desc=condition_desc,
                        action=action,
                        action_desc=action_desc,
                        confidence=confidence,
                        support_count=len(observations)
                    ))

        return invariants

    def _extract_pixel_mapping_invariants(self, observations: List[TransformationObservation],
                                         n_examples: int) -> List[InvariantPattern]:
        """Extract pixel-level color mapping invariants."""
        invariants = []

        # Count color mappings: source_color -> target_color
        mappings = defaultdict(lambda: defaultdict(int))

        for obs in observations:
            source_color = obs.source_condition.get('color', -1)
            target_color = obs.target_action.get('new_color', -1)
            if source_color != -1 and target_color != -1:
                mappings[source_color][target_color] += 1

        # Find consistent mappings
        for source_color, target_counts in mappings.items():
            target_color = max(target_counts.items(), key=lambda x: x[1])[0]
            support = target_counts[target_color]
            confidence = support / n_examples

            if confidence >= 0.8:  # Higher threshold for pixel mappings
                condition = lambda obj, all_objs, sc=source_color: obj.dominant_color == sc
                condition_desc = f"pixels with color {source_color}"
                action = lambda grid, obj, all_objs, tc=target_color: self._map_color(grid, source_color, tc)
                action_desc = f"map to color {target_color}"

                invariants.append(InvariantPattern(
                    transform_type='pixel_color_map',
                    condition=condition,
                    condition_desc=condition_desc,
                    action=action,
                    action_desc=action_desc,
                    confidence=confidence,
                    support_count=support
                ))

        return invariants

    # Helper methods for applying transformations

    def _recolor_object(self, grid: np.ndarray, obj: ArcObject, new_color: int) -> np.ndarray:
        """Recolor a specific object."""
        result = grid.copy()
        result[obj.mask] = new_color
        return result

    def _move_object(self, grid: np.ndarray, obj: ArcObject, dy: float, dx: float) -> np.ndarray:
        """Move a specific object."""
        result = grid.copy()

        # Clear old position
        result[obj.mask] = 0

        # Place at new position
        y1, x1, y2, x2 = obj.bbox
        new_y1 = int(y1 + dy)
        new_x1 = int(x1 + dx)
        new_y2 = int(y2 + dy)
        new_x2 = int(x2 + dx)

        # Check bounds
        if (new_y1 >= 0 and new_x1 >= 0 and
            new_y2 <= grid.shape[0] and new_x2 <= grid.shape[1]):

            # Shift the mask
            new_mask = np.zeros_like(result, dtype=bool)
            mask_h, mask_w = obj.mask.shape
            new_mask[new_y1:new_y1+mask_h, new_x1:new_x1+mask_w] = obj.mask[y1:y2, x1:x2]
            result[new_mask] = obj.dominant_color

        return result

    def _scale_object(self, grid: np.ndarray, obj: ArcObject, scale_factor: int) -> np.ndarray:
        """Scale a specific object."""
        result = grid.copy()

        # Simple scaling by repetition
        y1, x1, y2, x2 = obj.bbox
        obj_grid = grid[y1:y2, x1:x2]

        if scale_factor > 1:
            scaled = np.repeat(np.repeat(obj_grid, scale_factor, axis=0), scale_factor, axis=1)
        elif scale_factor < 1:
            scaled = obj_grid[::abs(scale_factor), ::abs(scale_factor)]
        else:
            return result

        # Clear old position
        result[obj.mask] = 0

        # Place scaled version (if it fits)
        if (y1 + scaled.shape[0] <= result.shape[0] and
            x1 + scaled.shape[1] <= result.shape[1]):
            result[y1:y1+scaled.shape[0], x1:x1+scaled.shape[1]] = scaled

        return result

    def _map_color(self, grid: np.ndarray, source_color: int, target_color: int) -> np.ndarray:
        """Map all pixels of one color to another."""
        result = grid.copy()
        result[result == source_color] = target_color
        return result


class PatternBasedHypothesisGenerator:
    """Generate hypotheses based on detected invariant patterns."""

    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer()
        self.object_detector = ObjectDetector()

    def generate_from_patterns(self, invariants: List[InvariantPattern],
                               test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """
        Generate hypotheses from invariant patterns.

        Returns: List of (Transform object, description, confidence)
        """
        hypotheses = []

        # Detect objects in test input
        test_objects = self.object_detector.detect_objects(test_input)

        for pattern in invariants:
            # Create a transformation function that applies this pattern
            def make_transform(pattern=pattern):
                def transform(grid: np.ndarray) -> np.ndarray:
                    result = grid.copy()
                    objects = self.object_detector.detect_objects(result)

                    # Apply pattern to each matching object
                    for obj in objects:
                        if pattern.condition(obj, objects):
                            result = pattern.action(result, obj, objects)

                    return result
                return transform

            transform_fn = make_transform()
            description = f"{pattern.condition_desc} → {pattern.action_desc}"

            # Create a proper Transform object
            transform_obj = Transform(
                name=f"pattern_{pattern.transform_type}",
                function=transform_fn,
                parameters={},
                category='pattern_inferred'
            )

            hypotheses.append((transform_obj, description, pattern.confidence))

        return hypotheses
