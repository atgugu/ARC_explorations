"""
Enhanced Conditional Pattern Inference

Key improvements:
1. Relaxed object matching (multiple strategies)
2. Conditional validation on training data
3. Richer conditional types (shape, count, relationships)
4. Learning from training patterns (not hardcoded)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter

from ..transformations.conditional_transforms import (
    ConditionalTransform, Condition, ConditionalAction,
    ConditionLibrary, ActionLibrary, ConditionalTransformBuilder
)
from ..core.object_reasoning import ObjectDetector, ArcObject
from ..transformations.arc_primitives import Transform


class ImprovedConditionalPatternAnalyzer:
    """
    Enhanced pattern analyzer with:
    - Relaxed object matching
    - Training validation
    - Richer conditional types
    - Learned patterns (not hardcoded)
    """

    def __init__(self):
        self.object_detector = ObjectDetector()
        self.condition_lib = ConditionLibrary()
        self.action_lib = ActionLibrary()
        self.builder = ConditionalTransformBuilder()

    def analyze_training_pairs(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[ConditionalTransform]:
        """
        Extract conditional patterns with validation.

        Process:
        1. Collect observations using relaxed matching
        2. Detect conditional patterns
        3. Validate on training data
        4. Return only high-confidence conditionals
        """
        # Collect observations
        all_observations = []
        for inp, out in train_pairs:
            observations = self._analyze_pair_relaxed(inp, out)
            all_observations.extend(observations)

        if not all_observations:
            return []

        # Detect patterns
        candidate_conditionals = self._detect_conditional_patterns(all_observations, train_pairs)

        # Validate on training data
        validated_conditionals = []
        for conditional in candidate_conditionals:
            accuracy = self._validate_on_training(conditional, train_pairs)
            if accuracy >= 0.8:  # Must work on 80%+ of training examples
                conditional.confidence = accuracy
                validated_conditionals.append(conditional)

        # Sort by confidence
        validated_conditionals.sort(key=lambda x: x.confidence, reverse=True)

        return validated_conditionals

    def _analyze_pair_relaxed(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze pair using relaxed object matching."""
        observations = []

        # Detect objects
        input_objects = self.object_detector.detect_objects(input_grid)
        output_objects = self.object_detector.detect_objects(output_grid)

        if not input_objects:
            return observations

        # Try multiple matching strategies
        matched_pairs = self._match_objects_multi_strategy(input_objects, output_objects, input_grid, output_grid)

        # Record observations
        for inp_obj, out_obj in matched_pairs:
            obs = {
                'obj_in': inp_obj,
                'obj_out': out_obj,
                'grid_in': input_grid,
                'grid_out': output_grid,
                'all_objs_in': input_objects,
                'all_objs_out': output_objects
            }
            observations.append(obs)

        # Handle removed objects
        matched_input_ids = {id(pair[0]) for pair in matched_pairs}
        for inp_obj in input_objects:
            if id(inp_obj) not in matched_input_ids:
                obs = {
                    'obj_in': inp_obj,
                    'obj_out': None,  # Removed
                    'grid_in': input_grid,
                    'grid_out': output_grid,
                    'all_objs_in': input_objects,
                    'all_objs_out': output_objects
                }
                observations.append(obs)

        return observations

    def _match_objects_multi_strategy(self, input_objects: List[ArcObject],
                                      output_objects: List[ArcObject],
                                      input_grid: np.ndarray,
                                      output_grid: np.ndarray) -> List[Tuple[ArcObject, ArcObject]]:
        """
        Try multiple matching strategies to find best matches.

        Strategies:
        1. Position + size similarity (current)
        2. Color + approximate size
        3. Relative order in list
        4. Bounding box overlap
        """
        if not input_objects or not output_objects:
            return []

        # Strategy 1: Position + size (existing)
        matches_1 = self._match_by_position_and_size(input_objects, output_objects)

        # Strategy 2: Color + size
        matches_2 = self._match_by_color_and_size(input_objects, output_objects)

        # Strategy 3: Relative order
        matches_3 = self._match_by_order(input_objects, output_objects)

        # Choose best strategy (most matches with highest confidence)
        all_strategies = [
            (matches_1, 'position_size'),
            (matches_2, 'color_size'),
            (matches_3, 'order')
        ]

        # Return strategy with most matches
        best_matches = max(all_strategies, key=lambda x: len(x[0]))[0]

        return best_matches

    def _match_by_position_and_size(self, input_objects: List[ArcObject],
                                    output_objects: List[ArcObject]) -> List[Tuple[ArcObject, ArcObject]]:
        """Match by position and size (original strategy)."""
        matched = []
        used_output = set()

        for inp_obj in input_objects:
            best_match = None
            best_score = float('inf')

            for i, out_obj in enumerate(output_objects):
                if i in used_output:
                    continue

                # Position distance
                dist = np.sqrt((inp_obj.position[0] - out_obj.position[0])**2 +
                             (inp_obj.position[1] - out_obj.position[1])**2)

                # Size similarity
                size_diff = abs(inp_obj.size - out_obj.size) / max(inp_obj.size, out_obj.size, 1)

                score = dist + size_diff * 5

                if score < best_score:
                    best_score = score
                    best_match = i

            # Relaxed threshold (was 20, now 30)
            if best_match is not None and best_score < 30:
                matched.append((inp_obj, output_objects[best_match]))
                used_output.add(best_match)

        return matched

    def _match_by_color_and_size(self, input_objects: List[ArcObject],
                                 output_objects: List[ArcObject]) -> List[Tuple[ArcObject, ArcObject]]:
        """Match by dominant color and approximate size."""
        matched = []
        used_output = set()

        for inp_obj in input_objects:
            best_match = None
            best_score = float('inf')

            for i, out_obj in enumerate(output_objects):
                if i in used_output:
                    continue

                # Color match (0 if same, 1 if different)
                color_diff = 0 if inp_obj.dominant_color == out_obj.dominant_color else 1

                # Size similarity
                size_ratio = out_obj.size / max(inp_obj.size, 1)
                size_diff = abs(1.0 - size_ratio)

                score = color_diff * 10 + size_diff * 5

                if score < best_score:
                    best_score = score
                    best_match = i

            if best_match is not None and best_score < 15:
                matched.append((inp_obj, output_objects[best_match]))
                used_output.add(best_match)

        return matched

    def _match_by_order(self, input_objects: List[ArcObject],
                       output_objects: List[ArcObject]) -> List[Tuple[ArcObject, ArcObject]]:
        """Match by relative order (1st input → 1st output, etc.)"""
        matched = []
        n = min(len(input_objects), len(output_objects))

        # Sort both by position (top-left to bottom-right)
        input_sorted = sorted(input_objects, key=lambda obj: (obj.position[0], obj.position[1]))
        output_sorted = sorted(output_objects, key=lambda obj: (obj.position[0], obj.position[1]))

        for i in range(n):
            matched.append((input_sorted[i], output_sorted[i]))

        return matched

    def _detect_conditional_patterns(self, observations: List[Dict[str, Any]],
                                    train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[ConditionalTransform]:
        """
        Detect conditional patterns from observations.

        Uses machine learning approach: find properties that correlate with different transformations.
        """
        conditionals = []

        # Group observations by transformation type
        color_changes = [obs for obs in observations if self._is_color_change(obs)]
        position_changes = [obs for obs in observations if self._is_position_change(obs)]
        size_changes = [obs for obs in observations if self._is_size_change(obs)]
        removals = [obs for obs in observations if obs['obj_out'] is None]

        # Detect color conditionals
        conditionals.extend(self._detect_color_conditionals(color_changes))

        # Detect position conditionals
        conditionals.extend(self._detect_position_conditionals(position_changes))

        # Detect removal conditionals
        conditionals.extend(self._detect_removal_conditionals(removals, observations))

        # Detect size conditionals
        conditionals.extend(self._detect_size_conditionals(size_changes))

        return conditionals

    def _is_color_change(self, obs: Dict) -> bool:
        """Check if observation shows color change."""
        if obs['obj_out'] is None:
            return False
        return obs['obj_in'].dominant_color != obs['obj_out'].dominant_color

    def _is_position_change(self, obs: Dict) -> bool:
        """Check if observation shows position change."""
        if obs['obj_out'] is None:
            return False
        dy = obs['obj_out'].position[0] - obs['obj_in'].position[0]
        dx = obs['obj_out'].position[1] - obs['obj_in'].position[1]
        return abs(dy) > 0.5 or abs(dx) > 0.5

    def _is_size_change(self, obs: Dict) -> bool:
        """Check if observation shows size change."""
        if obs['obj_out'] is None:
            return False
        return abs(obs['obj_in'].size - obs['obj_out'].size) > 2

    def _detect_color_conditionals(self, observations: List[Dict]) -> List[ConditionalTransform]:
        """
        Detect conditional color patterns.

        Strategy: Find properties that correlate with different output colors.
        """
        conditionals = []

        if len(observations) < 2:
            return conditionals

        # Analyze: Does output color depend on size?
        size_to_color = defaultdict(list)
        for obs in observations:
            size = obs['obj_in'].size
            out_color = obs['obj_out'].dominant_color
            size_to_color[size].append(out_color)

        # Check if different sizes get different colors
        sizes = sorted(size_to_color.keys())
        if len(sizes) >= 2:
            small_colors = Counter(size_to_color[sizes[0]])
            large_colors = Counter(size_to_color[sizes[-1]])

            small_color = small_colors.most_common(1)[0][0]
            large_color = large_colors.most_common(1)[0][0]

            if small_color != large_color:
                # Found size-based color conditional!
                threshold = (sizes[0] + sizes[-1]) // 2

                conditional = ConditionalTransform(
                    condition=self.condition_lib.size_greater_than(threshold),
                    then_action=self.action_lib.recolor_to(large_color),
                    else_action=self.action_lib.recolor_to(small_color),
                    confidence=0.7
                )
                conditionals.append(conditional)

        # Analyze: Does output color depend on position (edge vs center)?
        edge_colors = []
        center_colors = []

        for obs in observations:
            grid = obs['grid_in']
            obj = obs['obj_in']
            out_color = obs['obj_out'].dominant_color

            y1, x1, y2, x2 = obj.bbox
            h, w = grid.shape

            is_edge = (y1 < 2 or x1 < 2 or y2 > h-2 or x2 > w-2)

            if is_edge:
                edge_colors.append(out_color)
            else:
                center_colors.append(out_color)

        if edge_colors and center_colors:
            edge_color = Counter(edge_colors).most_common(1)[0][0]
            center_color = Counter(center_colors).most_common(1)[0][0]

            if edge_color != center_color:
                # Found edge-based color conditional!
                conditional = ConditionalTransform(
                    condition=self.condition_lib.near_edge(2),
                    then_action=self.action_lib.recolor_to(edge_color),
                    else_action=self.action_lib.recolor_to(center_color),
                    confidence=0.7
                )
                conditionals.append(conditional)

        return conditionals

    def _detect_position_conditionals(self, observations: List[Dict]) -> List[ConditionalTransform]:
        """Detect conditional position patterns."""
        conditionals = []

        if len(observations) < 2:
            return conditionals

        # Analyze: Does movement depend on initial position?
        by_quadrant = defaultdict(list)

        for obs in observations:
            grid = obs['grid_in']
            obj_in = obs['obj_in']
            obj_out = obs['obj_out']

            h, w = grid.shape
            obj_y, obj_x = obj_in.position

            quadrant = (
                'top' if obj_y < h/2 else 'bottom',
                'left' if obj_x < w/2 else 'right'
            )

            dy = obj_out.position[0] - obj_in.position[0]
            dx = obj_out.position[1] - obj_in.position[1]

            by_quadrant[quadrant].append((dy, dx))

        # Check if different quadrants move differently
        if len(by_quadrant) >= 2:
            movements = {}
            for quadrant, deltas in by_quadrant.items():
                avg_dy = int(np.mean([d[0] for d in deltas]))
                avg_dx = int(np.mean([d[1] for d in deltas]))
                movements[quadrant] = (avg_dy, avg_dx)

            # If movements are different, create conditionals
            unique_movements = set(movements.values())
            if len(unique_movements) > 1:
                # For simplicity, create top vs bottom conditional
                if ('top', 'left') in movements and ('bottom', 'left') in movements:
                    top_move = movements[('top', 'left')]
                    bottom_move = movements[('bottom', 'left')]

                    if top_move != bottom_move:
                        conditional = ConditionalTransform(
                            condition=self.condition_lib.in_quadrant('top_left'),
                            then_action=self.action_lib.move_by(*top_move),
                            else_action=self.action_lib.move_by(*bottom_move),
                            confidence=0.6
                        )
                        conditionals.append(conditional)

        return conditionals

    def _detect_removal_conditionals(self, removals: List[Dict], all_observations: List[Dict]) -> List[ConditionalTransform]:
        """Detect conditional removal patterns."""
        conditionals = []

        if not removals:
            return conditionals

        # What distinguishes removed vs kept objects?
        removed_sizes = [obs['obj_in'].size for obs in removals]
        kept_objects = [obs for obs in all_observations if obs['obj_out'] is not None]
        kept_sizes = [obs['obj_in'].size for obs in kept_objects]

        if removed_sizes and kept_sizes:
            avg_removed = np.mean(removed_sizes)
            avg_kept = np.mean(kept_sizes)

            if avg_removed < avg_kept * 0.7:  # Removed objects significantly smaller
                threshold = int((avg_removed + avg_kept) / 2)

                conditional = ConditionalTransform(
                    condition=self.condition_lib.size_less_than(threshold),
                    then_action=self.action_lib.remove(),
                    else_action=self.action_lib.keep(),
                    confidence=0.7
                )
                conditionals.append(conditional)

        return conditionals

    def _detect_size_conditionals(self, observations: List[Dict]) -> List[ConditionalTransform]:
        """Detect size-based transformation conditionals."""
        # For now, size conditionals are covered in color_conditionals
        # Can be expanded if needed
        return []

    def _validate_on_training(self, conditional: ConditionalTransform,
                             train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Validate conditional on training data.

        Returns: Accuracy (0.0 to 1.0)
        """
        correct = 0
        total = len(train_pairs)

        for inp, out in train_pairs:
            try:
                prediction = conditional.apply(inp)

                # Check if prediction matches output
                if np.array_equal(prediction, out):
                    correct += 1
                else:
                    # Partial credit for shape match
                    if prediction.shape == out.shape:
                        accuracy = (prediction == out).mean()
                        correct += accuracy

            except Exception:
                # If transformation fails, count as 0
                pass

        return correct / total if total > 0 else 0.0
