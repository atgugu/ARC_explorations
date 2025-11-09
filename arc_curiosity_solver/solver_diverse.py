"""
Diverse Pattern-Based ARC Curiosity Solver

Extends pattern-based solver with safe diversity improvements:
1. Parameter variations of detected patterns
2. Pattern combinations
3. Systematic exploration around detected invariants

Philosophy: If we detect "largest → color 2", the PATTERN is likely correct,
but maybe the PARAMETER is wrong. Try: color 1, 3, 4... or "smallest → color 2"

This stays true to example-driven generation while exploring parameter space.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from .solver_pattern_based import PatternBasedARCCuriositySolver
from .belief_dynamics.belief_space import Hypothesis
from .core.pattern_inference import InvariantPattern, PatternAnalyzer, PatternBasedHypothesisGenerator
from .transformations.arc_primitives import Transform
from .core.object_reasoning import ObjectDetector
import numpy


class DiverseARCCuriositySolver(PatternBasedARCCuriositySolver):
    """
    Enhanced pattern-based solver with safe diversity improvements.

    Key improvements:
    - Generates parameter variations of detected patterns
    - Tries pattern combinations
    - Explores neighborhood around detected invariants
    """

    def __init__(self):
        super().__init__()
        self.diversity_multiplier = 3  # Generate 3x variations per pattern
        self.max_combinations = 5  # Max pattern combinations to try
        self.force_diverse_selection = True  # Force diverse prediction selection

    def _generate_hypotheses(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                            test_input: np.ndarray) -> List[Hypothesis]:
        """
        Generate diverse hypotheses with parameter variations.

        Safe approach:
        1. Detect invariant patterns (from training examples)
        2. For each pattern, generate parameter variations
        3. Try pattern combinations
        4. Add diversity while staying example-driven
        """
        hypotheses = []

        # Step 1: Analyze training pairs to extract patterns
        self.detected_patterns = self.pattern_analyzer.analyze_training_pairs(train_pairs)

        if self.detected_patterns:
            self.used_pattern_inference = True

            if self.verbose:
                print(f"  🔍 Detected {len(self.detected_patterns)} invariant patterns:")
                for i, pattern in enumerate(self.detected_patterns[:5], 1):
                    print(f"     {i}. {pattern}")

            # Step 2: Generate BASE hypotheses from patterns (exact as detected)
            base_hypotheses = self.pattern_generator.generate_from_patterns(
                self.detected_patterns, test_input
            )

            for transform_obj, description, confidence in base_hypotheses:
                hyp = Hypothesis(
                    program=transform_obj,
                    name=f"pattern_{len(hypotheses)}",
                    parameters={'description': description, 'variant': 'exact'},
                    activation=confidence
                )
                hypotheses.append(hyp)

            if self.verbose:
                print(f"  Generated {len(hypotheses)} exact pattern hypotheses")

            # Step 3: Generate VARIATIONS of detected patterns (safe diversity)
            variations = self._generate_pattern_variations(
                self.detected_patterns, test_input, train_pairs
            )

            for transform_obj, description, confidence in variations:
                hyp = Hypothesis(
                    program=transform_obj,
                    name=f"variation_{len(hypotheses)}",
                    parameters={'description': description, 'variant': 'variation'},
                    activation=confidence * 0.7  # Lower confidence for variations
                )
                hypotheses.append(hyp)

            if self.verbose:
                print(f"  Generated {len(variations)} pattern variations")

            # Step 4: Generate COMBINATIONS of patterns (if multiple detected)
            if len(self.detected_patterns) >= 2:
                combinations = self._generate_pattern_combinations(
                    self.detected_patterns, test_input
                )

                for transform_obj, description, confidence in combinations:
                    hyp = Hypothesis(
                        program=transform_obj,
                        name=f"combo_{len(hypotheses)}",
                        parameters={'description': description, 'variant': 'combination'},
                        activation=confidence * 0.6  # Even lower for combinations
                    )
                    hypotheses.append(hyp)

                if self.verbose:
                    print(f"  Generated {len(combinations)} pattern combinations")

        else:
            # No patterns detected - fall back to base solver
            if self.verbose:
                print("  ⚠️  No invariant patterns detected, using generic hypotheses")

            hypotheses = super(PatternBasedARCCuriositySolver, self)._generate_hypotheses(
                train_pairs, test_input
            )

        return hypotheses

    def _generate_pattern_variations(self, patterns: List[InvariantPattern],
                                    test_input: np.ndarray,
                                    train_pairs: List[Tuple]) -> List[Tuple[Transform, str, float]]:
        """
        Generate safe variations of detected patterns.

        For each pattern, explore parameter space:
        - Color changes: Try adjacent colors
        - Size categories: Try opposite category (largest ↔ smallest)
        - Spatial patterns: Try different directions

        Safe because: Only varies parameters of DETECTED patterns
        """
        variations = []
        object_detector = ObjectDetector()

        # Collect colors used in training data
        training_colors = set()
        for inp, out in train_pairs:
            training_colors.update(inp.flatten())
            training_colors.update(out.flatten())
        training_colors = sorted(training_colors)

        for pattern in patterns:
            if pattern.transform_type == 'color_change':
                # Generate color variations
                color_vars = self._vary_color_pattern(pattern, training_colors, test_input)
                variations.extend(color_vars)

            elif pattern.transform_type == 'position_change':
                # Generate position variations
                pos_vars = self._vary_position_pattern(pattern, test_input)
                variations.extend(pos_vars)

            elif pattern.transform_type == 'pixel_color_map':
                # Generate pixel mapping variations
                map_vars = self._vary_pixel_mapping(pattern, training_colors, test_input)
                variations.extend(map_vars)

        # Limit total variations
        return variations[:len(patterns) * self.diversity_multiplier]

    def _vary_color_pattern(self, pattern: InvariantPattern,
                           training_colors: List[int],
                           test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """Generate color parameter variations."""
        variations = []

        # Extract original target color from action description
        # E.g., "recolor to 2" → try colors 0, 1, 3, 4 from training data

        for target_color in training_colors:
            # Create variation with different target color
            def make_color_variation(pattern=pattern, color=target_color):
                def transform(grid: np.ndarray) -> np.ndarray:
                    result = grid.copy()
                    detector = ObjectDetector()
                    objects = detector.detect_objects(result)

                    for obj in objects:
                        if pattern.condition(obj, objects):
                            # Apply recoloring with variation color
                            result[obj.mask] = color

                    return result
                return transform

            transform_fn = make_color_variation()
            description = f"{pattern.condition_desc} → recolor to {target_color}"

            transform_obj = Transform(
                name=f"color_var_{target_color}",
                function=transform_fn,
                parameters={'target_color': int(target_color)},
                category='pattern_variation'
            )

            # Confidence decreases with distance from original pattern
            confidence = pattern.confidence * 0.5
            variations.append((transform_obj, description, confidence))

        return variations

    def _vary_position_pattern(self, pattern: InvariantPattern,
                               test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """Generate position parameter variations."""
        variations = []

        # Try different movement magnitudes
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (0, 2)]

        for dy, dx in deltas:
            def make_position_variation(dy=dy, dx=dx):
                def transform(grid: np.ndarray) -> np.ndarray:
                    result = grid.copy()
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
                parameters={'dy': dy, 'dx': dx},
                category='pattern_variation'
            )

            confidence = pattern.confidence * 0.4
            variations.append((transform_obj, description, confidence))

        return variations[:3]  # Limit to 3 variations

    def _vary_pixel_mapping(self, pattern: InvariantPattern,
                           training_colors: List[int],
                           test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """Generate pixel color mapping variations."""
        variations = []

        # Try mapping each training color to each other training color
        for src_color in training_colors[:3]:  # Limit source colors
            for tgt_color in training_colors[:3]:  # Limit target colors
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
                    parameters={'source': int(src_color), 'target': int(tgt_color)},
                    category='pattern_variation'
                )

                confidence = pattern.confidence * 0.3
                variations.append((transform_obj, description, confidence))

        return variations[:5]  # Limit to 5 variations

    def _generate_pattern_combinations(self, patterns: List[InvariantPattern],
                                       test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """
        Generate combinations of detected patterns.

        If we detected both "recolor" and "move" patterns, try applying both.
        Safe because both patterns were independently detected in training data.
        """
        combinations = []

        # Only combine top patterns
        top_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)[:3]

        if len(top_patterns) < 2:
            return combinations

        # Try pairs of patterns
        for i in range(len(top_patterns)):
            for j in range(i + 1, len(top_patterns)):
                p1, p2 = top_patterns[i], top_patterns[j]

                # Only combine different types
                if p1.transform_type == p2.transform_type:
                    continue

                # Create combined transformation
                def make_combination(p1=p1, p2=p2):
                    def transform(grid: np.ndarray) -> np.ndarray:
                        result = grid.copy()
                        detector = ObjectDetector()
                        objects = detector.detect_objects(result)

                        # Apply first pattern
                        for obj in objects:
                            if p1.condition(obj, objects):
                                result = p1.action(result, obj, objects)

                        # Re-detect objects after first transformation
                        objects = detector.detect_objects(result)

                        # Apply second pattern
                        for obj in objects:
                            if p2.condition(obj, objects):
                                result = p2.action(result, obj, objects)

                        return result
                    return transform

                transform_fn = make_combination()
                description = f"({p1.condition_desc} → {p1.action_desc}) THEN ({p2.condition_desc} → {p2.action_desc})"

                transform_obj = Transform(
                    name=f"combo_{p1.transform_type}_{p2.transform_type}",
                    function=transform_fn,
                    parameters={'p1': p1.transform_type, 'p2': p2.transform_type},
                    category='pattern_combination'
                )

                # Confidence is product of individual confidences (both must be correct)
                confidence = min(p1.confidence, p2.confidence) * 0.5
                combinations.append((transform_obj, description, confidence))

                if len(combinations) >= self.max_combinations:
                    break

            if len(combinations) >= self.max_combinations:
                break

        return combinations

    def _select_diverse_hypotheses(self, hypotheses: List[Hypothesis], k: int = 2) -> List[Tuple[Hypothesis, float]]:
        """
        Select diverse hypotheses prioritizing different variant types.

        Strategy:
        1. Pick best exact pattern hypothesis
        2. Pick best variation hypothesis (different type)
        3. If not enough, fill with highest belief remaining

        This ensures we try both exact patterns AND variations.
        """
        if not hypotheses:
            return []

        # Separate hypotheses by variant type
        exact = [h for h in hypotheses if h.parameters.get('variant') == 'exact']
        variations = [h for h in hypotheses if h.parameters.get('variant') == 'variation']
        combinations = [h for h in hypotheses if h.parameters.get('variant') == 'combination']
        generic = [h for h in hypotheses if 'variant' not in h.parameters]

        # Sort each group by activation
        exact = sorted(exact, key=lambda h: h.activation, reverse=True)
        variations = sorted(variations, key=lambda h: h.activation, reverse=True)
        combinations = sorted(combinations, key=lambda h: h.activation, reverse=True)
        generic = sorted(generic, key=lambda h: h.activation, reverse=True)

        selected = []

        # Strategy: Pick one from each category to ensure diversity
        if exact:
            selected.append((exact[0], exact[0].activation))
        if variations and len(selected) < k:
            selected.append((variations[0], variations[0].activation))
        if combinations and len(selected) < k:
            selected.append((combinations[0], combinations[0].activation))
        if generic and len(selected) < k:
            selected.append((generic[0], generic[0].activation))

        # If we still need more, fill with highest activation
        if len(selected) < k:
            all_hyps = sorted(hypotheses, key=lambda h: h.activation, reverse=True)
            for hyp in all_hyps:
                if (hyp, hyp.activation) not in selected and len(selected) < k:
                    selected.append((hyp, hyp.activation))

        return selected[:k]

    def solve(self, train_pairs: List[Tuple], test_input: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve with diverse hypothesis selection.

        Uses diverse selection in parent's solve method.
        """
        # Temporarily override parent's _select_top_k_hypotheses with diverse selection
        original_select = self._original_select if hasattr(self, '_original_select') else None

        if self.force_diverse_selection and not original_select:
            # Store original and replace with diverse
            self._original_select = super()._select_top_k_hypotheses

        # Call parent solve
        result = super().solve(train_pairs, test_input, verbose)

        return result

    def _select_top_k_hypotheses(self, hypotheses: List[Hypothesis], k: int = 2) -> List[Tuple[Hypothesis, float]]:
        """
        Override parent's selection with diverse selection when pattern inference used.
        """
        if self.force_diverse_selection and self.used_pattern_inference:
            return self._select_diverse_hypotheses(hypotheses, k)
        else:
            # Standard selection
            return super()._select_top_k_hypotheses(hypotheses, k)
