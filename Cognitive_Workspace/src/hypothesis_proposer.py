"""
Hypothesis Proposer - Phase 4

Generates candidate programs by analyzing training pairs and proposing
transformations using the 65 DSL primitives.

Key components:
1. Pattern Analyzer - Detects patterns in input-output pairs
2. Hypothesis Generator - Proposes candidate programs
3. Beam Search - Explores program space efficiently
4. Scorer - Ranks hypotheses by likelihood
"""

from typing import List, Tuple, Dict, Callable, Optional, Any
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import sys
import os

# Import all DSL primitives
sys.path.insert(0, os.path.dirname(__file__))
from dsl.core_primitives import *


@dataclass
class Pattern:
    """Detected pattern in input-output transformation"""
    name: str
    confidence: float
    parameters: Dict[str, Any]
    description: str


@dataclass
class Hypothesis:
    """A candidate program hypothesis"""
    program: Callable
    primitives: List[str]
    parameters: Dict[str, Any]
    score: float
    description: str
    pattern: Optional[Pattern] = None


class PatternAnalyzer:
    """Analyzes input-output pairs to detect transformation patterns"""

    def __init__(self):
        self.patterns = []

    def analyze_pair(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Analyze a single input-output pair"""
        patterns = []

        # Size change patterns
        patterns.extend(self._detect_size_changes(input_grid, output_grid))

        # Color change patterns
        patterns.extend(self._detect_color_changes(input_grid, output_grid))

        # Object-based patterns
        patterns.extend(self._detect_object_patterns(input_grid, output_grid))

        # Spatial patterns
        patterns.extend(self._detect_spatial_patterns(input_grid, output_grid))

        # Tiling patterns
        patterns.extend(self._detect_tiling_patterns(input_grid, output_grid))

        # Symmetry patterns
        patterns.extend(self._detect_symmetry_patterns(input_grid, output_grid))

        # Cropping patterns
        patterns.extend(self._detect_crop_patterns(input_grid, output_grid))

        # Object selection patterns
        patterns.extend(self._detect_selection_patterns(input_grid, output_grid))

        # Fill and morphology patterns
        patterns.extend(self._detect_fill_patterns(input_grid, output_grid))

        # Color by property patterns
        patterns.extend(self._detect_color_by_property(input_grid, output_grid))

        # Overlay patterns
        patterns.extend(self._detect_overlay_patterns(input_grid, output_grid))

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def _detect_size_changes(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect size-related transformations"""
        patterns = []
        ih, iw = input_grid.shape
        oh, ow = output_grid.shape

        # Scaling
        if oh % ih == 0 and ow % iw == 0:
            scale_r = oh // ih
            scale_c = ow // iw
            if scale_r == scale_c:
                patterns.append(Pattern(
                    name="scale",
                    confidence=0.9,
                    parameters={"factor": scale_r},
                    description=f"Output is {scale_r}x scaled version of input"
                ))

        # Tiling
        if oh % ih == 0 and ow % iw == 0:
            tile_r = oh // ih
            tile_c = ow // iw
            patterns.append(Pattern(
                name="tile",
                confidence=0.8,
                parameters={"rows": tile_r, "cols": tile_c},
                description=f"Output tiles input {tile_r}x{tile_c}"
            ))

        # Cropping
        if oh <= ih and ow <= iw:
            patterns.append(Pattern(
                name="crop",
                confidence=0.7,
                parameters={"height": oh, "width": ow},
                description="Output is cropped version of input"
            ))

        return patterns

    def _detect_color_changes(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect color transformation patterns"""
        patterns = []

        if input_grid.shape != output_grid.shape:
            return patterns

        # Check for simple recoloring
        input_colors = set(input_grid.flatten()) - {0}
        output_colors = set(output_grid.flatten()) - {0}

        # Color swap
        if len(input_colors) == len(output_colors):
            color_map = {}
            for r in range(input_grid.shape[0]):
                for c in range(input_grid.shape[1]):
                    ic = input_grid[r, c]
                    oc = output_grid[r, c]
                    if ic != 0:
                        if ic in color_map:
                            if color_map[ic] != oc:
                                break
                        else:
                            color_map[ic] = oc
            else:
                if color_map:
                    patterns.append(Pattern(
                        name="recolor",
                        confidence=0.95,
                        parameters={"color_map": color_map},
                        description=f"Colors swapped: {color_map}"
                    ))

        # Check if output inverts input colors
        inverted = True
        for r in range(input_grid.shape[0]):
            for c in range(input_grid.shape[1]):
                if input_grid[r, c] != 0 and output_grid[r, c] != 0:
                    expected = 10 - input_grid[r, c]
                    if output_grid[r, c] != expected:
                        inverted = False
                        break
            if not inverted:
                break

        if inverted:
            patterns.append(Pattern(
                name="invert_colors",
                confidence=0.9,
                parameters={},
                description="Colors are inverted"
            ))

        return patterns

    def _detect_object_patterns(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect object-based transformations"""
        patterns = []

        # Get objects from input
        input_objects = []
        for color in range(1, 10):
            objs = select_by_color(input_grid, color)
            input_objects.extend(objs)

        output_objects = []
        for color in range(1, 10):
            objs = select_by_color(output_grid, color)
            output_objects.extend(objs)

        # Object count changes
        if len(output_objects) > len(input_objects):
            factor = len(output_objects) / len(input_objects)
            if factor == int(factor):
                patterns.append(Pattern(
                    name="duplicate_objects",
                    confidence=0.7,
                    parameters={"factor": int(factor)},
                    description=f"Objects duplicated {int(factor)}x"
                ))

        # Object size changes
        if input_objects and output_objects:
            avg_input_size = np.mean([len(obj) for obj in input_objects])
            avg_output_size = np.mean([len(obj) for obj in output_objects])

            if avg_output_size > avg_input_size * 1.5:
                patterns.append(Pattern(
                    name="grow_objects",
                    confidence=0.7,
                    parameters={"iterations": 1},
                    description="Objects are grown/dilated"
                ))
            elif avg_output_size < avg_input_size * 0.7:
                patterns.append(Pattern(
                    name="shrink_objects",
                    confidence=0.7,
                    parameters={"iterations": 1},
                    description="Objects are shrunk/eroded"
                ))

        return patterns

    def _detect_spatial_patterns(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect spatial transformation patterns"""
        patterns = []

        if input_grid.shape != output_grid.shape:
            return patterns

        # Rotation
        for angle in [90, 180, 270]:
            rotated = np.rot90(input_grid, k=angle // 90)
            if np.array_equal(rotated, output_grid):
                patterns.append(Pattern(
                    name="rotate",
                    confidence=1.0,
                    parameters={"angle": angle},
                    description=f"Rotated {angle} degrees"
                ))
                return patterns  # Early exit if exact match

        # Reflection
        h_flip = np.flip(input_grid, axis=0)
        v_flip = np.flip(input_grid, axis=1)

        if np.array_equal(h_flip, output_grid):
            patterns.append(Pattern(
                name="reflect",
                confidence=1.0,
                parameters={"axis": Axis.HORIZONTAL},
                description="Reflected horizontally"
            ))

        if np.array_equal(v_flip, output_grid):
            patterns.append(Pattern(
                name="reflect",
                confidence=1.0,
                parameters={"axis": Axis.VERTICAL},
                description="Reflected vertically"
            ))

        return patterns

    def _detect_tiling_patterns(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect tiling patterns"""
        patterns = []
        ih, iw = input_grid.shape
        oh, ow = output_grid.shape

        # Check if output is tiled version of input
        if oh % ih == 0 and ow % iw == 0:
            rows = oh // ih
            cols = ow // iw

            # Check if it's actually tiled
            is_tiled = True
            for r in range(rows):
                for c in range(cols):
                    tile = output_grid[r*ih:(r+1)*ih, c*iw:(c+1)*iw]
                    if not np.array_equal(tile, input_grid):
                        is_tiled = False
                        break
                if not is_tiled:
                    break

            if is_tiled and (rows > 1 or cols > 1):
                patterns.append(Pattern(
                    name="tile",
                    confidence=1.0,
                    parameters={"rows": rows, "cols": cols},
                    description=f"Tiled {rows}x{cols}"
                ))

        return patterns

    def _detect_symmetry_patterns(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect symmetry patterns"""
        patterns = []

        if input_grid.shape != output_grid.shape:
            return patterns

        # Check if output is symmetric version of input
        h_sym = symmetrize(input_grid, Axis.HORIZONTAL)
        v_sym = symmetrize(input_grid, Axis.VERTICAL)
        both_sym = symmetrize(input_grid, Axis.BOTH)

        if np.array_equal(h_sym, output_grid):
            patterns.append(Pattern(
                name="symmetrize",
                confidence=1.0,
                parameters={"axis": Axis.HORIZONTAL},
                description="Symmetrized horizontally"
            ))

        if np.array_equal(v_sym, output_grid):
            patterns.append(Pattern(
                name="symmetrize",
                confidence=1.0,
                parameters={"axis": Axis.VERTICAL},
                description="Symmetrized vertically"
            ))

        if np.array_equal(both_sym, output_grid):
            patterns.append(Pattern(
                name="symmetrize",
                confidence=1.0,
                parameters={"axis": Axis.BOTH},
                description="Symmetrized both axes"
            ))

        return patterns

    def _detect_crop_patterns(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect cropping patterns"""
        patterns = []

        # Check for crop_to_content
        try:
            cropped = crop_to_content(input_grid)
            if np.array_equal(cropped, output_grid):
                patterns.append(Pattern(
                    name="crop_to_content",
                    confidence=1.0,
                    parameters={},
                    description="Cropped to content"
                ))
        except:
            pass

        return patterns

    def _detect_selection_patterns(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect object selection and extraction patterns"""
        patterns = []

        # Check if output is a selected object from input
        # Get objects from input
        input_objs_by_color = {}
        for color in range(1, 10):
            objs = select_by_color(input_grid, color)
            if objs:
                input_objs_by_color[color] = objs

        # Check if output contains just the largest object
        try:
            all_input_objs = []
            for objs in input_objs_by_color.values():
                all_input_objs.extend(objs)

            if all_input_objs:
                largest = select_largest(all_input_objs, k=1)
                if largest:
                    # Create grid with just the largest object
                    test_grid = np.zeros(output_grid.shape, dtype=int)
                    for r, c in largest[0]:
                        if 0 <= r < test_grid.shape[0] and 0 <= c < test_grid.shape[1]:
                            test_grid[r, c] = input_grid[r, c]

                    if np.array_equal(test_grid, output_grid):
                        patterns.append(Pattern(
                            name="select_largest",
                            confidence=0.9,
                            parameters={},
                            description="Selected largest object"
                        ))
        except:
            pass

        # Check if output is just one color from input
        output_colors = set(output_grid.flatten()) - {0}
        input_colors = set(input_grid.flatten()) - {0}

        if len(output_colors) == 1 and len(input_colors) > 1:
            color = list(output_colors)[0]
            # Check if output is just this color from input
            test_grid = np.where(input_grid == color, color, 0)
            if np.array_equal(test_grid, output_grid):
                patterns.append(Pattern(
                    name="select_color",
                    confidence=0.95,
                    parameters={"color": color},
                    description=f"Selected color {color}"
                ))

        return patterns

    def _detect_fill_patterns(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect fill and morphology patterns"""
        patterns = []

        if input_grid.shape != output_grid.shape:
            return patterns

        # Check for fill_holes pattern
        # Get objects and see if output has holes filled
        try:
            for color in range(1, 10):
                input_objs = select_by_color(input_grid, color)
                output_objs = select_by_color(output_grid, color)

                if input_objs and output_objs:
                    # Check if each output object is filled version of input
                    filled_match = True
                    for i, inp_obj in enumerate(input_objs):
                        if i < len(output_objs):
                            filled = fill_holes(inp_obj, grid_shape=input_grid.shape)
                            # Check if filled matches output object area
                            if len(filled) > len(inp_obj):
                                # Holes were filled
                                test_grid = input_grid.copy()
                                for r, c in filled:
                                    if 0 <= r < test_grid.shape[0] and 0 <= c < test_grid.shape[1]:
                                        test_grid[r, c] = color

                                # Compare with output
                                if np.array_equal(test_grid, output_grid):
                                    patterns.append(Pattern(
                                        name="fill_holes",
                                        confidence=0.95,
                                        parameters={},
                                        description="Fill holes in objects"
                                    ))
                                    return patterns
        except:
            pass

        return patterns

    def _detect_color_by_property(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect patterns where objects are colored by their properties"""
        patterns = []

        if input_grid.shape != output_grid.shape:
            return patterns

        # Get objects from input
        input_objs_by_color = {}
        for color in range(1, 10):
            objs = select_by_color(input_grid, color)
            if objs:
                input_objs_by_color[color] = objs

        # Get objects from output
        output_objs_by_color = {}
        for color in range(1, 10):
            objs = select_by_color(output_grid, color)
            if objs:
                output_objs_by_color[color] = objs

        # Check if objects are colored by size
        if input_objs_by_color and output_objs_by_color:
            # Collect all input objects
            all_input_objs = []
            for objs in input_objs_by_color.values():
                all_input_objs.extend(objs)

            # Collect all output objects
            all_output_objs = []
            for objs in output_objs_by_color.values():
                all_output_objs.extend(objs)

            # Check if same positions but different colors based on size
            if len(all_input_objs) == len(all_output_objs):
                # Sort by size
                input_sizes = [(i, len(obj)) for i, obj in enumerate(all_input_objs)]
                output_colors = []

                # Get output color for each object
                for obj in all_output_objs:
                    if obj:
                        r, c = obj[0]
                        if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                            output_colors.append(output_grid[r, c])

                # Check if colors correlate with size order
                if len(output_colors) == len(input_sizes):
                    sorted_indices = sorted(range(len(input_sizes)), key=lambda i: input_sizes[i][1])
                    colors_match_size_order = True

                    # Simple heuristic: check if colors increase/decrease with size
                    patterns.append(Pattern(
                        name="color_by_size",
                        confidence=0.7,
                        parameters={},
                        description="Color objects by their size"
                    ))

        return patterns

    def _detect_overlay_patterns(self, input_grid: Grid, output_grid: Grid) -> List[Pattern]:
        """Detect overlay and combination patterns"""
        patterns = []

        # Check if output combines multiple objects from input
        # Look for cases where output has more colors/objects than any single input

        # Check if output is input with background filled
        input_colors = set(input_grid.flatten()) - {0}
        output_colors = set(output_grid.flatten()) - {0}

        if input_colors == output_colors and input_grid.shape == output_grid.shape:
            # Check if only background (0) pixels changed
            different_pixels = np.sum(input_grid != output_grid)
            total_pixels = input_grid.size

            if 0 < different_pixels < total_pixels * 0.5:
                # Some pixels changed, might be background fill
                # Check if all changes are from 0 to some color
                changes_from_zero = True
                for r in range(input_grid.shape[0]):
                    for c in range(input_grid.shape[1]):
                        if input_grid[r, c] != output_grid[r, c]:
                            if input_grid[r, c] != 0:
                                changes_from_zero = False
                                break
                    if not changes_from_zero:
                        break

                if changes_from_zero:
                    patterns.append(Pattern(
                        name="fill_background",
                        confidence=0.8,
                        parameters={},
                        description="Fill background with pattern"
                    ))

        return patterns


class HypothesisGenerator:
    """Generates candidate program hypotheses from detected patterns"""

    def __init__(self):
        self.analyzer = PatternAnalyzer()

    def generate_hypotheses(self, train_pairs: List[Dict]) -> List[Hypothesis]:
        """Generate hypotheses from training pairs"""
        hypotheses = []

        # Analyze all training pairs
        all_patterns = []
        for pair in train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            patterns = self.analyzer.analyze_pair(input_grid, output_grid)
            all_patterns.append(patterns)

        # Find common patterns across all training pairs
        common_patterns = self._find_common_patterns(all_patterns)

        # Generate hypotheses from common patterns
        for pattern in common_patterns:
            hyp = self._pattern_to_hypothesis(pattern)
            if hyp:
                hypotheses.append(hyp)

        # Add composite hypotheses (multi-step programs)
        hypotheses.extend(self._generate_composite_hypotheses(common_patterns))

        return sorted(hypotheses, key=lambda h: h.score, reverse=True)

    def _find_common_patterns(self, all_patterns: List[List[Pattern]]) -> List[Pattern]:
        """Find patterns that appear in all training examples"""
        if not all_patterns:
            return []

        # Count pattern occurrences
        pattern_counts = defaultdict(lambda: {'count': 0, 'avg_confidence': 0, 'params': []})

        for patterns in all_patterns:
            seen_names = set()
            for pattern in patterns:
                if pattern.name not in seen_names:
                    pattern_counts[pattern.name]['count'] += 1
                    pattern_counts[pattern.name]['avg_confidence'] += pattern.confidence
                    pattern_counts[pattern.name]['params'].append(pattern.parameters)
                    seen_names.add(pattern.name)

        # Get patterns that appear in all examples
        num_examples = len(all_patterns)
        common = []

        for name, data in pattern_counts.items():
            if data['count'] == num_examples:
                # Average confidence
                avg_conf = data['avg_confidence'] / num_examples

                # Check if parameters are consistent
                params = data['params']
                if self._parameters_consistent(params):
                    common.append(Pattern(
                        name=name,
                        confidence=avg_conf,
                        parameters=params[0],
                        description=f"{name} with {params[0]}"
                    ))

        return common

    def _parameters_consistent(self, params_list: List[Dict]) -> bool:
        """Check if parameters are consistent across examples"""
        if not params_list:
            return True

        first = params_list[0]
        for params in params_list[1:]:
            if params != first:
                return False
        return True

    def _pattern_to_hypothesis(self, pattern: Pattern) -> Optional[Hypothesis]:
        """Convert a pattern to a hypothesis"""
        name = pattern.name
        params = pattern.parameters

        # Map pattern names to programs
        if name == "rotate":
            angle = params.get("angle", 90)
            def program(grid):
                return np.rot90(grid, k=angle // 90)

            return Hypothesis(
                program=program,
                primitives=["rotate"],
                parameters=params,
                score=pattern.confidence,
                description=f"Rotate {angle} degrees",
                pattern=pattern
            )

        elif name == "reflect":
            axis = params.get("axis", Axis.HORIZONTAL)
            def program(grid):
                if axis == Axis.HORIZONTAL:
                    return np.flip(grid, axis=0)
                else:
                    return np.flip(grid, axis=1)

            return Hypothesis(
                program=program,
                primitives=["reflect"],
                parameters=params,
                score=pattern.confidence,
                description=f"Reflect {axis}",
                pattern=pattern
            )

        elif name == "tile":
            rows = params.get("rows", 2)
            cols = params.get("cols", 2)
            def program(grid):
                # Use numpy tile for grid-based tiling
                return np.tile(grid, (rows, cols))

            return Hypothesis(
                program=program,
                primitives=["tile"],
                parameters=params,
                score=pattern.confidence,
                description=f"Tile {rows}x{cols}",
                pattern=pattern
            )

        elif name == "recolor":
            color_map = params.get("color_map", {})
            def program(grid):
                result = grid.copy()
                for old_color, new_color in color_map.items():
                    result[grid == old_color] = new_color
                return result

            return Hypothesis(
                program=program,
                primitives=["recolor"],
                parameters=params,
                score=pattern.confidence,
                description=f"Recolor {color_map}",
                pattern=pattern
            )

        elif name == "symmetrize":
            axis = params.get("axis", Axis.HORIZONTAL)
            def program(grid):
                return symmetrize(grid, axis)

            return Hypothesis(
                program=program,
                primitives=["symmetrize"],
                parameters=params,
                score=pattern.confidence,
                description=f"Symmetrize {axis}",
                pattern=pattern
            )

        elif name == "invert_colors":
            def program(grid):
                return invert_colors(grid)

            return Hypothesis(
                program=program,
                primitives=["invert_colors"],
                parameters=params,
                score=pattern.confidence,
                description="Invert colors",
                pattern=pattern
            )

        elif name == "grow_objects":
            iterations = params.get("iterations", 1)
            def program(grid):
                result = grid.copy()
                for color in range(1, 10):
                    objs = select_by_color(result, color)
                    for obj in objs:
                        grown = grow(obj, iterations)
                        result = overlay(result, grown, color=color)
                return result

            return Hypothesis(
                program=program,
                primitives=["grow", "select_by_color"],
                parameters=params,
                score=pattern.confidence * 0.8,
                description="Grow all objects",
                pattern=pattern
            )

        elif name == "crop_to_content":
            def program(grid):
                return crop_to_content(grid)

            return Hypothesis(
                program=program,
                primitives=["crop_to_content"],
                parameters=params,
                score=pattern.confidence,
                description="Crop to content",
                pattern=pattern
            )

        elif name == "select_largest":
            def program(grid):
                # Get all objects
                all_objs = []
                for color in range(1, 10):
                    objs = select_by_color(grid, color)
                    all_objs.extend(objs)

                if not all_objs:
                    return grid

                # Get largest
                largest = select_largest(all_objs, k=1)
                if not largest:
                    return grid

                # Create output grid
                result = np.zeros(grid.shape, dtype=int)
                for r, c in largest[0]:
                    if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                        result[r, c] = grid[r, c]

                return result

            return Hypothesis(
                program=program,
                primitives=["select_largest", "select_by_color"],
                parameters=params,
                score=pattern.confidence,
                description="Select largest object",
                pattern=pattern
            )

        elif name == "select_color":
            color = params.get("color", 1)
            def program(grid):
                return np.where(grid == color, color, 0)

            return Hypothesis(
                program=program,
                primitives=["select_color"],
                parameters=params,
                score=pattern.confidence,
                description=f"Select color {color}",
                pattern=pattern
            )

        elif name == "fill_holes":
            def program(grid):
                result = grid.copy()
                for color in range(1, 10):
                    objs = select_by_color(grid, color)
                    for obj in objs:
                        filled = fill_holes(obj, grid_shape=grid.shape)
                        # Apply filled object
                        for r, c in filled:
                            if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                                result[r, c] = color
                return result

            return Hypothesis(
                program=program,
                primitives=["fill_holes", "select_by_color"],
                parameters=params,
                score=pattern.confidence,
                description="Fill holes in objects",
                pattern=pattern
            )

        elif name == "color_by_size":
            def program(grid):
                result = grid.copy()
                # Get all objects
                all_objs = []
                for color in range(1, 10):
                    objs = select_by_color(grid, color)
                    all_objs.extend([(obj, color) for obj in objs])

                if not all_objs:
                    return result

                # Sort by size
                all_objs.sort(key=lambda x: len(x[0]))

                # Assign colors based on size (smallest=1, largest=9)
                result = np.zeros(grid.shape, dtype=int)
                num_objs = len(all_objs)
                for i, (obj, _) in enumerate(all_objs):
                    # Map index to color (1-9)
                    new_color = min(9, max(1, int((i / max(num_objs - 1, 1)) * 8) + 1))
                    for r, c in obj:
                        if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                            result[r, c] = new_color

                return result

            return Hypothesis(
                program=program,
                primitives=["select_by_color"],
                parameters=params,
                score=pattern.confidence,
                description="Color objects by size",
                pattern=pattern
            )

        elif name == "fill_background":
            def program(grid):
                # Find most common non-zero color
                from collections import Counter
                colors = [c for c in grid.flatten() if c != 0]
                if not colors:
                    return grid

                common_color = Counter(colors).most_common(1)[0][0]

                # Fill all zeros with this color
                result = grid.copy()
                result[result == 0] = common_color
                return result

            return Hypothesis(
                program=program,
                primitives=[],
                parameters=params,
                score=pattern.confidence,
                description="Fill background",
                pattern=pattern
            )

        return None

    def _generate_composite_hypotheses(self, patterns: List[Pattern]) -> List[Hypothesis]:
        """Generate multi-step composite hypotheses (2-step and 3-step)"""
        composites = []

        # Limit patterns to top 5 to avoid combinatorial explosion
        top_patterns = patterns[:5]

        # Try combining pairs of patterns (2-step)
        for i, p1 in enumerate(top_patterns):
            for p2 in top_patterns[i+1:]:
                # Try p1 then p2
                h1 = self._pattern_to_hypothesis(p1)
                h2 = self._pattern_to_hypothesis(p2)

                if h1 and h2:
                    def composite_program(grid, h1=h1, h2=h2):
                        try:
                            intermediate = h1.program(grid)
                            return h2.program(intermediate)
                        except:
                            return grid

                    composites.append(Hypothesis(
                        program=composite_program,
                        primitives=h1.primitives + h2.primitives,
                        parameters={**h1.parameters, **h2.parameters},
                        score=h1.score * h2.score * 0.9,  # Penalty for complexity
                        description=f"{h1.description} then {h2.description}",
                        pattern=None
                    ))

        # Try 3-step compositions for top 3 patterns only
        top3_patterns = patterns[:3]
        for i, p1 in enumerate(top3_patterns):
            for j, p2 in enumerate(top3_patterns):
                if i != j:
                    for k, p3 in enumerate(top3_patterns):
                        if k != i and k != j:
                            h1 = self._pattern_to_hypothesis(p1)
                            h2 = self._pattern_to_hypothesis(p2)
                            h3 = self._pattern_to_hypothesis(p3)

                            if h1 and h2 and h3:
                                def composite_program_3(grid, h1=h1, h2=h2, h3=h3):
                                    try:
                                        step1 = h1.program(grid)
                                        step2 = h2.program(step1)
                                        return h3.program(step2)
                                    except:
                                        return grid

                                composites.append(Hypothesis(
                                    program=composite_program_3,
                                    primitives=h1.primitives + h2.primitives + h3.primitives,
                                    parameters={**h1.parameters, **h2.parameters, **h3.parameters},
                                    score=h1.score * h2.score * h3.score * 0.8,  # Higher penalty
                                    description=f"{h1.description}, {h2.description}, then {h3.description}",
                                    pattern=None
                                ))

        return composites


class HypothesisProposer:
    """Main hypothesis proposer - coordinates analysis and generation"""

    def __init__(self):
        self.generator = HypothesisGenerator()

    def propose(self, train_pairs: List[Dict], beam_size: int = 10) -> List[Hypothesis]:
        """
        Propose candidate hypotheses for a task.

        Args:
            train_pairs: List of training input-output pairs
            beam_size: Number of top hypotheses to keep

        Returns:
            List of top hypotheses ranked by score
        """
        # Generate hypotheses
        hypotheses = self.generator.generate_hypotheses(train_pairs)

        # Validate hypotheses on training pairs
        validated = []
        for hyp in hypotheses:
            score = self._validate_hypothesis(hyp, train_pairs)
            hyp.score = score
            validated.append(hyp)

        # Sort by validation score
        validated.sort(key=lambda h: h.score, reverse=True)

        # Return top-k
        return validated[:beam_size]

    def _validate_hypothesis(self, hypothesis: Hypothesis, train_pairs: List[Dict]) -> float:
        """Validate hypothesis on training pairs"""
        correct = 0
        total = len(train_pairs)

        for pair in train_pairs:
            input_grid = np.array(pair['input'])
            expected_output = np.array(pair['output'])

            try:
                predicted_output = hypothesis.program(input_grid)

                # Check if output matches
                if np.array_equal(predicted_output, expected_output):
                    correct += 1
                else:
                    # Partial credit for similar outputs
                    if predicted_output.shape == expected_output.shape:
                        similarity = np.sum(predicted_output == expected_output) / predicted_output.size
                        correct += similarity * 0.5
            except Exception as e:
                # Hypothesis failed, score 0 for this pair
                pass

        return correct / total if total > 0 else 0.0

    def solve(self, task: Dict) -> Optional[np.ndarray]:
        """
        Solve a task by proposing and selecting best hypothesis.

        Args:
            task: Task dictionary with 'train' and 'test' keys

        Returns:
            Predicted output grid or None if no solution found
        """
        train_pairs = task['train']
        test_input = np.array(task['test'][0]['input'])

        # Propose hypotheses
        hypotheses = self.propose(train_pairs, beam_size=10)

        if not hypotheses:
            return None

        # Use best hypothesis
        best = hypotheses[0]

        try:
            output = best.program(test_input)
            return output
        except Exception as e:
            # Try next best hypotheses
            for hyp in hypotheses[1:]:
                try:
                    output = hyp.program(test_input)
                    return output
                except:
                    continue

        return None
