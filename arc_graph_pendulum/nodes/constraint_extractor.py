"""
Constraint Extractor for SMT-Based Program Synthesis.

Analyzes input-output pairs to extract formal constraints that any valid
program must satisfy. These constraints guide SMT-based program search.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Any
from scipy.ndimage import label
from collections import Counter


class ConstraintExtractor:
    """
    Extracts constraints from ARC training examples.

    Constraint categories:
    1. Shape constraints (dimensions, transformations)
    2. Color constraints (palette, mappings)
    3. Spatial constraints (positions, relationships)
    4. Object constraints (count, properties)
    """

    def __init__(self):
        pass

    def extract(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        Extract all constraints from training examples.

        Args:
            train_examples: List of (input, output) pairs

        Returns:
            Dictionary of constraints organized by category
        """
        if not train_examples:
            return {}

        constraints = {
            'shape': self._extract_shape_constraints(train_examples),
            'color': self._extract_color_constraints(train_examples),
            'spatial': self._extract_spatial_constraints(train_examples),
            'object': self._extract_object_constraints(train_examples),
            'pixel': self._extract_pixel_constraints(train_examples),
            'meta': {
                'num_examples': len(train_examples),
                'deterministic': self._check_deterministic(train_examples)
            }
        }

        return constraints

    def _extract_shape_constraints(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Extract constraints related to grid dimensions and shape transformations."""
        constraints = {
            'input_shapes': [],
            'output_shapes': [],
            'shape_relationship': None,
            'preserves_shape': True,
            'scale_factor': None,
            'is_extraction': False,
            'is_expansion': False
        }

        for inp, out in train_examples:
            in_h, in_w = inp.shape
            out_h, out_w = out.shape

            constraints['input_shapes'].append((in_h, in_w))
            constraints['output_shapes'].append((out_h, out_w))

            if (in_h, in_w) != (out_h, out_w):
                constraints['preserves_shape'] = False

        # Determine shape relationship
        if constraints['preserves_shape']:
            constraints['shape_relationship'] = 'preserve'
        else:
            # Check for consistent scaling
            in_shapes = constraints['input_shapes']
            out_shapes = constraints['output_shapes']

            ratios_h = [out_h / in_h for (in_h, in_w), (out_h, out_w) in zip(in_shapes, out_shapes)]
            ratios_w = [out_w / in_w for (in_h, in_w), (out_h, out_w) in zip(in_shapes, out_shapes)]

            if len(set(ratios_h)) == 1 and len(set(ratios_w)) == 1:
                if ratios_h[0] == ratios_w[0]:
                    constraints['shape_relationship'] = 'uniform_scale'
                    constraints['scale_factor'] = ratios_h[0]
                else:
                    constraints['shape_relationship'] = 'non_uniform_scale'

            # Check if output is always smaller (extraction)
            all_smaller = all(out_h <= in_h and out_w <= in_w for (in_h, in_w), (out_h, out_w) in zip(in_shapes, out_shapes))
            if all_smaller:
                constraints['is_extraction'] = True

            # Check if output is always larger (expansion)
            all_larger = all(out_h >= in_h and out_w >= in_w for (in_h, in_w), (out_h, out_w) in zip(in_shapes, out_shapes))
            if all_larger:
                constraints['is_expansion'] = True

        return constraints

    def _extract_color_constraints(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Extract constraints related to color palettes and transformations."""
        constraints = {
            'input_palettes': [],
            'output_palettes': [],
            'preserves_palette': True,
            'adds_colors': set(),
            'removes_colors': set(),
            'color_mapping': None,
            'fixed_mapping': True,
            'background_color': None
        }

        # Collect palettes
        for inp, out in train_examples:
            in_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))

            constraints['input_palettes'].append(in_colors)
            constraints['output_palettes'].append(out_colors)

            if in_colors != out_colors:
                constraints['preserves_palette'] = False

        # Determine background color (most common across all inputs)
        all_input_colors = []
        for inp, _ in train_examples:
            all_input_colors.extend(inp.flatten().tolist())
        if all_input_colors:
            constraints['background_color'] = Counter(all_input_colors).most_common(1)[0][0]

        # Check for consistent color additions/removals
        if not constraints['preserves_palette']:
            all_added = None
            all_removed = None

            for in_pal, out_pal in zip(constraints['input_palettes'], constraints['output_palettes']):
                added = out_pal - in_pal
                removed = in_pal - out_pal

                if all_added is None:
                    all_added = added
                    all_removed = removed
                else:
                    if added != all_added:
                        all_added = set()
                    if removed != all_removed:
                        all_removed = set()

            if all_added:
                constraints['adds_colors'] = all_added
            if all_removed:
                constraints['removes_colors'] = all_removed

        # Try to infer color mapping
        color_mappings = []
        for inp, out in train_examples:
            if inp.shape == out.shape:
                mapping = {}
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        in_color = inp[i, j]
                        out_color = out[i, j]

                        if in_color in mapping:
                            if mapping[in_color] != out_color:
                                # Not a consistent mapping
                                mapping = None
                                break
                        else:
                            mapping[in_color] = out_color
                    if mapping is None:
                        break

                if mapping:
                    color_mappings.append(mapping)

        # Check if all examples have the same color mapping
        if color_mappings and len(color_mappings) == len(train_examples):
            first_mapping = color_mappings[0]
            if all(m == first_mapping for m in color_mappings):
                constraints['color_mapping'] = first_mapping
                constraints['fixed_mapping'] = True

        return constraints

    def _extract_spatial_constraints(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Extract constraints related to spatial transformations."""
        constraints = {
            'preserves_positions': [],
            'has_translation': False,
            'has_reflection': False,
            'has_rotation': False,
            'input_embedded': False
        }

        for inp, out in train_examples:
            # Check if input is embedded in output
            if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
                # Try to find input as subregion
                found_embedded = False
                for r in range(out.shape[0] - inp.shape[0] + 1):
                    for c in range(out.shape[1] - inp.shape[1] + 1):
                        subregion = out[r:r+inp.shape[0], c:c+inp.shape[1]]
                        if np.array_equal(subregion, inp):
                            found_embedded = True
                            break
                    if found_embedded:
                        break

                if found_embedded:
                    constraints['input_embedded'] = True

            # Check for reflections (if same shape)
            if inp.shape == out.shape:
                # Horizontal reflection
                if np.array_equal(out, np.fliplr(inp)):
                    constraints['has_reflection'] = 'horizontal'
                # Vertical reflection
                elif np.array_equal(out, np.flipud(inp)):
                    constraints['has_reflection'] = 'vertical'
                # 90-degree rotation
                elif np.array_equal(out, np.rot90(inp, k=1)):
                    constraints['has_rotation'] = 90
                # 180-degree rotation
                elif np.array_equal(out, np.rot90(inp, k=2)):
                    constraints['has_rotation'] = 180
                # 270-degree rotation
                elif np.array_equal(out, np.rot90(inp, k=3)):
                    constraints['has_rotation'] = 270

        return constraints

    def _extract_object_constraints(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Extract constraints related to connected component objects."""
        constraints = {
            'input_object_counts': [],
            'output_object_counts': [],
            'preserves_object_count': True,
            'increases_objects': False,
            'decreases_objects': False
        }

        for inp, out in train_examples:
            # Count objects in input
            in_obj_count = self._count_objects(inp)
            out_obj_count = self._count_objects(out)

            constraints['input_object_counts'].append(in_obj_count)
            constraints['output_object_counts'].append(out_obj_count)

            if in_obj_count != out_obj_count:
                constraints['preserves_object_count'] = False

                if out_obj_count > in_obj_count:
                    constraints['increases_objects'] = True
                else:
                    constraints['decreases_objects'] = True

        return constraints

    def _extract_pixel_constraints(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Extract pixel-level constraints (specific positions/values)."""
        constraints = {
            'fixed_pixels': [],  # Pixels that never change
            'changed_pixels': [],  # Pixels that always change
            'background_preserved': False
        }

        # Only extract if shape is preserved
        if not train_examples:
            return constraints

        first_inp, first_out = train_examples[0]
        if first_inp.shape != first_out.shape:
            return constraints

        # Check if all examples have same shape
        same_shape = all(inp.shape == first_inp.shape and out.shape == first_out.shape
                        for inp, out in train_examples)
        if not same_shape:
            return constraints

        # Find pixels that are always preserved
        h, w = first_inp.shape
        for i in range(h):
            for j in range(w):
                always_preserved = all(inp[i, j] == out[i, j] for inp, out in train_examples)
                always_changed = all(inp[i, j] != out[i, j] for inp, out in train_examples)

                if always_preserved:
                    constraints['fixed_pixels'].append((i, j))
                elif always_changed:
                    constraints['changed_pixels'].append((i, j))

        return constraints

    def _count_objects(self, grid: np.ndarray, background: int = 0) -> int:
        """Count connected component objects (excluding background)."""
        total_objects = 0
        colors = [c for c in np.unique(grid) if c != background]

        for color in colors:
            color_mask = (grid == color).astype(int)
            labeled, num_features = label(color_mask)
            total_objects += num_features

        return total_objects

    def _check_deterministic(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if same inputs always produce same outputs (should always be true for ARC)."""
        input_output_map = {}

        for inp, out in train_examples:
            inp_key = inp.tobytes()

            if inp_key in input_output_map:
                if not np.array_equal(input_output_map[inp_key], out):
                    return False
            else:
                input_output_map[inp_key] = out

        return True

    def summarize_constraints(self, constraints: Dict[str, Any]) -> str:
        """Generate human-readable summary of extracted constraints."""
        lines = []
        lines.append("=== EXTRACTED CONSTRAINTS ===")

        # Shape constraints
        shape = constraints.get('shape', {})
        lines.append(f"\n[Shape]")
        lines.append(f"  Relationship: {shape.get('shape_relationship', 'unknown')}")
        lines.append(f"  Preserves shape: {shape.get('preserves_shape', False)}")
        if shape.get('scale_factor'):
            lines.append(f"  Scale factor: {shape.get('scale_factor')}")

        # Color constraints
        color = constraints.get('color', {})
        lines.append(f"\n[Color]")
        lines.append(f"  Preserves palette: {color.get('preserves_palette', False)}")
        if color.get('color_mapping'):
            lines.append(f"  Fixed color mapping: {color.get('color_mapping')}")
        if color.get('adds_colors'):
            lines.append(f"  Adds colors: {color.get('adds_colors')}")
        if color.get('removes_colors'):
            lines.append(f"  Removes colors: {color.get('removes_colors')}")

        # Spatial constraints
        spatial = constraints.get('spatial', {})
        lines.append(f"\n[Spatial]")
        if spatial.get('has_reflection'):
            lines.append(f"  Reflection: {spatial.get('has_reflection')}")
        if spatial.get('has_rotation'):
            lines.append(f"  Rotation: {spatial.get('has_rotation')}°")
        lines.append(f"  Input embedded: {spatial.get('input_embedded', False)}")

        # Object constraints
        obj = constraints.get('object', {})
        lines.append(f"\n[Objects]")
        lines.append(f"  Preserves count: {obj.get('preserves_object_count', False)}")
        if obj.get('increases_objects'):
            lines.append(f"  Increases objects: True")
        if obj.get('decreases_objects'):
            lines.append(f"  Decreases objects: True")

        return "\n".join(lines)
