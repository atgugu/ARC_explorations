"""
Constraint-Based Program Synthesizer.

Uses extracted constraints to guide program synthesis, filtering the
program search space to only consider candidates that satisfy constraints.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Any, Callable, Optional
from scipy.ndimage import label
import itertools


class ConstraintBasedSynthesizer:
    """
    Synthesizes programs by matching constraints to primitive operations.

    Instead of searching all possible programs, we:
    1. Match constraints to compatible primitives
    2. Generate candidate programs that satisfy constraint profile
    3. Verify candidates against training examples
    4. Return programs sorted by simplicity
    """

    def __init__(self):
        # Map constraint patterns to compatible primitives
        self.constraint_to_primitives = self._build_constraint_mapping()

    def synthesize(self,
                   constraints: Dict[str, Any],
                   train_examples: List[Tuple[np.ndarray, np.ndarray]],
                   max_depth: int = 3,
                   max_candidates: int = 20) -> List[Dict[str, Any]]:
        """
        Synthesize programs that satisfy the given constraints.

        Args:
            constraints: Extracted constraints from training examples
            train_examples: Training examples for verification
            max_depth: Maximum program depth (number of operations)
            max_candidates: Maximum number of candidate programs to return

        Returns:
            List of candidate programs, sorted by score
        """
        if not train_examples:
            return []

        # Step 1: Match constraints to primitives
        compatible_primitives = self._match_constraints_to_primitives(constraints)

        if not compatible_primitives:
            return []

        # Step 2: Generate candidate programs using iterative deepening
        candidates = []

        for depth in range(1, max_depth + 1):
            depth_candidates = self._generate_candidates(
                compatible_primitives,
                depth,
                constraints
            )

            # Step 3: Verify candidates against training
            for candidate in depth_candidates:
                score = self._evaluate_candidate(candidate, train_examples)

                if score > 0.0:
                    candidates.append({
                        'function': candidate['function'],
                        'description': candidate['description'],
                        'depth': depth,
                        'score': score,
                        'primitive_type': candidate['primitive_type']
                    })

                    if len(candidates) >= max_candidates:
                        break

            if len(candidates) >= max_candidates:
                break

        # Sort by score (descending), then by depth (ascending)
        candidates.sort(key=lambda c: (-c['score'], c['depth']))

        return candidates

    def _match_constraints_to_primitives(self, constraints: Dict[str, Any]) -> List[str]:
        """Match extracted constraints to compatible primitive operations."""
        compatible = set()

        shape = constraints.get('shape', {})
        color = constraints.get('color', {})
        spatial = constraints.get('spatial', {})
        obj = constraints.get('object', {})

        # Shape-based matching
        if shape.get('preserves_shape'):
            compatible.update(['recolor', 'fill', 'pattern', 'mask'])

        if shape.get('is_extraction'):
            compatible.update(['extract', 'crop', 'select'])

        if shape.get('is_expansion'):
            compatible.update(['expand', 'pad', 'tile'])

        if shape.get('scale_factor'):
            compatible.update(['scale', 'replicate'])

        # Color-based matching
        if color.get('color_mapping'):
            compatible.add('color_mapping')

        if not color.get('preserves_palette'):
            compatible.update(['recolor', 'swap_colors'])

        # Spatial-based matching
        if spatial.get('has_reflection'):
            if spatial['has_reflection'] == 'horizontal':
                compatible.add('flip_horizontal')
            elif spatial['has_reflection'] == 'vertical':
                compatible.add('flip_vertical')

        if spatial.get('has_rotation'):
            compatible.add('rotate')

        if spatial.get('input_embedded'):
            compatible.add('embed')

        # Object-based matching
        if obj.get('increases_objects'):
            compatible.update(['copy_objects', 'tile_objects'])

        if obj.get('decreases_objects'):
            compatible.update(['filter_objects', 'merge_objects'])

        return list(compatible)

    def _generate_candidates(self,
                            primitives: List[str],
                            depth: int,
                            constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate programs of given depth."""
        candidates = []

        if depth == 1:
            # Single-operation programs
            for prim in primitives:
                candidate = self._create_program(prim, constraints)
                if candidate:
                    candidates.append(candidate)
        else:
            # Multi-operation programs (composition)
            # For now, limit to depth=1 for efficiency
            # TODO: Implement composition in future versions
            pass

        return candidates

    def _create_program(self, primitive_type: str, constraints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a program function for a given primitive type."""
        try:
            if primitive_type == 'flip_horizontal':
                return {
                    'function': lambda grid: np.fliplr(grid),
                    'description': 'Flip grid horizontally (left-right)',
                    'primitive_type': 'flip_horizontal'
                }

            elif primitive_type == 'flip_vertical':
                return {
                    'function': lambda grid: np.flipud(grid),
                    'description': 'Flip grid vertically (top-bottom)',
                    'primitive_type': 'flip_vertical'
                }

            elif primitive_type == 'rotate':
                rotation = constraints.get('spatial', {}).get('has_rotation', 90)
                k = rotation // 90
                return {
                    'function': lambda grid: np.rot90(grid, k=k),
                    'description': f'Rotate grid {rotation} degrees',
                    'primitive_type': 'rotate'
                }

            elif primitive_type == 'color_mapping':
                mapping = constraints.get('color', {}).get('color_mapping')
                if not mapping:
                    return None

                def apply_color_mapping(grid):
                    result = grid.copy()
                    for old_color, new_color in mapping.items():
                        result[grid == old_color] = new_color
                    return result

                return {
                    'function': apply_color_mapping,
                    'description': f'Apply color mapping: {mapping}',
                    'primitive_type': 'color_mapping'
                }

            elif primitive_type == 'extract':
                # Extract largest non-background object
                def extract_largest(grid):
                    background = self._infer_background(grid)
                    objects = self._detect_objects(grid, background)

                    if not objects:
                        return grid

                    largest = max(objects, key=lambda o: o['size'])
                    r1, c1, r2, c2 = largest['bbox']
                    return grid[r1:r2+1, c1:c2+1].copy()

                return {
                    'function': extract_largest,
                    'description': 'Extract largest object',
                    'primitive_type': 'extract'
                }

            elif primitive_type == 'crop':
                # Crop to non-background bounding box
                def crop_to_content(grid):
                    background = self._infer_background(grid)
                    non_bg = np.argwhere(grid != background)

                    if len(non_bg) == 0:
                        return grid

                    r_min, c_min = non_bg.min(axis=0)
                    r_max, c_max = non_bg.max(axis=0)

                    return grid[r_min:r_max+1, c_min:c_max+1].copy()

                return {
                    'function': crop_to_content,
                    'description': 'Crop to non-background content',
                    'primitive_type': 'crop'
                }

            elif primitive_type == 'tile':
                # Tile the input pattern
                scale = constraints.get('shape', {}).get('scale_factor', 2)
                if scale and scale > 1:
                    k = int(scale)
                    return {
                        'function': lambda grid: np.tile(grid, (k, k)),
                        'description': f'Tile pattern {k}x{k}',
                        'primitive_type': 'tile'
                    }

            elif primitive_type == 'scale':
                # Scale by repeating pixels
                scale = constraints.get('shape', {}).get('scale_factor', 2)
                if scale and scale > 1:
                    k = int(scale)

                    def scale_up(grid):
                        h, w = grid.shape
                        result = np.zeros((h * k, w * k), dtype=grid.dtype)
                        for i in range(h):
                            for j in range(w):
                                result[i*k:(i+1)*k, j*k:(j+1)*k] = grid[i, j]
                        return result

                    return {
                        'function': scale_up,
                        'description': f'Scale up by {k}x',
                        'primitive_type': 'scale'
                    }

            elif primitive_type == 'recolor':
                # Simple recoloring based on constraints
                adds_colors = constraints.get('color', {}).get('adds_colors', set())
                removes_colors = constraints.get('color', {}).get('removes_colors', set())

                if adds_colors or removes_colors:
                    def recolor_transform(grid):
                        result = grid.copy()
                        # This is a placeholder - would need more sophisticated logic
                        return result

                    return {
                        'function': recolor_transform,
                        'description': 'Recolor transformation',
                        'primitive_type': 'recolor'
                    }

            # Add more primitive implementations as needed

        except Exception:
            return None

        return None

    def _evaluate_candidate(self,
                           candidate: Dict[str, Any],
                           train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate a candidate program on training examples."""
        try:
            program_func = candidate['function']
            scores = []

            for inp, expected in train_examples:
                try:
                    output = program_func(inp.copy())

                    if output.shape == expected.shape:
                        iou = np.sum(output == expected) / output.size
                    else:
                        iou = 0.0

                    scores.append(iou)

                except Exception:
                    scores.append(0.0)

            return np.mean(scores) if scores else 0.0

        except Exception:
            return 0.0

    def _detect_objects(self, grid: np.ndarray, background: int = 0) -> List[Dict[str, Any]]:
        """Detect all connected component objects."""
        objects = []
        colors = [c for c in np.unique(grid) if c != background]

        for color in colors:
            color_mask = (grid == color).astype(int)
            labeled, num_features = label(color_mask)

            for obj_id in range(1, num_features + 1):
                obj_mask = labeled == obj_id
                positions = np.argwhere(obj_mask)

                if len(positions) == 0:
                    continue

                r_min, c_min = positions.min(axis=0)
                r_max, c_max = positions.max(axis=0)

                objects.append({
                    'color': color,
                    'size': len(positions),
                    'bbox': (r_min, c_min, r_max, c_max),
                    'positions': positions
                })

        return objects

    def _infer_background(self, grid: np.ndarray) -> int:
        """Infer the background color (most common)."""
        from collections import Counter
        counts = Counter(grid.flatten())
        return counts.most_common(1)[0][0]

    def _build_constraint_mapping(self) -> Dict[str, List[str]]:
        """Build mapping from constraint patterns to primitive types."""
        # This is primarily for documentation/reference
        return {
            'preserves_shape': ['recolor', 'fill', 'pattern', 'mask'],
            'is_extraction': ['extract', 'crop', 'select'],
            'is_expansion': ['expand', 'pad', 'tile'],
            'has_reflection': ['flip_horizontal', 'flip_vertical'],
            'has_rotation': ['rotate'],
            'color_mapping': ['color_mapping'],
            'increases_objects': ['copy_objects', 'tile_objects'],
            'decreases_objects': ['filter_objects', 'merge_objects']
        }
