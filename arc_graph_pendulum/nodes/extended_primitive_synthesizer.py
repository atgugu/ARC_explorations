"""
Extended Primitive Synthesizer - Generates programs using extended primitives.

Converts detected primitive patterns into executable programs.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from nodes.extended_primitives import ExtendedPrimitives


class ExtendedPrimitiveSynthesizer:
    """Synthesizes programs from extended primitive detections."""

    def __init__(self):
        """Initialize synthesizer."""
        self.primitives = ExtendedPrimitives()

    def synthesize(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate programs from primitive detections.

        Args:
            detections: List of detected primitives with parameters

        Returns:
            List of program dictionaries with functions and metadata
        """
        programs = []

        for detection in detections:
            primitive_name = detection['primitive']
            params = detection.get('params', {})
            confidence = detection.get('confidence', 0.8)
            description = detection.get('description', primitive_name)

            # Generate program function
            program_func = self._create_program(primitive_name, params)

            if program_func:
                programs.append({
                    'function': program_func,
                    'type': primitive_name,
                    'confidence': confidence,
                    'description': description,
                    'category': self._get_category(primitive_name)
                })

        return programs

    def _create_program(self, primitive_name: str, params: Dict[str, Any]) -> Callable:
        """
        Create executable program function for a primitive.

        Args:
            primitive_name: Name of the primitive
            params: Parameters for the primitive

        Returns:
            Executable function
        """
        # Map primitive names to functions
        primitive_map = {
            # Spatial operations
            'extract_leftmost_object': lambda grid: self.primitives.extract_leftmost_object(grid),
            'extract_rightmost_object': lambda grid: self.primitives.extract_rightmost_object(grid),
            'extract_topmost_object': lambda grid: self.primitives.extract_topmost_object(grid),
            'extract_bottommost_object': lambda grid: self.primitives.extract_bottommost_object(grid),
            'align_objects_to_grid': lambda grid: self.primitives.align_objects_to_grid(
                grid, spacing=params.get('spacing', 2)
            ),

            # Object operations
            'copy_object_horizontal': lambda grid: self.primitives.copy_object_horizontal(
                grid, n=params.get('n', 3)
            ),
            'copy_object_vertical': lambda grid: self.primitives.copy_object_vertical(
                grid, n=params.get('n', 3)
            ),
            'connect_objects_with_line': lambda grid: self.primitives.connect_objects_with_line(
                grid, color=params.get('color', 1)
            ),
            'object_intersection': lambda grid: self.primitives.object_intersection(grid),
            'object_union': lambda grid: self.primitives.object_union(grid),

            # Color operations
            'recolor_by_row': lambda grid: self.primitives.recolor_by_row(
                grid, start_color=params.get('start_color', 1)
            ),
            'recolor_by_column': lambda grid: self.primitives.recolor_by_column(
                grid, start_color=params.get('start_color', 1)
            ),
            'recolor_checkerboard': lambda grid: self.primitives.recolor_checkerboard(
                grid,
                color1=params.get('color1', 1),
                color2=params.get('color2', 2)
            ),
            'swap_colors_by_size': lambda grid: self.primitives.swap_colors_by_size(grid),
            'color_propagation': lambda grid: self.primitives.color_propagation(grid),

            # Pattern operations
            'apply_horizontal_symmetry': lambda grid: self.primitives.apply_horizontal_symmetry(grid),
            'apply_vertical_symmetry': lambda grid: self.primitives.apply_vertical_symmetry(grid),
            'apply_rotational_symmetry': lambda grid: self.primitives.apply_rotational_symmetry(
                grid, order=params.get('order', 4)
            ),
            'complete_partial_pattern': lambda grid: self.primitives.complete_partial_pattern(grid),
            'generate_periodic_tiling': lambda grid: self.primitives.generate_periodic_tiling(
                grid, tile_size=params.get('tile_size', 3)
            ),
        }

        return primitive_map.get(primitive_name)

    def _get_category(self, primitive_name: str) -> str:
        """Get category for a primitive."""
        spatial = ['extract_leftmost_object', 'extract_rightmost_object',
                  'extract_topmost_object', 'extract_bottommost_object',
                  'align_objects_to_grid']

        object_ops = ['copy_object_horizontal', 'copy_object_vertical',
                     'connect_objects_with_line', 'object_intersection', 'object_union']

        color_ops = ['recolor_by_row', 'recolor_by_column', 'recolor_checkerboard',
                    'swap_colors_by_size', 'color_propagation']

        pattern_ops = ['apply_horizontal_symmetry', 'apply_vertical_symmetry',
                      'apply_rotational_symmetry', 'complete_partial_pattern',
                      'generate_periodic_tiling']

        if primitive_name in spatial:
            return 'spatial'
        elif primitive_name in object_ops:
            return 'object'
        elif primitive_name in color_ops:
            return 'color'
        elif primitive_name in pattern_ops:
            return 'pattern'
        else:
            return 'unknown'
