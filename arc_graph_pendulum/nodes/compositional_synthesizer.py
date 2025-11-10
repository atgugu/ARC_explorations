"""
Compositional Program Synthesizer - Priority 2 Implementation
Generates programs for multi-step transformations.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional

from nodes.shape_transformation_synthesizer import ShapeTransformationSynthesizer
from nodes.enhanced_targeted_synthesizer import EnhancedTargetedSynthesizer
from core.dsl import DSLRegistry


class CompositionalSynthesizer:
    """
    Generates executable programs for compositional (multi-step) transformations.

    Handles:
    1. 2-step compositions
    2. 3-step compositions
    3. Chaining operations correctly
    """

    def __init__(self):
        self.shape_synthesizer = ShapeTransformationSynthesizer()
        self.enhanced_synthesizer = EnhancedTargetedSynthesizer()
        self.dsl = DSLRegistry()

    def synthesize(self, composition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate programs based on compositional analysis.

        Returns list of programs sorted by confidence.
        """
        composition_type = composition.get('transformation_type', '')
        programs = []

        if composition_type == 'compositional_2step':
            programs.extend(self._synthesize_2step(composition))
        elif composition_type == 'compositional_3step':
            programs.extend(self._synthesize_3step(composition))

        # Sort by confidence
        programs.sort(key=lambda p: p.get('confidence', 0.0), reverse=True)

        return programs

    def _synthesize_2step(self, composition: Dict[str, Any]) -> List[Dict]:
        """Generate programs for 2-step compositions."""
        programs = []

        step1 = composition.get('step1', {})
        step2 = composition.get('step2', {})
        confidence = composition.get('confidence', 0.8)

        # Get step 1 operation
        step1_op_name = step1.get('operation', '')
        step1_func = self._get_operation_func(step1_op_name)

        if step1_func is None:
            return programs

        # Get step 2 programs
        step2_programs = self._synthesize_step(step2)

        # Combine into compositional programs
        for step2_prog in step2_programs:
            step2_func = step2_prog['function']

            # Create composed function
            def make_composed_func(f1, f2):
                def composed(grid):
                    intermediate = f1(grid)
                    result = f2(intermediate)
                    return result
                return composed

            composed_func = make_composed_func(step1_func, step2_func)

            programs.append({
                'type': f'2step_{step1_op_name}_{step2_prog["type"]}',
                'function': composed_func,
                'confidence': confidence * step2_prog.get('confidence', 0.9),
                'description': f'{step1_op_name} → {step2_prog["description"]}',
                'composition': '2-step'
            })

        return programs

    def _synthesize_3step(self, composition: Dict[str, Any]) -> List[Dict]:
        """Generate programs for 3-step compositions."""
        programs = []

        step1 = composition.get('step1', {})
        step2 = composition.get('step2', {})
        step3 = composition.get('step3', {})
        confidence = composition.get('confidence', 0.7)

        # Get step 1 operation
        step1_op_name = step1.get('operation', '')
        step1_func = self._get_operation_func(step1_op_name)

        if step1_func is None:
            return programs

        # Get step 2 operation
        step2_op_name = step2.get('operation', '')
        step2_func = self._get_operation_func(step2_op_name)

        if step2_func is None:
            return programs

        # Get step 3 programs
        step3_programs = self._synthesize_step(step3)

        # Combine into compositional programs
        for step3_prog in step3_programs:
            step3_func = step3_prog['function']

            # Create composed function
            def make_composed_func(f1, f2, f3):
                def composed(grid):
                    intermediate1 = f1(grid)
                    intermediate2 = f2(intermediate1)
                    result = f3(intermediate2)
                    return result
                return composed

            composed_func = make_composed_func(step1_func, step2_func, step3_func)

            programs.append({
                'type': f'3step_{step1_op_name}_{step2_op_name}_{step3_prog["type"]}',
                'function': composed_func,
                'confidence': confidence * step3_prog.get('confidence', 0.9),
                'description': f'{step1_op_name} → {step2_op_name} → {step3_prog["description"]}',
                'composition': '3-step'
            })

        return programs

    def _synthesize_step(self, step_analysis: Dict[str, Any]) -> List[Dict]:
        """Synthesize programs for a single step based on its analysis."""
        step_type = step_analysis.get('transformation_type', '')

        # Try shape synthesizer for shape-changing transformations
        if step_type in ['object_extraction', 'cropping', 'region_selection', 'color_counting']:
            programs = self.shape_synthesizer.synthesize(step_analysis)
            if programs:
                return programs

        # Try enhanced synthesizer for same-shape transformations
        if step_type in ['color_remap', 'geometric', 'pattern_based_tiling', 'pattern_extraction']:
            # Convert to rule format expected by enhanced synthesizer
            rule = {
                'rule_type': step_type,
                'transformation_type': step_type,
                'parameters': step_analysis.get('parameters', {}),
                'confidence': step_analysis.get('confidence', 0.8),
                'description': step_analysis.get('description', '')
            }

            try:
                from core.node import NodeOutput
                synth_input = {'rule': rule}
                from nodes.enhanced_targeted_synthesizer import enhanced_targeted_synthesizer_func
                output = enhanced_targeted_synthesizer_func(synth_input)
                if output.result:
                    return output.result
            except Exception:
                pass

        # Fallback to identity
        return [{
            'type': 'identity',
            'function': lambda g: g,
            'confidence': 0.1,
            'description': 'Identity (fallback)'
        }]

    def _get_operation_func(self, op_name: str) -> Optional[Callable]:
        """Get the function for a named operation."""
        # Geometric transformations
        if op_name == 'rotate_90':
            return lambda g: np.rot90(g, k=1)
        elif op_name == 'rotate_180':
            return lambda g: np.rot90(g, k=2)
        elif op_name == 'rotate_270':
            return lambda g: np.rot90(g, k=3)
        elif op_name == 'flip_horizontal':
            return lambda g: np.fliplr(g)
        elif op_name == 'flip_vertical':
            return lambda g: np.flipud(g)
        elif op_name == 'transpose':
            return lambda g: np.transpose(g)

        # Color operations
        elif op_name == 'invert_colors':
            return self._make_invert_colors_func()

        # Crop operations
        elif op_name == 'crop_to_content':
            return self._make_crop_to_content_func()

        # Object extraction
        elif op_name == 'extract_largest_object':
            return self._make_extract_largest_func()
        elif op_name == 'extract_smallest_object':
            return self._make_extract_smallest_func()

        # DSL operations
        elif hasattr(self.dsl, 'operations') and op_name in self.dsl.operations:
            return self.dsl.get(op_name)

        return None

    # Factory methods

    def _make_invert_colors_func(self) -> Callable:
        """Create color inversion function."""
        def invert_colors(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            non_zero = result != 0
            if non_zero.any():
                unique = np.unique(result[non_zero])
                if len(unique) == 2:
                    result[result == unique[0]] = -1
                    result[result == unique[1]] = unique[0]
                    result[result == -1] = unique[1]
            return result
        return invert_colors

    def _make_crop_to_content_func(self) -> Callable:
        """Create crop to content function."""
        def crop_to_content(grid: np.ndarray) -> np.ndarray:
            non_zero_rows, non_zero_cols = np.where(grid != 0)
            if len(non_zero_rows) == 0:
                return grid

            min_row, max_row = non_zero_rows.min(), non_zero_rows.max()
            min_col, max_col = non_zero_cols.min(), non_zero_cols.max()

            return grid[min_row:max_row+1, min_col:max_col+1]
        return crop_to_content

    def _make_extract_largest_func(self) -> Callable:
        """Create extract largest object function."""
        def extract_largest(grid: np.ndarray) -> np.ndarray:
            from scipy import ndimage

            objects = []
            colors = np.unique(grid)
            colors = colors[colors != 0]

            for color in colors:
                mask = (grid == color).astype(int)
                labeled, num_objects = ndimage.label(mask)

                for obj_id in range(1, num_objects + 1):
                    obj_mask = (labeled == obj_id)
                    coords = np.argwhere(obj_mask)

                    if len(coords) > 0:
                        min_row, min_col = coords.min(axis=0)
                        max_row, max_col = coords.max(axis=0)

                        objects.append({
                            'size': len(coords),
                            'mask': obj_mask,
                            'bbox': (min_row, max_row, min_col, max_col)
                        })

            if not objects:
                return grid

            largest = max(objects, key=lambda o: o['size'])
            min_row, max_row, min_col, max_col = largest['bbox']
            bbox_region = grid[min_row:max_row+1, min_col:max_col+1].copy()
            obj_mask_region = largest['mask'][min_row:max_row+1, min_col:max_col+1]
            bbox_region[~obj_mask_region] = 0
            return bbox_region

        return extract_largest

    def _make_extract_smallest_func(self) -> Callable:
        """Create extract smallest object function."""
        def extract_smallest(grid: np.ndarray) -> np.ndarray:
            from scipy import ndimage

            objects = []
            colors = np.unique(grid)
            colors = colors[colors != 0]

            for color in colors:
                mask = (grid == color).astype(int)
                labeled, num_objects = ndimage.label(mask)

                for obj_id in range(1, num_objects + 1):
                    obj_mask = (labeled == obj_id)
                    coords = np.argwhere(obj_mask)

                    if len(coords) > 0:
                        min_row, min_col = coords.min(axis=0)
                        max_row, max_col = coords.max(axis=0)

                        objects.append({
                            'size': len(coords),
                            'mask': obj_mask,
                            'bbox': (min_row, max_row, min_col, max_col)
                        })

            if not objects:
                return grid

            smallest = min(objects, key=lambda o: o['size'])
            min_row, max_row, min_col, max_col = smallest['bbox']
            bbox_region = grid[min_row:max_row+1, min_col:max_col+1].copy()
            obj_mask_region = smallest['mask'][min_row:max_row+1, min_col:max_col+1]
            bbox_region[~obj_mask_region] = 0
            return bbox_region

        return extract_smallest
