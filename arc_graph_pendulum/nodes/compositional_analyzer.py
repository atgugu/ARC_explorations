"""
Compositional Transformation Analyzer - Priority 2 Implementation
Detects multi-step transformations: input → intermediate → output
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import ndimage

from nodes.differential_analyzer import DifferentialAnalyzer
from nodes.enhanced_differential_analyzer import EnhancedDifferentialAnalyzer
from nodes.shape_transformation_analyzer import ShapeTransformationAnalyzer
from utils.grid_utils import compute_iou
from core.dsl import DSLRegistry


class CompositionalTransformationAnalyzer:
    """
    Analyzes multi-step transformations by searching for intermediate states.

    Handles:
    1. 2-step compositions (extract → transform, transform → crop, etc.)
    2. 3-step compositions (extract → rotate → place, etc.)
    3. Intermediate state search (find plausible intermediate grids)
    """

    def __init__(self):
        self.base_analyzer = DifferentialAnalyzer()
        self.enhanced_analyzer = EnhancedDifferentialAnalyzer()
        self.shape_analyzer = ShapeTransformationAnalyzer()
        self.dsl = DSLRegistry()

    def analyze(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze if transformation is compositional (multi-step).

        Returns best matching composition with confidence.
        """
        # Try 2-step compositions first
        result_2step = self._check_2step_composition(input_grid, output_grid)
        if result_2step and result_2step['confidence'] >= 0.8:
            return result_2step

        # Try 3-step compositions if 2-step fails
        result_3step = self._check_3step_composition(input_grid, output_grid)
        if result_3step and result_3step['confidence'] >= 0.7:
            return result_3step

        return None

    def _check_2step_composition(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Check if transformation can be decomposed into 2 steps.

        Strategy:
        1. Generate plausible intermediate states by applying basic operations
        2. For each intermediate, check if intermediate → output is simple
        3. Score by quality of intermediate match
        """
        best_composition = None
        best_score = 0.0

        # Get common transformation operations
        operations = self._get_basic_operations()

        # Try each operation as step 1
        for op_name, op_func in operations:
            try:
                # Apply operation to get intermediate
                intermediate = op_func(input_grid)

                # Check if intermediate → output is a known transformation
                step2_analysis = self._analyze_single_step(intermediate, output_grid)

                if step2_analysis and step2_analysis['confidence'] >= 0.7:
                    # Check if intermediate is actually different from input
                    if not np.array_equal(intermediate, input_grid):
                        # Score the composition
                        score = step2_analysis['confidence'] * 0.9  # Slight penalty for composition

                        if score > best_score:
                            best_score = score
                            best_composition = {
                                'transformation_type': 'compositional_2step',
                                'step1': {
                                    'operation': op_name,
                                    'description': f'Apply {op_name}'
                                },
                                'step2': step2_analysis,
                                'intermediate_shape': intermediate.shape,
                                'confidence': score,
                                'description': f'2-step: {op_name} → {step2_analysis["transformation_type"]}'
                            }

            except Exception:
                continue

        return best_composition

    def _check_3step_composition(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Check if transformation requires 3 steps.

        Strategy:
        1. Try 2 operations in sequence
        2. Check if final state → output is simple
        """
        best_composition = None
        best_score = 0.0

        operations = self._get_basic_operations()

        # Try pairs of operations
        for i, (op1_name, op1_func) in enumerate(operations[:10]):  # Limit to first 10 for speed
            try:
                intermediate1 = op1_func(input_grid)

                if np.array_equal(intermediate1, input_grid):
                    continue

                for op2_name, op2_func in operations[:10]:
                    try:
                        intermediate2 = op2_func(intermediate1)

                        if np.array_equal(intermediate2, intermediate1):
                            continue

                        # Check if intermediate2 → output is simple
                        step3_analysis = self._analyze_single_step(intermediate2, output_grid)

                        if step3_analysis and step3_analysis['confidence'] >= 0.7:
                            score = step3_analysis['confidence'] * 0.8  # Larger penalty for 3-step

                            if score > best_score:
                                best_score = score
                                best_composition = {
                                    'transformation_type': 'compositional_3step',
                                    'step1': {
                                        'operation': op1_name,
                                        'description': f'Apply {op1_name}'
                                    },
                                    'step2': {
                                        'operation': op2_name,
                                        'description': f'Apply {op2_name}'
                                    },
                                    'step3': step3_analysis,
                                    'confidence': score,
                                    'description': f'3-step: {op1_name} → {op2_name} → {step3_analysis["transformation_type"]}'
                                }

                    except Exception:
                        continue

            except Exception:
                continue

        return best_composition

    def _analyze_single_step(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """Analyze a single transformation step using all available analyzers."""
        # Try shape analyzer first (for shape-changing)
        if input_grid.shape != output_grid.shape:
            shape_result = self.shape_analyzer.analyze(input_grid, output_grid)
            if shape_result and shape_result['confidence'] >= 0.7:
                return shape_result

        # Try enhanced analyzer (for same-shape complex patterns)
        enhanced_result = self.enhanced_analyzer.analyze(input_grid, output_grid)
        if enhanced_result and enhanced_result['confidence'] >= 0.7:
            return enhanced_result

        # Try base analyzer (for simple transformations)
        base_result = self.base_analyzer.analyze(input_grid, output_grid)
        if base_result and base_result['confidence'] >= 0.7:
            return base_result

        return None

    def _get_basic_operations(self) -> List[Tuple[str, callable]]:
        """
        Get list of basic operations to try for intermediate states.

        Returns list of (name, function) tuples.
        """
        operations = []

        # Geometric transformations
        operations.append(('rotate_90', lambda g: np.rot90(g, k=1)))
        operations.append(('rotate_180', lambda g: np.rot90(g, k=2)))
        operations.append(('rotate_270', lambda g: np.rot90(g, k=3)))
        operations.append(('flip_horizontal', lambda g: np.fliplr(g)))
        operations.append(('flip_vertical', lambda g: np.flipud(g)))
        operations.append(('transpose', lambda g: np.transpose(g)))

        # Color operations
        operations.append(('invert_colors', lambda g: self._invert_colors(g)))

        # Crop to non-zero bounding box
        operations.append(('crop_to_content', lambda g: self._crop_to_content(g)))

        # Extract objects
        operations.append(('extract_largest_object', lambda g: self._extract_largest_object(g)))
        operations.append(('extract_smallest_object', lambda g: self._extract_smallest_object(g)))

        # DSL operations
        if hasattr(self.dsl, 'operations'):
            for op_name in ['scale_2x', 'tile_2x2', 'keep_top_half', 'keep_bottom_half']:
                if op_name in self.dsl.operations:
                    operations.append((op_name, self.dsl.get(op_name)))

        return operations

    def _invert_colors(self, grid: np.ndarray) -> np.ndarray:
        """Invert non-zero colors (0→0, others swap)."""
        result = grid.copy()
        non_zero = result != 0
        if non_zero.any():
            unique = np.unique(result[non_zero])
            if len(unique) == 2:
                # Simple swap
                result[result == unique[0]] = -1
                result[result == unique[1]] = unique[0]
                result[result == -1] = unique[1]
        return result

    def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:
        """Crop to bounding box of non-zero content."""
        non_zero_rows, non_zero_cols = np.where(grid != 0)
        if len(non_zero_rows) == 0:
            return grid

        min_row, max_row = non_zero_rows.min(), non_zero_rows.max()
        min_col, max_col = non_zero_cols.min(), non_zero_cols.max()

        return grid[min_row:max_row+1, min_col:max_col+1]

    def _extract_largest_object(self, grid: np.ndarray) -> np.ndarray:
        """Extract largest connected component."""
        objects = self._detect_all_objects(grid)
        if not objects:
            return grid

        largest = max(objects, key=lambda o: o['size'])
        return self._extract_object_tight(largest, grid)

    def _extract_smallest_object(self, grid: np.ndarray) -> np.ndarray:
        """Extract smallest connected component."""
        objects = self._detect_all_objects(grid)
        if not objects:
            return grid

        smallest = min(objects, key=lambda o: o['size'])
        return self._extract_object_tight(smallest, grid)

    def _detect_all_objects(self, grid: np.ndarray) -> List[Dict]:
        """Detect all connected component objects."""
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
                        'color': int(color),
                        'coords': coords,
                        'size': len(coords),
                        'mask': obj_mask,
                        'bbox': (min_row, max_row, min_col, max_col)
                    })

        return objects

    def _extract_object_tight(self, obj: Dict, grid: np.ndarray) -> np.ndarray:
        """Extract object with tight bounding box."""
        min_row, max_row, min_col, max_col = obj['bbox']
        bbox_region = grid[min_row:max_row+1, min_col:max_col+1].copy()
        obj_mask_region = obj['mask'][min_row:max_row+1, min_col:max_col+1]
        bbox_region[~obj_mask_region] = 0
        return bbox_region
