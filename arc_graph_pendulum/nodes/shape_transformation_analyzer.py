"""
Shape Transformation Analyzer - Priority 1 Implementation
Handles object extraction, cropping, and region selection transformations.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import ndimage
from collections import Counter


class ShapeTransformationAnalyzer:
    """
    Specialized analyzer for shape-changing transformations.

    Handles:
    1. Object extraction (extract specific objects from input)
    2. Cropping/bounding box (crop to specific region)
    3. Region selection (extract rows/columns/patterns)
    """

    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze if transformation is a shape-changing operation.

        Returns best matching transformation with confidence.
        """
        results = []

        # Try all shape transformation types
        results.append(self._check_object_extraction(input_grid, output_grid))
        results.append(self._check_cropping(input_grid, output_grid))
        results.append(self._check_region_selection(input_grid, output_grid))
        results.append(self._check_color_counting(input_grid, output_grid))
        results.append(self._check_object_properties(input_grid, output_grid))

        # Filter and sort by confidence
        results = [r for r in results if r is not None and r['confidence'] > 0.5]

        if results:
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results[0]

        return None

    def _check_object_extraction(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect if output is a specific object extracted from input.

        Strategies:
        - Extract by color (e.g., "extract all red")
        - Extract by size (e.g., "extract largest")
        - Extract by position (e.g., "extract top-left object")
        - Extract by uniqueness (e.g., "extract unique colored object")
        """
        # Detect objects in input
        objects = self._detect_all_objects(input_grid)

        if not objects:
            return None

        # Try to match output with an extracted object
        best_match = None
        best_confidence = 0.0
        best_rule = None

        # Strategy 1: Extract by color
        for obj in objects:
            extracted = self._extract_object_tight(obj, input_grid)
            confidence = self._grid_similarity(extracted, output_grid)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = obj
                best_rule = {
                    'type': 'extract_by_color',
                    'color': obj['color'],
                    'description': f"Extract object of color {obj['color']}"
                }

        # Strategy 2: Extract largest object
        if objects:
            largest = max(objects, key=lambda o: o['size'])
            extracted = self._extract_object_tight(largest, input_grid)
            confidence = self._grid_similarity(extracted, output_grid)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = largest
                best_rule = {
                    'type': 'extract_largest',
                    'color': largest['color'],
                    'size': largest['size'],
                    'description': f"Extract largest object (color {largest['color']}, size {largest['size']})"
                }

        # Strategy 3: Extract by color uniqueness
        color_counts = Counter([obj['color'] for obj in objects])
        unique_colors = [color for color, count in color_counts.items() if count == 1]

        for obj in objects:
            if obj['color'] in unique_colors:
                extracted = self._extract_object_tight(obj, input_grid)
                confidence = self._grid_similarity(extracted, output_grid)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = obj
                    best_rule = {
                        'type': 'extract_unique_color',
                        'color': obj['color'],
                        'description': f"Extract unique colored object (color {obj['color']})"
                    }

        # Strategy 4: Extract smallest object
        if objects:
            smallest = min(objects, key=lambda o: o['size'])
            extracted = self._extract_object_tight(smallest, input_grid)
            confidence = self._grid_similarity(extracted, output_grid)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = smallest
                best_rule = {
                    'type': 'extract_smallest',
                    'color': smallest['color'],
                    'size': smallest['size'],
                    'description': f"Extract smallest object (color {smallest['color']}, size {smallest['size']})"
                }

        if best_confidence >= 0.7:
            return {
                'transformation_type': 'object_extraction',
                'parameters': {
                    'rule': best_rule,
                    'object': best_match,
                    'tight_crop': True
                },
                'confidence': best_confidence,
                'description': best_rule['description']
            }

        return None

    def _check_cropping(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect if output is a crop/bounding box of input.

        Strategies:
        - Crop to bounding box of non-zero pixels
        - Crop to bounding box of specific color
        - Crop to specific region (center, corner, etc.)
        """
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape

        # Check if output could be a contiguous crop
        if h_out > h_in or w_out > w_in:
            return None

        # Strategy 1: Find all possible matching crops in input
        best_match = None
        best_confidence = 0.0
        best_position = None

        for i in range(h_in - h_out + 1):
            for j in range(w_in - w_out + 1):
                crop = input_grid[i:i+h_out, j:j+w_out]
                confidence = self._grid_similarity(crop, output_grid)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_position = (i, j)
                    best_match = crop

        if best_confidence >= 0.9:
            # Determine cropping rule
            i, j = best_position
            rule_type = 'crop_region'
            description = f"Crop region at ({i}, {j}) with size ({h_out}, {w_out})"

            # Check if it's a special position
            if i == 0 and j == 0:
                rule_type = 'crop_top_left'
                description = f"Crop top-left corner ({h_out}x{w_out})"
            elif i == 0 and j + w_out == w_in:
                rule_type = 'crop_top_right'
                description = f"Crop top-right corner ({h_out}x{w_out})"
            elif i + h_out == h_in and j == 0:
                rule_type = 'crop_bottom_left'
                description = f"Crop bottom-left corner ({h_out}x{w_out})"
            elif i + h_out == h_in and j + w_out == w_in:
                rule_type = 'crop_bottom_right'
                description = f"Crop bottom-right corner ({h_out}x{w_out})"

            return {
                'transformation_type': 'cropping',
                'parameters': {
                    'rule_type': rule_type,
                    'position': best_position,
                    'size': (h_out, w_out)
                },
                'confidence': best_confidence,
                'description': description
            }

        # Strategy 2: Crop to bounding box of non-zero
        non_zero_rows, non_zero_cols = np.where(input_grid != 0)

        if len(non_zero_rows) > 0:
            min_row, max_row = non_zero_rows.min(), non_zero_rows.max()
            min_col, max_col = non_zero_cols.min(), non_zero_cols.max()

            bbox_crop = input_grid[min_row:max_row+1, min_col:max_col+1]
            confidence = self._grid_similarity(bbox_crop, output_grid)

            if confidence >= 0.9:
                return {
                    'transformation_type': 'cropping',
                    'parameters': {
                        'rule_type': 'crop_to_content_bbox',
                        'bbox': (min_row, max_row, min_col, max_col)
                    },
                    'confidence': confidence,
                    'description': f"Crop to bounding box of non-zero content"
                }

        # Strategy 3: Crop to bounding box of specific color
        colors = np.unique(input_grid)
        colors = colors[colors != 0]

        for color in colors:
            color_rows, color_cols = np.where(input_grid == color)

            if len(color_rows) > 0:
                min_row, max_row = color_rows.min(), color_rows.max()
                min_col, max_col = color_cols.min(), color_cols.max()

                bbox_crop = input_grid[min_row:max_row+1, min_col:max_col+1]
                confidence = self._grid_similarity(bbox_crop, output_grid)

                if confidence >= 0.9:
                    return {
                        'transformation_type': 'cropping',
                        'parameters': {
                            'rule_type': 'crop_to_color_bbox',
                            'color': int(color),
                            'bbox': (min_row, max_row, min_col, max_col)
                        },
                        'confidence': confidence,
                        'description': f"Crop to bounding box of color {color}"
                    }

        return None

    def _check_region_selection(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect if output selects specific regions/patterns from input.

        Examples:
        - Extract specific rows
        - Extract specific columns
        - Extract diagonal
        - Extract pattern-matching cells
        """
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape

        # Strategy 1: Row selection
        if w_in == w_out and h_out < h_in:
            selected_rows = []

            for out_row_idx in range(h_out):
                for in_row_idx in range(h_in):
                    if np.array_equal(input_grid[in_row_idx], output_grid[out_row_idx]):
                        selected_rows.append(in_row_idx)
                        break

            if len(selected_rows) == h_out:
                return {
                    'transformation_type': 'region_selection',
                    'parameters': {
                        'rule_type': 'select_rows',
                        'rows': selected_rows
                    },
                    'confidence': 1.0,
                    'description': f"Select rows {selected_rows}"
                }

        # Strategy 2: Column selection
        if h_in == h_out and w_out < w_in:
            selected_cols = []

            for out_col_idx in range(w_out):
                for in_col_idx in range(w_in):
                    if np.array_equal(input_grid[:, in_col_idx], output_grid[:, out_col_idx]):
                        selected_cols.append(in_col_idx)
                        break

            if len(selected_cols) == w_out:
                return {
                    'transformation_type': 'region_selection',
                    'parameters': {
                        'rule_type': 'select_columns',
                        'columns': selected_cols
                    },
                    'confidence': 1.0,
                    'description': f"Select columns {selected_cols}"
                }

        # Strategy 3: Diagonal extraction
        if h_out == 1 or w_out == 1:
            # Main diagonal
            if min(h_in, w_in) == max(h_out, w_out):
                diag = np.diag(input_grid)
                if h_out == 1:
                    output_flat = output_grid.flatten()
                else:
                    output_flat = output_grid.flatten()

                if np.array_equal(diag, output_flat):
                    return {
                        'transformation_type': 'region_selection',
                        'parameters': {
                            'rule_type': 'extract_diagonal',
                            'diagonal_type': 'main'
                        },
                        'confidence': 1.0,
                        'description': "Extract main diagonal"
                    }

        return None

    def _check_color_counting(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect if output is a color count or selection based on color properties.

        Examples:
        - Output is the most/least common color
        - Output is the unique color
        - Output is missing color
        """
        h_out, w_out = output_grid.shape

        # Check if output is very small (1x1 or similar)
        if h_out * w_out > 5:
            return None

        # Get unique value in output (if nearly uniform)
        out_values = output_grid.flatten()
        out_unique, out_counts = np.unique(out_values, return_counts=True)

        # Find dominant color in output
        if len(out_unique) > 0:
            dominant_out_color = out_unique[np.argmax(out_counts)]

            # Get input color statistics
            in_unique, in_counts = np.unique(input_grid, return_counts=True)
            in_color_counts = dict(zip(in_unique, in_counts))

            # Remove background (0)
            in_color_counts_no_bg = {k: v for k, v in in_color_counts.items() if k != 0}

            if not in_color_counts_no_bg:
                return None

            # Strategy 1: Most common non-zero color
            most_common_color = max(in_color_counts_no_bg.items(), key=lambda x: x[1])[0]
            if dominant_out_color == most_common_color:
                return {
                    'transformation_type': 'color_counting',
                    'parameters': {
                        'rule_type': 'most_common_color',
                        'color': int(dominant_out_color)
                    },
                    'confidence': 0.9,
                    'description': f"Output most common color: {dominant_out_color}"
                }

            # Strategy 2: Least common non-zero color
            least_common_color = min(in_color_counts_no_bg.items(), key=lambda x: x[1])[0]
            if dominant_out_color == least_common_color:
                return {
                    'transformation_type': 'color_counting',
                    'parameters': {
                        'rule_type': 'least_common_color',
                        'color': int(dominant_out_color)
                    },
                    'confidence': 0.9,
                    'description': f"Output least common color: {dominant_out_color}"
                }

            # Strategy 3: Unique color (appears only once as object)
            color_object_counts = {}
            for color in in_unique:
                if color != 0:
                    objects = self._detect_objects_by_color(input_grid, color)
                    color_object_counts[color] = len(objects)

            unique_color_objects = [c for c, count in color_object_counts.items() if count == 1]
            if len(unique_color_objects) == 1 and dominant_out_color in unique_color_objects:
                return {
                    'transformation_type': 'color_counting',
                    'parameters': {
                        'rule_type': 'unique_color_object',
                        'color': int(dominant_out_color)
                    },
                    'confidence': 0.85,
                    'description': f"Output unique color object: {dominant_out_color}"
                }

        return None

    def _check_object_properties(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect if output represents object properties.

        Examples:
        - Count of objects
        - Size of largest object
        - Color of smallest object
        """
        h_out, w_out = output_grid.shape

        # Only for very small outputs
        if h_out * w_out > 10:
            return None

        objects = self._detect_all_objects(input_grid)

        if not objects:
            return None

        # Check if output size matches object count
        if h_out * w_out == len(objects):
            # Could be counting objects
            return {
                'transformation_type': 'object_properties',
                'parameters': {
                    'rule_type': 'count_objects',
                    'count': len(objects)
                },
                'confidence': 0.7,
                'description': f"Count objects: {len(objects)}"
            }

        return None

    # Helper methods

    def _detect_all_objects(self, grid: np.ndarray) -> List[Dict]:
        """Detect all connected component objects in grid."""
        objects = []

        # For each non-zero color
        colors = np.unique(grid)
        colors = colors[colors != 0]

        for color in colors:
            objects.extend(self._detect_objects_by_color(grid, color))

        return objects

    def _detect_objects_by_color(self, grid: np.ndarray, color: int) -> List[Dict]:
        """Detect objects of a specific color."""
        objects = []

        # Find connected components
        mask = (grid == color).astype(int)
        labeled, num_objects = ndimage.label(mask)

        for obj_id in range(1, num_objects + 1):
            obj_mask = (labeled == obj_id)
            coords = np.argwhere(obj_mask)

            if len(coords) > 0:
                min_row, min_col = coords.min(axis=0)
                max_row, max_col = coords.max(axis=0)
                center = coords.mean(axis=0)

                objects.append({
                    'color': int(color),
                    'coords': coords,
                    'center': center,
                    'size': len(coords),
                    'mask': obj_mask,
                    'bbox': (min_row, max_row, min_col, max_col)
                })

        return objects

    def _extract_object_tight(self, obj: Dict, grid: np.ndarray) -> np.ndarray:
        """Extract object with tight bounding box."""
        min_row, max_row, min_col, max_col = obj['bbox']

        # Extract the bounding box region
        bbox_region = grid[min_row:max_row+1, min_col:max_col+1].copy()

        # Keep only the object pixels, zero out others
        obj_mask_region = obj['mask'][min_row:max_row+1, min_col:max_col+1]
        bbox_region[~obj_mask_region] = 0

        return bbox_region

    def _grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Compute similarity between two grids."""
        if grid1.shape != grid2.shape:
            return 0.0

        if grid1.size == 0:
            return 0.0

        matches = np.sum(grid1 == grid2)
        total = grid1.size

        return matches / total
