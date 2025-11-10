"""
Conditional Rule Synthesizer - Priority 3+ Implementation
Generates adaptive programs that examine input and choose parameters.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
from scipy import ndimage

from nodes.feature_extractor import FeatureExtractor


class ConditionalSynthesizer:
    """
    Generates conditional/adaptive programs based on meta-patterns.

    These programs inspect the test input and dynamically choose
    parameters based on learned rules.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def synthesize(self, meta_pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate conditional programs from meta-pattern.

        Returns list of adaptive programs sorted by confidence.
        """
        meta_type = meta_pattern.get('meta_pattern_type', '')
        programs = []

        if meta_type == 'conditional_row_selection':
            programs.extend(self._synthesize_conditional_row_selection(meta_pattern))
        elif meta_type == 'conditional_column_selection':
            programs.extend(self._synthesize_conditional_column_selection(meta_pattern))
        elif meta_type == 'conditional_object_extraction':
            programs.extend(self._synthesize_conditional_object_extraction(meta_pattern))
        elif meta_type == 'conditional_cropping':
            programs.extend(self._synthesize_conditional_cropping(meta_pattern))

        # Sort by confidence
        programs.sort(key=lambda p: p.get('confidence', 0.0), reverse=True)

        return programs

    def _synthesize_conditional_row_selection(self, meta_pattern: Dict) -> List[Dict]:
        """Generate programs for conditional row selection."""
        programs = []

        rule = meta_pattern.get('rule', {})
        rule_type = rule.get('type', '')
        confidence = meta_pattern.get('confidence', 0.8)

        if rule_type == 'first_non_empty_row':
            programs.append({
                'type': 'adaptive_first_non_empty_row',
                'function': self._make_first_non_empty_row_func(),
                'confidence': confidence,
                'description': 'Adaptively select first non-empty row',
                'adaptive': True
            })

        elif rule_type == 'first_multicolor_row':
            programs.append({
                'type': 'adaptive_first_multicolor_row',
                'function': self._make_first_multicolor_row_func(),
                'confidence': confidence,
                'description': 'Adaptively select first multicolor row',
                'adaptive': True
            })

        elif rule_type == 'first_row_with_color':
            color = rule.get('color', -1)
            programs.append({
                'type': f'adaptive_first_row_with_color_{color}',
                'function': self._make_first_row_with_color_func(color),
                'confidence': confidence,
                'description': f'Adaptively select first row with color {color}',
                'adaptive': True
            })

        elif rule_type == 'rows_with_objects':
            programs.append({
                'type': 'adaptive_rows_with_objects',
                'function': self._make_rows_with_objects_func(),
                'confidence': confidence,
                'description': 'Adaptively select rows with object centers',
                'adaptive': True
            })

        return programs

    def _synthesize_conditional_column_selection(self, meta_pattern: Dict) -> List[Dict]:
        """Generate programs for conditional column selection."""
        programs = []

        rule = meta_pattern.get('rule', {})
        rule_type = rule.get('type', '')
        confidence = meta_pattern.get('confidence', 0.8)

        if rule_type == 'first_non_empty_column':
            programs.append({
                'type': 'adaptive_first_non_empty_column',
                'function': self._make_first_non_empty_column_func(),
                'confidence': confidence,
                'description': 'Adaptively select first non-empty column',
                'adaptive': True
            })

        return programs

    def _synthesize_conditional_object_extraction(self, meta_pattern: Dict) -> List[Dict]:
        """Generate programs for conditional object extraction."""
        programs = []

        rule = meta_pattern.get('rule', {})
        rule_type = rule.get('type', '')
        confidence = meta_pattern.get('confidence', 0.8)

        if rule_type == 'extract_least_common_object_color':
            programs.append({
                'type': 'adaptive_extract_least_common_object',
                'function': self._make_extract_least_common_object_func(),
                'confidence': confidence,
                'description': 'Adaptively extract object with least common color',
                'adaptive': True
            })

        elif rule_type == 'extract_unique_count_color':
            programs.append({
                'type': 'adaptive_extract_unique_count_object',
                'function': self._make_extract_unique_count_object_func(),
                'confidence': confidence,
                'description': 'Adaptively extract object with unique count',
                'adaptive': True
            })

        return programs

    def _synthesize_conditional_cropping(self, meta_pattern: Dict) -> List[Dict]:
        """Generate programs for conditional cropping."""
        programs = []

        rule = meta_pattern.get('rule', {})
        rule_type = rule.get('type', '')
        confidence = meta_pattern.get('confidence', 0.8)

        if rule_type == 'crop_to_most_common_color':
            programs.append({
                'type': 'adaptive_crop_to_most_common_color',
                'function': self._make_crop_to_most_common_color_func(),
                'confidence': confidence,
                'description': 'Adaptively crop to most common color bbox',
                'adaptive': True
            })

        return programs

    # Factory methods for adaptive functions

    def _make_first_non_empty_row_func(self) -> Callable:
        """Create function that selects first non-empty row."""
        def select_first_non_empty_row(grid: np.ndarray) -> np.ndarray:
            # Find first non-empty row
            for i in range(grid.shape[0]):
                if np.any(grid[i, :] != 0):
                    return grid[i:i+1, :]
            return grid[:1, :]  # Fallback to first row

        return select_first_non_empty_row

    def _make_first_multicolor_row_func(self) -> Callable:
        """Create function that selects first multicolor row."""
        def select_first_multicolor_row(grid: np.ndarray) -> np.ndarray:
            # Find first row with multiple colors
            for i in range(grid.shape[0]):
                unique_colors = len(np.unique(grid[i, :]))
                if unique_colors > 1:
                    return grid[i:i+1, :]
            return grid[:1, :]  # Fallback

        return select_first_multicolor_row

    def _make_first_row_with_color_func(self, target_color: int) -> Callable:
        """Create function that selects first row containing target color."""
        def select_first_row_with_color(grid: np.ndarray) -> np.ndarray:
            # Find first row with target color
            for i in range(grid.shape[0]):
                if target_color in grid[i, :]:
                    return grid[i:i+1, :]
            return grid[:1, :]  # Fallback

        return select_first_row_with_color

    def _make_rows_with_objects_func(self) -> Callable:
        """Create function that selects rows with object centers."""
        def select_rows_with_objects(grid: np.ndarray) -> np.ndarray:
            # Detect objects and find rows with centers
            features = self.feature_extractor.extract_object_features(grid)
            positions = features.get('object_positions', [])

            if not positions:
                return grid[:1, :]  # Fallback

            # Get unique row indices
            row_indices = sorted(set(int(pos[0]) for pos in positions))

            # Extract those rows
            selected_rows = grid[row_indices, :]
            return selected_rows

        return select_rows_with_objects

    def _make_first_non_empty_column_func(self) -> Callable:
        """Create function that selects first non-empty column."""
        def select_first_non_empty_column(grid: np.ndarray) -> np.ndarray:
            # Find first non-empty column
            for j in range(grid.shape[1]):
                if np.any(grid[:, j] != 0):
                    return grid[:, j:j+1]
            return grid[:, :1]  # Fallback

        return select_first_non_empty_column

    def _make_extract_least_common_object_func(self) -> Callable:
        """Create function that extracts object with least common color."""
        def extract_least_common_object(grid: np.ndarray) -> np.ndarray:
            # Get object features
            features = self.feature_extractor.extract_object_features(grid)
            objects_per_color = features.get('objects_per_color', {})

            if not objects_per_color:
                return grid

            # Find color with least objects
            least_common_color = min(objects_per_color.items(), key=lambda x: x[1])[0]

            # Extract object of that color
            objects = self._detect_objects_by_color(grid, least_common_color)
            if not objects:
                return grid

            # Return largest object of that color
            largest = max(objects, key=lambda o: o['size'])
            return self._extract_object_tight(largest, grid)

        return extract_least_common_object

    def _make_extract_unique_count_object_func(self) -> Callable:
        """Create function that extracts object with unique count."""
        def extract_unique_count_object(grid: np.ndarray) -> np.ndarray:
            # Get object features
            features = self.feature_extractor.extract_object_features(grid)
            objects_per_color = features.get('objects_per_color', {})

            if not objects_per_color:
                return grid

            # Find colors with unique count (count == 1)
            unique_colors = [c for c, count in objects_per_color.items() if count == 1]

            if not unique_colors:
                return grid

            # Extract first unique color object
            target_color = unique_colors[0]
            objects = self._detect_objects_by_color(grid, target_color)

            if not objects:
                return grid

            return self._extract_object_tight(objects[0], grid)

        return extract_unique_count_object

    def _make_crop_to_most_common_color_func(self) -> Callable:
        """Create function that crops to most common color bbox."""
        def crop_to_most_common_color(grid: np.ndarray) -> np.ndarray:
            # Find most common non-zero color
            non_zero = grid[grid != 0]
            if len(non_zero) == 0:
                return grid

            from collections import Counter
            color_counts = Counter(non_zero)
            most_common_color = color_counts.most_common(1)[0][0]

            # Crop to bbox of this color
            color_rows, color_cols = np.where(grid == most_common_color)

            if len(color_rows) == 0:
                return grid

            min_row, max_row = color_rows.min(), color_rows.max()
            min_col, max_col = color_cols.min(), color_cols.max()

            return grid[min_row:max_row+1, min_col:max_col+1]

        return crop_to_most_common_color

    # Helper methods

    def _detect_objects_by_color(self, grid: np.ndarray, color: int) -> List[Dict]:
        """Detect objects of specific color."""
        objects = []
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
