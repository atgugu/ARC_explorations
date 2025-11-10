"""
Feature Extraction Framework
Extracts properties from grids and objects for meta-pattern learning.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import ndimage
from collections import Counter


class FeatureExtractor:
    """
    Extracts features from grids and objects for correlation analysis.

    Features include:
    - Grid-level: colors, sizes, patterns
    - Object-level: counts, sizes, positions, colors
    - Spatial: positions, distributions, clustering
    """

    def extract_grid_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract features from entire grid."""
        features = {}

        # Basic properties
        features['height'] = grid.shape[0]
        features['width'] = grid.shape[1]
        features['size'] = grid.size

        # Color statistics
        unique_colors = np.unique(grid)
        features['num_colors'] = len(unique_colors)
        features['colors'] = unique_colors.tolist()
        features['has_zero'] = 0 in unique_colors

        # Color counts
        color_counts = Counter(grid.flatten())
        features['color_counts'] = dict(color_counts)

        # Most/least common non-zero colors
        non_zero_counts = {k: v for k, v in color_counts.items() if k != 0}
        if non_zero_counts:
            features['most_common_color'] = max(non_zero_counts.items(), key=lambda x: x[1])[0]
            features['least_common_color'] = min(non_zero_counts.items(), key=lambda x: x[1])[0]

        # Density
        non_zero_count = np.sum(grid != 0)
        features['density'] = non_zero_count / grid.size if grid.size > 0 else 0

        # Symmetry checks
        features['is_symmetric_horizontal'] = np.array_equal(grid, np.fliplr(grid))
        features['is_symmetric_vertical'] = np.array_equal(grid, np.flipud(grid))

        return features

    def extract_object_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract object-level features."""
        features = {}

        # Detect all objects
        objects = self._detect_all_objects(grid)
        features['num_objects'] = len(objects)

        if not objects:
            return features

        # Object sizes
        sizes = [obj['size'] for obj in objects]
        features['object_sizes'] = sizes
        features['min_object_size'] = min(sizes)
        features['max_object_size'] = max(sizes)
        features['avg_object_size'] = np.mean(sizes)

        # Object colors
        colors = [obj['color'] for obj in objects]
        features['object_colors'] = colors
        features['num_unique_object_colors'] = len(set(colors))

        # Object color counts (how many objects per color)
        color_counts = Counter(colors)
        features['objects_per_color'] = dict(color_counts)

        # Find colors with unique object count
        unique_count_colors = [c for c, count in color_counts.items() if count == 1]
        features['unique_count_colors'] = unique_count_colors

        # Object positions (centers)
        positions = [obj['center'] for obj in objects]
        if positions:
            features['object_positions'] = positions
            features['leftmost_object_idx'] = int(np.argmin([p[1] for p in positions]))
            features['rightmost_object_idx'] = int(np.argmax([p[1] for p in positions]))
            features['topmost_object_idx'] = int(np.argmin([p[0] for p in positions]))
            features['bottommost_object_idx'] = int(np.argmax([p[0] for p in positions]))

        return features

    def extract_row_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract row-specific features."""
        features = {}

        h, w = grid.shape
        features['num_rows'] = h

        # For each row
        row_features = []
        for i in range(h):
            row = grid[i, :]
            row_feat = {
                'index': i,
                'unique_colors': len(np.unique(row)),
                'non_zero_count': int(np.sum(row != 0)),
                'is_empty': np.all(row == 0),
                'is_uniform': len(np.unique(row)) == 1,
                'dominant_color': int(Counter(row).most_common(1)[0][0]),
            }
            row_features.append(row_feat)

        features['rows'] = row_features

        # First non-empty row
        for i, rf in enumerate(row_features):
            if not rf['is_empty']:
                features['first_non_empty_row'] = i
                break

        # First row with specific properties
        for i, rf in enumerate(row_features):
            if rf['unique_colors'] > 1:
                features['first_multicolor_row'] = i
                break

        return features

    def extract_column_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract column-specific features."""
        # Transpose and use row logic
        transposed_features = self.extract_row_features(grid.T)

        # Rename keys
        features = {}
        features['num_columns'] = transposed_features.get('num_rows', 0)
        features['columns'] = transposed_features.get('rows', [])

        if 'first_non_empty_row' in transposed_features:
            features['first_non_empty_column'] = transposed_features['first_non_empty_row']
        if 'first_multicolor_row' in transposed_features:
            features['first_multicolor_column'] = transposed_features['first_multicolor_row']

        return features

    def extract_all_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract all features from grid."""
        features = {}

        features['grid'] = self.extract_grid_features(grid)
        features['objects'] = self.extract_object_features(grid)
        features['rows'] = self.extract_row_features(grid)
        features['columns'] = self.extract_column_features(grid)

        return features

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
