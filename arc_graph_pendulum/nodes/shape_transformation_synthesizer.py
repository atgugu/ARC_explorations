"""
Shape Transformation Synthesizer - Priority 1 Implementation
Generates programs for object extraction, cropping, and region selection.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Tuple
from scipy import ndimage


class ShapeTransformationSynthesizer:
    """
    Generates executable programs for shape-changing transformations.

    Handles:
    1. Object extraction programs
    2. Cropping programs
    3. Region selection programs
    4. Color counting programs
    """

    def synthesize(self, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate programs based on inferred rule.

        Returns list of programs sorted by confidence.
        """
        rule_type = rule.get('transformation_type', '')
        programs = []

        # Always include identity as last resort
        programs.append({
            'type': 'identity',
            'function': lambda grid: grid,
            'confidence': 0.1,
            'description': 'Identity (fallback)'
        })

        # Generate rule-specific programs
        if rule_type == 'object_extraction':
            programs.extend(self._synthesize_object_extraction(rule))
        elif rule_type == 'cropping':
            programs.extend(self._synthesize_cropping(rule))
        elif rule_type == 'region_selection':
            programs.extend(self._synthesize_region_selection(rule))
        elif rule_type == 'color_counting':
            programs.extend(self._synthesize_color_counting(rule))
        elif rule_type == 'object_properties':
            programs.extend(self._synthesize_object_properties(rule))

        # Sort by confidence (highest first)
        programs.sort(key=lambda p: p.get('confidence', 0.0), reverse=True)

        return programs

    def _synthesize_object_extraction(self, rule: Dict[str, Any]) -> List[Dict]:
        """Generate object extraction programs."""
        programs = []
        parameters = rule.get('parameters', {})
        extraction_rule = parameters.get('rule', {}) if 'rule' in parameters else {}
        rule_type = extraction_rule.get('type', rule.get('rule_type', ''))
        confidence = rule.get('confidence', 0.8)

        # Handle abstract rules from shape rule inferencer
        if rule_type == 'extract_smallest_by_count':
            programs.append({
                'type': 'extract_smallest_by_count',
                'function': self._make_extract_smallest_by_count_func(),
                'confidence': confidence,
                'description': 'Extract object with smallest pixel count'
            })

        elif rule_type == 'extract_largest_by_count':
            programs.append({
                'type': 'extract_largest_by_count',
                'function': self._make_extract_largest_by_count_func(),
                'confidence': confidence,
                'description': 'Extract object with largest pixel count'
            })

        elif rule_type == 'extract_by_color':
            color = extraction_rule.get('color', parameters.get('color'))
            if color is not None:
                programs.append({
                    'type': 'extract_by_color',
                    'function': self._make_extract_by_color_func(color),
                    'confidence': confidence,
                    'description': f'Extract objects of color {color}'
                })

        elif rule_type == 'extract_largest':
            programs.append({
                'type': 'extract_largest',
                'function': self._make_extract_largest_func(),
                'confidence': confidence,
                'description': 'Extract largest object'
            })

        elif rule_type == 'extract_smallest':
            programs.append({
                'type': 'extract_smallest',
                'function': self._make_extract_smallest_func(),
                'confidence': confidence,
                'description': 'Extract smallest object'
            })

        elif rule_type == 'extract_unique_color':
            programs.append({
                'type': 'extract_unique_color',
                'function': self._make_extract_unique_color_func(),
                'confidence': confidence,
                'description': 'Extract object with unique color'
            })

        # Add variations if no specific rule
        if not programs or confidence < 0.8:
            programs.append({
                'type': 'extract_smallest_by_count',
                'function': self._make_extract_smallest_by_count_func(),
                'confidence': 0.7,
                'description': 'Extract object with smallest pixel count'
            })

            programs.append({
                'type': 'extract_largest',
                'function': self._make_extract_largest_func(),
                'confidence': 0.7,
                'description': 'Extract largest object'
            })

        return programs

    def _synthesize_cropping(self, rule: Dict[str, Any]) -> List[Dict]:
        """Generate cropping programs."""
        programs = []
        parameters = rule.get('parameters', {})
        crop_rule_type = parameters.get('rule_type', rule.get('rule_type', ''))
        confidence = rule.get('confidence', 0.9)

        # Handle abstract rules from shape rule inferencer
        if crop_rule_type == 'crop_to_smallest_color_bbox':
            programs.append({
                'type': 'crop_to_smallest_color_bbox',
                'function': self._make_crop_to_smallest_color_bbox_func(),
                'confidence': confidence,
                'description': 'Crop to bounding box of color with smallest region'
            })

        elif crop_rule_type == 'crop_to_largest_color_bbox':
            programs.append({
                'type': 'crop_to_largest_color_bbox',
                'function': self._make_crop_to_largest_color_bbox_func(),
                'confidence': confidence,
                'description': 'Crop to bounding box of color with largest region'
            })

        elif crop_rule_type == 'crop_to_content_bbox':
            programs.append({
                'type': 'crop_to_content_bbox',
                'function': self._make_crop_to_content_bbox_func(),
                'confidence': confidence,
                'description': 'Crop to bounding box of non-zero content'
            })

        elif crop_rule_type == 'crop_to_color_bbox':
            color = parameters.get('color')
            if color is not None:
                programs.append({
                    'type': 'crop_to_color_bbox',
                    'function': self._make_crop_to_color_bbox_func(color),
                    'confidence': confidence,
                    'description': f'Crop to bounding box of color {color}'
                })

        elif crop_rule_type in ['crop_top_left', 'crop_top_right', 'crop_bottom_left', 'crop_bottom_right', 'crop_region']:
            position = parameters.get('position')
            size = parameters.get('size')

            if position and size:
                programs.append({
                    'type': crop_rule_type,
                    'function': self._make_crop_region_func(position, size),
                    'confidence': confidence,
                    'description': f'Crop region at {position} with size {size}'
                })

        # Add variations if uncertain
        if not programs or confidence < 0.8:
            programs.append({
                'type': 'crop_to_smallest_color_bbox',
                'function': self._make_crop_to_smallest_color_bbox_func(),
                'confidence': 0.7,
                'description': 'Crop to smallest color bbox'
            })

            programs.append({
                'type': 'crop_to_content_bbox',
                'function': self._make_crop_to_content_bbox_func(),
                'confidence': 0.6,
                'description': 'Crop to content bbox'
            })

        return programs

    def _synthesize_region_selection(self, rule: Dict[str, Any]) -> List[Dict]:
        """Generate region selection programs."""
        programs = []
        parameters = rule.get('parameters', {})
        selection_type = parameters.get('rule_type', '')
        confidence = rule.get('confidence', 1.0)

        if selection_type == 'select_rows':
            rows = parameters.get('rows', [])
            programs.append({
                'type': 'select_rows',
                'function': self._make_select_rows_func(rows),
                'confidence': confidence,
                'description': f'Select rows {rows}'
            })

        elif selection_type == 'select_columns':
            columns = parameters.get('columns', [])
            programs.append({
                'type': 'select_columns',
                'function': self._make_select_columns_func(columns),
                'confidence': confidence,
                'description': f'Select columns {columns}'
            })

        elif selection_type == 'extract_diagonal':
            diag_type = parameters.get('diagonal_type', 'main')
            programs.append({
                'type': 'extract_diagonal',
                'function': self._make_extract_diagonal_func(diag_type),
                'confidence': confidence,
                'description': f'Extract {diag_type} diagonal'
            })

        return programs

    def _synthesize_color_counting(self, rule: Dict[str, Any]) -> List[Dict]:
        """Generate color counting programs."""
        programs = []
        parameters = rule.get('parameters', {})
        count_rule_type = parameters.get('rule_type', '')
        confidence = rule.get('confidence', 0.9)

        if count_rule_type == 'most_common_color':
            programs.append({
                'type': 'most_common_color',
                'function': self._make_most_common_color_func(),
                'confidence': confidence,
                'description': 'Output most common non-zero color'
            })

        elif count_rule_type == 'least_common_color':
            programs.append({
                'type': 'least_common_color',
                'function': self._make_least_common_color_func(),
                'confidence': confidence,
                'description': 'Output least common non-zero color'
            })

        elif count_rule_type == 'unique_color_object':
            programs.append({
                'type': 'unique_color_object',
                'function': self._make_unique_color_object_func(),
                'confidence': confidence,
                'description': 'Output color with unique object count'
            })

        return programs

    def _synthesize_object_properties(self, rule: Dict[str, Any]) -> List[Dict]:
        """Generate object property programs."""
        programs = []
        parameters = rule.get('parameters', {})
        prop_rule_type = parameters.get('rule_type', '')
        confidence = rule.get('confidence', 0.7)

        if prop_rule_type == 'count_objects':
            programs.append({
                'type': 'count_objects',
                'function': self._make_count_objects_func(),
                'confidence': confidence,
                'description': 'Count number of objects'
            })

        return programs

    # Factory methods to create transformation functions

    def _make_extract_by_color_func(self, color: int) -> Callable:
        """Create function to extract objects of specific color."""
        def extract_by_color(grid: np.ndarray) -> np.ndarray:
            objects = self._detect_objects_by_color(grid, color)

            if not objects:
                return np.array([[0]])

            # Return largest object of that color
            largest = max(objects, key=lambda o: o['size'])
            return self._extract_object_tight(largest, grid)

        return extract_by_color

    def _make_extract_largest_func(self) -> Callable:
        """Create function to extract largest object."""
        def extract_largest(grid: np.ndarray) -> np.ndarray:
            objects = self._detect_all_objects(grid)

            if not objects:
                return np.array([[0]])

            largest = max(objects, key=lambda o: o['size'])
            return self._extract_object_tight(largest, grid)

        return extract_largest

    def _make_extract_smallest_func(self) -> Callable:
        """Create function to extract smallest object."""
        def extract_smallest(grid: np.ndarray) -> np.ndarray:
            objects = self._detect_all_objects(grid)

            if not objects:
                return np.array([[0]])

            smallest = min(objects, key=lambda o: o['size'])
            return self._extract_object_tight(smallest, grid)

        return extract_smallest

    def _make_extract_unique_color_func(self) -> Callable:
        """Create function to extract object with unique color."""
        def extract_unique_color(grid: np.ndarray) -> np.ndarray:
            objects = self._detect_all_objects(grid)

            if not objects:
                return np.array([[0]])

            # Count objects per color
            from collections import Counter
            color_counts = Counter([obj['color'] for obj in objects])

            # Find colors with exactly one object
            unique_colors = [c for c, count in color_counts.items() if count == 1]

            if unique_colors:
                # Extract first unique color object
                for obj in objects:
                    if obj['color'] in unique_colors:
                        return self._extract_object_tight(obj, grid)

            # Fallback to largest
            largest = max(objects, key=lambda o: o['size'])
            return self._extract_object_tight(largest, grid)

        return extract_unique_color

    def _make_extract_most_common_color_func(self) -> Callable:
        """Create function to extract objects of most common color."""
        def extract_most_common_color(grid: np.ndarray) -> np.ndarray:
            # Find most common non-zero color
            non_zero = grid[grid != 0]

            if len(non_zero) == 0:
                return np.array([[0]])

            from collections import Counter
            color_counts = Counter(non_zero)
            most_common_color = color_counts.most_common(1)[0][0]

            # Extract largest object of that color
            objects = self._detect_objects_by_color(grid, most_common_color)

            if not objects:
                return np.array([[0]])

            largest = max(objects, key=lambda o: o['size'])
            return self._extract_object_tight(largest, grid)

        return extract_most_common_color

    def _make_extract_smallest_by_count_func(self) -> Callable:
        """Create function to extract object with smallest pixel count."""
        def extract_smallest_by_count(grid: np.ndarray) -> np.ndarray:
            objects = self._detect_all_objects(grid)

            if not objects:
                return np.array([[0]])

            # Find object with smallest pixel count
            smallest = min(objects, key=lambda o: o['size'])
            return self._extract_object_tight(smallest, grid)

        return extract_smallest_by_count

    def _make_extract_largest_by_count_func(self) -> Callable:
        """Create function to extract object with largest pixel count."""
        def extract_largest_by_count(grid: np.ndarray) -> np.ndarray:
            objects = self._detect_all_objects(grid)

            if not objects:
                return np.array([[0]])

            # Find object with largest pixel count
            largest = max(objects, key=lambda o: o['size'])
            return self._extract_object_tight(largest, grid)

        return extract_largest_by_count

    def _make_crop_to_content_bbox_func(self) -> Callable:
        """Create function to crop to bounding box of non-zero content."""
        def crop_to_content_bbox(grid: np.ndarray) -> np.ndarray:
            non_zero_rows, non_zero_cols = np.where(grid != 0)

            if len(non_zero_rows) == 0:
                return grid

            min_row, max_row = non_zero_rows.min(), non_zero_rows.max()
            min_col, max_col = non_zero_cols.min(), non_zero_cols.max()

            return grid[min_row:max_row+1, min_col:max_col+1]

        return crop_to_content_bbox

    def _make_crop_to_color_bbox_func(self, color: int) -> Callable:
        """Create function to crop to bounding box of specific color."""
        def crop_to_color_bbox(grid: np.ndarray) -> np.ndarray:
            color_rows, color_cols = np.where(grid == color)

            if len(color_rows) == 0:
                return grid

            min_row, max_row = color_rows.min(), color_rows.max()
            min_col, max_col = color_cols.min(), color_cols.max()

            return grid[min_row:max_row+1, min_col:max_col+1]

        return crop_to_color_bbox

    def _make_crop_to_smallest_color_bbox_func(self) -> Callable:
        """Create function to crop to bounding box of color with smallest region."""
        def crop_to_smallest_color_bbox(grid: np.ndarray) -> np.ndarray:
            colors = np.unique(grid)
            colors = colors[colors != 0]

            if len(colors) == 0:
                return grid

            # Find bbox size for each color
            smallest_bbox = None
            smallest_size = float('inf')

            for color in colors:
                color_rows, color_cols = np.where(grid == color)

                if len(color_rows) > 0:
                    min_row, max_row = color_rows.min(), color_rows.max()
                    min_col, max_col = color_cols.min(), color_cols.max()

                    height = max_row - min_row + 1
                    width = max_col - min_col + 1
                    size = height * width

                    if size < smallest_size:
                        smallest_size = size
                        smallest_bbox = (min_row, max_row, min_col, max_col)

            if smallest_bbox:
                min_row, max_row, min_col, max_col = smallest_bbox
                return grid[min_row:max_row+1, min_col:max_col+1]

            return grid

        return crop_to_smallest_color_bbox

    def _make_crop_to_largest_color_bbox_func(self) -> Callable:
        """Create function to crop to bounding box of color with largest region."""
        def crop_to_largest_color_bbox(grid: np.ndarray) -> np.ndarray:
            colors = np.unique(grid)
            colors = colors[colors != 0]

            if len(colors) == 0:
                return grid

            # Find bbox size for each color
            largest_bbox = None
            largest_size = 0

            for color in colors:
                color_rows, color_cols = np.where(grid == color)

                if len(color_rows) > 0:
                    min_row, max_row = color_rows.min(), color_rows.max()
                    min_col, max_col = color_cols.min(), color_cols.max()

                    height = max_row - min_row + 1
                    width = max_col - min_col + 1
                    size = height * width

                    if size > largest_size:
                        largest_size = size
                        largest_bbox = (min_row, max_row, min_col, max_col)

            if largest_bbox:
                min_row, max_row, min_col, max_col = largest_bbox
                return grid[min_row:max_row+1, min_col:max_col+1]

            return grid

        return crop_to_largest_color_bbox

    def _make_crop_region_func(self, position: Tuple[int, int], size: Tuple[int, int]) -> Callable:
        """Create function to crop specific region."""
        def crop_region(grid: np.ndarray) -> np.ndarray:
            i, j = position
            h, w = size

            if i + h > grid.shape[0] or j + w > grid.shape[1]:
                return grid

            return grid[i:i+h, j:j+w]

        return crop_region

    def _make_select_rows_func(self, rows: List[int]) -> Callable:
        """Create function to select specific rows."""
        def select_rows(grid: np.ndarray) -> np.ndarray:
            valid_rows = [r for r in rows if r < grid.shape[0]]

            if not valid_rows:
                return grid

            return grid[valid_rows, :]

        return select_rows

    def _make_select_columns_func(self, columns: List[int]) -> Callable:
        """Create function to select specific columns."""
        def select_columns(grid: np.ndarray) -> np.ndarray:
            valid_cols = [c for c in columns if c < grid.shape[1]]

            if not valid_cols:
                return grid

            return grid[:, valid_cols]

        return select_columns

    def _make_extract_diagonal_func(self, diag_type: str) -> Callable:
        """Create function to extract diagonal."""
        def extract_diagonal(grid: np.ndarray) -> np.ndarray:
            if diag_type == 'main':
                diag = np.diag(grid)
            else:
                diag = np.diag(np.fliplr(grid))

            return diag.reshape(1, -1)

        return extract_diagonal

    def _make_most_common_color_func(self) -> Callable:
        """Create function to output most common color."""
        def most_common_color(grid: np.ndarray) -> np.ndarray:
            non_zero = grid[grid != 0]

            if len(non_zero) == 0:
                return np.array([[0]])

            from collections import Counter
            color_counts = Counter(non_zero)
            most_common = color_counts.most_common(1)[0][0]

            return np.array([[most_common]])

        return most_common_color

    def _make_least_common_color_func(self) -> Callable:
        """Create function to output least common color."""
        def least_common_color(grid: np.ndarray) -> np.ndarray:
            non_zero = grid[grid != 0]

            if len(non_zero) == 0:
                return np.array([[0]])

            from collections import Counter
            color_counts = Counter(non_zero)
            least_common = min(color_counts.items(), key=lambda x: x[1])[0]

            return np.array([[least_common]])

        return least_common_color

    def _make_unique_color_object_func(self) -> Callable:
        """Create function to output unique color (by object count)."""
        def unique_color_object(grid: np.ndarray) -> np.ndarray:
            objects = self._detect_all_objects(grid)

            if not objects:
                return np.array([[0]])

            from collections import Counter
            color_counts = Counter([obj['color'] for obj in objects])
            unique_colors = [c for c, count in color_counts.items() if count == 1]

            if unique_colors:
                return np.array([[unique_colors[0]]])

            return np.array([[0]])

        return unique_color_object

    def _make_count_objects_func(self) -> Callable:
        """Create function to count objects."""
        def count_objects(grid: np.ndarray) -> np.ndarray:
            objects = self._detect_all_objects(grid)
            count = len(objects)

            # Return grid of size count x 1 filled with 1s
            return np.ones((count, 1), dtype=int)

        return count_objects

    # Helper methods (same as in analyzer)

    def _detect_all_objects(self, grid: np.ndarray) -> List[Dict]:
        """Detect all connected component objects in grid."""
        objects = []

        colors = np.unique(grid)
        colors = colors[colors != 0]

        for color in colors:
            objects.extend(self._detect_objects_by_color(grid, color))

        return objects

    def _detect_objects_by_color(self, grid: np.ndarray, color: int) -> List[Dict]:
        """Detect objects of a specific color."""
        objects = []

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

        bbox_region = grid[min_row:max_row+1, min_col:max_col+1].copy()
        obj_mask_region = obj['mask'][min_row:max_row+1, min_col:max_col+1]
        bbox_region[~obj_mask_region] = 0

        return bbox_region


# Node creation functions
from core.node import Node, NodeOutput


def shape_transformation_synthesizer_func(data: Dict[str, Any]) -> NodeOutput:
    """Shape transformation synthesis function for node."""
    artifacts = {}
    telemetry = {'node_type': 'synthesizer', 'subtype': 'shape_transformation_synthesizer'}

    rule = data.get('rule', {})

    if not rule:
        telemetry['success'] = False
        return NodeOutput(result=[], artifacts=artifacts, telemetry=telemetry)

    synthesizer = ShapeTransformationSynthesizer()
    programs = synthesizer.synthesize(rule)

    artifacts['programs'] = programs
    artifacts['num_programs'] = len(programs)

    telemetry['success'] = True
    telemetry['num_programs'] = len(programs)
    telemetry['rule_type'] = rule.get('transformation_type', 'unknown')

    return NodeOutput(result=programs, artifacts=artifacts, telemetry=telemetry)


def create_shape_transformation_synthesizer_node() -> Node:
    """Create a shape transformation synthesizer node."""
    return Node(
        name="shape_transformation_synthesizer",
        func=shape_transformation_synthesizer_func,
        input_type="transformation_rule",
        output_type="programs",
        deterministic=True,
        category="synthesizer"
    )
