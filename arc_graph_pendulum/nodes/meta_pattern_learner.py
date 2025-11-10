"""
Meta-Pattern Learner - Priority 3+ Implementation
Learns conditional rules from variation in training examples.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from collections import Counter

from nodes.feature_extractor import FeatureExtractor


class MetaPatternLearner:
    """
    Learns meta-patterns from training example variation.

    Key insight: When transformation parameters vary across examples,
    they often correlate with input features.

    Example:
    - Train 1: extract row 2 (row 2 is first with color 5)
    - Train 2: extract row 0 (row 0 is first with color 5)
    - Train 3: extract row 1 (row 1 is first with color 5)

    Learns: extract first row containing specific color
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def analyze_variation(
        self,
        analyses: List[Dict[str, Any]],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze variation in transformation parameters across training examples.

        Returns meta-rule if variation pattern found, None otherwise.
        """
        if len(analyses) < 2:
            return None

        # Check if transformation type is consistent
        types = [a.get('transformation_type', '') for a in analyses]
        if len(set(types)) > 1:
            return None  # Different transformation types

        trans_type = types[0]

        # Dispatch to specific variation analyzers
        if trans_type == 'region_selection':
            return self._analyze_region_selection_variation(analyses, train_examples)
        elif trans_type == 'object_extraction':
            return self._analyze_object_extraction_variation(analyses, train_examples)
        elif trans_type == 'cropping':
            return self._analyze_cropping_variation(analyses, train_examples)
        elif trans_type == 'pattern_extraction':
            return self._analyze_pattern_extraction_variation(analyses, train_examples)

        return None

    def _analyze_region_selection_variation(
        self,
        analyses: List[Dict],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Analyze variation in region selection (row/column selection)."""
        rule_types = [a.get('parameters', {}).get('rule_type', '') for a in analyses]

        if 'select_rows' in rule_types:
            return self._analyze_row_selection_variation(analyses, train_examples)
        elif 'select_columns' in rule_types:
            return self._analyze_column_selection_variation(analyses, train_examples)

        return None

    def _analyze_row_selection_variation(
        self,
        analyses: List[Dict],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Analyze variation in row selection."""
        # Extract selected rows from each example
        selected_rows_per_example = []

        for analysis in analyses:
            params = analysis.get('parameters', {})
            rows = params.get('rows', [])
            selected_rows_per_example.append(rows)

        # Check if same rows selected every time
        if all(rows == selected_rows_per_example[0] for rows in selected_rows_per_example):
            return None  # No variation

        # Extract features from each input
        features_per_example = []
        for input_grid, _ in train_examples:
            features = self.feature_extractor.extract_all_features(input_grid)
            features_per_example.append(features)

        # Try to find correlation: selected_row = f(features)
        correlations = self._find_row_correlations(
            selected_rows_per_example,
            features_per_example,
            train_examples
        )

        if correlations:
            return {
                'meta_pattern_type': 'conditional_row_selection',
                'transformation_type': 'region_selection',
                'rule': correlations,
                'confidence': correlations.get('confidence', 0.8),
                'description': correlations.get('description', 'Conditional row selection')
            }

        return None

    def _find_row_correlations(
        self,
        selected_rows_per_example: List[List[int]],
        features_per_example: List[Dict],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Find correlation between selected rows and input features."""

        # Check Pattern 1: "First row with specific property"
        # E.g., first non-empty row, first multicolor row

        # Extract first selected row from each example
        first_rows = [rows[0] if rows else -1 for rows in selected_rows_per_example]

        # Check if it's always the first non-empty row
        first_non_empty_rows = [
            feat['rows'].get('first_non_empty_row', -1)
            for feat in features_per_example
        ]

        if first_rows == first_non_empty_rows and all(r >= 0 for r in first_rows):
            return {
                'type': 'first_non_empty_row',
                'confidence': 0.95,
                'description': 'Select first non-empty row'
            }

        # Check if it's always the first multicolor row
        first_multicolor_rows = [
            feat['rows'].get('first_multicolor_row', -1)
            for feat in features_per_example
        ]

        if first_rows == first_multicolor_rows and all(r >= 0 for r in first_rows):
            return {
                'type': 'first_multicolor_row',
                'confidence': 0.95,
                'description': 'Select first multicolor row'
            }

        # Check Pattern 2: "Row containing specific color"
        # Find which color the selected row contains

        selected_row_colors = []
        for i, (input_grid, _) in enumerate(train_examples):
            rows = selected_rows_per_example[i]
            if rows:
                row_idx = rows[0]
                if 0 <= row_idx < input_grid.shape[0]:
                    row = input_grid[row_idx, :]
                    unique_colors = np.unique(row[row != 0])
                    selected_row_colors.append(set(unique_colors))

        # Check if there's a common color across all selected rows
        if selected_row_colors:
            common_colors = set.intersection(*selected_row_colors)

            if common_colors:
                # Found a color that appears in all selected rows
                target_color = list(common_colors)[0]

                # Verify this is the first row containing this color
                is_first_with_color = True
                for i, (input_grid, _) in enumerate(train_examples):
                    rows = selected_rows_per_example[i]
                    if not rows:
                        continue

                    selected_row = rows[0]

                    # Find first row with target color
                    first_row_with_color = -1
                    for r in range(input_grid.shape[0]):
                        if target_color in input_grid[r, :]:
                            first_row_with_color = r
                            break

                    if selected_row != first_row_with_color:
                        is_first_with_color = False
                        break

                if is_first_with_color:
                    return {
                        'type': 'first_row_with_color',
                        'color': int(target_color),
                        'confidence': 0.90,
                        'description': f'Select first row containing color {target_color}'
                    }

        # Check Pattern 3: "Rows containing objects"
        # Check if selected rows correspond to rows with objects

        rows_with_objects = []
        for i, (input_grid, _) in enumerate(train_examples):
            features = features_per_example[i]
            obj_features = features.get('objects', {})

            if obj_features.get('num_objects', 0) > 0:
                # Find which rows contain object centers
                positions = obj_features.get('object_positions', [])
                rows_set = set()
                for pos in positions:
                    rows_set.add(int(pos[0]))  # row coordinate
                rows_with_objects.append(sorted(rows_set))

        # Check if selected rows match rows with objects
        if len(rows_with_objects) == len(selected_rows_per_example):
            match_count = sum(
                1 for sel, obj in zip(selected_rows_per_example, rows_with_objects)
                if sel == obj
            )

            if match_count >= len(selected_rows_per_example) * 0.8:
                return {
                    'type': 'rows_with_objects',
                    'confidence': 0.85,
                    'description': 'Select rows containing object centers'
                }

        return None

    def _analyze_column_selection_variation(
        self,
        analyses: List[Dict],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Analyze variation in column selection (similar to rows)."""
        # Similar logic to row selection but for columns
        # For brevity, using similar patterns

        selected_cols_per_example = []
        for analysis in analyses:
            params = analysis.get('parameters', {})
            cols = params.get('columns', [])
            selected_cols_per_example.append(cols)

        if all(cols == selected_cols_per_example[0] for cols in selected_cols_per_example):
            return None  # No variation

        # Extract first selected column
        first_cols = [cols[0] if cols else -1 for cols in selected_cols_per_example]

        features_per_example = []
        for input_grid, _ in train_examples:
            features = self.feature_extractor.extract_all_features(input_grid)
            features_per_example.append(features)

        # Check if it's first non-empty column
        first_non_empty_cols = [
            feat['columns'].get('first_non_empty_column', -1)
            for feat in features_per_example
        ]

        if first_cols == first_non_empty_cols and all(c >= 0 for c in first_cols):
            return {
                'meta_pattern_type': 'conditional_column_selection',
                'transformation_type': 'region_selection',
                'rule': {
                    'type': 'first_non_empty_column',
                    'confidence': 0.95,
                    'description': 'Select first non-empty column'
                },
                'confidence': 0.95,
                'description': 'Conditional column selection'
            }

        return None

    def _analyze_object_extraction_variation(
        self,
        analyses: List[Dict],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Analyze variation in object extraction."""
        # Extract which objects were selected
        extraction_rules = [a.get('parameters', {}).get('rule', {}) for a in analyses]

        rule_types = [r.get('type', '') for r in extraction_rules]

        # If all same type, no variation
        if len(set(rule_types)) == 1:
            return None

        # Check if color varies but follows pattern
        if all(rt == 'extract_by_color' for rt in rule_types):
            colors = [r.get('color', -1) for r in extraction_rules]

            # Extract features
            features_per_example = []
            for input_grid, _ in train_examples:
                features = self.feature_extractor.extract_all_features(input_grid)
                features_per_example.append(features)

            # Check if extracted color is always least common
            least_common_colors = []
            for features in features_per_example:
                obj_features = features.get('objects', {})
                objects_per_color = obj_features.get('objects_per_color', {})

                if objects_per_color:
                    least_common = min(objects_per_color.items(), key=lambda x: x[1])[0]
                    least_common_colors.append(least_common)

            if colors == least_common_colors:
                return {
                    'meta_pattern_type': 'conditional_object_extraction',
                    'transformation_type': 'object_extraction',
                    'rule': {
                        'type': 'extract_least_common_object_color',
                        'confidence': 0.90,
                        'description': 'Extract object of least common color'
                    },
                    'confidence': 0.90,
                    'description': 'Extract object with least common color'
                }

            # Check if extracted color has unique object count
            unique_count_colors_per_example = [
                f.get('objects', {}).get('unique_count_colors', [])
                for f in features_per_example
            ]

            all_match = all(
                color in unique_colors
                for color, unique_colors in zip(colors, unique_count_colors_per_example)
            )

            if all_match:
                return {
                    'meta_pattern_type': 'conditional_object_extraction',
                    'transformation_type': 'object_extraction',
                    'rule': {
                        'type': 'extract_unique_count_color',
                        'confidence': 0.90,
                        'description': 'Extract object with unique occurrence count'
                    },
                    'confidence': 0.90,
                    'description': 'Extract object with unique count'
                }

        return None

    def _analyze_cropping_variation(
        self,
        analyses: List[Dict],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Analyze variation in cropping operations."""
        # Extract crop parameters
        crop_rule_types = [
            a.get('parameters', {}).get('rule_type', '')
            for a in analyses
        ]

        # If all same, no variation
        if len(set(crop_rule_types)) == 1:
            return None

        # Check if cropping to different colors but following pattern
        if all('color' in crt for crt in crop_rule_types):
            colors = [a.get('parameters', {}).get('color', -1) for a in analyses]

            features_per_example = []
            for input_grid, _ in train_examples:
                features = self.feature_extractor.extract_all_features(input_grid)
                features_per_example.append(features)

            # Check if color is always most/least common
            most_common_colors = [
                f.get('grid', {}).get('most_common_color', -1)
                for f in features_per_example
            ]

            if colors == most_common_colors:
                return {
                    'meta_pattern_type': 'conditional_cropping',
                    'transformation_type': 'cropping',
                    'rule': {
                        'type': 'crop_to_most_common_color',
                        'confidence': 0.90,
                        'description': 'Crop to most common color bbox'
                    },
                    'confidence': 0.90,
                    'description': 'Crop to most common color'
                }

        return None

    def _analyze_pattern_extraction_variation(
        self,
        analyses: List[Dict],
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Analyze variation in pattern extraction."""
        # Extract row mappings
        row_mappings = []

        for analysis in analyses:
            params = analysis.get('parameters', {})
            mapping = params.get('row_mapping', [])
            row_mappings.append(mapping)

        # Check if mappings vary
        if all(m == row_mappings[0] for m in row_mappings):
            return None  # No variation

        # Try to find pattern in variation
        # For pattern extraction, this is complex - may need sequence analysis

        return None
