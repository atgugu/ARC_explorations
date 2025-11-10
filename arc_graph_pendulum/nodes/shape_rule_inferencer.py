"""
Shape Transformation Rule Inferencer
Infers abstract rules from multiple shape transformation analyses.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import Counter


class ShapeRuleInferencer:
    """
    Infers abstract transformation rules from shape analyses.

    Example:
    - Analyses: [extract color 8, extract color 1, extract color 6]
    - Inferred rule: Extract smallest object (by pixel count)
    """

    def infer_rule(self, analyses: List[Dict[str, Any]], train_examples: List) -> Dict[str, Any]:
        """
        Infer abstract rule from multiple analyses.

        Args:
            analyses: List of shape transformation analyses (one per example)
            train_examples: List of (input, output) tuples for reference

        Returns:
            Abstract rule dictionary
        """
        if not analyses:
            return self._default_rule()

        # Group by transformation type
        types = [a['transformation_type'] for a in analyses]
        most_common_type = Counter(types).most_common(1)[0][0]
        consistency = sum(1 for t in types if t == most_common_type) / len(types)

        # Get analyses of the most common type
        consensus_analyses = [a for a in analyses if a['transformation_type'] == most_common_type]

        # Infer rule based on type
        if most_common_type == 'object_extraction':
            return self._infer_object_extraction_rule(consensus_analyses, train_examples)
        elif most_common_type == 'cropping':
            return self._infer_cropping_rule(consensus_analyses, train_examples)
        elif most_common_type == 'region_selection':
            return self._infer_region_selection_rule(consensus_analyses, train_examples)
        elif most_common_type == 'color_counting':
            return self._infer_color_counting_rule(consensus_analyses, train_examples)
        else:
            return {
                'transformation_type': most_common_type,
                'rule_type': 'unknown',
                'confidence': consistency * 0.5,
                'description': f'{most_common_type} (no specific rule inferred)'
            }

    def _infer_object_extraction_rule(
        self,
        analyses: List[Dict],
        train_examples: List
    ) -> Dict[str, Any]:
        """Infer abstract rule for object extraction."""
        # Extract the specific extraction rules from analyses
        extraction_rules = []

        for analysis in analyses:
            params = analysis.get('parameters', {})
            rule = params.get('rule', {})
            extraction_rules.append(rule.get('type', 'unknown'))

        # Check if they're all the same type
        rule_types = Counter(extraction_rules)
        most_common_rule = rule_types.most_common(1)[0][0]

        if most_common_rule in ['extract_largest', 'extract_smallest', 'extract_unique_color']:
            # These are already abstract rules
            return {
                'transformation_type': 'object_extraction',
                'rule_type': most_common_rule,
                'confidence': 0.95,
                'description': f'Extract object: {most_common_rule.replace("_", " ")}'
            }

        # Check if it's extract_by_color but with different colors
        if most_common_rule == 'extract_by_color':
            # Analyze the pattern of colors extracted
            colors_extracted = []

            for i, (analysis, (input_grid, output_grid)) in enumerate(zip(analyses, train_examples)):
                params = analysis.get('parameters', {})
                rule = params.get('rule', {})
                color = rule.get('color')

                if color is not None:
                    # Count pixels of each color in input
                    from collections import defaultdict
                    color_counts = defaultdict(int)

                    unique_colors = np.unique(input_grid)
                    for c in unique_colors:
                        if c != 0:
                            color_counts[int(c)] = int(np.sum(input_grid == c))

                    colors_extracted.append({
                        'color': color,
                        'count': color_counts.get(color, 0),
                        'all_counts': color_counts
                    })

            # Check if extracted color is always smallest/largest
            if colors_extracted:
                all_smallest = all(
                    info['count'] == min(info['all_counts'].values())
                    for info in colors_extracted
                )

                all_largest = all(
                    info['count'] == max(info['all_counts'].values())
                    for info in colors_extracted
                )

                if all_smallest:
                    return {
                        'transformation_type': 'object_extraction',
                        'rule_type': 'extract_smallest_by_count',
                        'confidence': 0.95,
                        'description': 'Extract object with smallest pixel count'
                    }

                if all_largest:
                    return {
                        'transformation_type': 'object_extraction',
                        'rule_type': 'extract_largest_by_count',
                        'confidence': 0.95,
                        'description': 'Extract object with largest pixel count'
                    }

        # Default to first analysis
        return {
            'transformation_type': 'object_extraction',
            'rule_type': 'extract_by_color',
            'color': analyses[0].get('parameters', {}).get('rule', {}).get('color', 0),
            'confidence': 0.6,
            'description': f'Extract object (unclear pattern)'
        }

    def _infer_cropping_rule(
        self,
        analyses: List[Dict],
        train_examples: List
    ) -> Dict[str, Any]:
        """Infer abstract rule for cropping."""
        # Extract crop rule types
        crop_rules = []

        for analysis in analyses:
            params = analysis.get('parameters', {})
            rule_type = params.get('rule_type', 'unknown')
            crop_rules.append(rule_type)

        # Check if they're all the same
        rule_types = Counter(crop_rules)
        most_common_rule = rule_types.most_common(1)[0][0]

        if most_common_rule in ['crop_to_content_bbox', 'crop_top_left', 'crop_top_right',
                                'crop_bottom_left', 'crop_bottom_right']:
            # These are abstract enough
            return {
                'transformation_type': 'cropping',
                'rule_type': most_common_rule,
                'confidence': 0.90,
                'description': f'Crop: {most_common_rule.replace("_", " ")}'
            }

        # Check if it's crop_to_color_bbox with varying colors
        if most_common_rule == 'crop_to_color_bbox':
            # Analyze which color is being cropped to
            colors_cropped = []

            for i, (analysis, (input_grid, output_grid)) in enumerate(zip(analyses, train_examples)):
                params = analysis.get('parameters', {})
                color = params.get('color')

                if color is not None:
                    # Get bounding box sizes for each color
                    from collections import defaultdict
                    color_bbox_sizes = {}

                    unique_colors = np.unique(input_grid)
                    for c in unique_colors:
                        if c != 0:
                            rows, cols = np.where(input_grid == c)
                            if len(rows) > 0:
                                height = rows.max() - rows.min() + 1
                                width = cols.max() - cols.min() + 1
                                color_bbox_sizes[int(c)] = height * width

                    colors_cropped.append({
                        'color': color,
                        'size': color_bbox_sizes.get(color, 0),
                        'all_sizes': color_bbox_sizes
                    })

            # Check if cropped color is always smallest/largest bbox
            if colors_cropped:
                all_smallest = all(
                    info['size'] == min(info['all_sizes'].values())
                    for info in colors_cropped
                    if info['all_sizes']
                )

                all_largest = all(
                    info['size'] == max(info['all_sizes'].values())
                    for info in colors_cropped
                    if info['all_sizes']
                )

                if all_smallest:
                    return {
                        'transformation_type': 'cropping',
                        'rule_type': 'crop_to_smallest_color_bbox',
                        'confidence': 0.95,
                        'description': 'Crop to bounding box of color with smallest region'
                    }

                if all_largest:
                    return {
                        'transformation_type': 'cropping',
                        'rule_type': 'crop_to_largest_color_bbox',
                        'confidence': 0.95,
                        'description': 'Crop to bounding box of color with largest region'
                    }

        # Default
        return {
            'transformation_type': 'cropping',
            'rule_type': most_common_rule,
            'confidence': 0.6,
            'description': f'Crop (unclear pattern)'
        }

    def _infer_region_selection_rule(
        self,
        analyses: List[Dict],
        train_examples: List
    ) -> Dict[str, Any]:
        """Infer abstract rule for region selection."""
        # For now, use first analysis
        if analyses:
            return {
                'transformation_type': 'region_selection',
                'rule_type': analyses[0].get('parameters', {}).get('rule_type', 'unknown'),
                'parameters': analyses[0].get('parameters', {}),
                'confidence': 0.8,
                'description': analyses[0].get('description', 'Region selection')
            }

        return self._default_rule()

    def _infer_color_counting_rule(
        self,
        analyses: List[Dict],
        train_examples: List
    ) -> Dict[str, Any]:
        """Infer abstract rule for color counting."""
        # Extract count rule types
        count_rules = []

        for analysis in analyses:
            params = analysis.get('parameters', {})
            rule_type = params.get('rule_type', 'unknown')
            count_rules.append(rule_type)

        # Check if they're consistent
        rule_types = Counter(count_rules)
        most_common_rule = rule_types.most_common(1)[0][0]
        consistency = rule_types[most_common_rule] / len(count_rules)

        return {
            'transformation_type': 'color_counting',
            'rule_type': most_common_rule,
            'confidence': 0.85 * consistency,
            'description': f'Color counting: {most_common_rule.replace("_", " ")}'
        }

    def _default_rule(self) -> Dict[str, Any]:
        """Return default rule when inference fails."""
        return {
            'transformation_type': 'unknown',
            'rule_type': 'unknown',
            'confidence': 0.0,
            'description': 'No rule inferred'
        }
