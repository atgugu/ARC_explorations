"""
Enhanced Differential Analyzer with support for complex transformations.
Adds object-level, pattern-based, and multi-step detection.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import sys
import os
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.differential_analyzer import DifferentialAnalyzer as BaseDifferentialAnalyzer
from utils.grid_utils import compute_iou


class EnhancedDifferentialAnalyzer(BaseDifferentialAnalyzer):
    """
    Enhanced analyzer with support for:
    - Object-level operations
    - Pattern-based transformations
    - Conditional operations
    - Multi-step compositions
    """

    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Enhanced analysis with more transformation types."""
        results = []

        # Try all base transformations first
        base_result = super().analyze(input_grid, output_grid)
        if base_result['confidence'] >= 0.9:
            # High confidence in simple transformation
            return base_result

        # Try enhanced transformations
        results.append(base_result)
        results.append(self._check_pattern_based_tiling(input_grid, output_grid))
        results.append(self._check_pattern_extraction(input_grid, output_grid))
        results.append(self._check_object_operations(input_grid, output_grid))
        results.append(self._check_conditional_fill(input_grid, output_grid))
        results.append(self._check_spatial_pattern(input_grid, output_grid))

        # Sort by confidence
        results = [r for r in results if r is not None]
        results.sort(key=lambda x: x['confidence'], reverse=True)

        if results:
            return results[0]
        else:
            return base_result

    def _check_pattern_based_tiling(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Check for pattern-based tiling where input controls output pattern.
        E.g., 3x3 input → 9x9 output where each input cell controls a 3x3 block.
        """
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape

        # Check for integer scaling
        for scale_h in [2, 3, 4]:
            for scale_w in [2, 3, 4]:
                if h_out == h_in * scale_h and w_out == w_in * scale_w:
                    # Try pattern-based tiling
                    match_score = self._test_pattern_tiling(
                        input_grid, output_grid, scale_h, scale_w
                    )

                    if match_score >= 0.7:
                        return {
                            'transformation_type': 'pattern_based_tiling',
                            'parameters': {
                                'scale_h': scale_h,
                                'scale_w': scale_w,
                                'pattern_type': 'conditional_blocks'
                            },
                            'confidence': match_score,
                            'description': f'Pattern-based tiling {scale_h}x{scale_w} with conditional blocks'
                        }

        return None

    def _test_pattern_tiling(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        scale_h: int,
        scale_w: int
    ) -> float:
        """Test if output is pattern-based tiling of input."""
        h_in, w_in = input_grid.shape

        # Extract the "pattern" that gets replicated
        # Hypothesis: The pattern is determined by non-zero cells in input
        non_zero_mask = input_grid != 0

        if not non_zero_mask.any():
            return 0.0

        # Find a representative block (use the one corresponding to a non-zero input cell)
        for i in range(h_in):
            for j in range(w_in):
                if input_grid[i, j] != 0:
                    # Extract the corresponding block in output
                    block_i = i * scale_h
                    block_j = j * scale_w

                    if block_i + scale_h <= output_grid.shape[0] and \
                       block_j + scale_w <= output_grid.shape[1]:
                        pattern_block = output_grid[
                            block_i:block_i+scale_h,
                            block_j:block_j+scale_w
                        ].copy()

                        # Test if this pattern repeats for all non-zero cells
                        matches = 0
                        total = 0

                        for ii in range(h_in):
                            for jj in range(w_in):
                                total += 1
                                bi = ii * scale_h
                                bj = jj * scale_w

                                actual_block = output_grid[bi:bi+scale_h, bj:bj+scale_w]

                                if input_grid[ii, jj] != 0:
                                    # Should match pattern
                                    if np.array_equal(actual_block, pattern_block):
                                        matches += 1
                                else:
                                    # Should be all zeros
                                    if np.all(actual_block == 0):
                                        matches += 1

                        return matches / total if total > 0 else 0.0

        return 0.0

    def _check_pattern_extraction(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Check for pattern extraction and extension.
        E.g., extracting certain rows/columns and repeating them.
        """
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape

        # Check if width is same (row operations)
        if w_in == w_out:
            # Try to find row pattern
            result = self._find_row_pattern(input_grid, output_grid)
            if result:
                return result

        # Check if height is same (column operations)
        if h_in == h_out:
            # Try to find column pattern
            result = self._find_column_pattern(input_grid, output_grid)
            if result:
                return result

        return None

    def _find_row_pattern(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """Find if output rows are extracted/repeated from input."""
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape

        # Account for color remapping
        color_mapping = self._learn_color_mapping_loose(input_grid, output_grid)

        # Try to match each output row to an input row
        row_mapping = []

        for out_row_idx in range(h_out):
            out_row = output_grid[out_row_idx]
            best_match = -1
            best_score = 0.0

            for in_row_idx in range(h_in):
                in_row = input_grid[in_row_idx]

                # Apply color mapping
                mapped_in_row = in_row.copy()
                for from_color, to_color in color_mapping.items():
                    mapped_in_row[in_row == from_color] = to_color

                # Check match
                if np.array_equal(mapped_in_row, out_row):
                    best_match = in_row_idx
                    best_score = 1.0
                    break

            row_mapping.append((best_match, best_score))

        # Check if we found a valid pattern
        avg_score = np.mean([score for _, score in row_mapping])

        if avg_score >= 0.7:
            extracted_rows = [idx for idx, score in row_mapping if score > 0.5]

            return {
                'transformation_type': 'pattern_extraction',
                'parameters': {
                    'axis': 'rows',
                    'row_mapping': extracted_rows,
                    'color_mapping': color_mapping,
                    'extension_factor': h_out / h_in
                },
                'confidence': avg_score,
                'description': f'Row pattern extraction with {len(set(extracted_rows))} unique rows'
            }

        return None

    def _find_column_pattern(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """Find if output columns are extracted/repeated from input."""
        # Transpose and use row logic
        result = self._find_row_pattern(input_grid.T, output_grid.T)

        if result:
            result['parameters']['axis'] = 'columns'
            result['description'] = result['description'].replace('Row', 'Column')
            return result

        return None

    def _learn_color_mapping_loose(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Dict[int, int]:
        """Learn color mapping even when shapes don't match exactly."""
        mapping = {}

        # Count color co-occurrences
        in_colors = np.unique(input_grid)
        out_colors = np.unique(output_grid)

        # Simple heuristic: map colors based on frequency similarity
        in_counts = {c: np.sum(input_grid == c) for c in in_colors}
        out_counts = {c: np.sum(output_grid == c) for c in out_colors}

        # Sort by frequency
        in_sorted = sorted(in_counts.items(), key=lambda x: x[1], reverse=True)
        out_sorted = sorted(out_counts.items(), key=lambda x: x[1], reverse=True)

        # Map in order
        for (in_color, _), (out_color, _) in zip(in_sorted, out_sorted):
            mapping[int(in_color)] = int(out_color)

        return mapping

    def _check_object_operations(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Check for object-level operations.
        Detects if objects are moved, rotated, or transformed individually.
        """
        if input_grid.shape != output_grid.shape:
            return None  # Object ops usually preserve shape

        # Detect objects in input
        in_objects = self._detect_objects(input_grid)
        out_objects = self._detect_objects(output_grid)

        if len(in_objects) == 0 or len(out_objects) == 0:
            return None

        if len(in_objects) != len(out_objects):
            # Different number of objects - might be filtering/adding
            return None

        # Try to match objects
        object_transformations = []

        for in_obj in in_objects:
            best_match = None
            best_score = 0.0

            for out_obj in out_objects:
                # Check if same color
                if in_obj['color'] == out_obj['color']:
                    # Check if same shape (possibly translated)
                    score = self._object_similarity(in_obj, out_obj)
                    if score > best_score:
                        best_score = score
                        best_match = out_obj

            if best_match:
                # Compute transformation
                dy = best_match['center'][0] - in_obj['center'][0]
                dx = best_match['center'][1] - in_obj['center'][1]

                object_transformations.append({
                    'type': 'translate',
                    'offset': (dy, dx),
                    'score': best_score
                })

        if object_transformations:
            avg_score = np.mean([t['score'] for t in object_transformations])

            if avg_score >= 0.7:
                # Check if all objects have same transformation
                offsets = [t['offset'] for t in object_transformations]
                unique_offsets = set(offsets)

                if len(unique_offsets) == 1:
                    return {
                        'transformation_type': 'object_translate_all',
                        'parameters': {'offset': offsets[0]},
                        'confidence': avg_score,
                        'description': f'All objects translated by {offsets[0]}'
                    }
                else:
                    return {
                        'transformation_type': 'object_operations',
                        'parameters': {'per_object_transforms': object_transformations},
                        'confidence': avg_score * 0.8,
                        'description': f'{len(in_objects)} objects with individual transformations'
                    }

        return None

    def _detect_objects(self, grid: np.ndarray) -> List[Dict]:
        """Detect connected component objects in grid."""
        objects = []

        # For each non-zero color
        colors = np.unique(grid)
        colors = colors[colors != 0]

        for color in colors:
            # Find connected components
            mask = (grid == color).astype(int)
            labeled, num_objects = ndimage.label(mask)

            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)
                coords = np.argwhere(obj_mask)

                if len(coords) > 0:
                    center = coords.mean(axis=0)

                    objects.append({
                        'color': int(color),
                        'coords': coords,
                        'center': center,
                        'size': len(coords),
                        'mask': obj_mask
                    })

        return objects

    def _object_similarity(self, obj1: Dict, obj2: Dict) -> float:
        """Compute similarity between two objects."""
        if obj1['color'] != obj2['color']:
            return 0.0

        if obj1['size'] != obj2['size']:
            return 0.0

        # Check if shapes match (allowing translation)
        # Normalize to same reference point
        coords1 = obj1['coords'] - obj1['coords'].min(axis=0)
        coords2 = obj2['coords'] - obj2['coords'].min(axis=0)

        # Sort coordinates
        coords1 = coords1[np.lexsort((coords1[:, 1], coords1[:, 0]))]
        coords2 = coords2[np.lexsort((coords2[:, 1], coords2[:, 0]))]

        if np.array_equal(coords1, coords2):
            return 1.0

        return 0.0

    def _check_conditional_fill(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Check for conditional fill operations.
        E.g., "fill all cells of color X with pattern Y"
        """
        # This is complex - skip for now
        return None

    def _check_spatial_pattern(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> Optional[Dict]:
        """
        Check for spatial pattern transformations.
        E.g., "move all objects to opposite quadrant"
        """
        # This is complex - skip for now
        return None


# Update the node creation function
from core.node import Node, NodeOutput


def enhanced_differential_analyzer_func(data: Dict[str, Any]) -> NodeOutput:
    """Enhanced differential analysis function for node."""
    artifacts = {}
    telemetry = {'node_type': 'extractor', 'subtype': 'enhanced_differential_analyzer'}

    train_examples = data.get('train', [])

    if not train_examples:
        telemetry['success'] = False
        return NodeOutput(result=[], artifacts=artifacts, telemetry=telemetry)

    analyzer = EnhancedDifferentialAnalyzer()
    analyses = []

    for i, (input_grid, output_grid) in enumerate(train_examples):
        analysis = analyzer.analyze(input_grid, output_grid)
        analyses.append(analysis)

        # Log to telemetry
        telemetry[f'example_{i}_type'] = analysis['transformation_type']
        telemetry[f'example_{i}_confidence'] = analysis['confidence']

    artifacts['transformation_analyses'] = analyses

    # Compute consensus
    types = [a['transformation_type'] for a in analyses]
    if types:
        from collections import Counter
        most_common_type = Counter(types).most_common(1)[0][0]
        consensus_count = sum(1 for t in types if t == most_common_type)
        consensus_ratio = consensus_count / len(types)

        artifacts['consensus_type'] = most_common_type
        artifacts['consensus_ratio'] = consensus_ratio
        telemetry['consensus_type'] = most_common_type
        telemetry['consensus_ratio'] = consensus_ratio

    telemetry['success'] = True
    telemetry['num_examples_analyzed'] = len(analyses)

    return NodeOutput(result=analyses, artifacts=artifacts, telemetry=telemetry)


def create_enhanced_differential_analyzer_node() -> Node:
    """Create an enhanced differential analyzer node."""
    return Node(
        name="enhanced_differential_analyzer",
        func=enhanced_differential_analyzer_func,
        input_type="task_data",
        output_type="transformation_analyses",
        deterministic=True,
        category="extractor"
    )
