"""
Differential Analyzer: Analyzes what changed between input and output grids.
This is the foundation of V3's example-driven approach.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput
from utils.grid_utils import compute_iou


class DifferentialAnalyzer:
    """
    Analyzes the transformation between input and output grids.
    Detects specific changes rather than just extracting features.
    """

    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """
        Analyze what transformation occurred.

        Returns:
            Dictionary with:
            - transformation_type: Main category
            - parameters: Specific details
            - confidence: How certain we are
            - description: Human-readable explanation
        """
        results = []

        # Check all transformation types
        results.append(self._check_identity(input_grid, output_grid))
        results.append(self._check_geometric(input_grid, output_grid))
        results.append(self._check_color_remap(input_grid, output_grid))
        results.append(self._check_translation(input_grid, output_grid))
        results.append(self._check_size_change(input_grid, output_grid))
        results.append(self._check_tiling(input_grid, output_grid))

        # Sort by confidence and return best match
        results = [r for r in results if r is not None]
        results.sort(key=lambda x: x['confidence'], reverse=True)

        if len(results) > 0:
            return results[0]
        else:
            return {
                'transformation_type': 'unknown',
                'parameters': {},
                'confidence': 0.0,
                'description': 'No clear transformation detected'
            }

    def _check_identity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Check if output is identical to input."""
        if input_grid.shape != output_grid.shape:
            return None

        iou = compute_iou(input_grid, output_grid)
        if iou >= 0.99:
            return {
                'transformation_type': 'identity',
                'parameters': {},
                'confidence': 1.0,
                'description': 'Output is identical to input'
            }
        return None

    def _check_geometric(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Check for rotations and flips."""
        if input_grid.shape != output_grid.shape:
            # Rotation by 90 or 270 changes shape
            h_in, w_in = input_grid.shape
            h_out, w_out = output_grid.shape

            if h_in == w_out and w_in == h_out:
                # Possible 90 or 270 rotation
                # Check 90
                rotated_90 = np.rot90(input_grid, k=-1)
                if np.array_equal(rotated_90, output_grid):
                    return {
                        'transformation_type': 'rotate',
                        'parameters': {'angle': 90},
                        'confidence': 1.0,
                        'description': 'Rotate 90° clockwise'
                    }

                # Check 270
                rotated_270 = np.rot90(input_grid, k=-3)
                if np.array_equal(rotated_270, output_grid):
                    return {
                        'transformation_type': 'rotate',
                        'parameters': {'angle': 270},
                        'confidence': 1.0,
                        'description': 'Rotate 270° clockwise'
                    }
        else:
            # Same shape - check 180, flips, transpose
            # Check 180
            rotated_180 = np.rot90(input_grid, k=2)
            if np.array_equal(rotated_180, output_grid):
                return {
                    'transformation_type': 'rotate',
                    'parameters': {'angle': 180},
                    'confidence': 1.0,
                    'description': 'Rotate 180°'
                }

            # Check horizontal flip
            flipped_h = np.fliplr(input_grid)
            if np.array_equal(flipped_h, output_grid):
                return {
                    'transformation_type': 'flip',
                    'parameters': {'axis': 'horizontal'},
                    'confidence': 1.0,
                    'description': 'Flip horizontally'
                }

            # Check vertical flip
            flipped_v = np.flipud(input_grid)
            if np.array_equal(flipped_v, output_grid):
                return {
                    'transformation_type': 'flip',
                    'parameters': {'axis': 'vertical'},
                    'confidence': 1.0,
                    'description': 'Flip vertically'
                }

            # Check transpose
            transposed = input_grid.T
            if np.array_equal(transposed, output_grid):
                return {
                    'transformation_type': 'transpose',
                    'parameters': {},
                    'confidence': 1.0,
                    'description': 'Transpose grid'
                }

        return None

    def _check_color_remap(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Check if it's a color remapping (same structure, different colors)."""
        if input_grid.shape != output_grid.shape:
            return None

        # Learn color mapping
        mapping = {}
        reverse_mapping = {}

        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = int(input_grid[i, j])
                out_color = int(output_grid[i, j])

                # Check for conflicts
                if in_color in mapping:
                    if mapping[in_color] != out_color:
                        # Same input color maps to different output colors - not a valid mapping
                        return None
                else:
                    mapping[in_color] = out_color

                # Also check reverse (for bijection)
                if out_color in reverse_mapping:
                    if reverse_mapping[out_color] != in_color:
                        # Multiple input colors map to same output - still valid, but lower confidence
                        pass
                else:
                    reverse_mapping[out_color] = in_color

        # Apply mapping and check if it produces output
        remapped = np.vectorize(mapping.get)(input_grid)
        accuracy = np.mean(remapped == output_grid)

        if accuracy >= 0.99:
            # Check if it's a trivial mapping (identity on colors)
            is_identity_mapping = all(k == v for k, v in mapping.items())

            if is_identity_mapping:
                return None  # This is just identity, handled elsewhere

            return {
                'transformation_type': 'color_remap',
                'parameters': {'mapping': mapping, 'is_bijection': len(mapping) == len(reverse_mapping)},
                'confidence': 0.95,
                'description': f'Color remapping: {mapping}'
            }

        return None

    def _check_translation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """
        Check if output is a translation of input.
        This handles cases where the grid size might change or stay the same.
        """
        # If shapes are different, might be translation with different canvas size
        # For simplicity, check if same shape first

        if input_grid.shape != output_grid.shape:
            # Could be translation with cropping/padding - too complex for now
            return None

        # Try small offsets
        best_offset = None
        best_iou = 0.0

        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dy == 0 and dx == 0:
                    continue

                # Translate input
                translated = np.zeros_like(output_grid)

                # Source region
                src_y_start = max(0, -dy)
                src_y_end = min(input_grid.shape[0], input_grid.shape[0] - dy)
                src_x_start = max(0, -dx)
                src_x_end = min(input_grid.shape[1], input_grid.shape[1] - dx)

                # Destination region
                dst_y_start = max(0, dy)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_start = max(0, dx)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)

                if src_y_end > src_y_start and src_x_end > src_x_start:
                    translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        input_grid[src_y_start:src_y_end, src_x_start:src_x_end]

                    iou = compute_iou(translated, output_grid)
                    if iou > best_iou:
                        best_iou = iou
                        best_offset = (dy, dx)

        if best_iou >= 0.90:
            dy, dx = best_offset
            return {
                'transformation_type': 'translate',
                'parameters': {'offset': (dy, dx)},
                'confidence': min(best_iou, 0.95),
                'description': f'Translate by ({dy}, {dx})'
            }

        return None

    def _check_size_change(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Check for scaling (repeating pixels) or downsampling."""
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape

        # Check for upscaling (2x, 3x)
        for factor in [2, 3]:
            if h_out == h_in * factor and w_out == w_in * factor:
                # Check if it's pixel repetition
                scaled = np.repeat(np.repeat(input_grid, factor, axis=0), factor, axis=1)
                if np.array_equal(scaled, output_grid):
                    return {
                        'transformation_type': 'scale',
                        'parameters': {'factor': factor},
                        'confidence': 1.0,
                        'description': f'Scale {factor}x (repeat pixels)'
                    }

        # Check for downsampling
        for factor in [2, 3]:
            if h_in == h_out * factor and w_in == w_out * factor:
                downsampled = input_grid[::factor, ::factor]
                if np.array_equal(downsampled, output_grid):
                    return {
                        'transformation_type': 'downsample',
                        'parameters': {'factor': factor},
                        'confidence': 1.0,
                        'description': f'Downsample {factor}x'
                    }

        return None

    def _check_tiling(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Check if output is a tiling of input."""
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape

        # Check for 2x2, 3x3, 2x1, 1x2 tiling
        for tile_h in [1, 2, 3]:
            for tile_w in [1, 2, 3]:
                if tile_h == 1 and tile_w == 1:
                    continue

                if h_out == h_in * tile_h and w_out == w_in * tile_w:
                    tiled = np.tile(input_grid, (tile_h, tile_w))
                    if np.array_equal(tiled, output_grid):
                        return {
                            'transformation_type': 'tile',
                            'parameters': {'tile_h': tile_h, 'tile_w': tile_w},
                            'confidence': 1.0,
                            'description': f'Tile {tile_h}x{tile_w}'
                        }

        return None


def differential_analyzer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Differential analysis function for node.

    Args:
        data: Dictionary with 'train' examples (list of (input, output) pairs)

    Returns:
        NodeOutput with analysis for each training example
    """
    artifacts = {}
    telemetry = {'node_type': 'extractor', 'subtype': 'differential_analyzer'}

    train_examples = data.get('train', [])

    if not train_examples:
        telemetry['success'] = False
        return NodeOutput(result=[], artifacts=artifacts, telemetry=telemetry)

    analyzer = DifferentialAnalyzer()
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


def create_differential_analyzer_node() -> Node:
    """Create a differential analyzer node."""
    return Node(
        name="differential_analyzer",
        func=differential_analyzer_func,
        input_type="task_data",
        output_type="transformation_analyses",
        deterministic=True,
        category="extractor"
    )
