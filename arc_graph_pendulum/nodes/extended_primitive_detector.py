"""
Extended Primitive Detector - Detects which extended primitives apply to a task.

Analyzes training examples to determine which of the 20 new primitives might
solve the transformation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from nodes.extended_primitives import ExtendedPrimitives


class ExtendedPrimitiveDetector:
    """Detects applicable extended primitives for a task."""

    def __init__(self):
        """Initialize detector."""
        self.primitives = ExtendedPrimitives()

    def detect(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Detect which extended primitives might apply.

        Args:
            train_examples: List of (input, output) training pairs

        Returns:
            List of detected primitive applications with confidence scores
        """
        detections = []

        # Analyze each category
        detections.extend(self._detect_spatial(train_examples))
        detections.extend(self._detect_object_operations(train_examples))
        detections.extend(self._detect_color_operations(train_examples))
        detections.extend(self._detect_pattern_operations(train_examples))

        # Sort by confidence
        detections.sort(key=lambda d: d.get('confidence', 0), reverse=True)

        return detections

    def _detect_spatial(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """Detect spatial primitive applications."""
        detections = []

        # Check for positional object extraction
        for inp, out in train_examples:
            # If output is smaller and contains one object
            if out.size < inp.size and np.any(out != 0):
                # Check which position extraction matches
                leftmost = self.primitives.extract_leftmost_object(inp)
                if np.array_equal(leftmost, out):
                    detections.append({
                        'primitive': 'extract_leftmost_object',
                        'confidence': 0.95,
                        'description': 'Extract leftmost object'
                    })
                    break

                rightmost = self.primitives.extract_rightmost_object(inp)
                if np.array_equal(rightmost, out):
                    detections.append({
                        'primitive': 'extract_rightmost_object',
                        'confidence': 0.95,
                        'description': 'Extract rightmost object'
                    })
                    break

                topmost = self.primitives.extract_topmost_object(inp)
                if np.array_equal(topmost, out):
                    detections.append({
                        'primitive': 'extract_topmost_object',
                        'confidence': 0.95,
                        'description': 'Extract topmost object'
                    })
                    break

                bottommost = self.primitives.extract_bottommost_object(inp)
                if np.array_equal(bottommost, out):
                    detections.append({
                        'primitive': 'extract_bottommost_object',
                        'confidence': 0.95,
                        'description': 'Extract bottommost object'
                    })
                    break

        # Check for grid alignment
        if self._looks_like_grid_alignment(train_examples):
            detections.append({
                'primitive': 'align_objects_to_grid',
                'confidence': 0.85,
                'description': 'Align objects to regular grid',
                'params': {'spacing': 2}
            })

        return detections

    def _detect_object_operations(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """Detect object operation applications."""
        detections = []

        for inp, out in train_examples:
            # Check for horizontal repetition
            if out.shape[1] > inp.shape[1] and out.shape[0] == inp.shape[0]:
                ratio = out.shape[1] / inp.shape[1]
                if abs(ratio - round(ratio)) < 0.1:
                    n = int(round(ratio))
                    test = self.primitives.copy_object_horizontal(inp, n=n)
                    if np.array_equal(test, out):
                        detections.append({
                            'primitive': 'copy_object_horizontal',
                            'confidence': 0.92,
                            'description': f'Copy object {n} times horizontally',
                            'params': {'n': n}
                        })
                        break

            # Check for vertical repetition
            if out.shape[0] > inp.shape[0] and out.shape[1] == inp.shape[1]:
                ratio = out.shape[0] / inp.shape[0]
                if abs(ratio - round(ratio)) < 0.1:
                    n = int(round(ratio))
                    test = self.primitives.copy_object_vertical(inp, n=n)
                    if np.array_equal(test, out):
                        detections.append({
                            'primitive': 'copy_object_vertical',
                            'confidence': 0.92,
                            'description': f'Copy object {n} times vertically',
                            'params': {'n': n}
                        })
                        break

        # Check for connecting objects
        if self._looks_like_connection(train_examples):
            detections.append({
                'primitive': 'connect_objects_with_line',
                'confidence': 0.80,
                'description': 'Connect objects with lines',
                'params': {'color': 1}
            })

        return detections

    def _detect_color_operations(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """Detect color operation applications."""
        detections = []

        for inp, out in train_examples:
            if inp.shape != out.shape:
                continue

            # Check for row-based recoloring
            if self._check_row_gradient(inp, out):
                detections.append({
                    'primitive': 'recolor_by_row',
                    'confidence': 0.88,
                    'description': 'Recolor by row (gradient)',
                    'params': {'start_color': 1}
                })
                break

            # Check for column-based recoloring
            if self._check_column_gradient(inp, out):
                detections.append({
                    'primitive': 'recolor_by_column',
                    'confidence': 0.88,
                    'description': 'Recolor by column (gradient)',
                    'params': {'start_color': 1}
                })
                break

            # Check for checkerboard
            if self._check_checkerboard(inp, out):
                detections.append({
                    'primitive': 'recolor_checkerboard',
                    'confidence': 0.90,
                    'description': 'Recolor in checkerboard pattern',
                    'params': {'color1': 1, 'color2': 2}
                })
                break

        return detections

    def _detect_pattern_operations(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """Detect pattern operation applications."""
        detections = []

        for inp, out in train_examples:
            if inp.shape != out.shape:
                continue

            # Check for horizontal symmetry
            if self._check_horizontal_symmetry(out):
                test = self.primitives.apply_horizontal_symmetry(inp)
                if np.array_equal(test, out):
                    detections.append({
                        'primitive': 'apply_horizontal_symmetry',
                        'confidence': 0.93,
                        'description': 'Apply horizontal (left-right) symmetry'
                    })
                    break

            # Check for vertical symmetry
            if self._check_vertical_symmetry(out):
                test = self.primitives.apply_vertical_symmetry(inp)
                if np.array_equal(test, out):
                    detections.append({
                        'primitive': 'apply_vertical_symmetry',
                        'confidence': 0.93,
                        'description': 'Apply vertical (top-bottom) symmetry'
                    })
                    break

        return detections

    # Helper methods

    def _looks_like_grid_alignment(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if examples look like grid alignment."""
        for inp, out in train_examples:
            # Regular spacing in output
            if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
                # Output is larger and possibly has regular structure
                return True
        return False

    def _looks_like_connection(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if examples look like objects being connected."""
        for inp, out in train_examples:
            if inp.shape == out.shape:
                # More non-zero pixels in output (lines added)
                inp_count = np.sum(inp != 0)
                out_count = np.sum(out != 0)
                if out_count > inp_count:
                    return True
        return False

    def _check_row_gradient(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output has row-based gradient."""
        # Different colors in different rows
        for i in range(out.shape[0]):
            row_colors = set(out[i, :][out[i, :] != 0])
            if len(row_colors) > 1:
                return False  # Multiple colors in one row

        # Each row should have different color
        row_colors_list = [set(out[i, :][out[i, :] != 0]) for i in range(out.shape[0])]
        row_colors_list = [c for c in row_colors_list if c]

        if len(row_colors_list) > 1:
            # Check if they're different
            return len(set(tuple(c) for c in row_colors_list)) > 1

        return False

    def _check_column_gradient(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output has column-based gradient."""
        for j in range(out.shape[1]):
            col_colors = set(out[:, j][out[:, j] != 0])
            if len(col_colors) > 1:
                return False

        col_colors_list = [set(out[:, j][out[:, j] != 0]) for j in range(out.shape[1])]
        col_colors_list = [c for c in col_colors_list if c]

        if len(col_colors_list) > 1:
            return len(set(tuple(c) for c in col_colors_list)) > 1

        return False

    def _check_checkerboard(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output has checkerboard pattern."""
        # Sample a few positions
        colors_even = set()
        colors_odd = set()

        for i in range(min(5, out.shape[0])):
            for j in range(min(5, out.shape[1])):
                if out[i, j] != 0:
                    if (i + j) % 2 == 0:
                        colors_even.add(out[i, j])
                    else:
                        colors_odd.add(out[i, j])

        # Checkerboard has at most 2 colors in specific pattern
        return len(colors_even) <= 2 and len(colors_odd) <= 2 and colors_even != colors_odd

    def _check_horizontal_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid is horizontally symmetric."""
        return np.array_equal(grid, np.fliplr(grid))

    def _check_vertical_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid is vertically symmetric."""
        return np.array_equal(grid, np.flipud(grid))
