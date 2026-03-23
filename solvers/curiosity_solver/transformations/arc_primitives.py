"""
ARC-Specific Transformation Primitives

Implements basic transformation operations for ARC tasks:
- Spatial: translate, rotate, reflect, scale
- Color: recolor, swap, filter
- Logical: AND, OR, conditional
- Topological: connect, separate, fill
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
from scipy.ndimage import rotate as scipy_rotate, zoom


@dataclass
class Transform:
    """Represents a transformation operation."""
    name: str
    function: Callable
    parameters: dict
    category: str  # 'spatial', 'color', 'logical', 'topological'


class ARCPrimitives:
    """Collection of primitive ARC transformation operations."""

    # ========== Spatial Transformations ==========

    @staticmethod
    def translate(grid: np.ndarray, dx: int, dy: int, fill_value: int = 0) -> np.ndarray:
        """Translate grid by (dx, dy)."""
        h, w = grid.shape
        result = np.full_like(grid, fill_value)

        # Compute valid source and destination regions
        src_y_start = max(0, -dy)
        src_y_end = min(h, h - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(w, w - dx)

        dst_y_start = max(0, dy)
        dst_y_end = min(h, h + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(w, w + dx)

        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            grid[src_y_start:src_y_end, src_x_start:src_x_end]

        return result

    @staticmethod
    def rotate_90(grid: np.ndarray, times: int = 1) -> np.ndarray:
        """Rotate grid 90 degrees clockwise, times times."""
        return np.rot90(grid, k=-times % 4)

    @staticmethod
    def reflect_horizontal(grid: np.ndarray) -> np.ndarray:
        """Reflect grid horizontally."""
        return np.fliplr(grid)

    @staticmethod
    def reflect_vertical(grid: np.ndarray) -> np.ndarray:
        """Reflect grid vertically."""
        return np.flipud(grid)

    @staticmethod
    def scale(grid: np.ndarray, factor: int) -> np.ndarray:
        """Scale grid by repeating pixels."""
        if factor <= 0:
            return grid

        h, w = grid.shape
        result = np.zeros((h * factor, w * factor), dtype=grid.dtype)

        for i in range(h):
            for j in range(w):
                result[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = grid[i, j]

        return result

    # ========== Color Transformations ==========

    @staticmethod
    def recolor(grid: np.ndarray, color_map: dict) -> np.ndarray:
        """Recolor grid according to mapping."""
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
        return result

    @staticmethod
    def recolor_by_condition(grid: np.ndarray,
                            condition_fn: Callable,
                            new_color: int) -> np.ndarray:
        """Recolor cells matching condition."""
        result = grid.copy()
        mask = condition_fn(grid)
        result[mask] = new_color
        return result

    @staticmethod
    def swap_colors(grid: np.ndarray, color1: int, color2: int) -> np.ndarray:
        """Swap two colors."""
        result = grid.copy()
        mask1 = grid == color1
        mask2 = grid == color2
        result[mask1] = color2
        result[mask2] = color1
        return result

    @staticmethod
    def filter_color(grid: np.ndarray, keep_color: int, fill_value: int = 0) -> np.ndarray:
        """Keep only specified color, fill rest."""
        result = np.full_like(grid, fill_value)
        result[grid == keep_color] = keep_color
        return result

    # ========== Logical Operations ==========

    @staticmethod
    def logical_and(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Element-wise AND (non-zero AND non-zero)."""
        return ((grid1 != 0) & (grid2 != 0)).astype(grid1.dtype) * grid1

    @staticmethod
    def logical_or(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Element-wise OR (take non-zero value)."""
        result = grid1.copy()
        result[grid2 != 0] = grid2[grid2 != 0]
        return result

    @staticmethod
    def logical_xor(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Element-wise XOR."""
        return ((grid1 != 0) ^ (grid2 != 0)).astype(grid1.dtype)

    # ========== Topological Operations ==========

    @staticmethod
    def fill_region(grid: np.ndarray, start_pos: Tuple[int, int], fill_color: int) -> np.ndarray:
        """Flood fill from start position."""
        result = grid.copy()
        h, w = grid.shape
        y, x = start_pos

        if not (0 <= y < h and 0 <= x < w):
            return result

        target_color = grid[y, x]
        if target_color == fill_color:
            return result

        # BFS flood fill
        from collections import deque
        queue = deque([(y, x)])
        visited = set()

        while queue:
            cy, cx = queue.popleft()

            if (cy, cx) in visited:
                continue
            if not (0 <= cy < h and 0 <= cx < w):
                continue
            if result[cy, cx] != target_color:
                continue

            visited.add((cy, cx))
            result[cy, cx] = fill_color

            # Add neighbors
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                queue.append((cy + dy, cx + dx))

        return result

    @staticmethod
    def detect_objects(grid: np.ndarray, connectivity: int = 4) -> List[np.ndarray]:
        """
        Detect connected components (objects) in grid.

        Args:
            grid: Input grid
            connectivity: 4 or 8 (neighbor connectivity)

        Returns:
            List of binary masks for each object
        """
        from scipy.ndimage import label

        # Consider non-zero pixels as foreground
        foreground = grid != 0

        if connectivity == 4:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:
            structure = np.ones((3, 3))

        labeled, n_objects = label(foreground, structure=structure)

        objects = []
        for i in range(1, n_objects + 1):
            mask = labeled == i
            objects.append(mask)

        return objects

    @staticmethod
    def extract_object(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract object using mask."""
        result = np.zeros_like(grid)
        result[mask] = grid[mask]
        return result

    @staticmethod
    def bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of mask (y1, x1, y2, x2)."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return (0, 0, 0, 0)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return (int(y1), int(x1), int(y2 + 1), int(x2 + 1))

    # ========== Pattern Operations ==========

    @staticmethod
    def tile_pattern(pattern: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Tile pattern to fill target shape."""
        h, w = pattern.shape
        th, tw = target_shape

        result = np.zeros(target_shape, dtype=pattern.dtype)

        for i in range(0, th, h):
            for j in range(0, tw, w):
                end_i = min(i + h, th)
                end_j = min(j + w, tw)
                result[i:end_i, j:end_j] = pattern[:end_i-i, :end_j-j]

        return result

    @staticmethod
    def count_colors(grid: np.ndarray) -> dict:
        """Count occurrences of each color."""
        unique, counts = np.unique(grid, return_counts=True)
        return dict(zip(unique, counts))

    @staticmethod
    def most_common_color(grid: np.ndarray, exclude: int = 0) -> int:
        """Get most common color (excluding background)."""
        counts = ARCPrimitives.count_colors(grid)
        if exclude in counts:
            del counts[exclude]

        if not counts:
            return 0

        return max(counts.items(), key=lambda x: x[1])[0]


class CompositeTransform:
    """
    Composite transformation combining multiple primitives.
    """

    def __init__(self, name: str = "composite"):
        self.name = name
        self.transforms: List[Transform] = []

    def add(self, transform: Transform):
        """Add a transformation to the sequence."""
        self.transforms.append(transform)

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply all transformations in sequence."""
        result = grid.copy()

        for transform in self.transforms:
            try:
                result = transform.function(result, **transform.parameters)
            except Exception as e:
                # Skip failed transformations
                continue

        return result

    def __repr__(self):
        transform_names = [t.name for t in self.transforms]
        return f"CompositeTransform({' -> '.join(transform_names)})"


class TransformLibrary:
    """
    Library of all available transformations.
    """

    def __init__(self):
        self.primitives = ARCPrimitives()
        self.library = self._build_library()

    def _build_library(self) -> dict:
        """Build library of named transforms."""
        lib = {}

        # Spatial
        lib['translate_right'] = Transform('translate_right', self.primitives.translate,
                                          {'dx': 1, 'dy': 0}, 'spatial')
        lib['translate_left'] = Transform('translate_left', self.primitives.translate,
                                         {'dx': -1, 'dy': 0}, 'spatial')
        lib['translate_up'] = Transform('translate_up', self.primitives.translate,
                                       {'dx': 0, 'dy': -1}, 'spatial')
        lib['translate_down'] = Transform('translate_down', self.primitives.translate,
                                         {'dx': 0, 'dy': 1}, 'spatial')

        lib['rotate_90'] = Transform('rotate_90', self.primitives.rotate_90,
                                    {'times': 1}, 'spatial')
        lib['rotate_180'] = Transform('rotate_180', self.primitives.rotate_90,
                                     {'times': 2}, 'spatial')
        lib['rotate_270'] = Transform('rotate_270', self.primitives.rotate_90,
                                     {'times': 3}, 'spatial')

        lib['reflect_h'] = Transform('reflect_h', self.primitives.reflect_horizontal,
                                    {}, 'spatial')
        lib['reflect_v'] = Transform('reflect_v', self.primitives.reflect_vertical,
                                    {}, 'spatial')

        lib['scale_2x'] = Transform('scale_2x', self.primitives.scale,
                                   {'factor': 2}, 'spatial')
        lib['scale_3x'] = Transform('scale_3x', self.primitives.scale,
                                   {'factor': 3}, 'spatial')

        # Color (parameterized - need to be instantiated with specific params)
        # These are templates
        lib['swap_colors'] = Transform('swap_colors', self.primitives.swap_colors,
                                      {}, 'color')
        lib['recolor'] = Transform('recolor', self.primitives.recolor,
                                  {}, 'color')

        return lib

    def get_transform(self, name: str) -> Optional[Transform]:
        """Get transform by name."""
        return self.library.get(name)

    def get_all_spatial(self) -> List[Transform]:
        """Get all spatial transforms."""
        return [t for t in self.library.values() if t.category == 'spatial']

    def get_all_color(self) -> List[Transform]:
        """Get all color transforms."""
        return [t for t in self.library.values() if t.category == 'color']

    def create_composite(self, transform_names: List[str]) -> Optional[CompositeTransform]:
        """Create composite transformation from list of names."""
        composite = CompositeTransform()

        for name in transform_names:
            transform = self.get_transform(name)
            if transform:
                composite.add(transform)

        return composite if composite.transforms else None
