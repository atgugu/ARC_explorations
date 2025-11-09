"""
Grid utilities for ARC tasks.
"""

import numpy as np
from typing import Tuple, List, Optional, Set
from dataclasses import dataclass


# ARC color palette
ARC_COLORS = {
    0: (0, 0, 0),         # Black
    1: (0, 116, 217),     # Blue
    2: (255, 65, 54),     # Red
    3: (46, 204, 64),     # Green
    4: (255, 220, 0),     # Yellow
    5: (170, 170, 170),   # Gray
    6: (240, 18, 190),    # Magenta
    7: (255, 133, 27),    # Orange
    8: (127, 219, 255),   # Azure
    9: (135, 12, 37),     # Maroon
}


@dataclass
class Grid:
    """Wrapper for grid operations."""

    data: np.ndarray

    def __init__(self, data):
        """Initialize with numpy array or list."""
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.int32)
        else:
            self.data = data.astype(np.int32)

    @property
    def shape(self) -> Tuple[int, int]:
        """Get grid shape."""
        return self.data.shape

    @property
    def height(self) -> int:
        """Get grid height."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Get grid width."""
        return self.data.shape[1]

    def copy(self) -> 'Grid':
        """Create a copy of the grid."""
        return Grid(self.data.copy())

    def get_colors(self) -> Set[int]:
        """Get unique colors in the grid."""
        return set(self.data.flatten())

    def count_color(self, color: int) -> int:
        """Count occurrences of a color."""
        return int(np.sum(self.data == color))

    def get_color_histogram(self) -> np.ndarray:
        """Get histogram of colors (10-dimensional)."""
        hist = np.zeros(10, dtype=np.int32)
        for color in range(10):
            hist[color] = self.count_color(color)
        return hist

    def crop(self, y1: int, y2: int, x1: int, x2: int) -> 'Grid':
        """Crop the grid."""
        return Grid(self.data[y1:y2, x1:x2])

    def find_objects(self, background_color: int = 0) -> List[np.ndarray]:
        """
        Find connected components (objects) in the grid.

        Args:
            background_color: Color to treat as background

        Returns:
            List of binary masks for each object
        """
        from scipy import ndimage

        # Create binary mask (non-background)
        binary = self.data != background_color

        # Label connected components
        labeled, num_features = ndimage.label(binary)

        # Extract each object
        objects = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            objects.append(mask)

        return objects

    def apply_mask(self, mask: np.ndarray, color: int) -> 'Grid':
        """Apply a mask to set colors."""
        result = self.copy()
        result.data[mask] = color
        return result

    def rotate_90(self, k: int = 1) -> 'Grid':
        """Rotate grid 90 degrees k times."""
        return Grid(np.rot90(self.data, k))

    def flip_horizontal(self) -> 'Grid':
        """Flip grid horizontally."""
        return Grid(np.fliplr(self.data))

    def flip_vertical(self) -> 'Grid':
        """Flip grid vertically."""
        return Grid(np.flipud(self.data))

    def has_symmetry_vertical(self) -> bool:
        """Check if grid has vertical symmetry."""
        return np.array_equal(self.data, np.fliplr(self.data))

    def has_symmetry_horizontal(self) -> bool:
        """Check if grid has horizontal symmetry."""
        return np.array_equal(self.data, np.flipud(self.data))

    def has_symmetry_diagonal(self) -> bool:
        """Check if grid has diagonal symmetry."""
        if self.height != self.width:
            return False
        return np.array_equal(self.data, self.data.T)

    def to_array(self) -> np.ndarray:
        """Get underlying numpy array."""
        return self.data


def visualize_grid(grid: np.ndarray, return_string: bool = True) -> Optional[str]:
    """
    Create a simple ASCII visualization of a grid.

    Args:
        grid: Numpy array representing the grid
        return_string: If True, return string; otherwise print

    Returns:
        String representation if return_string=True
    """
    if isinstance(grid, Grid):
        grid = grid.data

    # Simple character mapping
    chars = ' ●◆◇○◎▪▫♦♢'

    lines = []
    for row in grid:
        line = ''.join(chars[val] if val < len(chars) else '?' for val in row)
        lines.append(line)

    result = '\n'.join(lines)

    if return_string:
        return result
    else:
        print(result)


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Intersection over Union between prediction and target.

    Args:
        pred: Predicted grid
        target: Target grid

    Returns:
        IoU score (0-1)
    """
    if pred.shape != target.shape:
        return 0.0

    intersection = np.sum(pred == target)
    union = pred.size

    return intersection / union if union > 0 else 0.0


def compute_hamming_distance(pred: np.ndarray, target: np.ndarray) -> int:
    """
    Compute Hamming distance between grids.

    Args:
        pred: Predicted grid
        target: Target grid

    Returns:
        Number of differing cells
    """
    if pred.shape != target.shape:
        return pred.size + target.size

    return int(np.sum(pred != target))


def extract_bounding_box(grid: np.ndarray, color: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box of a specific color.

    Args:
        grid: Input grid
        color: Color to find

    Returns:
        (y1, y2, x1, x2) or None if color not found
    """
    mask = grid == color
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return (y1, y2 + 1, x1, x2 + 1)


def detect_periodicity(grid: np.ndarray) -> Tuple[int, int]:
    """
    Detect if grid has periodic pattern.

    Args:
        grid: Input grid

    Returns:
        (period_y, period_x) where 0 means no periodicity
    """
    h, w = grid.shape

    # Check horizontal periodicity
    period_x = 0
    for px in range(1, w // 2 + 1):
        if w % px == 0:
            # Check if pattern repeats
            repeated = np.tile(grid[:, :px], (1, w // px))
            if np.array_equal(grid, repeated):
                period_x = px
                break

    # Check vertical periodicity
    period_y = 0
    for py in range(1, h // 2 + 1):
        if h % py == 0:
            repeated = np.tile(grid[:py, :], (h // py, 1))
            if np.array_equal(grid, repeated):
                period_y = py
                break

    return (period_y, period_x)
