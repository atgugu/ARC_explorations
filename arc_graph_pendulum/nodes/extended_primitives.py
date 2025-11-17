"""
Extended Primitive Library - 20 New Transformation Primitives

Based on analysis of 37 high-quality tasks (0.80-0.95 IoU) that need more
precise transformations. Organized into 4 categories:

1. Spatial Operations (5): Position-based object selection and alignment
2. Object Operations (5): Multi-object manipulation and composition
3. Color Operations (5): Advanced color transformations
4. Pattern Operations (5): Symmetry and pattern completion

Expected impact: +2-3% evaluation solve rate (1.7% → 4-5%)
Target: Convert 8-12 of 37 high-quality tasks to perfect solves
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from scipy.ndimage import label, find_objects
from collections import Counter


class ExtendedPrimitives:
    """Extended primitive library with 20 new transformations."""

    def __init__(self):
        """Initialize extended primitives."""
        pass

    # =========================================================================
    # CATEGORY 1: SPATIAL OPERATIONS (5 primitives)
    # =========================================================================

    def extract_leftmost_object(self, grid: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Extract the leftmost non-background object.

        Args:
            grid: Input grid
            background: Background color (default 0)

        Returns:
            Tight bounding box of leftmost object
        """
        # Find all objects
        objects = self._detect_all_objects(grid, background)

        if not objects:
            return grid.copy()

        # Find leftmost (minimum column position)
        leftmost = min(objects, key=lambda obj: obj['bbox'][1])

        # Extract tight bounding box
        r1, c1, r2, c2 = leftmost['bbox']
        return grid[r1:r2+1, c1:c2+1].copy()

    def extract_rightmost_object(self, grid: np.ndarray, background: int = 0) -> np.ndarray:
        """Extract the rightmost non-background object."""
        objects = self._detect_all_objects(grid, background)

        if not objects:
            return grid.copy()

        # Find rightmost (maximum column position)
        rightmost = max(objects, key=lambda obj: obj['bbox'][3])

        r1, c1, r2, c2 = rightmost['bbox']
        return grid[r1:r2+1, c1:c2+1].copy()

    def extract_topmost_object(self, grid: np.ndarray, background: int = 0) -> np.ndarray:
        """Extract the topmost non-background object."""
        objects = self._detect_all_objects(grid, background)

        if not objects:
            return grid.copy()

        # Find topmost (minimum row position)
        topmost = min(objects, key=lambda obj: obj['bbox'][0])

        r1, c1, r2, c2 = topmost['bbox']
        return grid[r1:r2+1, c1:c2+1].copy()

    def extract_bottommost_object(self, grid: np.ndarray, background: int = 0) -> np.ndarray:
        """Extract the bottommost non-background object."""
        objects = self._detect_all_objects(grid, background)

        if not objects:
            return grid.copy()

        # Find bottommost (maximum row position)
        bottommost = max(objects, key=lambda obj: obj['bbox'][2])

        r1, c1, r2, c2 = bottommost['bbox']
        return grid[r1:r2+1, c1:c2+1].copy()

    def align_objects_to_grid(self, grid: np.ndarray, spacing: int = 2,
                             background: int = 0) -> np.ndarray:
        """
        Align objects to a regular grid with spacing.

        Args:
            grid: Input grid
            spacing: Grid spacing
            background: Background color

        Returns:
            Grid with objects aligned
        """
        objects = self._detect_all_objects(grid, background)

        if not objects:
            return grid.copy()

        # Calculate new grid size
        max_obj_height = max(obj['bbox'][2] - obj['bbox'][0] + 1 for obj in objects)
        max_obj_width = max(obj['bbox'][3] - obj['bbox'][1] + 1 for obj in objects)

        # Arrange objects in grid
        objects_per_row = int(np.sqrt(len(objects))) + 1
        new_height = objects_per_row * (max_obj_height + spacing)
        new_width = objects_per_row * (max_obj_width + spacing)

        result = np.full((new_height, new_width), background, dtype=np.int32)

        # Place objects
        for i, obj in enumerate(objects):
            row_idx = i // objects_per_row
            col_idx = i % objects_per_row

            r_start = row_idx * (max_obj_height + spacing)
            c_start = col_idx * (max_obj_width + spacing)

            # Extract object pixels
            r1, c1, r2, c2 = obj['bbox']
            obj_region = grid[r1:r2+1, c1:c2+1]

            # Place in result
            r_end = r_start + obj_region.shape[0]
            c_end = c_start + obj_region.shape[1]

            if r_end <= new_height and c_end <= new_width:
                result[r_start:r_end, c_start:c_end] = obj_region

        return result

    # =========================================================================
    # CATEGORY 2: OBJECT OPERATIONS (5 primitives)
    # =========================================================================

    def copy_object_horizontal(self, grid: np.ndarray, n: int = 3,
                               background: int = 0) -> np.ndarray:
        """
        Copy largest object N times horizontally.

        Args:
            grid: Input grid
            n: Number of copies (including original)
            background: Background color

        Returns:
            Grid with object repeated horizontally
        """
        objects = self._detect_all_objects(grid, background)

        if not objects:
            return grid.copy()

        # Get largest object
        largest = max(objects, key=lambda obj: obj['size'])
        r1, c1, r2, c2 = largest['bbox']
        obj_region = grid[r1:r2+1, c1:c2+1]

        # Create result grid
        obj_height = r2 - r1 + 1
        obj_width = c2 - c1 + 1
        new_width = obj_width * n
        new_height = obj_height

        result = np.full((new_height, new_width), background, dtype=np.int32)

        # Copy object n times
        for i in range(n):
            c_start = i * obj_width
            c_end = c_start + obj_width
            result[:, c_start:c_end] = obj_region

        return result

    def copy_object_vertical(self, grid: np.ndarray, n: int = 3,
                            background: int = 0) -> np.ndarray:
        """Copy largest object N times vertically."""
        objects = self._detect_all_objects(grid, background)

        if not objects:
            return grid.copy()

        largest = max(objects, key=lambda obj: obj['size'])
        r1, c1, r2, c2 = largest['bbox']
        obj_region = grid[r1:r2+1, c1:c2+1]

        obj_height = r2 - r1 + 1
        obj_width = c2 - c1 + 1
        new_height = obj_height * n
        new_width = obj_width

        result = np.full((new_height, new_width), background, dtype=np.int32)

        for i in range(n):
            r_start = i * obj_height
            r_end = r_start + obj_height
            result[r_start:r_end, :] = obj_region

        return result

    def connect_objects_with_line(self, grid: np.ndarray, color: int = 1,
                                  background: int = 0) -> np.ndarray:
        """
        Connect all objects with horizontal/vertical lines.

        Args:
            grid: Input grid
            color: Line color
            background: Background color

        Returns:
            Grid with objects connected
        """
        result = grid.copy()
        objects = self._detect_all_objects(grid, background)

        if len(objects) < 2:
            return result

        # Connect consecutive objects
        for i in range(len(objects) - 1):
            obj1 = objects[i]
            obj2 = objects[i + 1]

            # Get centers
            r1_center = (obj1['bbox'][0] + obj1['bbox'][2]) // 2
            c1_center = (obj1['bbox'][1] + obj1['bbox'][3]) // 2
            r2_center = (obj2['bbox'][0] + obj2['bbox'][2]) // 2
            c2_center = (obj2['bbox'][1] + obj2['bbox'][3]) // 2

            # Draw line (Manhattan path)
            # Horizontal first
            c_start, c_end = sorted([c1_center, c2_center])
            result[r1_center, c_start:c_end+1] = color

            # Then vertical
            r_start, r_end = sorted([r1_center, r2_center])
            result[r_start:r_end+1, c2_center] = color

        return result

    def object_intersection(self, grid: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Find intersection of overlapping colored regions.

        Args:
            grid: Input grid
            background: Background color

        Returns:
            Grid with only overlapping pixels
        """
        # Find all colors
        colors = [c for c in np.unique(grid) if c != background]

        if len(colors) < 2:
            return np.full_like(grid, background)

        # Create masks for each color
        masks = [grid == color for color in colors]

        # Intersection is where all masks are True
        intersection = np.all(masks, axis=0)

        result = np.full_like(grid, background)
        result[intersection] = colors[0]  # Use first color

        return result

    def object_union(self, grid: np.ndarray, output_color: int = 1,
                    background: int = 0) -> np.ndarray:
        """
        Create union of all colored regions.

        Args:
            grid: Input grid
            output_color: Color for union
            background: Background color

        Returns:
            Grid with union of all objects
        """
        result = np.full_like(grid, background)

        # Any non-background pixel becomes output_color
        non_bg = grid != background
        result[non_bg] = output_color

        return result

    # =========================================================================
    # CATEGORY 3: COLOR OPERATIONS (5 primitives)
    # =========================================================================

    def recolor_by_row(self, grid: np.ndarray, start_color: int = 1,
                       background: int = 0) -> np.ndarray:
        """
        Recolor each row with incrementing colors (gradient).

        Args:
            grid: Input grid
            start_color: Starting color
            background: Background color (preserved)

        Returns:
            Grid with row-based gradient
        """
        result = grid.copy()

        for i in range(grid.shape[0]):
            color = (start_color + i) % 10  # Cycle through 0-9
            if color == background:
                color = (color + 1) % 10

            # Recolor non-background pixels in this row
            row_mask = result[i, :] != background
            result[i, row_mask] = color

        return result

    def recolor_by_column(self, grid: np.ndarray, start_color: int = 1,
                         background: int = 0) -> np.ndarray:
        """Recolor each column with incrementing colors."""
        result = grid.copy()

        for j in range(grid.shape[1]):
            color = (start_color + j) % 10
            if color == background:
                color = (color + 1) % 10

            col_mask = result[:, j] != background
            result[col_mask, j] = color

        return result

    def recolor_checkerboard(self, grid: np.ndarray, color1: int = 1,
                            color2: int = 2, background: int = 0) -> np.ndarray:
        """
        Recolor non-background pixels in checkerboard pattern.

        Args:
            grid: Input grid
            color1: First checkerboard color
            color2: Second checkerboard color
            background: Background color (preserved)

        Returns:
            Grid with checkerboard coloring
        """
        result = grid.copy()

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if result[i, j] != background:
                    # Checkerboard: (i + j) % 2 determines color
                    if (i + j) % 2 == 0:
                        result[i, j] = color1
                    else:
                        result[i, j] = color2

        return result

    def swap_colors_by_size(self, grid: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Swap colors so largest region becomes smallest color (and vice versa).

        Args:
            grid: Input grid
            background: Background color (ignored)

        Returns:
            Grid with colors swapped by size
        """
        # Count pixels of each color
        colors = [c for c in np.unique(grid) if c != background]

        if len(colors) < 2:
            return grid.copy()

        color_counts = {c: np.sum(grid == c) for c in colors}

        # Sort by count
        sorted_colors = sorted(colors, key=lambda c: color_counts[c])

        # Create mapping: largest → 1, next → 2, etc.
        color_map = {c: i + 1 for i, c in enumerate(sorted_colors)}

        # Apply mapping
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color

        return result

    def color_propagation(self, grid: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Propagate colors to fill adjacent background cells (flood fill).

        Args:
            grid: Input grid
            background: Background color to fill

        Returns:
            Grid with colors propagated
        """
        from scipy.ndimage import binary_dilation

        result = grid.copy()
        colors = [c for c in np.unique(grid) if c != background]

        # Expand each color by one pixel
        for color in colors:
            color_mask = grid == color
            expanded = binary_dilation(color_mask, iterations=1)

            # Only fill background pixels
            fill_mask = expanded & (result == background)
            result[fill_mask] = color

        return result

    # =========================================================================
    # CATEGORY 4: PATTERN OPERATIONS (5 primitives)
    # =========================================================================

    def apply_horizontal_symmetry(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply horizontal (left-right) symmetry by mirroring left half.

        Args:
            grid: Input grid

        Returns:
            Horizontally symmetric grid
        """
        result = grid.copy()
        mid = grid.shape[1] // 2

        # Mirror left half to right (handle odd widths)
        left_half = result[:, :mid]
        flipped = np.fliplr(left_half)

        # For odd widths, skip middle column
        if grid.shape[1] % 2 == 0:
            result[:, mid:] = flipped
        else:
            result[:, mid+1:] = flipped

        return result

    def apply_vertical_symmetry(self, grid: np.ndarray) -> np.ndarray:
        """Apply vertical (top-bottom) symmetry by mirroring top half."""
        result = grid.copy()
        mid = grid.shape[0] // 2

        # Mirror top half to bottom
        top_half = result[:mid, :]
        flipped = np.flipud(top_half)

        # For odd heights, skip middle row
        if grid.shape[0] % 2 == 0:
            result[mid:, :] = flipped
        else:
            result[mid+1:, :] = flipped  # Skip middle row for odd heights

        return result

    def apply_rotational_symmetry(self, grid: np.ndarray, order: int = 4) -> np.ndarray:
        """
        Apply rotational symmetry (90-degree rotations).

        Args:
            grid: Input grid (must be square)
            order: Symmetry order (2 or 4)

        Returns:
            Rotationally symmetric grid
        """
        if grid.shape[0] != grid.shape[1]:
            return grid.copy()

        result = grid.copy()

        # Apply 90-degree rotations
        for i in range(1, order):
            rotated = np.rot90(result, k=i)
            # Combine with OR-like operation (keep non-zero)
            result = np.where(result == 0, rotated, result)

        return result

    def complete_partial_pattern(self, grid: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Complete a partial repeating pattern by detecting and extending it.

        Args:
            grid: Input grid with partial pattern
            background: Background color

        Returns:
            Grid with completed pattern
        """
        # Detect pattern size by finding first repetition
        pattern_height, pattern_width = self._detect_pattern_size(grid, background)

        if pattern_height == 0 or pattern_width == 0:
            return grid.copy()

        # Extract pattern
        pattern = grid[:pattern_height, :pattern_width]

        # Tile pattern to fill grid
        result = np.tile(pattern, (
            (grid.shape[0] + pattern_height - 1) // pattern_height,
            (grid.shape[1] + pattern_width - 1) // pattern_width
        ))

        # Crop to original size
        return result[:grid.shape[0], :grid.shape[1]]

    def generate_periodic_tiling(self, grid: np.ndarray, tile_size: int = 3,
                                background: int = 0) -> np.ndarray:
        """
        Generate periodic tiling from top-left tile.

        Args:
            grid: Input grid
            tile_size: Size of square tile
            background: Background color

        Returns:
            Grid with periodic tiling
        """
        # Extract top-left tile
        tile = grid[:tile_size, :tile_size]

        # Tile it across the grid
        result = np.tile(tile, (
            (grid.shape[0] + tile_size - 1) // tile_size,
            (grid.shape[1] + tile_size - 1) // tile_size
        ))

        return result[:grid.shape[0], :grid.shape[1]]

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _detect_all_objects(self, grid: np.ndarray,
                           background: int = 0) -> List[Dict[str, Any]]:
        """Detect all connected component objects."""
        objects = []

        # Find all non-background colors
        colors = [c for c in np.unique(grid) if c != background]

        for color in colors:
            color_mask = (grid == color).astype(int)
            labeled, num_features = label(color_mask)

            for obj_id in range(1, num_features + 1):
                obj_mask = labeled == obj_id
                positions = np.argwhere(obj_mask)

                if len(positions) == 0:
                    continue

                r_min, c_min = positions.min(axis=0)
                r_max, c_max = positions.max(axis=0)

                objects.append({
                    'color': color,
                    'size': len(positions),
                    'bbox': (int(r_min), int(c_min), int(r_max), int(c_max)),
                    'positions': positions
                })

        return objects

    def _detect_pattern_size(self, grid: np.ndarray,
                            background: int = 0) -> Tuple[int, int]:
        """Detect repeating pattern size."""
        # Try different pattern sizes
        for h in range(1, min(10, grid.shape[0] + 1)):
            for w in range(1, min(10, grid.shape[1] + 1)):
                if self._is_repeating_pattern(grid, h, w):
                    return h, w

        return 0, 0

    def _is_repeating_pattern(self, grid: np.ndarray, h: int, w: int) -> bool:
        """Check if grid has repeating pattern of size h×w."""
        pattern = grid[:h, :w]

        # Check if pattern repeats across grid
        for i in range(0, grid.shape[0], h):
            for j in range(0, grid.shape[1], w):
                tile = grid[i:min(i+h, grid.shape[0]),
                           j:min(j+w, grid.shape[1])]

                pattern_crop = pattern[:tile.shape[0], :tile.shape[1]]

                if not np.array_equal(tile, pattern_crop):
                    return False

        return True
