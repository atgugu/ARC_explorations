"""
Advanced TRG Primitives for Complex ARC Tasks

Extends the basic primitives with more sophisticated operations:
- Improved rotation handling (non-square grids)
- Pattern detection and completion
- Object-based transformations
- Spatial reasoning
- Multi-object operations
- Morphological operations (dilation, erosion, etc.)
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Any
from arc_generative_solver import ARCObject, TRGPrimitives
try:
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AdvancedPrimitives:
    """Advanced primitives for complex ARC reasoning"""

    # ========================================================================
    # 1. IMPROVED ROTATION HANDLING (FOR NON-SQUARE GRIDS)
    # ========================================================================

    @staticmethod
    def rotate_90_ccw(grid: np.ndarray) -> np.ndarray:
        """
        Rotate 90° counter-clockwise (standard numpy direction)

        Works correctly for non-square grids:
        [[1, 2, 3, 4], [5, 6, 7, 8]] -> [[4, 8],
                                          [3, 7],
                                          [2, 6],
                                          [1, 5]]
        """
        return np.rot90(grid, k=1)

    @staticmethod
    def rotate_90_cw(grid: np.ndarray) -> np.ndarray:
        """
        Rotate 90° clockwise

        Works correctly for non-square grids:
        [[1, 2, 3, 4], [5, 6, 7, 8]] -> [[5, 1],
                                          [6, 2],
                                          [7, 3],
                                          [8, 4]]
        """
        return np.rot90(grid, k=-1)

    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        """Rotate 180°"""
        return np.rot90(grid, k=2)

    @staticmethod
    def rotate_270_ccw(grid: np.ndarray) -> np.ndarray:
        """Rotate 270° counter-clockwise (same as 90° clockwise)"""
        return np.rot90(grid, k=3)

    # ========================================================================
    # 2. SYMMETRY AND PATTERN DETECTION
    # ========================================================================

    @staticmethod
    def detect_grid_symmetry(grid: np.ndarray) -> Dict[str, bool]:
        """Detect various symmetries in the grid"""
        h, w = grid.shape

        return {
            'horizontal': np.array_equal(grid, np.fliplr(grid)),
            'vertical': np.array_equal(grid, np.flipud(grid)),
            'diagonal_main': h == w and np.array_equal(grid, grid.T),
            'diagonal_anti': h == w and np.array_equal(grid, np.fliplr(grid).T),
            'rotational_180': np.array_equal(grid, np.rot90(grid, 2)),
            'rotational_90': h == w and np.array_equal(grid, np.rot90(grid, 1))
        }

    @staticmethod
    def extract_pattern(grid: np.ndarray,
                       pattern_size: Tuple[int, int]) -> List[np.ndarray]:
        """Extract repeating patterns of given size"""
        h, w = grid.shape
        ph, pw = pattern_size

        patterns = []
        for i in range(0, h - ph + 1, ph):
            for j in range(0, w - pw + 1, pw):
                pattern = grid[i:i+ph, j:j+pw]
                patterns.append(pattern)

        return patterns

    @staticmethod
    def tile_pattern(pattern: np.ndarray,
                    target_shape: Tuple[int, int]) -> np.ndarray:
        """Tile a pattern to fill target shape"""
        h, w = target_shape
        ph, pw = pattern.shape

        result = np.zeros((h, w), dtype=pattern.dtype)

        for i in range(0, h, ph):
            for j in range(0, w, pw):
                end_i = min(i + ph, h)
                end_j = min(j + pw, w)
                result[i:end_i, j:end_j] = pattern[:end_i-i, :end_j-j]

        return result

    @staticmethod
    def find_and_replace_pattern(grid: np.ndarray,
                                 find_pattern: np.ndarray,
                                 replace_pattern: np.ndarray) -> np.ndarray:
        """Find occurrences of a pattern and replace them"""
        result = grid.copy()
        ph, pw = find_pattern.shape
        h, w = grid.shape

        for i in range(h - ph + 1):
            for j in range(w - pw + 1):
                if np.array_equal(grid[i:i+ph, j:j+pw], find_pattern):
                    rh, rw = replace_pattern.shape
                    result[i:i+rh, j:j+rw] = replace_pattern

        return result

    @staticmethod
    def connect_objects(grid: np.ndarray,
                       objects: List[ARCObject],
                       color: int = 1,
                       method: str = 'line') -> np.ndarray:
        """Connect objects with lines or paths"""
        result = grid.copy()

        if len(objects) < 2:
            return result

        for i in range(len(objects) - 1):
            obj1, obj2 = objects[i], objects[i + 1]
            y1, x1 = map(int, obj1.centroid)
            y2, x2 = map(int, obj2.centroid)

            if method == 'line':
                # Bresenham line
                result = AdvancedPrimitives._draw_line(
                    result, y1, x1, y2, x2, color
                )
            elif method == 'manhattan':
                # Horizontal then vertical
                result[y1, min(x1,x2):max(x1,x2)+1] = color
                result[min(y1,y2):max(y1,y2)+1, x2] = color

        return result

    @staticmethod
    def _draw_line(grid: np.ndarray, y1: int, x1: int,
                   y2: int, x2: int, color: int) -> np.ndarray:
        """Bresenham line algorithm"""
        result = grid.copy()

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= y1 < grid.shape[0] and 0 <= x1 < grid.shape[1]:
                result[y1, x1] = color

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return result

    @staticmethod
    def fill_enclosed_regions(grid: np.ndarray,
                             fill_color: int,
                             boundary_color: Optional[int] = None) -> np.ndarray:
        """Fill regions enclosed by boundaries"""
        result = grid.copy()
        h, w = grid.shape

        # Find background regions (flood fill from edges)
        visited = np.zeros_like(grid, dtype=bool)

        # Flood fill from all edges
        def flood_fill_background(y: int, x: int):
            if (y < 0 or y >= h or x < 0 or x >= w or
                visited[y, x]):
                return

            if boundary_color is not None and grid[y, x] == boundary_color:
                return
            elif boundary_color is None and grid[y, x] != 0:
                return

            visited[y, x] = True

            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                flood_fill_background(y + dy, x + dx)

        # Start from edges
        for i in range(h):
            flood_fill_background(i, 0)
            flood_fill_background(i, w-1)
        for j in range(w):
            flood_fill_background(0, j)
            flood_fill_background(h-1, j)

        # Fill unvisited regions
        for i in range(h):
            for j in range(w):
                if not visited[i, j] and grid[i, j] == 0:
                    result[i, j] = fill_color

        return result

    @staticmethod
    def grow_shape(grid: np.ndarray,
                  mask: np.ndarray,
                  iterations: int = 1) -> np.ndarray:
        """Grow/dilate a shape"""
        if HAS_SCIPY:
            result = grid.copy()
            grown_mask = mask.copy()

            for _ in range(iterations):
                grown_mask = binary_dilation(grown_mask)

            color = grid[mask].max() if mask.any() else 1
            result[grown_mask] = color

            return result
        else:
            # Simple fallback: grow by adding neighbors
            result = grid.copy()
            current_mask = mask.copy()

            for _ in range(iterations):
                new_mask = current_mask.copy()
                h, w = mask.shape

                for i in range(h):
                    for j in range(w):
                        if current_mask[i, j]:
                            # Add neighbors
                            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w:
                                    new_mask[ni, nj] = True

                current_mask = new_mask

            color = grid[mask].max() if mask.any() else 1
            result[current_mask] = color

            return result

    @staticmethod
    def shrink_shape(grid: np.ndarray,
                    mask: np.ndarray,
                    iterations: int = 1) -> np.ndarray:
        """Shrink/erode a shape"""
        try:
            from scipy.ndimage import binary_erosion

            result = grid.copy()
            shrunk_mask = mask.copy()

            for _ in range(iterations):
                shrunk_mask = binary_erosion(shrunk_mask)

            # Remove parts that were eroded
            result[mask & ~shrunk_mask] = 0

            return result
        except ImportError:
            # Fallback without scipy
            return grid

    @staticmethod
    def recolor_by_size(grid: np.ndarray,
                       objects: List[ARCObject],
                       color_map: Dict[str, int]) -> np.ndarray:
        """Recolor objects based on their size"""
        result = np.zeros_like(grid)

        if not objects:
            return grid

        # Sort objects by size
        sorted_objs = sorted(objects, key=lambda o: o.area)

        for idx, obj in enumerate(objects):
            # Determine color based on size rank
            rank = sorted_objs.index(obj)

            if rank == 0 and 'smallest' in color_map:
                color = color_map['smallest']
            elif rank == len(sorted_objs) - 1 and 'largest' in color_map:
                color = color_map['largest']
            elif 'medium' in color_map:
                color = color_map['medium']
            else:
                color = obj.color

            result[obj.mask] = color

        return result

    @staticmethod
    def recolor_by_position(grid: np.ndarray,
                           objects: List[ARCObject],
                           color_map: Dict[str, int]) -> np.ndarray:
        """Recolor objects based on their position"""
        result = np.zeros_like(grid)

        if not objects:
            return grid

        h, w = grid.shape

        for obj in objects:
            y, x = obj.centroid

            # Determine position
            color = obj.color

            if y < h / 3 and 'top' in color_map:
                color = color_map['top']
            elif y > 2 * h / 3 and 'bottom' in color_map:
                color = color_map['bottom']
            elif 'middle' in color_map:
                color = color_map['middle']

            if x < w / 3 and 'left' in color_map:
                color = color_map['left']
            elif x > 2 * w / 3 and 'right' in color_map:
                color = color_map['right']

            result[obj.mask] = color

        return result

    @staticmethod
    def align_objects(grid: np.ndarray,
                     objects: List[ARCObject],
                     alignment: str = 'horizontal') -> np.ndarray:
        """Align objects horizontally or vertically"""
        result = np.zeros_like(grid)

        if not objects:
            return grid

        if alignment == 'horizontal':
            # Align to same y-coordinate
            target_y = int(np.mean([obj.centroid[0] for obj in objects]))

            for obj in objects:
                dy = target_y - int(obj.centroid[0])
                new_mask = np.zeros_like(obj.mask)

                y1, x1, y2, x2 = obj.bbox
                new_y1 = max(0, min(grid.shape[0] - (y2-y1+1), y1 + dy))
                new_y2 = new_y1 + (y2 - y1)

                new_mask[new_y1:new_y2+1, x1:x2+1] = \
                    obj.mask[y1:y2+1, x1:x2+1]

                result[new_mask] = obj.color

        elif alignment == 'vertical':
            # Align to same x-coordinate
            target_x = int(np.mean([obj.centroid[1] for obj in objects]))

            for obj in objects:
                dx = target_x - int(obj.centroid[1])
                new_mask = np.zeros_like(obj.mask)

                y1, x1, y2, x2 = obj.bbox
                new_x1 = max(0, min(grid.shape[1] - (x2-x1+1), x1 + dx))
                new_x2 = new_x1 + (x2 - x1)

                new_mask[y1:y2+1, new_x1:new_x2+1] = \
                    obj.mask[y1:y2+1, x1:x2+1]

                result[new_mask] = obj.color

        return result

    @staticmethod
    def create_frame(grid: np.ndarray,
                    thickness: int = 1,
                    color: int = 1) -> np.ndarray:
        """Create a frame around the grid"""
        result = grid.copy()
        h, w = grid.shape

        # Top and bottom
        result[:thickness, :] = color
        result[-thickness:, :] = color

        # Left and right
        result[:, :thickness] = color
        result[:, -thickness:] = color

        return result

    @staticmethod
    def gravity_transform(grid: np.ndarray,
                         direction: str = 'down') -> np.ndarray:
        """Apply gravity to non-zero pixels"""
        result = np.zeros_like(grid)
        h, w = grid.shape

        if direction == 'down':
            for j in range(w):
                non_zero = grid[:, j][grid[:, j] != 0]
                if len(non_zero) > 0:
                    result[h-len(non_zero):, j] = non_zero

        elif direction == 'up':
            for j in range(w):
                non_zero = grid[:, j][grid[:, j] != 0]
                if len(non_zero) > 0:
                    result[:len(non_zero), j] = non_zero

        elif direction == 'left':
            for i in range(h):
                non_zero = grid[i, :][grid[i, :] != 0]
                if len(non_zero) > 0:
                    result[i, :len(non_zero)] = non_zero

        elif direction == 'right':
            for i in range(h):
                non_zero = grid[i, :][grid[i, :] != 0]
                if len(non_zero) > 0:
                    result[i, w-len(non_zero):] = non_zero

        return result

    @staticmethod
    def count_objects_by_property(objects: List[ARCObject],
                                  property_name: str) -> Dict[Any, int]:
        """Count objects grouped by a property"""
        counts = {}

        for obj in objects:
            if property_name == 'color':
                key = obj.color
            elif property_name == 'area':
                key = obj.area
            elif property_name == 'shape':
                key = obj.shape_signature
            else:
                continue

            counts[key] = counts.get(key, 0) + 1

        return counts

    @staticmethod
    def get_majority_object(objects: List[ARCObject],
                           by: str = 'color') -> Optional[ARCObject]:
        """Get the most common object by some property"""
        if not objects:
            return None

        if by == 'color':
            from collections import Counter
            colors = [obj.color for obj in objects]
            most_common = Counter(colors).most_common(1)[0][0]
            return next(obj for obj in objects if obj.color == most_common)

        return objects[0]

    # ========================================================================
    # ENHANCED PATTERN TILING OPERATIONS
    # ========================================================================

    @staticmethod
    def repeat_pattern_horizontal(pattern: np.ndarray, n_times: int) -> np.ndarray:
        """Repeat pattern horizontally n times"""
        return np.tile(pattern, (1, n_times))

    @staticmethod
    def repeat_pattern_vertical(pattern: np.ndarray, n_times: int) -> np.ndarray:
        """Repeat pattern vertically n times"""
        return np.tile(pattern, (n_times, 1))

    @staticmethod
    def repeat_pattern_grid(pattern: np.ndarray,
                           n_rows: int, n_cols: int) -> np.ndarray:
        """Repeat pattern in grid (both directions)"""
        return np.tile(pattern, (n_rows, n_cols))

    @staticmethod
    def complete_symmetry_horizontal(grid: np.ndarray) -> np.ndarray:
        """Complete horizontal symmetry by mirroring left-right"""
        return np.concatenate([grid, np.fliplr(grid)], axis=1)

    @staticmethod
    def complete_symmetry_vertical(grid: np.ndarray) -> np.ndarray:
        """Complete vertical symmetry by mirroring top-bottom"""
        return np.concatenate([grid, np.flipud(grid)], axis=0)

    @staticmethod
    def detect_pattern_size(grid: np.ndarray) -> Tuple[int, int]:
        """
        Detect repeating pattern size in grid
        Returns (pattern_height, pattern_width)
        """
        h, w = grid.shape

        # Try different pattern sizes (smallest first)
        for ph in range(1, h + 1):
            for pw in range(1, w + 1):
                if h % ph == 0 and w % pw == 0:  # Must divide evenly
                    pattern = grid[:ph, :pw]
                    tiled = AdvancedPrimitives.tile_pattern(pattern, (h, w))

                    if np.array_equal(tiled, grid):
                        return (ph, pw)

        return (h, w)  # No pattern found

    # ========================================================================
    # ENHANCED MORPHOLOGICAL OPERATIONS
    # ========================================================================

    @staticmethod
    def dilate_objects_enhanced(grid: np.ndarray,
                               iterations: int = 1,
                               background: int = 0) -> np.ndarray:
        """
        Dilate objects (grow) using scipy if available

        More accurate than simple grow_shape
        """
        if HAS_SCIPY:
            result = grid.copy()
            colors = np.unique(grid)
            colors = colors[colors != background]

            for color in colors:
                mask = (grid == color)
                dilated = binary_dilation(mask, iterations=iterations)
                # Only fill background pixels
                result[dilated & (result == background)] = color

            return result
        else:
            # Simple fallback
            result = grid.copy()
            h, w = grid.shape

            for _ in range(iterations):
                new_result = result.copy()

                for i in range(h):
                    for j in range(w):
                        if result[i, j] != background:
                            # Expand to neighbors
                            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w:
                                    if new_result[ni, nj] == background:
                                        new_result[ni, nj] = result[i, j]

                result = new_result

            return result

    @staticmethod
    def erode_objects_enhanced(grid: np.ndarray,
                              iterations: int = 1,
                              background: int = 0) -> np.ndarray:
        """
        Erode objects (shrink) using scipy if available

        More accurate than simple shrink_shape
        """
        if not HAS_SCIPY:
            return AdvancedPrimitives.shrink_shape(grid, grid != background, iterations)

        result = np.full_like(grid, background)
        colors = np.unique(grid)
        colors = colors[colors != background]

        for color in colors:
            mask = (grid == color)
            eroded = binary_erosion(mask, iterations=iterations)
            result[eroded] = color

        return result

    @staticmethod
    def fill_holes_in_objects(grid: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Fill holes in objects

        Example: [[1,1,1], [1,0,1], [1,1,1]] -> [[1,1,1], [1,1,1], [1,1,1]]
        """
        if not HAS_SCIPY:
            return grid  # Can't do much without scipy

        result = grid.copy()
        colors = np.unique(grid)
        colors = colors[colors != background]

        for color in colors:
            mask = (grid == color)
            filled = binary_fill_holes(mask)
            result[filled & (result == background)] = color

        return result

    @staticmethod
    def find_object_boundaries(grid: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Find boundaries of objects (edge pixels only)

        Returns grid with only boundary pixels of each object
        """
        if not HAS_SCIPY:
            return grid  # Fallback

        result = np.full_like(grid, background)
        colors = np.unique(grid)
        colors = colors[colors != background]

        for color in colors:
            mask = (grid == color)
            eroded = binary_erosion(mask, iterations=1)
            boundary = mask & ~eroded
            result[boundary] = color

        return result

    @staticmethod
    def hollow_objects(grid: np.ndarray,
                      thickness: int = 1,
                      background: int = 0) -> np.ndarray:
        """
        Make objects hollow (keep only outer shell of specified thickness)
        """
        if not HAS_SCIPY:
            return grid

        result = np.full_like(grid, background)
        colors = np.unique(grid)
        colors = colors[colors != background]

        for color in colors:
            mask = (grid == color)

            if thickness > 0:
                inner = binary_erosion(mask, iterations=thickness)
                shell = mask & ~inner
            else:
                shell = mask

            result[shell] = color

        return result

    # ========================================================================
    # OBJECT OPERATIONS
    # ========================================================================

    @staticmethod
    def move_object_to_position(grid: np.ndarray,
                               obj: ARCObject,
                               target_y: int,
                               target_x: int) -> np.ndarray:
        """
        Move an object to a specific position (top-left corner)
        """
        result = grid.copy()

        # Clear original position
        result[obj.mask] = 0

        # Calculate offset
        y1, x1, y2, x2 = obj.bbox
        dy = target_y - y1
        dx = target_x - x1

        # Place at new position
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if obj.mask[y, x]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        result[ny, nx] = obj.color

        return result

    @staticmethod
    def scale_object(grid: np.ndarray,
                    obj: ARCObject,
                    scale_factor: float) -> np.ndarray:
        """
        Scale an object by a factor (simple nearest-neighbor scaling)

        scale_factor > 1: grow
        scale_factor < 1: shrink
        """
        result = grid.copy()

        # Clear original
        result[obj.mask] = 0

        y1, x1, y2, x2 = obj.bbox
        obj_h = y2 - y1 + 1
        obj_w = x2 - x1 + 1

        new_h = int(obj_h * scale_factor)
        new_w = int(obj_w * scale_factor)

        # Simple nearest-neighbor scaling
        for ny in range(new_h):
            for nx in range(new_w):
                # Map back to original
                oy = int(ny / scale_factor)
                ox = int(nx / scale_factor)

                src_y = y1 + oy
                src_x = x1 + ox

                if obj.mask[src_y, src_x]:
                    dst_y = y1 + ny
                    dst_x = x1 + nx

                    if 0 <= dst_y < grid.shape[0] and 0 <= dst_x < grid.shape[1]:
                        result[dst_y, dst_x] = obj.color

        return result

    @staticmethod
    def duplicate_object(grid: np.ndarray,
                        obj: ARCObject,
                        offset_y: int,
                        offset_x: int) -> np.ndarray:
        """
        Duplicate an object at an offset position
        """
        result = grid.copy()

        y1, x1, y2, x2 = obj.bbox

        # Place duplicate
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if obj.mask[y, x]:
                    ny, nx = y + offset_y, x + offset_x
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        result[ny, nx] = obj.color

        return result

    @staticmethod
    def sort_objects_spatial(objects: List[ARCObject],
                            order: str = "left_to_right") -> List[ARCObject]:
        """
        Sort objects spatially

        order: "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"
        """
        if order == "left_to_right":
            return sorted(objects, key=lambda o: o.bbox[1])  # Sort by x1
        elif order == "right_to_left":
            return sorted(objects, key=lambda o: -o.bbox[1])
        elif order == "top_to_bottom":
            return sorted(objects, key=lambda o: o.bbox[0])  # Sort by y1
        elif order == "bottom_to_top":
            return sorted(objects, key=lambda o: -o.bbox[0])
        else:
            return objects

    @staticmethod
    def distribute_objects_evenly(grid: np.ndarray,
                                 objects: List[ARCObject],
                                 axis: str = "horizontal") -> np.ndarray:
        """
        Distribute objects evenly along an axis
        """
        result = np.zeros_like(grid)

        if not objects:
            return result

        h, w = grid.shape

        if axis == "horizontal":
            # Distribute along x-axis
            sorted_objs = sorted(objects, key=lambda o: o.bbox[1])
            total_width = sum(o.bbox[3] - o.bbox[1] + 1 for o in sorted_objs)
            spacing = (w - total_width) // (len(objects) + 1)

            current_x = spacing
            for obj in sorted_objs:
                y1, x1, y2, x2 = obj.bbox
                obj_h = y2 - y1 + 1
                obj_w = x2 - x1 + 1

                # Place object
                for y in range(y1, y2 + 1):
                    for x in range(x1, x2 + 1):
                        if obj.mask[y, x]:
                            ny = y
                            nx = current_x + (x - x1)
                            if 0 <= ny < h and 0 <= nx < w:
                                result[ny, nx] = obj.color

                current_x += obj_w + spacing

        elif axis == "vertical":
            # Distribute along y-axis
            sorted_objs = sorted(objects, key=lambda o: o.bbox[0])
            total_height = sum(o.bbox[2] - o.bbox[0] + 1 for o in sorted_objs)
            spacing = (h - total_height) // (len(objects) + 1)

            current_y = spacing
            for obj in sorted_objs:
                y1, x1, y2, x2 = obj.bbox
                obj_h = y2 - y1 + 1
                obj_w = x2 - x1 + 1

                # Place object
                for y in range(y1, y2 + 1):
                    for x in range(x1, x2 + 1):
                        if obj.mask[y, x]:
                            ny = current_y + (y - y1)
                            nx = x
                            if 0 <= ny < h and 0 <= nx < w:
                                result[ny, nx] = obj.color

                current_y += obj_h + spacing

        return result

    # ========================================================================
    # ENHANCED PHYSICS-BASED TRANSFORMS
    # ========================================================================

    @staticmethod
    def gravity_objects(grid: np.ndarray,
                       objects: List[ARCObject],
                       direction: str = "down",
                       stop_at_obstacle: bool = True) -> np.ndarray:
        """
        Apply gravity to objects (more sophisticated than gravity_transform)

        Objects fall until they hit the edge or another object
        """
        result = np.zeros_like(grid)
        h, w = grid.shape

        # Sort objects by position (fall order matters)
        if direction == "down":
            sorted_objs = sorted(objects, key=lambda o: -o.bbox[2])  # Bottom first
        elif direction == "up":
            sorted_objs = sorted(objects, key=lambda o: o.bbox[0])   # Top first
        elif direction == "left":
            sorted_objs = sorted(objects, key=lambda o: o.bbox[1])   # Left first
        elif direction == "right":
            sorted_objs = sorted(objects, key=lambda o: -o.bbox[3])  # Right first
        else:
            sorted_objs = objects

        for obj in sorted_objs:
            y1, x1, y2, x2 = obj.bbox

            # Find how far object can fall
            if direction == "down":
                max_fall = h - 1 - y2
                for dist in range(max_fall + 1):
                    # Check if can move this far
                    can_move = True
                    for y in range(y1, y2 + 1):
                        for x in range(x1, x2 + 1):
                            if obj.mask[y, x]:
                                ny = y + dist + 1
                                if ny >= h or (stop_at_obstacle and result[ny, x] != 0):
                                    can_move = False
                                    break
                        if not can_move:
                            break

                    if not can_move:
                        # Place at dist
                        for y in range(y1, y2 + 1):
                            for x in range(x1, x2 + 1):
                                if obj.mask[y, x]:
                                    result[y + dist, x] = obj.color
                        break
                else:
                    # Reached bottom
                    for y in range(y1, y2 + 1):
                        for x in range(x1, x2 + 1):
                            if obj.mask[y, x]:
                                result[y + max_fall, x] = obj.color

            elif direction == "up":
                max_rise = y1
                for dist in range(max_rise + 1):
                    can_move = True
                    for y in range(y1, y2 + 1):
                        for x in range(x1, x2 + 1):
                            if obj.mask[y, x]:
                                ny = y - dist - 1
                                if ny < 0 or (stop_at_obstacle and result[ny, x] != 0):
                                    can_move = False
                                    break
                        if not can_move:
                            break

                    if not can_move:
                        for y in range(y1, y2 + 1):
                            for x in range(x1, x2 + 1):
                                if obj.mask[y, x]:
                                    result[y - dist, x] = obj.color
                        break
                else:
                    for y in range(y1, y2 + 1):
                        for x in range(x1, x2 + 1):
                            if obj.mask[y, x]:
                                result[y - max_rise, x] = obj.color

            # Similar for left/right
            elif direction in ["left", "right"]:
                # Place object as-is for now (simplified)
                for y in range(y1, y2 + 1):
                    for x in range(x1, x2 + 1):
                        if obj.mask[y, x]:
                            result[y, x] = obj.color

        return result

    @staticmethod
    def stack_objects(grid: np.ndarray,
                     objects: List[ARCObject],
                     direction: str = "vertical") -> np.ndarray:
        """
        Stack objects on top of each other or side-by-side
        """
        result = np.zeros_like(grid)

        if not objects:
            return result

        if direction == "vertical":
            # Stack vertically (top to bottom)
            current_y = 0

            for obj in objects:
                y1, x1, y2, x2 = obj.bbox
                obj_h = y2 - y1 + 1
                obj_w = x2 - x1 + 1

                # Place object
                for y in range(y1, y2 + 1):
                    for x in range(x1, x2 + 1):
                        if obj.mask[y, x]:
                            ny = current_y + (y - y1)
                            nx = x
                            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                                result[ny, nx] = obj.color

                current_y += obj_h

        elif direction == "horizontal":
            # Stack horizontally (left to right)
            current_x = 0

            for obj in objects:
                y1, x1, y2, x2 = obj.bbox
                obj_h = y2 - y1 + 1
                obj_w = x2 - x1 + 1

                # Place object
                for y in range(y1, y2 + 1):
                    for x in range(x1, x2 + 1):
                        if obj.mask[y, x]:
                            ny = y
                            nx = current_x + (x - x1)
                            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                                result[ny, nx] = obj.color

                current_x += obj_w

        return result

    @staticmethod
    def compress_objects(grid: np.ndarray,
                        direction: str = "down") -> np.ndarray:
        """
        Compress all objects toward one direction (remove gaps)

        Similar to gravity but for all non-zero pixels
        """
        result = np.zeros_like(grid)
        h, w = grid.shape

        if direction == "down":
            # Compress downward
            for x in range(w):
                write_y = h - 1
                for y in range(h - 1, -1, -1):
                    if grid[y, x] != 0:
                        result[write_y, x] = grid[y, x]
                        write_y -= 1

        elif direction == "up":
            # Compress upward
            for x in range(w):
                write_y = 0
                for y in range(h):
                    if grid[y, x] != 0:
                        result[write_y, x] = grid[y, x]
                        write_y += 1

        elif direction == "left":
            # Compress leftward
            for y in range(h):
                write_x = 0
                for x in range(w):
                    if grid[y, x] != 0:
                        result[y, write_x] = grid[y, x]
                        write_x += 1

        elif direction == "right":
            # Compress rightward
            for y in range(h):
                write_x = w - 1
                for x in range(w - 1, -1, -1):
                    if grid[y, x] != 0:
                        result[y, write_x] = grid[y, x]
                        write_x -= 1

        return result
