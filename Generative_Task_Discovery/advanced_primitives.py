"""
Advanced TRG Primitives for Complex ARC Tasks

Extends the basic primitives with more sophisticated operations:
- Pattern detection and completion
- Object-based transformations
- Spatial reasoning
- Multi-object operations
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from arc_generative_solver import ARCObject, TRGPrimitives


class AdvancedPrimitives:
    """Advanced primitives for complex ARC reasoning"""

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
        from scipy.ndimage import binary_dilation

        result = grid.copy()
        grown_mask = mask.copy()

        for _ in range(iterations):
            grown_mask = binary_dilation(grown_mask)

        color = grid[mask].max() if mask.any() else 1
        result[grown_mask] = color

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
