"""
Core DSL Primitives - Phase 1 Implementation
Top 20 most frequently used primitives for ARC tasks

This module implements the essential operations that cover 70-80% of ARC tasks.
"""

from typing import List, Tuple, Union, Optional
from enum import Enum
import numpy as np
from scipy.ndimage import label, rotate as scipy_rotate
from scipy.ndimage import binary_dilation, binary_erosion
from collections import defaultdict

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

Grid = np.ndarray
Object = List[Tuple[int, int]]
ObjectSet = List[Object]
Color = int
Point = Tuple[int, int]


class Direction(Enum):
    """Cardinal and diagonal directions"""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    DIAG_NE = (-1, 1)
    DIAG_NW = (-1, -1)
    DIAG_SE = (1, 1)
    DIAG_SW = (1, -1)


class Axis(Enum):
    """Reflection/alignment axes"""
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2
    ANTI_DIAGONAL = 3
    BOTH = 4


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def object_to_grid(objects: Union[Object, ObjectSet], color: Union[Color, List[Color]],
                   grid_shape: Tuple[int, int]) -> Grid:
    """
    Convert object(s) to grid representation.

    Args:
        objects: Single object or list of objects
        color: Color(s) for the object(s)
        grid_shape: Output grid shape

    Returns:
        Grid with objects rendered
    """
    grid = np.zeros(grid_shape, dtype=int)

    # Handle single object
    if isinstance(objects[0], tuple) and len(objects[0]) == 2 and isinstance(objects[0][0], int):
        objects = [objects]

    # Handle single color or list of colors
    if isinstance(color, int):
        colors = [color] * len(objects)
    else:
        colors = color

    for obj, col in zip(objects, colors):
        for r, c in obj:
            if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                grid[r, c] = col

    return grid


def get_object_bounds(object: Object) -> Tuple[int, int, int, int]:
    """Get bounding box of object: (min_row, min_col, max_row, max_col)"""
    if not object:
        return (0, 0, 0, 0)
    rows, cols = zip(*object)
    return (min(rows), min(cols), max(rows), max(cols))


def compute_centroid(object: Object) -> Tuple[float, float]:
    """Compute centroid of object"""
    if not object:
        return (0.0, 0.0)
    rows, cols = zip(*object)
    return (sum(rows) / len(rows), sum(cols) / len(cols))


def get_object_color(object: Object, grid: Grid) -> Color:
    """Get the color of an object from grid"""
    if not object:
        return 0
    r, c = object[0]
    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        return int(grid[r, c])
    return 0


# ============================================================================
# 1. SELECTION & FILTERING
# ============================================================================

def select_by_color(grid: Grid, color: Color, connectivity: int = 4) -> ObjectSet:
    """
    Extract all connected components of a specific color.

    Args:
        grid: Input grid
        color: Color to select (1-9)
        connectivity: 4 or 8 connectivity

    Returns:
        List of objects (each object is a list of (row, col) coordinates)

    Example:
        >>> grid = np.array([[0,1,0], [1,1,0], [0,0,2]])
        >>> objects = select_by_color(grid, 1)
        >>> len(objects)
        1
    """
    # Create binary mask for the color
    mask = (grid == color).astype(int)

    # Use scipy's label function for connected components
    if connectivity == 4:
        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    else:  # 8-connectivity
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    labeled_array, num_features = label(mask, structure=structure)

    # Extract objects
    objects = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        obj = [(int(r), int(c)) for r, c in coords]
        objects.append(obj)

    return objects


def select_by_size(objects: ObjectSet, size: int, comparator: str = '==') -> ObjectSet:
    """
    Filter objects by size.

    Args:
        objects: Input objects
        size: Size threshold
        comparator: Comparison operator ('==', '>', '<', '>=', '<=', '!=')

    Returns:
        Filtered objects
    """
    comparators = {
        '==': lambda x, y: x == y,
        '>': lambda x, y: x > y,
        '<': lambda x, y: x < y,
        '>=': lambda x, y: x >= y,
        '<=': lambda x, y: x <= y,
        '!=': lambda x, y: x != y,
    }

    if comparator not in comparators:
        raise ValueError(f"Invalid comparator: {comparator}")

    comp_fn = comparators[comparator]
    return [obj for obj in objects if comp_fn(len(obj), size)]


def select_largest(objects: ObjectSet, k: int = 1) -> ObjectSet:
    """
    Select k largest objects.

    Args:
        objects: Input objects
        k: Number of objects to select

    Returns:
        k largest objects (sorted by size, largest first)
    """
    if not objects:
        return []

    sorted_objects = sorted(objects, key=len, reverse=True)
    return sorted_objects[:k]


def select_smallest(objects: ObjectSet, k: int = 1) -> ObjectSet:
    """
    Select k smallest objects.

    Args:
        objects: Input objects
        k: Number of objects to select

    Returns:
        k smallest objects (sorted by size, smallest first)
    """
    if not objects:
        return []

    sorted_objects = sorted(objects, key=len)
    return sorted_objects[:k]


def select_by_position(objects: ObjectSet, position: str, grid_shape: Tuple[int, int]) -> ObjectSet:
    """
    Filter objects by grid position.

    Args:
        objects: Input objects
        position: Position descriptor ("corner", "edge", "center", "top", "bottom", "left", "right")
        grid_shape: Shape of the grid (height, width)

    Returns:
        Objects in the specified position
    """
    H, W = grid_shape
    result = []

    for obj in objects:
        min_r, min_c, max_r, max_c = get_object_bounds(obj)
        centroid_r, centroid_c = compute_centroid(obj)

        if position == "corner":
            # Object touches two edges
            touches_top = (min_r == 0)
            touches_bottom = (max_r == H - 1)
            touches_left = (min_c == 0)
            touches_right = (max_c == W - 1)

            if (touches_top or touches_bottom) and (touches_left or touches_right):
                result.append(obj)

        elif position == "edge":
            # Object touches at least one edge
            if min_r == 0 or max_r == H - 1 or min_c == 0 or max_c == W - 1:
                result.append(obj)

        elif position == "center":
            # Object is in center region (middle third)
            if H/3 <= centroid_r <= 2*H/3 and W/3 <= centroid_c <= 2*W/3:
                result.append(obj)

        elif position == "top":
            if centroid_r < H / 3:
                result.append(obj)

        elif position == "bottom":
            if centroid_r > 2 * H / 3:
                result.append(obj)

        elif position == "left":
            if centroid_c < W / 3:
                result.append(obj)

        elif position == "right":
            if centroid_c > 2 * W / 3:
                result.append(obj)

    return result


# ============================================================================
# 2. SPATIAL TRANSFORMATIONS
# ============================================================================

def translate(object: Object, delta_row: int, delta_col: int) -> Object:
    """
    Move object by offset.

    Args:
        object: Input object
        delta_row: Row offset
        delta_col: Column offset

    Returns:
        Translated object
    """
    return [(r + delta_row, c + delta_col) for r, c in object]


def rotate(object: Object, angle: int, center: Optional[Point] = None) -> Object:
    """
    Rotate object by angle around center.

    Args:
        object: Input object
        angle: Rotation angle (90, 180, 270, or -90, -180, -270)
        center: Center of rotation (default: object's centroid)

    Returns:
        Rotated object

    Example:
        >>> obj = [(0,0), (0,1), (1,0), (1,1)]  # 2x2 square
        >>> rotated = rotate(obj, 90)
    """
    if not object:
        return []

    # Normalize angle to 0, 90, 180, 270
    angle = angle % 360

    if center is None:
        center_r, center_c = compute_centroid(object)
    else:
        center_r, center_c = center

    # Translate to origin
    translated = [(r - center_r, c - center_c) for r, c in object]

    # Rotate around origin
    rotated = []
    if angle == 90:
        rotated = [(-c, r) for r, c in translated]
    elif angle == 180:
        rotated = [(-r, -c) for r, c in translated]
    elif angle == 270:
        rotated = [(c, -r) for r, c in translated]
    else:  # 0 or 360
        rotated = translated

    # Translate back and round to integers
    result = [(int(round(r + center_r)), int(round(c + center_c))) for r, c in rotated]

    return result


def reflect(object: Object, axis: Axis, position: Optional[float] = None) -> Object:
    """
    Reflect object across axis.

    Args:
        object: Input object
        axis: Reflection axis (HORIZONTAL or VERTICAL)
        position: Axis position (default: through object centroid)

    Returns:
        Reflected object

    Example:
        >>> obj = [(0,0), (0,1), (1,0)]
        >>> reflected = reflect(obj, Axis.VERTICAL)
    """
    if not object:
        return []

    if position is None:
        center_r, center_c = compute_centroid(object)
    else:
        if axis == Axis.VERTICAL:
            center_c = position
            center_r = compute_centroid(object)[0]
        else:
            center_r = position
            center_c = compute_centroid(object)[1]

    reflected = []

    if axis == Axis.HORIZONTAL:
        # Reflect across horizontal line
        reflected = [(int(round(2 * center_r - r)), c) for r, c in object]
    elif axis == Axis.VERTICAL:
        # Reflect across vertical line
        reflected = [(r, int(round(2 * center_c - c))) for r, c in object]
    elif axis == Axis.DIAGONAL:
        # Reflect across main diagonal (swap r and c)
        reflected = [(c, r) for r, c in object]
    elif axis == Axis.ANTI_DIAGONAL:
        # Reflect across anti-diagonal
        # First reflect across main diagonal, then rotate 180
        reflected = [(-c, -r) for r, c in object]

    return reflected


def scale(object: Object, factor: int, center: Optional[Point] = None) -> Object:
    """
    Scale object by integer factor.

    Args:
        object: Input object
        factor: Scale factor (integer >= 1)
        center: Center of scaling (default: object's top-left corner)

    Returns:
        Scaled object

    Example:
        >>> obj = [(0,0), (0,1), (1,0), (1,1)]
        >>> scaled = scale(obj, 2)  # Each pixel becomes 2×2
    """
    if not object or factor <= 0:
        return []

    if factor == 1:
        return object.copy()

    if center is None:
        min_r, min_c, _, _ = get_object_bounds(object)
        center = (min_r, min_c)

    center_r, center_c = center

    # Translate to origin
    translated = [(r - center_r, c - center_c) for r, c in object]

    # Scale each pixel to factor×factor block
    scaled = []
    for r, c in translated:
        for dr in range(factor):
            for dc in range(factor):
                scaled.append((r * factor + dr, c * factor + dc))

    # Translate back
    result = [(int(r + center_r), int(c + center_c)) for r, c in scaled]

    return result


def move_to(object: Object, position: Union[str, Point], grid_shape: Tuple[int, int]) -> Object:
    """
    Move object to absolute position.

    Args:
        object: Input object
        position: Target position ("top_left", "top_right", "bottom_left", "bottom_right",
                  "center", or (row, col) tuple)
        grid_shape: Grid dimensions

    Returns:
        Moved object
    """
    if not object:
        return []

    H, W = grid_shape
    min_r, min_c, max_r, max_c = get_object_bounds(object)
    obj_h, obj_w = max_r - min_r + 1, max_c - min_c + 1

    # Determine target position
    if isinstance(position, tuple):
        target_r, target_c = position
    elif position == "top_left":
        target_r, target_c = 0, 0
    elif position == "top_right":
        target_r, target_c = 0, W - obj_w
    elif position == "bottom_left":
        target_r, target_c = H - obj_h, 0
    elif position == "bottom_right":
        target_r, target_c = H - obj_h, W - obj_w
    elif position == "center":
        target_r, target_c = (H - obj_h) // 2, (W - obj_w) // 2
    else:
        raise ValueError(f"Unknown position: {position}")

    # Translate object
    delta_r = target_r - min_r
    delta_c = target_c - min_c

    return translate(object, delta_r, delta_c)


# ============================================================================
# 3. COLOR OPERATIONS
# ============================================================================

def recolor(object: Object, new_color: Color, grid: Grid) -> Grid:
    """
    Change object to new color in grid.

    Args:
        object: Input object
        new_color: Target color
        grid: Grid to modify (will be copied)

    Returns:
        Modified grid
    """
    result = grid.copy()
    for r, c in object:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            result[r, c] = new_color
    return result


def swap_colors(grid: Grid, color1: Color, color2: Color) -> Grid:
    """
    Swap two colors globally in grid.

    Args:
        grid: Input grid
        color1: First color
        color2: Second color

    Returns:
        Grid with swapped colors
    """
    result = grid.copy()
    mask1 = grid == color1
    mask2 = grid == color2
    result[mask1] = color2
    result[mask2] = color1
    return result


def recolor_by_rule(objects: ObjectSet, rule: str, grid: Grid) -> Grid:
    """
    Recolor objects based on rule.

    Args:
        objects: Input objects
        rule: Recoloring rule ("size_ascending", "size_descending", "position_top_to_bottom",
              "position_left_to_right")
        grid: Grid to modify

    Returns:
        Modified grid
    """
    result = grid.copy()

    if rule == "size_ascending":
        sorted_objs = sorted(objects, key=len)
        colors = list(range(1, len(objects) + 1))
    elif rule == "size_descending":
        sorted_objs = sorted(objects, key=len, reverse=True)
        colors = list(range(1, len(objects) + 1))
    elif rule == "position_top_to_bottom":
        sorted_objs = sorted(objects, key=lambda obj: compute_centroid(obj)[0])
        colors = list(range(1, len(objects) + 1))
    elif rule == "position_left_to_right":
        sorted_objs = sorted(objects, key=lambda obj: compute_centroid(obj)[1])
        colors = list(range(1, len(objects) + 1))
    else:
        raise ValueError(f"Unknown rule: {rule}")

    for obj, color in zip(sorted_objs, colors):
        result = recolor(obj, color, result)

    return result


# ============================================================================
# 4. PATTERN OPERATIONS
# ============================================================================

def tile(object: Object, rows: int, cols: int, color: Optional[Color] = None,
         grid_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Tile object in rows×cols pattern.

    Args:
        object: Input object
        rows: Number of rows
        cols: Number of columns
        color: Color to use (if None, will try to infer from object)
        grid_shape: Output grid shape (auto-computed if None)

    Returns:
        Tiled grid
    """
    if not object:
        return np.zeros((rows, cols), dtype=int)

    # Get object dimensions
    min_r, min_c, max_r, max_c = get_object_bounds(object)
    obj_h, obj_w = max_r - min_r + 1, max_c - min_c + 1

    # Compute output size if not provided
    if grid_shape is None:
        grid_shape = (obj_h * rows, obj_w * cols)

    # Normalize object to start at (0, 0)
    normalized = translate(object, -min_r, -min_c)

    # Create output grid
    grid = np.zeros(grid_shape, dtype=int)

    # Tile the object
    for i in range(rows):
        for j in range(cols):
            offset_r = i * obj_h
            offset_c = j * obj_w
            tiled_obj = translate(normalized, offset_r, offset_c)

            # Draw the object
            for r, c in tiled_obj:
                if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                    if color is not None:
                        grid[r, c] = color
                    else:
                        grid[r, c] = 1  # Default color

    return grid


def copy_to_positions(object: Object, positions: List[Point], grid_shape: Tuple[int, int],
                     color: Optional[Color] = None) -> Grid:
    """
    Copy object to specific positions.

    Args:
        object: Input object
        positions: Target positions (top-left corners)
        grid_shape: Output grid shape
        color: Optional color override

    Returns:
        Grid with copied objects
    """
    grid = np.zeros(grid_shape, dtype=int)

    if not object:
        return grid

    # Normalize object to start at (0, 0)
    min_r, min_c, _, _ = get_object_bounds(object)
    normalized = translate(object, -min_r, -min_c)

    # Copy to each position
    for pos_r, pos_c in positions:
        copied_obj = translate(normalized, pos_r, pos_c)
        for r, c in copied_obj:
            if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                if color is not None:
                    grid[r, c] = color
                else:
                    grid[r, c] = 1  # Default color

    return grid


# ============================================================================
# 5. GRID OPERATIONS
# ============================================================================

def overlay(grid1: Grid, grid2: Grid, mode: str = 'replace') -> Grid:
    """
    Combine two grids.

    Args:
        grid1: Base grid
        grid2: Overlay grid
        mode: Overlay mode ('replace', 'or', 'and', 'xor', 'add')

    Returns:
        Combined grid
    """
    result = grid1.copy()

    # Ensure grids are same size (pad if needed)
    h = max(grid1.shape[0], grid2.shape[0])
    w = max(grid1.shape[1], grid2.shape[1])

    if grid1.shape != (h, w):
        padded1 = np.zeros((h, w), dtype=int)
        padded1[:grid1.shape[0], :grid1.shape[1]] = grid1
        result = padded1

    if grid2.shape != (h, w):
        padded2 = np.zeros((h, w), dtype=int)
        padded2[:grid2.shape[0], :grid2.shape[1]] = grid2
        grid2 = padded2

    if mode == 'replace':
        # grid2 overwrites grid1 where non-zero
        mask = grid2 != 0
        result[mask] = grid2[mask]
    elif mode == 'or':
        # Logical OR (any non-zero becomes 1)
        result = ((grid1 != 0) | (grid2 != 0)).astype(int)
    elif mode == 'and':
        # Logical AND
        result = ((grid1 != 0) & (grid2 != 0)).astype(int)
    elif mode == 'xor':
        # Logical XOR
        result = ((grid1 != 0) ^ (grid2 != 0)).astype(int)
    elif mode == 'add':
        # Addition (clamped to 9)
        result = np.clip(grid1 + grid2, 0, 9)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return result


def crop(grid: Grid, top: int, left: int, height: int, width: int) -> Grid:
    """
    Extract rectangular region from grid.

    Args:
        grid: Input grid
        top: Top row
        left: Left column
        height: Height of region
        width: Width of region

    Returns:
        Cropped grid
    """
    bottom = min(top + height, grid.shape[0])
    right = min(left + width, grid.shape[1])

    return grid[top:bottom, left:right].copy()


def crop_to_content(grid: Grid, background: int = 0) -> Grid:
    """
    Crop to bounding box of non-background pixels.

    Args:
        grid: Input grid
        background: Background color to ignore

    Returns:
        Cropped grid
    """
    # Find non-background pixels
    non_bg = np.argwhere(grid != background)

    if len(non_bg) == 0:
        return grid.copy()

    # Get bounding box
    min_r, min_c = non_bg.min(axis=0)
    max_r, max_c = non_bg.max(axis=0)

    return grid[min_r:max_r+1, min_c:max_c+1].copy()


# ============================================================================
# 6. TOPOLOGICAL OPERATIONS
# ============================================================================

def fill_holes(object: Object, grid_shape: Optional[Tuple[int, int]] = None) -> Object:
    """
    Fill interior holes in object.

    Args:
        object: Input object
        grid_shape: Grid dimensions (auto-computed if None)

    Returns:
        Object with holes filled
    """
    if not object:
        return []

    # Create binary grid
    if grid_shape is None:
        min_r, min_c, max_r, max_c = get_object_bounds(object)
        grid_shape = (max_r + 2, max_c + 2)  # Add padding

    grid = np.zeros(grid_shape, dtype=int)
    for r, c in object:
        if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
            grid[r, c] = 1

    # Flood fill from outside to mark external region
    filled = grid.copy()
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(grid).astype(int)

    # Extract filled object
    result = []
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            if filled[r, c] == 1:
                result.append((r, c))

    return result


def grow(object: Object, amount: int = 1, grid_shape: Optional[Tuple[int, int]] = None) -> Object:
    """
    Dilate object by pixels (morphological dilation).

    Args:
        object: Input object
        amount: Number of pixels to expand
        grid_shape: Grid dimensions

    Returns:
        Expanded object
    """
    if not object or amount <= 0:
        return object.copy()

    # Create binary grid
    if grid_shape is None:
        min_r, min_c, max_r, max_c = get_object_bounds(object)
        grid_shape = (max_r + amount + 2, max_c + amount + 2)

    grid = np.zeros(grid_shape, dtype=int)
    for r, c in object:
        if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
            grid[r, c] = 1

    # Apply binary dilation
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(grid, iterations=amount).astype(int)

    # Extract result
    result = []
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            if dilated[r, c] == 1:
                result.append((r, c))

    return result


def shrink(object: Object, amount: int = 1, grid_shape: Optional[Tuple[int, int]] = None) -> Object:
    """
    Erode object (morphological erosion).

    Args:
        object: Input object
        amount: Number of pixels to contract
        grid_shape: Grid dimensions

    Returns:
        Contracted object
    """
    if not object or amount <= 0:
        return object.copy()

    # Create binary grid
    if grid_shape is None:
        min_r, min_c, max_r, max_c = get_object_bounds(object)
        grid_shape = (max_r + 2, max_c + 2)

    grid = np.zeros(grid_shape, dtype=int)
    for r, c in object:
        if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
            grid[r, c] = 1

    # Apply binary erosion
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(grid, iterations=amount).astype(int)

    # Extract result
    result = []
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            if eroded[r, c] == 1:
                result.append((r, c))

    return result


# ============================================================================
# 7. UTILITY OPERATIONS
# ============================================================================

def count(objects: ObjectSet, predicate: Optional[callable] = None) -> int:
    """
    Count objects or objects matching predicate.

    Args:
        objects: Input objects
        predicate: Optional filter function

    Returns:
        Count of objects
    """
    if predicate is None:
        return len(objects)

    return sum(1 for obj in objects if predicate(obj))


def measure(object: Object, metric: str) -> float:
    """
    Measure object property.

    Args:
        object: Input object
        metric: Property to measure ("area", "width", "height", "aspect_ratio",
                "perimeter", "centroid_r", "centroid_c")

    Returns:
        Measured value
    """
    if not object:
        return 0.0

    if metric == "area":
        return float(len(object))

    min_r, min_c, max_r, max_c = get_object_bounds(object)

    if metric == "width":
        return float(max_c - min_c + 1)
    elif metric == "height":
        return float(max_r - min_r + 1)
    elif metric == "aspect_ratio":
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        return float(w) / float(h) if h > 0 else 0.0
    elif metric == "perimeter":
        # Approximate perimeter as boundary pixels
        obj_set = set(object)
        boundary = 0
        for r, c in object:
            neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
            if any(n not in obj_set for n in neighbors):
                boundary += 1
        return float(boundary)
    elif metric == "centroid_r":
        return compute_centroid(object)[0]
    elif metric == "centroid_c":
        return compute_centroid(object)[1]
    else:
        raise ValueError(f"Unknown metric: {metric}")


def sort_objects(objects: ObjectSet, key: str, order: str = 'ascending') -> ObjectSet:
    """
    Sort objects by property.

    Args:
        objects: Input objects
        key: Property to sort by ("size", "width", "height", "centroid_r", "centroid_c")
        order: Sort order ("ascending" or "descending")

    Returns:
        Sorted objects
    """
    reverse = (order == 'descending')

    if key == "size":
        return sorted(objects, key=len, reverse=reverse)
    elif key == "width":
        return sorted(objects, key=lambda obj: measure(obj, "width"), reverse=reverse)
    elif key == "height":
        return sorted(objects, key=lambda obj: measure(obj, "height"), reverse=reverse)
    elif key == "centroid_r":
        return sorted(objects, key=lambda obj: compute_centroid(obj)[0], reverse=reverse)
    elif key == "centroid_c":
        return sorted(objects, key=lambda obj: compute_centroid(obj)[1], reverse=reverse)
    else:
        raise ValueError(f"Unknown key: {key}")
