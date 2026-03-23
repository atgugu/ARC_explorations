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


# ============================================================================
# 8. LINE & PATH OPERATIONS
# ============================================================================

def connect(obj1: Object, obj2: Object, pattern: str = 'line', color: Color = 1) -> Object:
    """
    Draw connection between two objects.

    Args:
        obj1: First object
        obj2: Second object
        pattern: Connection type ('line', 'horizontal', 'vertical', 'manhattan')
        color: Color for the connection (used in rendering)

    Returns:
        Object representing the connection

    Example:
        >>> obj1 = [(0, 0)]
        >>> obj2 = [(5, 5)]
        >>> line = connect(obj1, obj2, 'line')
    """
    if not obj1 or not obj2:
        return []

    # Get centroids
    c1_r, c1_c = compute_centroid(obj1)
    c2_r, c2_c = compute_centroid(obj2)

    # Use closest points for better connections
    start = (int(round(c1_r)), int(round(c1_c)))
    end = (int(round(c2_r)), int(round(c2_c)))

    if pattern == 'line':
        return draw_line(start, end, color)
    elif pattern == 'horizontal':
        # Horizontal then vertical
        mid = (start[0], end[1])
        h_line = draw_line(start, mid, color)
        v_line = draw_line(mid, end, color)
        return h_line + v_line
    elif pattern == 'vertical':
        # Vertical then horizontal
        mid = (end[0], start[1])
        v_line = draw_line(start, mid, color)
        h_line = draw_line(mid, end, color)
        return v_line + h_line
    elif pattern == 'manhattan':
        # Manhattan distance path
        return draw_line(start, end, color)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def draw_line(start: Point, end: Point, color: Color = 1, thickness: int = 1) -> Object:
    """
    Draw line from point to point using Bresenham's algorithm.

    Args:
        start: Starting point (row, col)
        end: Ending point (row, col)
        color: Line color (for rendering)
        thickness: Line thickness (1 = single pixel)

    Returns:
        Object representing the line

    Example:
        >>> line = draw_line((0, 0), (5, 5), color=2)
    """
    r0, c0 = start
    r1, c1 = end

    points = []

    # Bresenham's line algorithm
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    r_step = 1 if r0 < r1 else -1
    c_step = 1 if c0 < c1 else -1

    if dr > dc:
        # More vertical
        error = dr / 2
        c = c0
        for r in range(r0, r1 + r_step, r_step):
            points.append((r, c))
            error -= dc
            if error < 0:
                c += c_step
                error += dr
    else:
        # More horizontal
        error = dc / 2
        r = r0
        for c in range(c0, c1 + c_step, c_step):
            points.append((r, c))
            error -= dr
            if error < 0:
                r += r_step
                error += dc

    # Apply thickness if > 1
    if thickness > 1:
        thick_points = []
        for r, c in points:
            for dr in range(-thickness//2, thickness//2 + 1):
                for dc in range(-thickness//2, thickness//2 + 1):
                    thick_points.append((r + dr, c + dc))
        points = list(set(thick_points))

    return points


def draw_rectangle(top_left: Point, bottom_right: Point, fill: bool = False, color: Color = 1) -> Object:
    """
    Draw rectangle.

    Args:
        top_left: Top-left corner
        bottom_right: Bottom-right corner
        fill: Whether to fill the rectangle
        color: Color for drawing

    Returns:
        Object representing the rectangle

    Example:
        >>> rect = draw_rectangle((2, 2), (8, 8), fill=True, color=3)
    """
    r0, c0 = top_left
    r1, c1 = bottom_right

    points = []

    if fill:
        # Filled rectangle
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                points.append((r, c))
    else:
        # Outline only
        for c in range(c0, c1 + 1):
            points.append((r0, c))  # Top edge
            points.append((r1, c))  # Bottom edge
        for r in range(r0 + 1, r1):
            points.append((r, c0))  # Left edge
            points.append((r, c1))  # Right edge

    return points


def fill_region(grid: Grid, seed_point: Point, new_color: Color) -> Grid:
    """
    Flood fill from seed point.

    Args:
        grid: Input grid
        seed_point: Starting point for flood fill
        new_color: Color to fill with

    Returns:
        Grid with filled region

    Example:
        >>> result = fill_region(grid, (5, 5), 3)
    """
    result = grid.copy()
    r, c = seed_point

    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return result

    old_color = grid[r, c]
    if old_color == new_color:
        return result

    # Flood fill using BFS
    from collections import deque
    queue = deque([(r, c)])
    visited = set()

    while queue:
        curr_r, curr_c = queue.popleft()

        if (curr_r, curr_c) in visited:
            continue

        if not (0 <= curr_r < grid.shape[0] and 0 <= curr_c < grid.shape[1]):
            continue

        if result[curr_r, curr_c] != old_color:
            continue

        visited.add((curr_r, curr_c))
        result[curr_r, curr_c] = new_color

        # Add neighbors (4-connectivity)
        queue.append((curr_r - 1, curr_c))
        queue.append((curr_r + 1, curr_c))
        queue.append((curr_r, curr_c - 1))
        queue.append((curr_r, curr_c + 1))

    return result


def extend_line(line_object: Object, direction: Direction, length: Union[int, str] = 'edge',
                grid_shape: Optional[Tuple[int, int]] = None) -> Object:
    """
    Extend line in direction.

    Args:
        line_object: Line object to extend
        direction: Direction to extend
        length: Extension length (int or 'edge' for grid edge)
        grid_shape: Grid dimensions (required if length='edge')

    Returns:
        Extended line object

    Example:
        >>> extended = extend_line(line, Direction.UP, 'edge', grid.shape)
    """
    if not line_object:
        return []

    # Find the endpoint in the given direction
    dr, dc = direction.value

    # Get the extreme point in the direction
    if dr < 0:  # UP
        endpoint = min(line_object, key=lambda p: p[0])
    elif dr > 0:  # DOWN
        endpoint = max(line_object, key=lambda p: p[0])
    elif dc < 0:  # LEFT
        endpoint = min(line_object, key=lambda p: p[1])
    elif dc > 0:  # RIGHT
        endpoint = max(line_object, key=lambda p: p[1])
    else:
        endpoint = line_object[0]

    # Determine extension length
    if length == 'edge':
        if grid_shape is None:
            raise ValueError("grid_shape required when length='edge'")

        H, W = grid_shape
        r, c = endpoint

        if dr < 0:
            ext_len = r
        elif dr > 0:
            ext_len = H - 1 - r
        elif dc < 0:
            ext_len = c
        elif dc > 0:
            ext_len = W - 1 - c
        else:
            ext_len = 0
    else:
        ext_len = length

    # Create extension
    extension = []
    r, c = endpoint
    for i in range(1, ext_len + 1):
        extension.append((r + i * dr, c + i * dc))

    return line_object + extension


def detect_lines(grid: Grid, direction: Optional[Direction] = None, min_length: int = 2) -> ObjectSet:
    """
    Extract all lines from grid.

    Args:
        grid: Input grid
        direction: Direction filter (None = all directions)
        min_length: Minimum line length

    Returns:
        List of line objects

    Example:
        >>> lines = detect_lines(grid, Direction.HORIZONTAL, min_length=3)
    """
    lines = []

    # Detect horizontal lines
    if direction is None or direction == Direction.RIGHT:
        for r in range(grid.shape[0]):
            line = []
            prev_color = 0
            for c in range(grid.shape[1]):
                color = grid[r, c]
                if color != 0:
                    if color == prev_color or not line:
                        line.append((r, c))
                        prev_color = color
                    else:
                        if len(line) >= min_length:
                            lines.append(line)
                        line = [(r, c)]
                        prev_color = color
                else:
                    if len(line) >= min_length:
                        lines.append(line)
                    line = []
                    prev_color = 0
            if len(line) >= min_length:
                lines.append(line)

    # Detect vertical lines
    if direction is None or direction == Direction.DOWN:
        for c in range(grid.shape[1]):
            line = []
            prev_color = 0
            for r in range(grid.shape[0]):
                color = grid[r, c]
                if color != 0:
                    if color == prev_color or not line:
                        line.append((r, c))
                        prev_color = color
                    else:
                        if len(line) >= min_length:
                            lines.append(line)
                        line = [(r, c)]
                        prev_color = color
                else:
                    if len(line) >= min_length:
                        lines.append(line)
                    line = []
                    prev_color = 0
            if len(line) >= min_length:
                lines.append(line)

    return lines


def trace_boundary(object: Object, color: Color = 1, thickness: int = 1) -> Object:
    """
    Draw outline around object.

    Args:
        object: Input object
        color: Boundary color
        thickness: Boundary thickness

    Returns:
        Object representing the boundary

    Example:
        >>> boundary = trace_boundary(obj, color=2, thickness=1)
    """
    if not object:
        return []

    obj_set = set(object)
    boundary = []

    for r, c in object:
        # Check if on boundary (has at least one non-object neighbor)
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        if any(n not in obj_set for n in neighbors):
            boundary.append((r, c))

    # Apply thickness
    if thickness > 1:
        thick_boundary = []
        for r, c in boundary:
            for dr in range(-thickness//2, thickness//2 + 1):
                for dc in range(-thickness//2, thickness//2 + 1):
                    thick_boundary.append((r + dr, c + dc))
        boundary = list(set(thick_boundary))

    return boundary


# ============================================================================
# EXTENDED SPATIAL TRANSFORMATIONS
# ============================================================================

def gravity(objects: ObjectSet, direction: Direction, until: str = 'edge',
            grid_shape: Optional[Tuple[int, int]] = None) -> ObjectSet:
    """
    Apply gravity - move objects until collision.

    Args:
        objects: Input objects
        direction: Gravity direction
        until: Stop condition ('edge', 'first')
        grid_shape: Grid dimensions (required for 'edge')

    Returns:
        Objects after gravity applied

    Example:
        >>> fallen = gravity(objects, Direction.DOWN, 'edge', grid.shape)
    """
    if not objects:
        return []

    if until == 'edge' and grid_shape is None:
        raise ValueError("grid_shape required when until='edge'")

    dr, dc = direction.value
    result = []

    for obj in objects:
        # Move until collision or edge
        moved = obj.copy()

        if until == 'edge':
            H, W = grid_shape

            # Calculate max movement
            max_move = float('inf')
            for r, c in moved:
                if dr < 0:  # UP
                    max_move = min(max_move, r)
                elif dr > 0:  # DOWN
                    max_move = min(max_move, H - 1 - r)
                elif dc < 0:  # LEFT
                    max_move = min(max_move, c)
                elif dc > 0:  # RIGHT
                    max_move = min(max_move, W - 1 - c)

            # Apply movement
            moved = translate(moved, dr * max_move, dc * max_move)

        result.append(moved)

    return result


def align(objects: ObjectSet, axis: Axis, spacing: int = 0) -> ObjectSet:
    """
    Align objects along axis with spacing.

    Args:
        objects: Input objects
        axis: Alignment axis
        spacing: Gap between objects

    Returns:
        Aligned objects

    Example:
        >>> aligned = align(objects, Axis.HORIZONTAL, spacing=2)
    """
    if not objects or len(objects) <= 1:
        return objects

    result = []
    current_pos = 0

    # Sort objects by position along axis
    if axis == Axis.HORIZONTAL:
        sorted_objs = sorted(objects, key=lambda obj: compute_centroid(obj)[1])
    else:  # VERTICAL
        sorted_objs = sorted(objects, key=lambda obj: compute_centroid(obj)[0])

    for i, obj in enumerate(sorted_objs):
        bounds = get_object_bounds(obj)

        if i == 0:
            # First object stays at original position
            result.append(obj)
            if axis == Axis.HORIZONTAL:
                current_pos = bounds[3] + 1 + spacing  # max_col + 1 + spacing
            else:
                current_pos = bounds[2] + 1 + spacing  # max_row + 1 + spacing
        else:
            # Move subsequent objects
            if axis == Axis.HORIZONTAL:
                delta = current_pos - bounds[1]  # Align to current_pos
                moved = translate(obj, 0, delta)
                new_bounds = get_object_bounds(moved)
                current_pos = new_bounds[3] + 1 + spacing
            else:  # VERTICAL
                delta = current_pos - bounds[0]  # Align to current_pos
                moved = translate(obj, delta, 0)
                new_bounds = get_object_bounds(moved)
                current_pos = new_bounds[2] + 1 + spacing

            result.append(moved)

    return result


def center(object: Object, grid_shape: Tuple[int, int]) -> Object:
    """
    Center object in grid.

    Args:
        object: Input object
        grid_shape: Grid dimensions

    Returns:
        Centered object

    Example:
        >>> centered = center(obj, (10, 10))
    """
    return move_to(object, "center", grid_shape)


# ============================================================================
# EXTENDED SELECTION PRIMITIVES
# ============================================================================

def select_by_shape(objects: ObjectSet, shape: str) -> ObjectSet:
    """
    Filter objects by shape type.

    Args:
        objects: Input objects
        shape: Shape type ("rectangle", "line", "square", "L", "T", "plus", "irregular")

    Returns:
        Objects matching the shape

    Example:
        >>> rectangles = select_by_shape(objects, "rectangle")
    """
    result = []

    for obj in objects:
        detected_shape = detect_shape(obj)
        if detected_shape == shape:
            result.append(obj)

    return result


def detect_shape(object: Object) -> str:
    """
    Detect the shape type of an object.

    Args:
        object: Input object

    Returns:
        Shape type as string
    """
    if not object:
        return "empty"

    if len(object) == 1:
        return "point"

    min_r, min_c, max_r, max_c = get_object_bounds(object)
    width = max_c - min_c + 1
    height = max_r - min_r + 1
    area = len(object)

    # Check for line
    if width == 1 or height == 1:
        if area >= 2:
            return "line"

    # Check for rectangle/square
    if area == width * height:
        if width == height:
            return "square"
        else:
            return "rectangle"

    # Check for hollow rectangle
    expected_hollow = 2 * (width + height) - 4
    if area == expected_hollow and width > 2 and height > 2:
        return "hollow_rectangle"

    # Otherwise irregular
    return "irregular"


def select_touching(reference: Object, objects: ObjectSet, connectivity: int = 4) -> ObjectSet:
    """
    Select objects that touch (are adjacent to) the reference object.

    Args:
        reference: Reference object
        objects: Candidate objects
        connectivity: 4 or 8 connectivity

    Returns:
        Objects touching the reference

    Example:
        >>> neighbors = select_touching(obj, all_objects)
    """
    if not reference:
        return []

    ref_set = set(reference)
    result = []

    for obj in objects:
        if obj == reference:
            continue

        # Check if any pixel of obj is adjacent to any pixel of reference
        touches = False
        for r, c in obj:
            if connectivity == 4:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            else:  # 8-connectivity
                neighbors = [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)]

            if any(n in ref_set for n in neighbors):
                touches = True
                break

        if touches:
            result.append(obj)

    return result


def select_aligned(objects: ObjectSet, axis: Axis, tolerance: float = 0.5) -> ObjectSet:
    """
    Select objects aligned along an axis.

    Args:
        objects: Input objects
        axis: Alignment axis
        tolerance: Maximum deviation from perfect alignment

    Returns:
        Aligned objects

    Example:
        >>> row_objects = select_aligned(objects, Axis.HORIZONTAL, tolerance=1.0)
    """
    if not objects:
        return []

    # Compute centroids
    centroids = [compute_centroid(obj) for obj in objects]

    # Group by alignment
    if axis == Axis.HORIZONTAL:
        # Group by similar row position
        groups = defaultdict(list)
        for obj, (r, c) in zip(objects, centroids):
            key = round(r / tolerance) * tolerance
            groups[key].append(obj)

        # Return largest group
        if groups:
            return max(groups.values(), key=len)
    else:  # VERTICAL
        # Group by similar column position
        groups = defaultdict(list)
        for obj, (r, c) in zip(objects, centroids):
            key = round(c / tolerance) * tolerance
            groups[key].append(obj)

        # Return largest group
        if groups:
            return max(groups.values(), key=len)

    return []


# ============================================================================
# EXTENDED PATTERN OPERATIONS (7 primitives)
# ============================================================================

def tile_with_spacing(object: Object, rows: int, cols: int, spacing: int = 1,
                     color: Optional[Color] = None,
                     grid_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Tile an object with spacing between copies.

    Args:
        object: Object to tile
        rows: Number of rows in tiling
        cols: Number of columns in tiling
        spacing: Gap between tiles (in cells)
        color: Color to use (if None, uses color from object context)
        grid_shape: Output grid size (auto-computed if None)

    Returns:
        Grid with tiled pattern
    """
    if not object:
        if grid_shape:
            return np.zeros(grid_shape, dtype=int)
        return np.zeros((10, 10), dtype=int)

    # Get object bounds
    min_r, min_c, max_r, max_c = get_object_bounds(object)
    obj_height = max_r - min_r + 1
    obj_width = max_c - min_c + 1

    # Normalize object to start at (0, 0)
    normalized = [(r - min_r, c - min_c) for r, c in object]

    # Calculate grid size if not provided
    if grid_shape is None:
        total_height = rows * obj_height + (rows - 1) * spacing
        total_width = cols * obj_width + (cols - 1) * spacing
        grid_shape = (total_height, total_width)

    result = np.zeros(grid_shape, dtype=int)

    # Default color
    if color is None:
        color = 1

    # Tile with spacing
    for r in range(rows):
        for c in range(cols):
            offset_r = r * (obj_height + spacing)
            offset_c = c * (obj_width + spacing)

            for obj_r, obj_c in normalized:
                new_r = offset_r + obj_r
                new_c = offset_c + obj_c

                if 0 <= new_r < grid_shape[0] and 0 <= new_c < grid_shape[1]:
                    result[new_r, new_c] = color

    return result


def copy_to_pattern(object: Object, pattern_object: Object, color: Color = 1,
                   grid_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Copy an object to each position in a pattern object.

    Args:
        object: Object to copy
        pattern_object: Object defining positions where copies should be placed
        color: Color for copied objects
        grid_shape: Output grid size

    Returns:
        Grid with copies at pattern positions
    """
    if not object or not pattern_object:
        if grid_shape:
            return np.zeros(grid_shape, dtype=int)
        return np.zeros((10, 10), dtype=int)

    # Normalize object to (0, 0)
    min_r, min_c, max_r, max_c = get_object_bounds(object)
    normalized = [(r - min_r, c - min_c) for r, c in object]

    # Determine grid size
    if grid_shape is None:
        all_positions = pattern_object + object
        min_r, min_c, max_r, max_c = get_object_bounds(all_positions)
        grid_shape = (max_r + 5, max_c + 5)

    result = np.zeros(grid_shape, dtype=int)

    # Copy object to each pattern position
    for pattern_r, pattern_c in pattern_object:
        for obj_r, obj_c in normalized:
            new_r = pattern_r + obj_r
            new_c = pattern_c + obj_c

            if 0 <= new_r < grid_shape[0] and 0 <= new_c < grid_shape[1]:
                result[new_r, new_c] = color

    return result


def symmetrize(grid: Grid, axis: Axis) -> Grid:
    """
    Make grid symmetric by mirroring across specified axis.

    Args:
        grid: Input grid
        axis: Axis to mirror across (HORIZONTAL, VERTICAL, BOTH, DIAGONAL, ANTI_DIAGONAL)

    Returns:
        Symmetric grid
    """
    result = grid.copy()
    h, w = grid.shape

    if axis == Axis.HORIZONTAL:
        # Mirror top to bottom
        for r in range(h // 2):
            result[h - 1 - r, :] = result[r, :]

    elif axis == Axis.VERTICAL:
        # Mirror left to right
        for c in range(w // 2):
            result[:, w - 1 - c] = result[:, c]

    elif axis == Axis.BOTH:
        # Mirror both axes
        result = symmetrize(result, Axis.HORIZONTAL)
        result = symmetrize(result, Axis.VERTICAL)

    elif axis == Axis.DIAGONAL:
        # Mirror across main diagonal (top-left to bottom-right)
        size = max(h, w)
        temp = np.zeros((size, size), dtype=int)
        temp[:h, :w] = grid

        for r in range(size):
            for c in range(r + 1, size):
                temp[c, r] = temp[r, c]

        result = temp[:h, :w]

    elif axis == Axis.ANTI_DIAGONAL:
        # Mirror across anti-diagonal (top-right to bottom-left)
        size = max(h, w)
        temp = np.zeros((size, size), dtype=int)
        temp[:h, :w] = grid

        for r in range(size):
            for c in range(size - r):
                temp[size - 1 - c, size - 1 - r] = temp[r, c]

        result = temp[:h, :w]

    return result


def extend_pattern(grid: Grid, direction: Direction, steps: int = 1) -> Grid:
    """
    Detect and extend repeating pattern in specified direction.

    Args:
        grid: Input grid with pattern
        direction: Direction to extend
        steps: Number of pattern repetitions to add

    Returns:
        Extended grid
    """
    h, w = grid.shape
    dr, dc = direction.value

    # Try to detect pattern period
    if direction in [Direction.DOWN, Direction.UP]:
        # Vertical pattern - try different periods
        for period in range(1, h // 2 + 1):
            # Check if pattern repeats with this period
            match = True
            for r in range(period, h - period):
                if not np.array_equal(grid[r, :], grid[r % period, :]):
                    match = False
                    break

            if match:
                # Found period, extend
                if direction == Direction.DOWN:
                    # Need enough repetitions to cover original + extension
                    num_reps = (h + steps * period + period - 1) // period
                    extension = np.tile(grid[:period, :], (num_reps, 1))
                    return extension[:h + steps * period, :]
                else:  # UP
                    num_reps = (h + steps * period + period - 1) // period
                    extension = np.tile(grid[-period:, :], (num_reps, 1))
                    return extension[-h - steps * period:, :]

    elif direction in [Direction.RIGHT, Direction.LEFT]:
        # Horizontal pattern
        for period in range(1, w // 2 + 1):
            match = True
            for c in range(period, w - period):
                if not np.array_equal(grid[:, c], grid[:, c % period]):
                    match = False
                    break

            if match:
                if direction == Direction.RIGHT:
                    num_reps = (w + steps * period + period - 1) // period
                    extension = np.tile(grid[:, :period], (1, num_reps))
                    return extension[:, :w + steps * period]
                else:  # LEFT
                    num_reps = (w + steps * period + period - 1) // period
                    extension = np.tile(grid[:, -period:], (1, num_reps))
                    return extension[:, -w - steps * period:]

    # No pattern detected, just tile the whole grid
    if direction in [Direction.DOWN, Direction.UP]:
        return np.tile(grid, (steps + 1, 1))
    else:  # RIGHT or LEFT
        return np.tile(grid, (1, steps + 1))


def rotate_pattern(objects: ObjectSet, center: Point, angles: List[int],
                  colors: Optional[List[Color]] = None,
                  grid_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Create rotational copies of objects around a center point.

    Args:
        objects: Objects to rotate
        center: Center point for rotation
        angles: List of angles to rotate to (in degrees)
        colors: Colors for each rotated copy (optional)
        grid_shape: Output grid size

    Returns:
        Grid with rotational pattern
    """
    if not objects:
        if grid_shape:
            return np.zeros(grid_shape, dtype=int)
        return np.zeros((10, 10), dtype=int)

    # Determine grid size
    if grid_shape is None:
        all_points = []
        for obj in objects:
            all_points.extend(obj)
        min_r, min_c, max_r, max_c = get_object_bounds(all_points)
        grid_shape = (max(max_r + 10, 20), max(max_c + 10, 20))

    result = np.zeros(grid_shape, dtype=int)

    # Default colors
    if colors is None:
        colors = [1] * len(angles)

    # Rotate and place each copy
    for angle, color in zip(angles, colors):
        for obj in objects:
            rotated = rotate(obj, angle, center)
            for r, c in rotated:
                if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                    result[r, c] = color

    return result


def kaleidoscope(object: Object, order: int, color: Optional[Color] = None,
                grid_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Create n-fold rotational symmetry pattern (kaleidoscope effect).

    Args:
        object: Object to create pattern from
        order: Order of symmetry (e.g., 4 for 4-fold symmetry)
        color: Color to use
        grid_shape: Output grid size

    Returns:
        Grid with kaleidoscope pattern
    """
    if not object or order < 2:
        if grid_shape:
            return np.zeros(grid_shape, dtype=int)
        return np.zeros((10, 10), dtype=int)

    # Determine grid size and center
    if grid_shape is None:
        min_r, min_c, max_r, max_c = get_object_bounds(object)
        size = max(max_r - min_r, max_c - min_c) * 3
        grid_shape = (size, size)

    center = (grid_shape[0] // 2, grid_shape[1] // 2)

    # Calculate angles for n-fold symmetry
    angles = [i * (360 // order) for i in range(order)]

    # Default color
    if color is None:
        color = 1

    # Create rotational pattern
    return rotate_pattern([object], center, angles, [color] * len(angles), grid_shape)


def tessellate(objects: ObjectSet, pattern: str = 'square',
              colors: Optional[List[Color]] = None,
              grid_shape: Tuple[int, int] = (30, 30)) -> Grid:
    """
    Arrange objects in tessellation pattern.

    Args:
        objects: Objects to tessellate
        pattern: Tessellation type ('square', 'hexagonal', 'brick', 'triangular')
        colors: Colors for objects (cycles if fewer than objects)
        grid_shape: Output grid size

    Returns:
        Grid with tessellation
    """
    if not objects:
        return np.zeros(grid_shape, dtype=int)

    result = np.zeros(grid_shape, dtype=int)

    # Default colors
    if colors is None:
        colors = [1]

    # Get object size
    obj = objects[0]
    min_r, min_c, max_r, max_c = get_object_bounds(obj)
    obj_height = max_r - min_r + 1
    obj_width = max_c - min_c + 1
    normalized = [(r - min_r, c - min_c) for r, c in obj]

    obj_idx = 0

    if pattern == 'square':
        # Regular square grid
        for r in range(0, grid_shape[0], obj_height):
            for c in range(0, grid_shape[1], obj_width):
                color = colors[obj_idx % len(colors)]
                for obj_r, obj_c in normalized:
                    new_r, new_c = r + obj_r, c + obj_c
                    if 0 <= new_r < grid_shape[0] and 0 <= new_c < grid_shape[1]:
                        result[new_r, new_c] = color
                obj_idx += 1

    elif pattern == 'brick':
        # Brick pattern (offset every other row)
        row_idx = 0
        for r in range(0, grid_shape[0], obj_height):
            offset = (obj_width // 2) if row_idx % 2 == 1 else 0
            for c in range(-offset, grid_shape[1], obj_width):
                color = colors[obj_idx % len(colors)]
                for obj_r, obj_c in normalized:
                    new_r, new_c = r + obj_r, c + obj_c
                    if 0 <= new_r < grid_shape[0] and 0 <= new_c < grid_shape[1]:
                        result[new_r, new_c] = color
                obj_idx += 1
            row_idx += 1

    elif pattern == 'hexagonal':
        # Hexagonal pattern (approximate with offset rows)
        row_idx = 0
        spacing_v = max(1, obj_height * 3 // 4)
        for r in range(0, grid_shape[0], spacing_v):
            offset = (obj_width // 2) if row_idx % 2 == 1 else 0
            for c in range(-offset, grid_shape[1], obj_width):
                color = colors[obj_idx % len(colors)]
                for obj_r, obj_c in normalized:
                    new_r, new_c = r + obj_r, c + obj_c
                    if 0 <= new_r < grid_shape[0] and 0 <= new_c < grid_shape[1]:
                        result[new_r, new_c] = color
                obj_idx += 1
            row_idx += 1

    elif pattern == 'triangular':
        # Triangular pattern (alternating orientation)
        row_idx = 0
        for r in range(0, grid_shape[0], obj_height):
            offset = (obj_width // 2) if row_idx % 2 == 1 else 0
            for c in range(-offset, grid_shape[1], obj_width):
                color = colors[obj_idx % len(colors)]
                # Alternate between normal and rotated
                if (row_idx + c // obj_width) % 2 == 0:
                    obj_to_use = normalized
                else:
                    obj_to_use = [(obj_width - 1 - obj_c, obj_r) for obj_r, obj_c in normalized]

                for obj_r, obj_c in obj_to_use:
                    new_r, new_c = r + obj_r, c + obj_c
                    if 0 <= new_r < grid_shape[0] and 0 <= new_c < grid_shape[1]:
                        result[new_r, new_c] = color
                obj_idx += 1
            row_idx += 1

    return result


# ============================================================================
# EXTENDED GRID OPERATIONS (4 primitives)
# ============================================================================

def split_grid(grid: Grid, rows: int, cols: int) -> List[Grid]:
    """
    Split grid into rows × cols subgrids.

    Args:
        grid: Input grid to split
        rows: Number of rows to split into
        cols: Number of columns to split into

    Returns:
        List of subgrids (row-major order)
    """
    h, w = grid.shape

    # Calculate subgrid dimensions
    subgrid_h = h // rows
    subgrid_w = w // cols

    subgrids = []

    for r in range(rows):
        for c in range(cols):
            start_r = r * subgrid_h
            end_r = start_r + subgrid_h if r < rows - 1 else h
            start_c = c * subgrid_w
            end_c = start_c + subgrid_w if c < cols - 1 else w

            subgrid = grid[start_r:end_r, start_c:end_c].copy()
            subgrids.append(subgrid)

    return subgrids


def merge_grids(subgrids: List[Grid], rows: int, cols: int) -> Grid:
    """
    Merge subgrids into single grid.

    Args:
        subgrids: List of subgrids in row-major order
        rows: Number of rows in layout
        cols: Number of columns in layout

    Returns:
        Merged grid
    """
    if not subgrids:
        return np.zeros((10, 10), dtype=int)

    if len(subgrids) != rows * cols:
        raise ValueError(f"Expected {rows * cols} subgrids, got {len(subgrids)}")

    # Get dimensions
    subgrid_heights = []
    subgrid_widths = []

    for r in range(rows):
        h = subgrids[r * cols].shape[0]
        subgrid_heights.append(h)

    for c in range(cols):
        w = subgrids[c].shape[1]
        subgrid_widths.append(w)

    total_h = sum(subgrid_heights)
    total_w = sum(subgrid_widths)

    result = np.zeros((total_h, total_w), dtype=int)

    # Place subgrids
    curr_r = 0
    for r in range(rows):
        curr_c = 0
        for c in range(cols):
            idx = r * cols + c
            subgrid = subgrids[idx]
            h, w = subgrid.shape

            result[curr_r:curr_r + h, curr_c:curr_c + w] = subgrid

            curr_c += w
        curr_r += subgrid_heights[r]

    return result


def pad(grid: Grid, padding: Union[int, Tuple[int, int, int, int]],
       fill_color: Color = 0) -> Grid:
    """
    Add padding border around grid.

    Args:
        grid: Input grid
        padding: Either single value (all sides) or (top, bottom, left, right)
        fill_color: Color for padding

    Returns:
        Padded grid
    """
    if isinstance(padding, int):
        top = bottom = left = right = padding
    else:
        top, bottom, left, right = padding

    h, w = grid.shape
    new_h = h + top + bottom
    new_w = w + left + right

    result = np.full((new_h, new_w), fill_color, dtype=int)
    result[top:top + h, left:left + w] = grid

    return result


def resize_grid(grid: Grid, new_shape: Tuple[int, int],
               method: str = 'nearest') -> Grid:
    """
    Resize grid to new dimensions.

    Args:
        grid: Input grid
        new_shape: Target (height, width)
        method: Resize method ('nearest', 'crop', 'pad')

    Returns:
        Resized grid
    """
    h, w = grid.shape
    new_h, new_w = new_shape

    if method == 'crop':
        # Crop to fit
        result = grid[:new_h, :new_w].copy()
        if result.shape[0] < new_h or result.shape[1] < new_w:
            # Need padding too
            padded = np.zeros(new_shape, dtype=int)
            padded[:result.shape[0], :result.shape[1]] = result
            return padded
        return result

    elif method == 'pad':
        # Pad to fit
        result = np.zeros(new_shape, dtype=int)
        copy_h = min(h, new_h)
        copy_w = min(w, new_w)
        result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
        return result

    else:  # 'nearest' - nearest neighbor interpolation
        result = np.zeros(new_shape, dtype=int)

        for r in range(new_h):
            for c in range(new_w):
                # Map to original coordinates
                orig_r = int(r * h / new_h)
                orig_c = int(c * w / new_w)

                # Clamp to valid range
                orig_r = min(orig_r, h - 1)
                orig_c = min(orig_c, w - 1)

                result[r, c] = grid[orig_r, orig_c]

        return result


# ============================================================================
# EXTENDED COLOR OPERATIONS (5 primitives)
# ============================================================================

def gradient_color(grid: Grid, start_color: Color, end_color: Color,
                  direction: Axis = Axis.HORIZONTAL,
                  target_color: Optional[Color] = None) -> Grid:
    """
    Apply color gradient from start_color to end_color.

    Args:
        grid: Input grid
        start_color: Starting color
        end_color: Ending color
        direction: Direction of gradient
        target_color: Only apply gradient to cells of this color (None = all non-zero)

    Returns:
        Grid with gradient applied
    """
    result = grid.copy()
    h, w = grid.shape

    # Create color range
    num_steps = max(h, w)
    if start_color == end_color:
        return result

    if direction == Axis.HORIZONTAL:
        for c in range(w):
            # Calculate color for this column
            ratio = c / (w - 1) if w > 1 else 0
            color = int(start_color + ratio * (end_color - start_color))
            color = min(max(color, 0), 9)  # Clamp to valid range

            for r in range(h):
                if target_color is None:
                    if result[r, c] != 0:
                        result[r, c] = color
                else:
                    if result[r, c] == target_color:
                        result[r, c] = color

    elif direction == Axis.VERTICAL:
        for r in range(h):
            ratio = r / (h - 1) if h > 1 else 0
            color = int(start_color + ratio * (end_color - start_color))
            color = min(max(color, 0), 9)

            for c in range(w):
                if target_color is None:
                    if result[r, c] != 0:
                        result[r, c] = color
                else:
                    if result[r, c] == target_color:
                        result[r, c] = color

    return result


def recolor_by_neighbor(grid: Grid, object: Object,
                       rule: str = 'most_common') -> Grid:
    """
    Recolor object based on neighboring colors.

    Args:
        grid: Input grid (provides color context)
        object: Object to recolor
        rule: Recoloring rule ('most_common', 'sum', 'max', 'min')

    Returns:
        Grid with recolored object
    """
    result = grid.copy()

    if not object:
        return result

    # Get neighbor colors for the object
    neighbor_colors = []

    for r, c in object:
        # Check 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if (nr, nc) not in object:  # External neighbor
                    color = grid[nr, nc]
                    if color != 0:
                        neighbor_colors.append(color)

    if not neighbor_colors:
        return result

    # Apply rule
    if rule == 'most_common':
        from collections import Counter
        new_color = Counter(neighbor_colors).most_common(1)[0][0]

    elif rule == 'sum':
        new_color = sum(neighbor_colors) % 10  # Wrap around

    elif rule == 'max':
        new_color = max(neighbor_colors)

    elif rule == 'min':
        new_color = min(neighbor_colors)

    else:
        new_color = neighbor_colors[0]

    # Recolor object
    for r, c in object:
        if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
            result[r, c] = new_color

    return result


def palette_reduce(grid: Grid, num_colors: int = 3) -> Grid:
    """
    Reduce color palette to num_colors using clustering.

    Args:
        grid: Input grid
        num_colors: Target number of colors (excluding background)

    Returns:
        Grid with reduced palette
    """
    result = grid.copy()

    # Get unique colors (excluding 0)
    unique_colors = sorted(set(grid.flatten()) - {0})

    if len(unique_colors) <= num_colors:
        return result

    # Simple quantization: map to evenly spaced colors
    target_colors = [i * 9 // (num_colors - 1) for i in range(num_colors)]
    if target_colors[-1] == 0:
        target_colors[-1] = 9

    # Map each color to nearest target
    color_map = {}
    for color in unique_colors:
        closest = min(target_colors, key=lambda c: abs(c - color))
        color_map[color] = closest

    # Apply mapping
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if result[r, c] in color_map:
                result[r, c] = color_map[result[r, c]]

    return result


def color_cycle(objects: ObjectSet, colors: List[Color],
               grid_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Assign colors to objects cyclically from color list.

    Args:
        objects: Objects to color
        colors: List of colors to cycle through
        grid_shape: Output grid size

    Returns:
        Grid with colored objects
    """
    if not objects or not colors:
        if grid_shape:
            return np.zeros(grid_shape, dtype=int)
        return np.zeros((10, 10), dtype=int)

    # Determine grid size
    if grid_shape is None:
        all_points = []
        for obj in objects:
            all_points.extend(obj)
        min_r, min_c, max_r, max_c = get_object_bounds(all_points)
        grid_shape = (max_r + 1, max_c + 1)

    result = np.zeros(grid_shape, dtype=int)

    # Assign colors cyclically
    for i, obj in enumerate(objects):
        color = colors[i % len(colors)]
        for r, c in obj:
            if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                result[r, c] = color

    return result


def invert_colors(grid: Grid, palette: Optional[List[Color]] = None) -> Grid:
    """
    Invert color mapping.

    Args:
        grid: Input grid
        palette: Color palette to invert (None = use all colors 1-9)

    Returns:
        Grid with inverted colors
    """
    result = grid.copy()

    if palette is None:
        # Invert all non-zero colors: map c -> (10 - c)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if result[r, c] != 0:
                    result[r, c] = 10 - result[r, c]
    else:
        # Invert within palette
        color_map = {}
        n = len(palette)
        for i, color in enumerate(palette):
            color_map[color] = palette[n - 1 - i]

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if result[r, c] in color_map:
                    result[r, c] = color_map[result[r, c]]

    return result


# ============================================================================
# EXTENDED TOPOLOGICAL OPERATIONS (3 primitives)
# ============================================================================

def hollow(object: Object) -> Object:
    """
    Keep only the outline of an object (remove interior).

    Args:
        object: Object to hollow

    Returns:
        Object with only boundary pixels
    """
    if not object:
        return []

    object_set = set(object)
    boundary = []

    # A pixel is on boundary if it has at least one non-object neighbor
    for r, c in object:
        is_boundary = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in object_set:
                is_boundary = True
                break

        if is_boundary:
            boundary.append((r, c))

    return boundary


def convex_hull(object: Object) -> Object:
    """
    Compute convex hull of object.

    Args:
        object: Input object

    Returns:
        Convex hull as object
    """
    if len(object) < 3:
        return object.copy()

    # Simple gift wrapping algorithm for integer coordinates
    points = sorted(object)

    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate and remove duplicates
    hull_points = lower[:-1] + upper[:-1]

    # Fill the hull to get all interior points
    if len(hull_points) < 3:
        return object.copy()

    # Convert hull to filled object
    min_r = min(p[0] for p in hull_points)
    max_r = max(p[0] for p in hull_points)
    min_c = min(p[1] for p in hull_points)
    max_c = max(p[1] for p in hull_points)

    result = []

    # Simple scan-line fill
    for r in range(min_r, max_r + 1):
        # Find intersections with hull boundary at this row
        intersections = []

        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]

            r1, c1 = p1
            r2, c2 = p2

            # Check if edge crosses this row
            if min(r1, r2) <= r <= max(r1, r2):
                if r2 != r1:
                    # Compute column intersection
                    c = c1 + (r - r1) * (c2 - c1) / (r2 - r1)
                    intersections.append(c)
                elif r == r1:
                    intersections.append(c1)

        if intersections:
            intersections = sorted(set(intersections))

            # Fill between pairs of intersections
            for i in range(0, len(intersections) - 1, 2):
                c_start = int(intersections[i])
                c_end = int(intersections[i + 1]) if i + 1 < len(intersections) else int(intersections[i])

                for c in range(c_start, c_end + 1):
                    result.append((r, c))

    # Remove duplicates
    return sorted(list(set(result)))


def skeleton(object: Object, grid_shape: Optional[Tuple[int, int]] = None) -> Object:
    """
    Extract medial axis (skeleton) of object using thinning.

    Args:
        object: Input object
        grid_shape: Grid size for computation

    Returns:
        Skeleton of object
    """
    if not object:
        return []

    # Determine grid size
    if grid_shape is None:
        min_r, min_c, max_r, max_c = get_object_bounds(object)
        grid_shape = (max_r + 2, max_c + 2)

    # Convert to grid
    grid = np.zeros(grid_shape, dtype=int)
    for r, c in object:
        if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
            grid[r, c] = 1

    # Simple thinning algorithm (Zhang-Suen)
    def neighbors(r, c):
        """Get 8-connected neighbors in order"""
        return [
            grid[r - 1, c] if r > 0 else 0,
            grid[r - 1, c + 1] if r > 0 and c < grid_shape[1] - 1 else 0,
            grid[r, c + 1] if c < grid_shape[1] - 1 else 0,
            grid[r + 1, c + 1] if r < grid_shape[0] - 1 and c < grid_shape[1] - 1 else 0,
            grid[r + 1, c] if r < grid_shape[0] - 1 else 0,
            grid[r + 1, c - 1] if r < grid_shape[0] - 1 and c > 0 else 0,
            grid[r, c - 1] if c > 0 else 0,
            grid[r - 1, c - 1] if r > 0 and c > 0 else 0,
        ]

    def transitions(neighbors_list):
        """Count 0->1 transitions"""
        count = 0
        for i in range(8):
            if neighbors_list[i] == 0 and neighbors_list[(i + 1) % 8] == 1:
                count += 1
        return count

    # Iterative thinning
    changed = True
    max_iterations = 100
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        # Step 1
        to_remove = []
        for r in range(1, grid_shape[0] - 1):
            for c in range(1, grid_shape[1] - 1):
                if grid[r, c] == 1:
                    n = neighbors(r, c)
                    num_neighbors = sum(n)
                    num_transitions = transitions(n)

                    if (2 <= num_neighbors <= 6 and
                        num_transitions == 1 and
                        n[0] * n[2] * n[4] == 0 and
                        n[2] * n[4] * n[6] == 0):
                        to_remove.append((r, c))

        for r, c in to_remove:
            grid[r, c] = 0
            changed = True

        # Step 2
        to_remove = []
        for r in range(1, grid_shape[0] - 1):
            for c in range(1, grid_shape[1] - 1):
                if grid[r, c] == 1:
                    n = neighbors(r, c)
                    num_neighbors = sum(n)
                    num_transitions = transitions(n)

                    if (2 <= num_neighbors <= 6 and
                        num_transitions == 1 and
                        n[0] * n[2] * n[6] == 0 and
                        n[0] * n[4] * n[6] == 0):
                        to_remove.append((r, c))

        for r, c in to_remove:
            grid[r, c] = 0
            changed = True

    # Convert back to object
    result = []
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            if grid[r, c] == 1:
                result.append((r, c))

    return result


# ============================================================================
# REMAINING PRIMITIVES (9 final primitives)
# ============================================================================

def select_by_property(objects: ObjectSet, property_name: str,
                      comparator: str, threshold: Union[int, float]) -> ObjectSet:
    """
    Filter objects by computed property.

    Args:
        objects: Objects to filter
        property_name: Property to compute ('area', 'perimeter', 'compactness', 'aspect_ratio')
        comparator: Comparison operator ('==', '<', '>', '<=', '>=', '!=')
        threshold: Value to compare against

    Returns:
        Filtered objects
    """
    result = []

    for obj in objects:
        # Compute property
        if property_name == 'area':
            value = len(obj)

        elif property_name == 'perimeter':
            boundary = hollow(obj)
            value = len(boundary)

        elif property_name == 'compactness':
            area = len(obj)
            boundary = hollow(obj)
            perimeter = len(boundary)
            if perimeter > 0:
                value = (4 * np.pi * area) / (perimeter ** 2)
            else:
                value = 0

        elif property_name == 'aspect_ratio':
            min_r, min_c, max_r, max_c = get_object_bounds(obj)
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            value = width / height if height > 0 else 0

        else:
            continue

        # Apply comparator
        if comparator == '==':
            match = (value == threshold)
        elif comparator == '<':
            match = (value < threshold)
        elif comparator == '>':
            match = (value > threshold)
        elif comparator == '<=':
            match = (value <= threshold)
        elif comparator == '>=':
            match = (value >= threshold)
        elif comparator == '!=':
            match = (value != threshold)
        else:
            match = False

        if match:
            result.append(obj)

    return result


def select_unique_color(grid: Grid) -> ObjectSet:
    """
    Select objects that have a unique color (appear only once).

    Args:
        grid: Input grid

    Returns:
        Objects with unique colors
    """
    from collections import Counter

    # Count objects per color
    color_counts = Counter()

    for color in range(1, 10):
        objects = select_by_color(grid, color)
        if objects:
            color_counts[color] = len(objects)

    # Select colors that appear exactly once
    unique_colors = [color for color, count in color_counts.items() if count == 1]

    result = []
    for color in unique_colors:
        objects = select_by_color(grid, color)
        result.extend(objects)

    return result


def select_by_distance(objects: ObjectSet, reference: Object,
                      min_dist: int = 0, max_dist: int = 100) -> ObjectSet:
    """
    Select objects within distance range from reference.

    Args:
        objects: Objects to filter
        reference: Reference object
        min_dist: Minimum distance
        max_dist: Maximum distance

    Returns:
        Filtered objects
    """
    if not reference:
        return []

    # Compute reference centroid
    ref_r, ref_c = compute_centroid(reference)

    result = []

    for obj in objects:
        # Compute object centroid
        obj_r, obj_c = compute_centroid(obj)

        # Compute distance
        dist = np.sqrt((obj_r - ref_r) ** 2 + (obj_c - ref_c) ** 2)

        if min_dist <= dist <= max_dist:
            result.append(obj)

    return result


def select_background(grid: Grid) -> Object:
    """
    Extract background pattern (most common color regions).

    Args:
        grid: Input grid

    Returns:
        Background object
    """
    # Find most common non-zero color
    from collections import Counter

    colors = grid.flatten()
    color_counts = Counter(colors)

    # Remove background (0)
    if 0 in color_counts:
        del color_counts[0]

    if not color_counts:
        return []

    # Most common color is likely background
    most_common_color = color_counts.most_common(1)[0][0]

    # Get largest connected component of that color
    objects = select_by_color(grid, most_common_color)

    if not objects:
        return []

    # Return largest component
    largest = select_largest(objects, k=1)
    return largest[0] if largest else []


def scale_to_fit(object: Object, target_width: int, target_height: int) -> Object:
    """
    Scale object to fit within target dimensions.

    Args:
        object: Object to scale
        target_width: Target width
        target_height: Target height

    Returns:
        Scaled object
    """
    if not object:
        return []

    # Get current bounds
    min_r, min_c, max_r, max_c = get_object_bounds(object)
    curr_height = max_r - min_r + 1
    curr_width = max_c - min_c + 1

    # Compute scale factor
    scale_h = target_height / curr_height
    scale_w = target_width / curr_width
    scale_factor = min(scale_h, scale_w)

    # Scale using existing scale primitive
    return scale(object, int(scale_factor))


def orbit(object: Object, center: Point, angle: int, radius: Optional[int] = None) -> Object:
    """
    Move object in circular orbit around center.

    Args:
        object: Object to move
        center: Center of orbit
        angle: Angle to rotate to (degrees)
        radius: Orbit radius (None = use current distance)

    Returns:
        Orbited object
    """
    if not object:
        return []

    # Compute object centroid
    obj_r, obj_c = compute_centroid(object)
    center_r, center_c = center

    # Current radius
    curr_radius = np.sqrt((obj_r - center_r) ** 2 + (obj_c - center_c) ** 2)

    if radius is None:
        radius = curr_radius

    # Compute new position
    angle_rad = np.radians(angle)
    new_r = center_r + radius * np.sin(angle_rad)
    new_c = center_c + radius * np.cos(angle_rad)

    # Translate object
    dr = int(new_r - obj_r)
    dc = int(new_c - obj_c)

    return translate(object, dr, dc)


def majority_vote(objects: ObjectSet, property_name: str = 'color',
                 grid: Optional[Grid] = None) -> Union[int, str]:
    """
    Return most common property value among objects.

    Args:
        objects: Objects to analyze
        property_name: Property to vote on ('color', 'size', 'shape')
        grid: Grid context (needed for color)

    Returns:
        Most common value
    """
    from collections import Counter

    values = []

    for obj in objects:
        if property_name == 'size':
            values.append(len(obj))

        elif property_name == 'shape':
            shape = detect_shape(obj)
            values.append(shape)

        elif property_name == 'color' and grid is not None:
            # Get color of object
            if obj:
                r, c = obj[0]
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    values.append(grid[r, c])

    if not values:
        return 0 if property_name in ['color', 'size'] else 'unknown'

    most_common = Counter(values).most_common(1)[0][0]
    return most_common


def distribute_evenly(objects: ObjectSet, grid_shape: Tuple[int, int],
                     axis: Axis = Axis.HORIZONTAL) -> ObjectSet:
    """
    Space objects evenly across grid.

    Args:
        objects: Objects to distribute
        grid_shape: Grid dimensions
        axis: Axis to distribute along

    Returns:
        Objects at new positions
    """
    if not objects:
        return []

    h, w = grid_shape
    n = len(objects)

    result = []

    if axis == Axis.HORIZONTAL:
        # Distribute horizontally
        spacing = w // (n + 1)

        for i, obj in enumerate(objects):
            target_c = (i + 1) * spacing

            # Get current centroid
            obj_r, obj_c = compute_centroid(obj)

            # Translate to target column
            dc = target_c - int(obj_c)
            new_obj = translate(obj, 0, dc)
            result.append(new_obj)

    elif axis == Axis.VERTICAL:
        # Distribute vertically
        spacing = h // (n + 1)

        for i, obj in enumerate(objects):
            target_r = (i + 1) * spacing

            # Get current centroid
            obj_r, obj_c = compute_centroid(obj)

            # Translate to target row
            dr = target_r - int(obj_r)
            new_obj = translate(obj, dr, 0)
            result.append(new_obj)

    else:
        # Both axes - arrange in grid pattern
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        spacing_r = h // (rows + 1)
        spacing_c = w // (cols + 1)

        for i, obj in enumerate(objects):
            row = i // cols
            col = i % cols

            target_r = (row + 1) * spacing_r
            target_c = (col + 1) * spacing_c

            obj_r, obj_c = compute_centroid(obj)

            dr = target_r - int(obj_r)
            dc = target_c - int(obj_c)

            new_obj = translate(obj, dr, dc)
            result.append(new_obj)

    return result


def shortest_path(grid: Grid, start: Point, end: Point,
                 obstacle_color: Optional[Color] = None) -> Object:
    """
    Find shortest path between two points using A*.

    Args:
        grid: Input grid
        start: Start position
        end: End position
        obstacle_color: Color that blocks paths (None = all non-zero)

    Returns:
        Path as object (list of points)
    """
    from heapq import heappush, heappop

    h, w = grid.shape
    start_r, start_c = start
    end_r, end_c = end

    # Validate positions
    if not (0 <= start_r < h and 0 <= start_c < w):
        return []
    if not (0 <= end_r < h and 0 <= end_c < w):
        return []

    # A* search
    def heuristic(r, c):
        return abs(r - end_r) + abs(c - end_c)

    # Priority queue: (f_score, g_score, r, c)
    open_set = [(heuristic(start_r, start_c), 0, start_r, start_c)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, g, r, c = heappop(open_set)

        if (r, c) == end:
            # Reconstruct path
            path = []
            current = end
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        # Check neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < h and 0 <= nc < w):
                continue

            # Check if blocked
            if obstacle_color is None:
                blocked = (grid[nr, nc] != 0)
            else:
                blocked = (grid[nr, nc] == obstacle_color)

            if blocked:
                continue

            tentative_g = g + 1

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative_g
                f_score = tentative_g + heuristic(nr, nc)
                heappush(open_set, (f_score, tentative_g, nr, nc))
                came_from[(nr, nc)] = (r, c)

    # No path found
    return []
