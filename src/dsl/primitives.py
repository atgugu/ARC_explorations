"""
ARC DSL Primitives - Core Implementation

This module implements the fundamental operations for ARC task solving.
65 primitives organized into 8 categories.

Type System:
- Grid: np.ndarray (H×W, dtype=int, values 0-9)
- Object: List[Tuple[int, int]] (list of (row, col) coordinates)
- ObjectSet: List[Object]
- Color: int (0-9, where 0 is background)
"""

from typing import List, Tuple, Union, Callable, Literal, Optional
from enum import Enum
import numpy as np
from dataclasses import dataclass

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


class OverlayMode(Enum):
    """Grid overlay modes"""
    REPLACE = "replace"  # grid2 overwrites grid1 (non-zero)
    OR = "or"            # logical OR
    AND = "and"          # logical AND
    XOR = "xor"          # logical XOR
    ADD = "add"          # addition (clamped to 9)
    BLEND = "blend"      # grid2 where nonzero, else grid1


@dataclass
class ObjectProperties:
    """Computed properties of an object"""
    size: int
    color: Color
    bounding_box: Tuple[int, int, int, int]  # (top, left, height, width)
    centroid: Tuple[float, float]
    shape_type: str  # "rectangle", "line", "L", "T", "plus", "irregular"
    is_symmetric_h: bool
    is_symmetric_v: bool
    has_hole: bool
    is_convex: bool
    aspect_ratio: float


# ============================================================================
# 1. SELECTION & FILTERING (12 primitives)
# ============================================================================

def select_by_color(grid: Grid, color: Color) -> ObjectSet:
    """
    Extract all connected components of a specific color.

    Args:
        grid: Input grid
        color: Color to select (1-9)

    Returns:
        List of objects (each object is a list of (row, col) coordinates)

    Example:
        >>> grid = np.array([[0,1,0], [1,1,0], [0,0,2]])
        >>> objects = select_by_color(grid, 1)
        >>> len(objects)  # One connected blue object
        1
    """
    # TODO: Implement using connected components (4-connectivity or 8-connectivity)
    raise NotImplementedError


def select_by_size(objects: ObjectSet, size: int, comparator: str = '==') -> ObjectSet:
    """
    Filter objects by size.

    Args:
        objects: Input objects
        size: Size threshold
        comparator: Comparison operator ('==', '>', '<', '>=', '<=', '!=')

    Returns:
        Filtered objects

    Example:
        >>> large_objects = select_by_size(objects, 10, '>')
    """
    # TODO: Implement size filtering
    raise NotImplementedError


def select_largest(objects: ObjectSet, k: int = 1) -> ObjectSet:
    """
    Select k largest objects.

    Args:
        objects: Input objects
        k: Number of objects to select

    Returns:
        k largest objects (sorted by size, largest first)

    Example:
        >>> top3 = select_largest(objects, 3)
    """
    # TODO: Implement top-k selection
    raise NotImplementedError


def select_smallest(objects: ObjectSet, k: int = 1) -> ObjectSet:
    """
    Select k smallest objects.

    Args:
        objects: Input objects
        k: Number of objects to select

    Returns:
        k smallest objects (sorted by size, smallest first)
    """
    # TODO: Implement bottom-k selection
    raise NotImplementedError


def select_by_shape(objects: ObjectSet, shape: str) -> ObjectSet:
    """
    Filter objects by shape type.

    Args:
        objects: Input objects
        shape: Shape type ("rectangle", "line", "L", "T", "plus", "square", "irregular")

    Returns:
        Objects matching the shape

    Example:
        >>> rectangles = select_by_shape(objects, "rectangle")
    """
    # TODO: Implement shape classification and filtering
    raise NotImplementedError


def select_by_position(objects: ObjectSet, position: str, grid_shape: Tuple[int, int]) -> ObjectSet:
    """
    Filter objects by grid position.

    Args:
        objects: Input objects
        position: Position descriptor ("corner", "edge", "center", "top", "bottom", "left", "right")
        grid_shape: Shape of the grid (height, width)

    Returns:
        Objects in the specified position

    Example:
        >>> corner_objects = select_by_position(objects, "corner", grid.shape)
    """
    # TODO: Implement position-based filtering
    raise NotImplementedError


def select_by_property(objects: ObjectSet, property_name: str, value) -> ObjectSet:
    """
    Filter objects by computed property.

    Args:
        objects: Input objects
        property_name: Property to check ("has_hole", "is_symmetric", "is_convex", etc.)
        value: Expected value

    Returns:
        Objects with property matching value

    Example:
        >>> symmetric_objects = select_by_property(objects, "is_symmetric_h", True)
    """
    # TODO: Implement property-based filtering
    raise NotImplementedError


def select_unique_color(objects: ObjectSet, grid: Grid) -> ObjectSet:
    """
    Select objects that are the only one of their color in the grid.

    Args:
        objects: Input objects
        grid: The grid (to determine colors)

    Returns:
        Objects with unique colors

    Example:
        >>> unique = select_unique_color(objects, grid)
    """
    # TODO: Implement unique color detection
    raise NotImplementedError


def select_touching(object: Object, objects: ObjectSet) -> ObjectSet:
    """
    Select objects that touch (are adjacent to) the given object.

    Args:
        object: Reference object
        objects: Candidate objects

    Returns:
        Objects touching the reference object

    Example:
        >>> neighbors = select_touching(obj, all_objects)
    """
    # TODO: Implement adjacency detection
    raise NotImplementedError


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
        >>> row_objects = select_aligned(objects, Axis.HORIZONTAL)
    """
    # TODO: Implement alignment detection
    raise NotImplementedError


def select_by_distance(reference: Object, objects: ObjectSet, distance: float, comparator: str = '==') -> ObjectSet:
    """
    Select objects at specific distance from reference.

    Args:
        reference: Reference object
        objects: Candidate objects
        distance: Distance threshold
        comparator: Comparison operator

    Returns:
        Objects satisfying distance constraint

    Example:
        >>> nearby = select_by_distance(obj, others, 5, '<')
    """
    # TODO: Implement distance-based filtering
    raise NotImplementedError


def select_background(grid: Grid) -> Grid:
    """
    Extract the background pattern (most common color or pattern).

    Args:
        grid: Input grid

    Returns:
        Grid with only background

    Example:
        >>> bg = select_background(grid)
    """
    # TODO: Implement background detection
    raise NotImplementedError


# ============================================================================
# 2. SPATIAL TRANSFORMATIONS (10 primitives)
# ============================================================================

def rotate(object: Object, angle: int, center: Optional[Point] = None) -> Object:
    """
    Rotate object by angle around center.

    Args:
        object: Input object
        angle: Rotation angle (90, 180, 270, or arbitrary)
        center: Center of rotation (default: object's centroid)

    Returns:
        Rotated object

    Example:
        >>> rotated = rotate(obj, 90)
    """
    # TODO: Implement rotation
    raise NotImplementedError


def reflect(object: Object, axis: Axis, position: Optional[float] = None) -> Object:
    """
    Reflect object across axis.

    Args:
        object: Input object
        axis: Reflection axis
        position: Axis position (default: through object centroid)

    Returns:
        Reflected object

    Example:
        >>> mirrored = reflect(obj, Axis.VERTICAL)
    """
    # TODO: Implement reflection
    raise NotImplementedError


def translate(object: Object, delta_row: int, delta_col: int) -> Object:
    """
    Move object by offset.

    Args:
        object: Input object
        delta_row: Row offset
        delta_col: Column offset

    Returns:
        Translated object

    Example:
        >>> moved = translate(obj, 2, -3)
    """
    return [(r + delta_row, c + delta_col) for r, c in object]


def scale(object: Object, factor: int, center: Optional[Point] = None) -> Object:
    """
    Scale object by integer factor.

    Args:
        object: Input object
        factor: Scale factor (integer)
        center: Center of scaling (default: object's top-left)

    Returns:
        Scaled object

    Example:
        >>> bigger = scale(obj, 2)
    """
    # TODO: Implement scaling
    raise NotImplementedError


def scale_to_fit(object: Object, width: int, height: int) -> Object:
    """
    Scale object to specific dimensions.

    Args:
        object: Input object
        width: Target width
        height: Target height

    Returns:
        Resized object

    Example:
        >>> resized = scale_to_fit(obj, 5, 5)
    """
    # TODO: Implement resize
    raise NotImplementedError


def move_to(object: Object, position: Union[str, Point], grid_shape: Tuple[int, int]) -> Object:
    """
    Move object to absolute position.

    Args:
        object: Input object
        position: Target position ("top_left", "center", etc. or (row, col))
        grid_shape: Grid dimensions

    Returns:
        Moved object

    Example:
        >>> centered = move_to(obj, "center", (10, 10))
    """
    # TODO: Implement absolute positioning
    raise NotImplementedError


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
        >>> aligned = align(objects, Axis.HORIZONTAL, 1)
    """
    # TODO: Implement alignment
    raise NotImplementedError


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
    # TODO: Implement centering
    raise NotImplementedError


def gravity(objects: ObjectSet, direction: Direction, until: str = 'edge', grid_shape: Optional[Tuple[int, int]] = None) -> ObjectSet:
    """
    Apply gravity - move objects until collision.

    Args:
        objects: Input objects
        direction: Gravity direction
        until: Stop condition ('edge', 'object', 'first')
        grid_shape: Grid dimensions (needed for 'edge')

    Returns:
        Objects after gravity applied

    Example:
        >>> fallen = gravity(objects, Direction.DOWN, 'edge', grid.shape)
    """
    # TODO: Implement gravity simulation
    raise NotImplementedError


def orbit(object: Object, center: Point, angle: int) -> Object:
    """
    Rotate object around a point.

    Args:
        object: Input object
        center: Center point
        angle: Rotation angle

    Returns:
        Rotated object

    Example:
        >>> orbited = orbit(obj, (5, 5), 90)
    """
    # TODO: Implement orbit
    raise NotImplementedError


# ============================================================================
# 3. COLOR OPERATIONS (8 primitives)
# ============================================================================

def recolor(object: Object, new_color: Color, grid: Grid) -> Grid:
    """
    Change object to new color.

    Args:
        object: Input object
        new_color: Target color
        grid: Grid to modify

    Returns:
        Modified grid

    Example:
        >>> grid = recolor(obj, 3, grid)
    """
    # TODO: Implement recoloring
    raise NotImplementedError


def recolor_by_rule(objects: ObjectSet, rule: str, grid: Grid) -> Grid:
    """
    Recolor objects based on rule.

    Args:
        objects: Input objects
        rule: Recoloring rule ("size_ascending", "position_based", "distance_from_center", etc.)
        grid: Grid to modify

    Returns:
        Modified grid

    Example:
        >>> grid = recolor_by_rule(objects, "size_ascending", grid)
    """
    # TODO: Implement rule-based recoloring
    raise NotImplementedError


def swap_colors(grid: Grid, color1: Color, color2: Color) -> Grid:
    """
    Swap two colors globally in grid.

    Args:
        grid: Input grid
        color1: First color
        color2: Second color

    Returns:
        Grid with swapped colors

    Example:
        >>> swapped = swap_colors(grid, 1, 2)
    """
    result = grid.copy()
    mask1 = grid == color1
    mask2 = grid == color2
    result[mask1] = color2
    result[mask2] = color1
    return result


def gradient_color(objects: ObjectSet, start_color: Color, end_color: Color, axis: Axis, grid: Grid) -> Grid:
    """
    Apply color gradient to objects.

    Args:
        objects: Input objects
        start_color: Starting color
        end_color: Ending color
        axis: Gradient direction
        grid: Grid to modify

    Returns:
        Modified grid

    Example:
        >>> grid = gradient_color(objects, 1, 9, Axis.HORIZONTAL, grid)
    """
    # TODO: Implement gradient
    raise NotImplementedError


def recolor_by_neighbor(objects: ObjectSet, rule: str, grid: Grid) -> Grid:
    """
    Recolor based on neighboring colors.

    Args:
        objects: Input objects
        rule: Neighbor rule ("majority", "sum", "max", "min")
        grid: Grid to modify

    Returns:
        Modified grid

    Example:
        >>> grid = recolor_by_neighbor(objects, "majority", grid)
    """
    # TODO: Implement neighbor-based recoloring
    raise NotImplementedError


def palette_reduce(grid: Grid, num_colors: int) -> Grid:
    """
    Reduce to top-k most common colors.

    Args:
        grid: Input grid
        num_colors: Number of colors to keep

    Returns:
        Grid with reduced palette

    Example:
        >>> reduced = palette_reduce(grid, 3)
    """
    # TODO: Implement palette reduction
    raise NotImplementedError


def color_cycle(objects: ObjectSet, colors: List[Color], start: int = 0, grid: Grid) -> Grid:
    """
    Assign colors cyclically to objects.

    Args:
        objects: Input objects
        colors: List of colors to cycle through
        start: Starting index in color list
        grid: Grid to modify

    Returns:
        Modified grid

    Example:
        >>> grid = color_cycle(objects, [1, 2, 3], 0, grid)
    """
    # TODO: Implement color cycling
    raise NotImplementedError


def invert_colors(grid: Grid, palette: Optional[List[int]] = None) -> Grid:
    """
    Invert color mapping.

    Args:
        grid: Input grid
        palette: Color pairs to invert (default: all non-zero colors)

    Returns:
        Grid with inverted colors

    Example:
        >>> inverted = invert_colors(grid, [0, 9])
    """
    # TODO: Implement color inversion
    raise NotImplementedError


# ============================================================================
# 4. PATTERN OPERATIONS (9 primitives)
# ============================================================================

def tile(object: Object, rows: int, cols: int, grid_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Tile object in rows×cols pattern.

    Args:
        object: Input object
        rows: Number of rows
        cols: Number of columns
        grid_shape: Output grid shape (auto-computed if None)

    Returns:
        Tiled grid

    Example:
        >>> tiled = tile(obj, 3, 3)
    """
    # TODO: Implement tiling
    raise NotImplementedError


def tile_with_spacing(object: Object, rows: int, cols: int, spacing: int, grid_shape: Optional[Tuple[int, int]] = None) -> Grid:
    """
    Tile with gap between copies.

    Args:
        object: Input object
        rows: Number of rows
        cols: Number of columns
        spacing: Gap size
        grid_shape: Output grid shape

    Returns:
        Tiled grid with spacing

    Example:
        >>> tiled = tile_with_spacing(obj, 2, 2, 1)
    """
    # TODO: Implement tiling with spacing
    raise NotImplementedError


def copy_to_positions(object: Object, positions: List[Point], grid_shape: Tuple[int, int], color: Optional[Color] = None) -> Grid:
    """
    Copy object to specific positions.

    Args:
        object: Input object
        positions: Target positions (top-left corners)
        grid_shape: Output grid shape
        color: Optional color override

    Returns:
        Grid with copied objects

    Example:
        >>> grid = copy_to_positions(obj, [(0,0), (5,5)], (10,10))
    """
    # TODO: Implement copying to positions
    raise NotImplementedError


def copy_to_pattern(object: Object, pattern_object: Object, grid: Grid) -> Grid:
    """
    Copy object to all positions where pattern exists.

    Args:
        object: Object to copy
        pattern_object: Pattern to match
        grid: Grid containing pattern

    Returns:
        Modified grid

    Example:
        >>> grid = copy_to_pattern(obj, pattern, grid)
    """
    # TODO: Implement pattern-based copying
    raise NotImplementedError


def symmetrize(grid: Grid, axis: Axis) -> Grid:
    """
    Make grid symmetric by mirroring.

    Args:
        grid: Input grid
        axis: Symmetry axis

    Returns:
        Symmetric grid

    Example:
        >>> symmetric = symmetrize(grid, Axis.VERTICAL)
    """
    # TODO: Implement symmetrization
    raise NotImplementedError


def extend_pattern(grid: Grid, direction: Direction, steps: int) -> Grid:
    """
    Detect and extend repeating pattern.

    Args:
        grid: Input grid
        direction: Extension direction
        steps: Number of pattern repetitions to add

    Returns:
        Extended grid

    Example:
        >>> extended = extend_pattern(grid, Direction.RIGHT, 3)
    """
    # TODO: Implement pattern extension
    raise NotImplementedError


def rotate_pattern(objects: ObjectSet, center: Point, angles: List[int], grid_shape: Tuple[int, int]) -> Grid:
    """
    Create rotational copies around center.

    Args:
        objects: Input objects
        center: Center point
        angles: List of rotation angles
        grid_shape: Output grid shape

    Returns:
        Grid with rotational pattern

    Example:
        >>> grid = rotate_pattern(objects, (5,5), [90, 180, 270], (10,10))
    """
    # TODO: Implement rotational pattern
    raise NotImplementedError


def kaleidoscope(object: Object, order: int, grid_shape: Tuple[int, int]) -> Grid:
    """
    Create kaleidoscope pattern with n-fold symmetry.

    Args:
        object: Input object
        order: Symmetry order (2, 4, 6, 8, etc.)
        grid_shape: Output grid shape

    Returns:
        Kaleidoscope pattern

    Example:
        >>> pattern = kaleidoscope(obj, 4, (20,20))
    """
    # TODO: Implement kaleidoscope
    raise NotImplementedError


def tessellate(objects: ObjectSet, pattern: str, grid_shape: Tuple[int, int]) -> Grid:
    """
    Arrange objects in tessellation pattern.

    Args:
        objects: Input objects
        pattern: Tessellation type ("square", "hex", "brick", "triangular")
        grid_shape: Output grid shape

    Returns:
        Tessellated grid

    Example:
        >>> grid = tessellate(objects, "hexagonal", (20,20))
    """
    # TODO: Implement tessellation
    raise NotImplementedError


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def object_to_grid(object: Object, color: Color, grid_shape: Tuple[int, int]) -> Grid:
    """Convert object coordinates to grid representation."""
    grid = np.zeros(grid_shape, dtype=int)
    for r, c in object:
        if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
            grid[r, c] = color
    return grid


def get_object_color(object: Object, grid: Grid) -> Color:
    """Get the color of an object from grid."""
    if not object:
        return 0
    r, c = object[0]
    return int(grid[r, c])


def compute_properties(object: Object, grid: Grid) -> ObjectProperties:
    """Compute all properties of an object."""
    # TODO: Implement property computation
    raise NotImplementedError


# ============================================================================
# NOTE: This file contains skeleton implementations.
# Each function marked with "raise NotImplementedError" needs to be implemented.
# See DSL_PRIMITIVES.md for full specifications and examples.
# ============================================================================
