"""
Domain-Specific Language (DSL) for ARC transformations.
Rich set of primitive operations for grid manipulation.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from scipy import ndimage


class Operation:
    """Base class for DSL operations."""

    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Execute the operation."""
        try:
            return self.func(grid, **kwargs)
        except Exception as e:
            # Return input on error
            return grid.copy()

    def __repr__(self):
        return f"Operation({self.name})"


class CompositeOperation(Operation):
    """Composite operation that sequences multiple operations."""

    def __init__(self, operations: List[Operation], name: str = None):
        self.operations = operations
        names = [op.name for op in operations]
        name = name or " → ".join(names)
        description = f"Sequence: {' then '.join(names)}"
        super().__init__(name, self._execute, description)

    def _execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Execute operations in sequence."""
        result = grid.copy()
        for op in self.operations:
            result = op(result, **kwargs)
        return result


# ============================================================================
# GRID OPERATIONS
# ============================================================================

def identity(grid: np.ndarray) -> np.ndarray:
    """Return grid unchanged."""
    return grid.copy()


def rotate_90(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)


def rotate_180(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 180 degrees."""
    return np.rot90(grid, k=2)


def rotate_270(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 270 degrees clockwise."""
    return np.rot90(grid, k=-3)


def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    """Flip grid horizontally."""
    return np.fliplr(grid)


def flip_vertical(grid: np.ndarray) -> np.ndarray:
    """Flip grid vertically."""
    return np.flipud(grid)


def transpose_grid(grid: np.ndarray) -> np.ndarray:
    """Transpose grid."""
    return grid.T


def tile_2x2(grid: np.ndarray) -> np.ndarray:
    """Tile grid 2x2."""
    return np.tile(grid, (2, 2))


def tile_3x3(grid: np.ndarray) -> np.ndarray:
    """Tile grid 3x3."""
    return np.tile(grid, (3, 3))


def tile_2x1(grid: np.ndarray) -> np.ndarray:
    """Tile grid 2x1 (vertical)."""
    return np.tile(grid, (2, 1))


def tile_1x2(grid: np.ndarray) -> np.ndarray:
    """Tile grid 1x2 (horizontal)."""
    return np.tile(grid, (1, 2))


# ============================================================================
# OBJECT OPERATIONS
# ============================================================================

def extract_largest_object(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Extract the largest connected component."""
    binary = grid != background
    labeled, num_features = ndimage.label(binary)

    if num_features == 0:
        return grid.copy()

    # Find largest
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1

    # Create mask
    mask = labeled == largest_label
    result = np.zeros_like(grid)
    result[mask] = grid[mask]

    return result


def extract_by_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Extract only cells of a specific color."""
    result = np.zeros_like(grid)
    result[grid == color] = color
    return result


def remove_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Remove all cells of a specific color (set to 0)."""
    result = grid.copy()
    result[grid == color] = 0
    return result


def keep_top_half(grid: np.ndarray) -> np.ndarray:
    """Keep only top half of grid."""
    h = grid.shape[0]
    result = grid[:h//2, :].copy()
    return result


def keep_bottom_half(grid: np.ndarray) -> np.ndarray:
    """Keep only bottom half of grid."""
    h = grid.shape[0]
    result = grid[h//2:, :].copy()
    return result


def keep_left_half(grid: np.ndarray) -> np.ndarray:
    """Keep only left half of grid."""
    w = grid.shape[1]
    result = grid[:, :w//2].copy()
    return result


def keep_right_half(grid: np.ndarray) -> np.ndarray:
    """Keep only right half of grid."""
    w = grid.shape[1]
    result = grid[:, w//2:].copy()
    return result


# ============================================================================
# COLOR OPERATIONS
# ============================================================================

def invert_colors(grid: np.ndarray, max_color: int = 9) -> np.ndarray:
    """Invert colors (0→9, 1→8, etc)."""
    return max_color - grid


def swap_colors_01(grid: np.ndarray) -> np.ndarray:
    """Swap colors 0 and 1."""
    result = grid.copy()
    result[grid == 0] = 10  # Temp
    result[grid == 1] = 0
    result[result == 10] = 1
    return result


def swap_colors_12(grid: np.ndarray) -> np.ndarray:
    """Swap colors 1 and 2."""
    result = grid.copy()
    result[grid == 1] = 10
    result[grid == 2] = 1
    result[result == 10] = 2
    return result


def increment_colors(grid: np.ndarray) -> np.ndarray:
    """Increment all colors by 1 (mod 10)."""
    return (grid + 1) % 10


def decrement_colors(grid: np.ndarray) -> np.ndarray:
    """Decrement all colors by 1 (mod 10)."""
    return (grid - 1) % 10


def replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
    """Replace one color with another."""
    result = grid.copy()
    result[grid == old_color] = new_color
    return result


# ============================================================================
# SPATIAL OPERATIONS
# ============================================================================

def translate_up(grid: np.ndarray, amount: int = 1) -> np.ndarray:
    """Translate grid up."""
    if amount >= grid.shape[0]:
        return np.zeros_like(grid)

    result = np.zeros_like(grid)
    result[:-amount, :] = grid[amount:, :]
    return result


def translate_down(grid: np.ndarray, amount: int = 1) -> np.ndarray:
    """Translate grid down."""
    if amount >= grid.shape[0]:
        return np.zeros_like(grid)

    result = np.zeros_like(grid)
    result[amount:, :] = grid[:-amount, :]
    return result


def translate_left(grid: np.ndarray, amount: int = 1) -> np.ndarray:
    """Translate grid left."""
    if amount >= grid.shape[1]:
        return np.zeros_like(grid)

    result = np.zeros_like(grid)
    result[:, :-amount] = grid[:, amount:]
    return result


def translate_right(grid: np.ndarray, amount: int = 1) -> np.ndarray:
    """Translate grid right."""
    if amount >= grid.shape[1]:
        return np.zeros_like(grid)

    result = np.zeros_like(grid)
    result[:, amount:] = grid[:, :-amount]
    return result


def scale_2x(grid: np.ndarray) -> np.ndarray:
    """Scale grid 2x by repeating pixels."""
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def scale_3x(grid: np.ndarray) -> np.ndarray:
    """Scale grid 3x by repeating pixels."""
    return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)


def downsample_2x(grid: np.ndarray) -> np.ndarray:
    """Downsample grid by 2x."""
    return grid[::2, ::2]


def downsample_3x(grid: np.ndarray) -> np.ndarray:
    """Downsample grid by 3x."""
    return grid[::3, ::3]


# ============================================================================
# PATTERN OPERATIONS
# ============================================================================

def fill_zeros_with_pattern(grid: np.ndarray) -> np.ndarray:
    """Fill zeros with most common non-zero color."""
    nonzero = grid[grid != 0]
    if len(nonzero) == 0:
        return grid.copy()

    # Find most common
    counts = np.bincount(nonzero)
    most_common = np.argmax(counts)

    result = grid.copy()
    result[grid == 0] = most_common
    return result


def outline_objects(grid: np.ndarray, background: int = 0, outline_color: int = 1) -> np.ndarray:
    """Outline objects (find edges)."""
    binary = grid != background
    from scipy.ndimage import binary_dilation, binary_erosion

    dilated = binary_dilation(binary)
    eroded = binary_erosion(binary)
    edges = dilated & ~eroded

    result = grid.copy()
    result[edges] = outline_color
    return result


def fill_enclosed_regions(grid: np.ndarray, fill_color: int = 1) -> np.ndarray:
    """Fill enclosed regions."""
    from scipy.ndimage import binary_fill_holes

    result = grid.copy()
    for color in range(1, 10):
        binary = grid == color
        if binary.any():
            filled = binary_fill_holes(binary)
            result[filled & ~binary] = fill_color

    return result


# ============================================================================
# DSL REGISTRY
# ============================================================================

class DSLRegistry:
    """Registry of all DSL operations."""

    def __init__(self):
        self.operations: Dict[str, Operation] = {}
        self._register_all()

    def _register_all(self):
        """Register all primitive operations."""

        # Grid operations
        self.register(Operation("identity", identity, "No change"))
        self.register(Operation("rotate_90", rotate_90, "Rotate 90° clockwise"))
        self.register(Operation("rotate_180", rotate_180, "Rotate 180°"))
        self.register(Operation("rotate_270", rotate_270, "Rotate 270° clockwise"))
        self.register(Operation("flip_h", flip_horizontal, "Flip horizontally"))
        self.register(Operation("flip_v", flip_vertical, "Flip vertically"))
        self.register(Operation("transpose", transpose_grid, "Transpose"))
        self.register(Operation("tile_2x2", tile_2x2, "Tile 2x2"))
        self.register(Operation("tile_3x3", tile_3x3, "Tile 3x3"))
        self.register(Operation("tile_2x1", tile_2x1, "Tile 2x1"))
        self.register(Operation("tile_1x2", tile_1x2, "Tile 1x2"))

        # Object operations
        self.register(Operation("largest_object", extract_largest_object, "Extract largest object"))
        self.register(Operation("top_half", keep_top_half, "Keep top half"))
        self.register(Operation("bottom_half", keep_bottom_half, "Keep bottom half"))
        self.register(Operation("left_half", keep_left_half, "Keep left half"))
        self.register(Operation("right_half", keep_right_half, "Keep right half"))

        # Color operations
        self.register(Operation("invert_colors", invert_colors, "Invert colors"))
        self.register(Operation("swap_01", swap_colors_01, "Swap colors 0↔1"))
        self.register(Operation("swap_12", swap_colors_12, "Swap colors 1↔2"))
        self.register(Operation("increment", increment_colors, "Increment colors"))
        self.register(Operation("decrement", decrement_colors, "Decrement colors"))

        # Spatial operations
        self.register(Operation("translate_up", translate_up, "Translate up 1"))
        self.register(Operation("translate_down", translate_down, "Translate down 1"))
        self.register(Operation("translate_left", translate_left, "Translate left 1"))
        self.register(Operation("translate_right", translate_right, "Translate right 1"))
        self.register(Operation("scale_2x", scale_2x, "Scale 2x"))
        self.register(Operation("scale_3x", scale_3x, "Scale 3x"))
        self.register(Operation("downsample_2x", downsample_2x, "Downsample 2x"))
        self.register(Operation("downsample_3x", downsample_3x, "Downsample 3x"))

        # Pattern operations
        self.register(Operation("fill_zeros", fill_zeros_with_pattern, "Fill zeros with pattern"))
        self.register(Operation("outline", outline_objects, "Outline objects"))
        self.register(Operation("fill_enclosed", fill_enclosed_regions, "Fill enclosed regions"))

    def register(self, operation: Operation):
        """Register an operation."""
        self.operations[operation.name] = operation

    def get(self, name: str) -> Optional[Operation]:
        """Get an operation by name."""
        return self.operations.get(name)

    def get_all(self) -> List[Operation]:
        """Get all operations."""
        return list(self.operations.values())

    def create_composite(self, op_names: List[str], name: str = None) -> Optional[CompositeOperation]:
        """Create a composite operation from a list of operation names."""
        ops = []
        for name in op_names:
            op = self.get(name)
            if op is None:
                return None
            ops.append(op)

        return CompositeOperation(ops, name)
