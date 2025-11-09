"""
Object-Based Reasoning Module

Adds object-level understanding to the curiosity-driven solver.

Key Design:
- Objects are detected and extracted from grids
- Objects have properties: position, size, color, shape
- Transformations can operate on objects or grids
- Belief dynamics work over both pixel-level and object-level hypotheses
"""

import numpy as np
from scipy.ndimage import label, center_of_mass
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ArcObject:
    """Represents a detected object in an ARC grid."""
    mask: np.ndarray  # Binary mask of object
    grid: np.ndarray  # Original grid values under mask
    bbox: Tuple[int, int, int, int]  # (y1, x1, y2, x2)

    # Properties
    position: Tuple[float, float] = field(default=(0, 0))  # Center of mass
    size: int = 0  # Number of pixels
    colors: List[int] = field(default_factory=list)  # Unique colors
    dominant_color: int = 0  # Most common color

    # Shape descriptors
    height: int = 0
    width: int = 0
    aspect_ratio: float = 1.0
    density: float = 1.0  # Filled area / bounding box area


class ObjectDetector:
    """Detects and extracts objects from ARC grids."""

    def __init__(self, background_color: int = 0, connectivity: int = 4):
        self.background_color = background_color
        self.connectivity = connectivity

    def detect_objects(self, grid: np.ndarray) -> List[ArcObject]:
        """
        Detect all objects in grid.

        Objects are connected components of non-background pixels.
        """
        # Create foreground mask
        foreground = grid != self.background_color

        # Label connected components
        if self.connectivity == 4:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:
            structure = np.ones((3, 3))

        labeled, n_objects = label(foreground, structure=structure)

        objects = []
        for obj_id in range(1, n_objects + 1):
            obj_mask = labeled == obj_id
            obj = self._extract_object(grid, obj_mask)
            objects.append(obj)

        return objects

    def _extract_object(self, grid: np.ndarray, mask: np.ndarray) -> ArcObject:
        """Extract object properties from mask."""
        # Bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            # Empty object
            return ArcObject(
                mask=mask,
                grid=np.zeros_like(grid),
                bbox=(0, 0, 0, 0)
            )

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        bbox = (int(y1), int(x1), int(y2 + 1), int(x2 + 1))

        # Extract grid values
        obj_grid = grid * mask

        # Properties
        position = center_of_mass(mask)
        size = int(np.sum(mask))

        # Colors
        obj_values = grid[mask]
        colors = list(np.unique(obj_values))
        unique, counts = np.unique(obj_values, return_counts=True)
        dominant_color = int(unique[np.argmax(counts)])

        # Shape
        height = y2 - y1 + 1
        width = x2 - x1 + 1
        aspect_ratio = height / width if width > 0 else 1.0
        density = size / (height * width) if (height * width) > 0 else 1.0

        return ArcObject(
            mask=mask,
            grid=obj_grid,
            bbox=bbox,
            position=position,
            size=size,
            colors=colors,
            dominant_color=dominant_color,
            height=height,
            width=width,
            aspect_ratio=aspect_ratio,
            density=density
        )

    def detect_by_color(self, grid: np.ndarray) -> Dict[int, List[ArcObject]]:
        """Group objects by their dominant color."""
        objects = self.detect_objects(grid)

        by_color = {}
        for obj in objects:
            color = obj.dominant_color
            if color not in by_color:
                by_color[color] = []
            by_color[color].append(obj)

        return by_color

    def detect_largest(self, grid: np.ndarray) -> Optional[ArcObject]:
        """Detect the largest object."""
        objects = self.detect_objects(grid)
        if not objects:
            return None

        return max(objects, key=lambda obj: obj.size)

    def count_objects(self, grid: np.ndarray) -> int:
        """Count number of objects."""
        return len(self.detect_objects(grid))


class ObjectTransformations:
    """Object-level transformation operations."""

    def __init__(self):
        self.detector = ObjectDetector()

    def recolor_objects(self,
                       grid: np.ndarray,
                       condition_fn: callable,
                       new_color: int) -> np.ndarray:
        """
        Recolor objects matching a condition.

        Example:
            recolor_objects(grid, lambda obj: obj.size > 10, color=2)
            # Recolors all objects with more than 10 pixels to color 2
        """
        objects = self.detector.detect_objects(grid)
        result = grid.copy()

        for obj in objects:
            if condition_fn(obj):
                result[obj.mask] = new_color

        return result

    def move_objects(self,
                    grid: np.ndarray,
                    target_fn: callable,
                    fill_value: int = 0) -> np.ndarray:
        """
        Move objects to target positions.

        Example:
            move_objects(grid, lambda obj: (0, obj.position[1]))
            # Moves all objects to top row
        """
        objects = self.detector.detect_objects(grid)
        result = np.full_like(grid, fill_value)

        for obj in objects:
            # Get target position
            target_pos = target_fn(obj)

            # Extract object from bounding box
            y1, x1, y2, x2 = obj.bbox
            obj_bbox = obj.grid[y1:y2, x1:x2]
            obj_mask_bbox = obj.mask[y1:y2, x1:x2]

            # Calculate offset
            current_center = obj.position
            dy = int(target_pos[0] - current_center[0])
            dx = int(target_pos[1] - current_center[1])

            # New position
            new_y1 = y1 + dy
            new_x1 = x1 + dx
            new_y2 = new_y1 + (y2 - y1)
            new_x2 = new_x1 + (x2 - x1)

            # Place object if within bounds
            if (0 <= new_y1 and new_y2 <= grid.shape[0] and
                0 <= new_x1 and new_x2 <= grid.shape[1]):
                result[new_y1:new_y2, new_x1:new_x2][obj_mask_bbox] = obj_bbox[obj_mask_bbox]

        return result

    def scale_objects(self,
                     grid: np.ndarray,
                     scale_factor: int,
                     fill_value: int = 0) -> np.ndarray:
        """
        Scale all objects by factor.

        Scales the grid and each object independently.
        """
        from arc_curiosity_solver.transformations.arc_primitives import ARCPrimitives

        objects = self.detector.detect_objects(grid)

        # Scale grid
        new_shape = (grid.shape[0] * scale_factor, grid.shape[1] * scale_factor)
        result = np.full(new_shape, fill_value)

        for obj in objects:
            # Extract object
            y1, x1, y2, x2 = obj.bbox
            obj_bbox = obj.grid[y1:y2, x1:x2]

            # Scale object
            scaled_obj = ARCPrimitives.scale(obj_bbox, scale_factor)

            # Place in new grid
            new_y1 = y1 * scale_factor
            new_x1 = x1 * scale_factor
            new_y2 = new_y1 + scaled_obj.shape[0]
            new_x2 = new_x1 + scaled_obj.shape[1]

            if new_y2 <= result.shape[0] and new_x2 <= result.shape[1]:
                # Only copy non-zero pixels
                mask = scaled_obj != fill_value
                result[new_y1:new_y2, new_x1:new_x2][mask] = scaled_obj[mask]

        return result

    def duplicate_largest(self,
                         grid: np.ndarray,
                         n_copies: int,
                         fill_value: int = 0) -> np.ndarray:
        """
        Duplicate the largest object n times.

        Places copies in a row.
        """
        largest = self.detector.detect_largest(grid)
        if largest is None:
            return grid.copy()

        # Extract object
        y1, x1, y2, x2 = largest.bbox
        obj_bbox = largest.grid[y1:y2, x1:x2]
        obj_mask_bbox = largest.mask[y1:y2, x1:x2]

        # Create result grid (wide enough for copies)
        obj_height = y2 - y1
        obj_width = x2 - x1

        result_width = grid.shape[1] + obj_width * (n_copies - 1)
        result = np.full((grid.shape[0], result_width), fill_value)

        # Place original grid
        result[:, :grid.shape[1]] = grid

        # Add copies
        for i in range(1, n_copies):
            offset_x = grid.shape[1] + obj_width * (i - 1)
            if offset_x + obj_width <= result.shape[1]:
                result[y1:y2, offset_x:offset_x+obj_width][obj_mask_bbox] = obj_bbox[obj_mask_bbox]

        return result

    def extract_objects_to_grid(self,
                               grid: np.ndarray,
                               rows: int,
                               cols: int) -> np.ndarray:
        """
        Extract objects and arrange in a grid layout.

        Useful for compression tasks.
        """
        objects = self.detector.detect_objects(grid)

        if not objects:
            return np.zeros((rows, cols))

        # Sort by position (top-left to bottom-right)
        objects.sort(key=lambda obj: (obj.position[0], obj.position[1]))

        # Determine cell size
        max_height = max(obj.height for obj in objects)
        max_width = max(obj.width for obj in objects)

        cell_height = max(max_height, rows)
        cell_width = max(max_width, cols)

        # Create result grid
        result = np.zeros((rows * cell_height, cols * cell_width), dtype=grid.dtype)

        # Place objects in grid
        for idx, obj in enumerate(objects[:rows * cols]):
            row = idx // cols
            col = idx % cols

            # Extract object bbox
            y1, x1, y2, x2 = obj.bbox
            obj_bbox = obj.grid[y1:y2, x1:x2]
            obj_mask_bbox = obj.mask[y1:y2, x1:x2]

            # Target position in result
            target_y = row * cell_height
            target_x = col * cell_width

            # Place object
            end_y = min(target_y + obj.height, result.shape[0])
            end_x = min(target_x + obj.width, result.shape[1])

            target_mask = obj_mask_bbox[:end_y-target_y, :end_x-target_x]
            result[target_y:end_y, target_x:end_x][target_mask] = obj_bbox[:end_y-target_y, :end_x-target_x][target_mask]

        return result

    def filter_objects_by_property(self,
                                  grid: np.ndarray,
                                  condition_fn: callable,
                                  fill_value: int = 0) -> np.ndarray:
        """
        Keep only objects matching condition, remove others.

        Example:
            filter_objects(grid, lambda obj: obj.size > 5)
            # Keeps only objects with more than 5 pixels
        """
        objects = self.detector.detect_objects(grid)
        result = np.full_like(grid, fill_value)

        for obj in objects:
            if condition_fn(obj):
                result[obj.mask] = grid[obj.mask]

        return result


def analyze_object_patterns(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """
    Analyze what object-level transformations are happening.

    Returns insights about object-level changes.
    """
    detector = ObjectDetector()
    patterns = {
        'object_count_changes': [],
        'size_changes': [],
        'color_changes': [],
        'position_changes': [],
        'needs_object_reasoning': False
    }

    for inp, out in train_pairs:
        # Detect objects
        inp_objects = detector.detect_objects(inp)
        out_objects = detector.detect_objects(out)

        # Count changes
        patterns['object_count_changes'].append((len(inp_objects), len(out_objects)))

        # Check if individual objects are transformed
        if len(inp_objects) > 0 and len(out_objects) > 0:
            # Size changes
            inp_sizes = [obj.size for obj in inp_objects]
            out_sizes = [obj.size for obj in out_objects]
            if inp_sizes != out_sizes:
                patterns['size_changes'].append(True)

            # Color changes
            inp_colors = set(obj.dominant_color for obj in inp_objects)
            out_colors = set(obj.dominant_color for obj in out_objects)
            if inp_colors != out_colors:
                patterns['color_changes'].append(True)

            # Position changes
            if len(inp_objects) == len(out_objects):
                inp_positions = [obj.position for obj in inp_objects]
                out_positions = [obj.position for obj in out_objects]
                if inp_positions != out_positions:
                    patterns['position_changes'].append(True)

    # Determine if object reasoning is needed
    patterns['needs_object_reasoning'] = (
        len(patterns['object_count_changes']) > 0 and
        any(inp != out for inp, out in patterns['object_count_changes'])
    ) or len(patterns['color_changes']) > 0 or len(patterns['position_changes']) > 0

    return patterns
