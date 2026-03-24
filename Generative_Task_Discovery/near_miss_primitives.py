"""
Near-Miss Primitives Implementation Plan

Targets 3 tasks that are 75-83% accurate (SO CLOSE!):
1. Extract largest object (83% accuracy)
2. Connect objects (75% accuracy)
3. Align objects (75% accuracy)

Expected impact: +3 tasks (40% → 48.6%)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from arc_generative_solver import ARCObject, TRGPrimitives


class NearMissPrimitives:
    """
    Primitives for near-miss tasks (75-83% accuracy)

    These are the "quick wins" - tasks that are almost working
    but need specific primitives to reach 100%.
    """

    # ========================================================================
    # 1. OBJECT SELECTION BY SIZE (for extract_largest task)
    # ========================================================================

    @staticmethod
    def select_largest_object(objects: List[ARCObject]) -> Optional[ARCObject]:
        """
        Select the largest object by area

        Current accuracy: 83%
        Issue: Can identify objects but can't filter by size
        Fix: Add size-based selection
        """
        if not objects:
            return None

        return max(objects, key=lambda obj: obj.area)

    @staticmethod
    def select_smallest_object(objects: List[ARCObject]) -> Optional[ARCObject]:
        """Select the smallest object by area"""
        if not objects:
            return None

        return min(objects, key=lambda obj: obj.area)

    @staticmethod
    def select_objects_by_size(objects: List[ARCObject],
                               criterion: str) -> List[ARCObject]:
        """
        Select objects by size criterion

        Args:
            criterion: "largest", "smallest", "medium", or "all"
        """
        if not objects:
            return []

        if criterion == "largest":
            max_area = max(obj.area for obj in objects)
            return [obj for obj in objects if obj.area == max_area]
        elif criterion == "smallest":
            min_area = min(obj.area for obj in objects)
            return [obj for obj in objects if obj.area == min_area]
        elif criterion == "medium":
            # Objects that are neither largest nor smallest
            max_area = max(obj.area for obj in objects)
            min_area = min(obj.area for obj in objects)
            return [obj for obj in objects
                   if obj.area != max_area and obj.area != min_area]
        else:  # "all"
            return objects

    @staticmethod
    def filter_objects_by_size_range(objects: List[ARCObject],
                                     min_area: int,
                                     max_area: int) -> List[ARCObject]:
        """Filter objects by area range"""
        return [obj for obj in objects
                if min_area <= obj.area <= max_area]

    @staticmethod
    def extract_largest_to_grid(grid: np.ndarray,
                                objects: List[ARCObject]) -> np.ndarray:
        """
        Extract only the largest object(s) to a new grid

        This is the specific operation needed for extract_largest task
        """
        result = np.zeros_like(grid)

        if not objects:
            return result

        largest_objs = NearMissPrimitives.select_objects_by_size(
            objects, "largest"
        )

        for obj in largest_objs:
            result[obj.mask] = obj.color

        return result

    # ========================================================================
    # 2. OBJECT CONNECTION (for connect_nearest task)
    # ========================================================================

    @staticmethod
    def draw_line_bresenham(grid: np.ndarray,
                           y1: int, x1: int,
                           y2: int, x2: int,
                           color: int) -> np.ndarray:
        """
        Draw line using Bresenham's algorithm

        Current accuracy: 75%
        Issue: Can identify objects but can't draw connecting lines
        Fix: Add line drawing primitive
        """
        result = grid.copy()

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            # Draw pixel if within bounds
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
    def connect_two_objects(grid: np.ndarray,
                           obj1: ARCObject,
                           obj2: ARCObject,
                           line_color: int = 3,
                           preserve_objects: bool = True) -> np.ndarray:
        """
        Connect two objects with a line

        Connects centroids of the two objects
        """
        y1, x1 = map(int, obj1.centroid)
        y2, x2 = map(int, obj2.centroid)

        result = NearMissPrimitives.draw_line_bresenham(
            grid, y1, x1, y2, x2, line_color
        )

        # Preserve original objects (don't overwrite them with line)
        if preserve_objects:
            result[obj1.mask] = obj1.color
            result[obj2.mask] = obj2.color

        return result

    @staticmethod
    def connect_nearest_objects(grid: np.ndarray,
                               objects: List[ARCObject],
                               line_color: int = 3) -> np.ndarray:
        """
        Connect each pair of nearest objects

        For connect_nearest task
        """
        result = grid.copy()

        if len(objects) < 2:
            return result

        # Connect adjacent pairs (preserving original objects)
        for i in range(len(objects) - 1):
            result = NearMissPrimitives.connect_two_objects(
                result, objects[i], objects[i + 1], line_color,
                preserve_objects=True
            )

        return result

    @staticmethod
    def connect_all_objects(grid: np.ndarray,
                           objects: List[ARCObject],
                           line_color: int = 3) -> np.ndarray:
        """Connect all objects to each other (complete graph)"""
        result = grid.copy()

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                result = NearMissPrimitives.connect_two_objects(
                    result, objects[i], objects[j], line_color
                )

        return result

    # ========================================================================
    # 3. OBJECT ALIGNMENT (for align_objects task)
    # ========================================================================

    @staticmethod
    def align_objects_horizontal(grid: np.ndarray,
                                objects: List[ARCObject],
                                alignment: str = "center") -> np.ndarray:
        """
        Align objects horizontally

        Current accuracy: 75%
        Issue: Can identify objects but can't align them
        Fix: Add alignment primitive

        Args:
            alignment: "top", "center", "bottom"
        """
        result = np.zeros_like(grid)

        if not objects:
            return result

        # Determine target y-coordinate
        if alignment == "top":
            target_y = min(obj.bbox[0] for obj in objects)
        elif alignment == "bottom":
            target_y = max(obj.bbox[2] for obj in objects)
        else:  # center
            target_y = int(np.mean([obj.centroid[0] for obj in objects]))

        # Move each object to target y
        for obj in objects:
            dy = target_y - int(obj.centroid[0])

            # Get object's local mask
            y1, x1, y2, x2 = obj.bbox
            local_mask = obj.mask[y1:y2+1, x1:x2+1]

            # Calculate new position
            new_y1 = max(0, min(grid.shape[0] - (y2-y1+1), y1 + dy))
            new_y2 = new_y1 + (y2 - y1)

            # Place object at new position
            if 0 <= new_y1 < grid.shape[0] and new_y2 < grid.shape[0]:
                result[new_y1:new_y2+1, x1:x2+1] = np.where(
                    local_mask,
                    obj.color,
                    result[new_y1:new_y2+1, x1:x2+1]
                )

        return result

    @staticmethod
    def align_objects_vertical(grid: np.ndarray,
                              objects: List[ARCObject],
                              alignment: str = "center") -> np.ndarray:
        """
        Align objects vertically

        Args:
            alignment: "left", "center", "right"
        """
        result = np.zeros_like(grid)

        if not objects:
            return result

        # Determine target x-coordinate
        if alignment == "left":
            target_x = min(obj.bbox[1] for obj in objects)
        elif alignment == "right":
            target_x = max(obj.bbox[3] for obj in objects)
        else:  # center
            target_x = int(np.mean([obj.centroid[1] for obj in objects]))

        # Move each object to target x
        for obj in objects:
            dx = target_x - int(obj.centroid[1])

            # Get object's local mask
            y1, x1, y2, x2 = obj.bbox
            local_mask = obj.mask[y1:y2+1, x1:x2+1]

            # Calculate new position
            new_x1 = max(0, min(grid.shape[1] - (x2-x1+1), x1 + dx))
            new_x2 = new_x1 + (x2 - x1)

            # Place object at new position
            if 0 <= new_x1 < grid.shape[1] and new_x2 < grid.shape[1]:
                result[y1:y2+1, new_x1:new_x2+1] = np.where(
                    local_mask,
                    obj.color,
                    result[y1:y2+1, new_x1:new_x2+1]
                )

        return result

    @staticmethod
    def align_objects_to_row(grid: np.ndarray,
                            objects: List[ARCObject],
                            row: int = 0) -> np.ndarray:
        """Align all objects to the same row"""
        result = np.zeros_like(grid)

        for obj in objects:
            y1, x1, y2, x2 = obj.bbox
            obj_height = y2 - y1 + 1
            obj_width = x2 - x1 + 1
            local_mask = obj.mask[y1:y2+1, x1:x2+1]

            # Place entire object starting at specified row
            if 0 <= row < grid.shape[0] and row + obj_height <= grid.shape[0]:
                result[row:row+obj_height, x1:x1+obj_width] = np.where(
                    local_mask,
                    obj.color,
                    result[row:row+obj_height, x1:x1+obj_width]
                )

        return result

    @staticmethod
    def pack_objects_horizontal(grid: np.ndarray,
                               objects: List[ARCObject],
                               row: int = 0,
                               spacing: int = 0) -> np.ndarray:
        """
        Pack objects horizontally at specified row without overlap

        This is for tasks like aligning objects to top row and
        arranging them side-by-side horizontally.

        Args:
            row: Row to place objects at
            spacing: Space between objects (0 = adjacent)
        """
        result = np.zeros_like(grid)

        if not objects:
            return result

        # Sort objects by their original x position
        sorted_objects = sorted(objects, key=lambda obj: obj.bbox[1])

        current_x = 0

        for obj in sorted_objects:
            y1, x1, y2, x2 = obj.bbox
            obj_height = y2 - y1 + 1
            obj_width = x2 - x1 + 1
            local_mask = obj.mask[y1:y2+1, x1:x2+1]

            # Place object at current_x position
            if (0 <= row < grid.shape[0] and
                row + obj_height <= grid.shape[0] and
                current_x + obj_width <= grid.shape[1]):

                result[row:row+obj_height, current_x:current_x+obj_width] = np.where(
                    local_mask,
                    obj.color,
                    result[row:row+obj_height, current_x:current_x+obj_width]
                )

            # Move x position for next object
            current_x += obj_width + spacing

        return result

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    @staticmethod
    def find_distance(obj1: ARCObject, obj2: ARCObject) -> float:
        """Calculate Euclidean distance between object centroids"""
        y1, x1 = obj1.centroid
        y2, x2 = obj2.centroid
        return np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    @staticmethod
    def find_nearest_object(obj: ARCObject,
                           others: List[ARCObject]) -> Optional[ARCObject]:
        """Find the nearest object to the given object"""
        if not others:
            return None

        return min(others, key=lambda o: NearMissPrimitives.find_distance(obj, o))


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demo_near_miss_primitives():
    """Demonstrate the near-miss primitives"""

    print("="*70)
    print("NEAR-MISS PRIMITIVES DEMONSTRATION")
    print("="*70)

    # Demo 1: Object selection by size
    print("\n1. Object Selection by Size")
    print("-"*70)

    # Create sample grid with objects of different sizes
    grid1 = np.array([
        [1, 0, 2, 2],
        [0, 0, 2, 2],
        [3, 0, 0, 0]
    ])

    primitives = TRGPrimitives()
    objects = primitives.components(grid1)

    print(f"Input grid:\n{grid1}")
    print(f"Found {len(objects)} objects:")
    for i, obj in enumerate(objects):
        print(f"  Object {i+1}: color={obj.color}, area={obj.area}")

    # Extract largest
    largest = NearMissPrimitives.select_largest_object(objects)
    if largest:
        print(f"Largest object: color={largest.color}, area={largest.area}")

        result = NearMissPrimitives.extract_largest_to_grid(grid1, objects)
        print(f"Extracted largest:\n{result}")

    # Demo 2: Object connection
    print("\n2. Object Connection (Line Drawing)")
    print("-"*70)

    grid2 = np.array([
        [1, 0, 0, 0, 2],
        [0, 0, 0, 0, 0]
    ])

    objects2 = primitives.components(grid2)
    print(f"Input grid:\n{grid2}")
    print(f"Objects to connect: {len(objects2)}")

    if len(objects2) >= 2:
        result2 = NearMissPrimitives.connect_nearest_objects(
            grid2, objects2, line_color=3
        )
        print(f"Connected:\n{result2}")

    # Demo 3: Object alignment
    print("\n3. Object Alignment")
    print("-"*70)

    grid3 = np.array([
        [1, 0],
        [0, 0],
        [0, 2]
    ])

    objects3 = primitives.components(grid3)
    print(f"Input grid:\n{grid3}")
    print(f"Objects to align: {len(objects3)}")

    if len(objects3) >= 2:
        # Horizontal alignment
        result3 = NearMissPrimitives.align_objects_horizontal(
            grid3, objects3, alignment="top"
        )
        print(f"Aligned horizontally (top):\n{result3}")

        # To same row
        result4 = NearMissPrimitives.align_objects_to_row(
            grid3, objects3, row=0
        )
        print(f"Aligned to row 0:\n{result4}")


if __name__ == "__main__":
    demo_near_miss_primitives()
