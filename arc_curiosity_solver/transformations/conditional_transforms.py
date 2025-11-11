"""
Conditional Transformation System for ARC-AGI

Implements IF-THEN-ELSE logic for transformations, enabling:
- "IF object.size > 3 THEN recolor to 2 ELSE recolor to 5"
- "IF near_edge(object) THEN move to center ELSE keep"
- "IF touching(obj1, obj2) THEN merge ELSE separate"

This is the CRITICAL missing piece preventing breakthrough past 1% solve rate.
"""

import numpy as np
from typing import List, Callable, Optional, Any, Dict
from dataclasses import dataclass
from ..core.object_reasoning import ArcObject


@dataclass
class Condition:
    """A conditional predicate that can be evaluated on objects."""
    name: str
    predicate: Callable[[ArcObject, List[ArcObject], np.ndarray], bool]
    description: str

    def __call__(self, obj: ArcObject, all_objects: List[ArcObject], grid: np.ndarray) -> bool:
        """Evaluate this condition."""
        return self.predicate(obj, all_objects, grid)


@dataclass
class ConditionalAction:
    """An action that can be applied to an object/grid."""
    name: str
    apply_fn: Callable[[np.ndarray, ArcObject, List[ArcObject]], np.ndarray]
    description: str
    parameters: Dict[str, Any]

    def apply(self, grid: np.ndarray, obj: ArcObject, all_objects: List[ArcObject]) -> np.ndarray:
        """Apply this action."""
        return self.apply_fn(grid, obj, all_objects)


@dataclass
class ConditionalTransform:
    """
    A conditional transformation: IF condition THEN action1 ELSE action2

    Example:
        IF object.size > 3:
            THEN recolor to blue
            ELSE recolor to red
    """
    condition: Condition
    then_action: ConditionalAction
    else_action: Optional[ConditionalAction] = None
    description: str = ""
    confidence: float = 1.0

    def __post_init__(self):
        if not self.description:
            else_part = f" ELSE {self.else_action.description}" if self.else_action else ""
            self.description = f"IF {self.condition.description} THEN {self.then_action.description}{else_part}"

    def apply(self, grid: np.ndarray, objects: Optional[List[ArcObject]] = None) -> np.ndarray:
        """
        Apply this conditional transformation to the grid.

        Args:
            grid: Input grid
            objects: Pre-detected objects (optional, will detect if not provided)

        Returns:
            Transformed grid
        """
        from ..core.object_reasoning import ObjectDetector

        if objects is None:
            detector = ObjectDetector()
            objects = detector.detect_objects(grid)

        result = grid.copy()

        # Apply transformation to each object based on condition
        for obj in objects:
            if self.condition(obj, objects, grid):
                # Condition is TRUE - apply THEN action
                result = self.then_action.apply(result, obj, objects)
            elif self.else_action is not None:
                # Condition is FALSE - apply ELSE action
                result = self.else_action.apply(result, obj, objects)

        return result

    def __repr__(self):
        return f"ConditionalTransform({self.description})"


class ConditionLibrary:
    """Library of reusable condition predicates."""

    @staticmethod
    def size_greater_than(threshold: int) -> Condition:
        """Object size > threshold."""
        return Condition(
            name=f"size_gt_{threshold}",
            predicate=lambda obj, all_objs, grid: obj.size > threshold,
            description=f"size > {threshold}"
        )

    @staticmethod
    def size_less_than(threshold: int) -> Condition:
        """Object size < threshold."""
        return Condition(
            name=f"size_lt_{threshold}",
            predicate=lambda obj, all_objs, grid: obj.size < threshold,
            description=f"size < {threshold}"
        )

    @staticmethod
    def is_largest() -> Condition:
        """Object is the largest."""
        return Condition(
            name="is_largest",
            predicate=lambda obj, all_objs, grid: (
                obj.size == max(o.size for o in all_objs) if all_objs else True
            ),
            description="is largest"
        )

    @staticmethod
    def is_smallest() -> Condition:
        """Object is the smallest."""
        return Condition(
            name="is_smallest",
            predicate=lambda obj, all_objs, grid: (
                obj.size == min(o.size for o in all_objs) if all_objs else True
            ),
            description="is smallest"
        )

    @staticmethod
    def has_color(color: int) -> Condition:
        """Object has specific color."""
        return Condition(
            name=f"color_{color}",
            predicate=lambda obj, all_objs, grid: obj.dominant_color == color,
            description=f"color == {color}"
        )

    @staticmethod
    def near_edge(margin: int = 2) -> Condition:
        """Object is near the grid edge."""
        def is_near_edge(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            h, w = grid.shape
            y1, x1, y2, x2 = obj.bbox
            return (y1 < margin or x1 < margin or
                   y2 > h - margin or x2 > w - margin)

        return Condition(
            name=f"near_edge_{margin}",
            predicate=is_near_edge,
            description=f"near edge (margin={margin})"
        )

    @staticmethod
    def near_center(threshold: float = 0.3) -> Condition:
        """Object is near the grid center."""
        def is_near_center(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            h, w = grid.shape
            center_y, center_x = h / 2, w / 2
            obj_y, obj_x = obj.position

            # Distance from center (normalized)
            dist = np.sqrt(((obj_y - center_y) / h)**2 + ((obj_x - center_x) / w)**2)
            return dist < threshold

        return Condition(
            name=f"near_center_{threshold}",
            predicate=is_near_center,
            description=f"near center (threshold={threshold})"
        )

    @staticmethod
    def touching(other_color: Optional[int] = None) -> Condition:
        """Object is touching another object (optionally of specific color)."""
        def is_touching(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox

            # Expand bbox by 1 pixel to detect adjacency
            y1_exp = max(0, y1 - 1)
            x1_exp = max(0, x1 - 1)
            y2_exp = min(grid.shape[0], y2 + 1)
            x2_exp = min(grid.shape[1], x2 + 1)

            # Check neighbors
            for other_obj in all_objs:
                if other_obj is obj:
                    continue

                # Check if bboxes overlap when expanded
                oy1, ox1, oy2, ox2 = other_obj.bbox

                if (y1_exp < oy2 and y2_exp > oy1 and
                    x1_exp < ox2 and x2_exp > ox1):

                    if other_color is None or other_obj.dominant_color == other_color:
                        return True

            return False

        color_desc = f" color {other_color}" if other_color is not None else ""
        return Condition(
            name=f"touching_{other_color if other_color is not None else 'any'}",
            predicate=is_touching,
            description=f"touching{color_desc}"
        )

    @staticmethod
    def in_quadrant(quadrant: str) -> Condition:
        """Object is in specific quadrant (top_left, top_right, bottom_left, bottom_right)."""
        def is_in_quadrant(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            h, w = grid.shape
            obj_y, obj_x = obj.position

            in_top = obj_y < h / 2
            in_left = obj_x < w / 2

            if quadrant == "top_left":
                return in_top and in_left
            elif quadrant == "top_right":
                return in_top and not in_left
            elif quadrant == "bottom_left":
                return not in_top and in_left
            elif quadrant == "bottom_right":
                return not in_top and not in_left
            return False

        return Condition(
            name=f"in_{quadrant}",
            predicate=is_in_quadrant,
            description=f"in {quadrant}"
        )

    @staticmethod
    def is_rectangular() -> Condition:
        """Object has rectangular/square shape."""
        def is_rect(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            bbox_area = (y2 - y1) * (x2 - x1)
            fill_ratio = obj.size / max(bbox_area, 1)
            return fill_ratio > 0.9  # >90% filled = rectangular

        return Condition(
            name="is_rectangular",
            predicate=is_rect,
            description="is rectangular"
        )

    @staticmethod
    def is_line_shaped() -> Condition:
        """Object is line-shaped (long and thin)."""
        def is_line(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            height = y2 - y1
            width = x2 - x1
            if min(height, width) == 0:
                return False
            aspect_ratio = max(height, width) / min(height, width)
            return aspect_ratio > 3  # At least 3:1 ratio

        return Condition(
            name="is_line_shaped",
            predicate=is_line,
            description="is line-shaped"
        )

    @staticmethod
    def object_count_equals(count: int) -> Condition:
        """Number of objects equals count."""
        def count_equals(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            return len(all_objs) == count

        return Condition(
            name=f"count_eq_{count}",
            predicate=count_equals,
            description=f"object count == {count}"
        )

    @staticmethod
    def object_count_greater_than(count: int) -> Condition:
        """Number of objects > count."""
        def count_gt(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            return len(all_objs) > count

        return Condition(
            name=f"count_gt_{count}",
            predicate=count_gt,
            description=f"object count > {count}"
        )

    @staticmethod
    def is_symmetric_horizontal() -> Condition:
        """Object is horizontally symmetric."""
        def is_h_sym(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            obj_grid = grid[y1:y2, x1:x2].copy()
            # Mask to object only
            obj_grid[~obj.mask[y1:y2, x1:x2]] = 0
            # Check horizontal symmetry
            return np.array_equal(obj_grid, np.fliplr(obj_grid))

        return Condition(
            name="is_symmetric_h",
            predicate=is_h_sym,
            description="is horizontally symmetric"
        )

    @staticmethod
    def is_symmetric_vertical() -> Condition:
        """Object is vertically symmetric."""
        def is_v_sym(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            obj_grid = grid[y1:y2, x1:x2].copy()
            # Mask to object only
            obj_grid[~obj.mask[y1:y2, x1:x2]] = 0
            # Check vertical symmetry
            return np.array_equal(obj_grid, np.flipud(obj_grid))

        return Condition(
            name="is_symmetric_v",
            predicate=is_v_sym,
            description="is vertically symmetric"
        )

    @staticmethod
    def aligned_with_grid() -> Condition:
        """Object is aligned with grid (horizontal or vertical lines)."""
        def is_aligned(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            # Check if object is aligned (single row or column)
            return (y2 - y1 == 1) or (x2 - x1 == 1)

        return Condition(
            name="is_aligned",
            predicate=is_aligned,
            description="aligned with grid"
        )

    @staticmethod
    def color_count_greater_than(color: int, threshold: int) -> Condition:
        """Object has more than threshold pixels of specific color."""
        def color_count_gt(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            obj_region = grid[y1:y2, x1:x2]
            obj_mask = obj.mask[y1:y2, x1:x2]
            color_pixels = np.sum((obj_region == color) & obj_mask)
            return color_pixels > threshold

        return Condition(
            name=f"color_{color}_count_gt_{threshold}",
            predicate=color_count_gt,
            description=f"has >{threshold} pixels of color {color}"
        )

    # ========== PHASE 4: RICHER PREDICATES ==========

    # === TOPOLOGICAL CONDITIONS ===

    @staticmethod
    def has_hole() -> Condition:
        """Object has interior hole(s)."""
        def has_interior_hole(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            from scipy import ndimage
            y1, x1, y2, x2 = obj.bbox
            if y2 - y1 < 3 or x2 - x1 < 3:  # Too small for hole
                return False

            # Get object region
            obj_mask = obj.mask[y1:y2, x1:x2]

            # Invert mask (background becomes foreground)
            inverted = ~obj_mask

            # Label background regions
            labeled, num_regions = ndimage.label(inverted)

            # If > 1 region, there's a hole (excluding outer background)
            # Check if any region is completely surrounded (no edge touching)
            h, w = obj_mask.shape
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled == region_id)
                # Check if region touches edge
                touches_edge = (
                    np.any(region_mask[0, :]) or
                    np.any(region_mask[-1, :]) or
                    np.any(region_mask[:, 0]) or
                    np.any(region_mask[:, -1])
                )
                if not touches_edge:
                    return True  # Interior hole found

            return False

        return Condition(
            name="has_hole",
            predicate=has_interior_hole,
            description="has interior hole"
        )

    @staticmethod
    def is_hollow() -> Condition:
        """Object is hollow (only perimeter, no interior)."""
        def is_perimeter_only(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            from scipy import ndimage
            y1, x1, y2, x2 = obj.bbox
            if y2 - y1 < 3 or x2 - x1 < 3:  # Too small to be hollow
                return False

            obj_mask = obj.mask[y1:y2, x1:x2]

            # Erode by 1 pixel
            eroded = ndimage.binary_erosion(obj_mask)

            # Count pixels before and after erosion
            original_pixels = np.sum(obj_mask)
            eroded_pixels = np.sum(eroded)

            # If erosion removes >80% of pixels, it's hollow
            if original_pixels > 0:
                return (original_pixels - eroded_pixels) / original_pixels > 0.8
            return False

        return Condition(
            name="is_hollow",
            predicate=is_perimeter_only,
            description="is hollow (perimeter only)"
        )

    @staticmethod
    def is_connected() -> Condition:
        """Object is fully connected (single component)."""
        def is_single_component(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            from scipy import ndimage
            y1, x1, y2, x2 = obj.bbox
            obj_mask = obj.mask[y1:y2, x1:x2]

            # Label connected components
            labeled, num_components = ndimage.label(obj_mask)

            return num_components == 1

        return Condition(
            name="is_connected",
            predicate=is_single_component,
            description="is fully connected"
        )

    @staticmethod
    def is_fragmented() -> Condition:
        """Object has multiple disconnected parts."""
        def is_multi_component(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            from scipy import ndimage
            y1, x1, y2, x2 = obj.bbox
            obj_mask = obj.mask[y1:y2, x1:x2]

            # Label connected components
            labeled, num_components = ndimage.label(obj_mask)

            return num_components > 1

        return Condition(
            name="is_fragmented",
            predicate=is_multi_component,
            description="has multiple disconnected parts"
        )

    # === RELATIONAL CONDITIONS ===

    @staticmethod
    def touching_color(color: int) -> Condition:
        """Object is touching (adjacent to) specific color."""
        def touches_specific_color(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            h, w = grid.shape

            # Expand bbox by 1 pixel
            y1_exp = max(0, y1 - 1)
            x1_exp = max(0, x1 - 1)
            y2_exp = min(h, y2 + 1)
            x2_exp = min(w, x2 + 1)

            # Get expanded region
            expanded_region = grid[y1_exp:y2_exp, x1_exp:x2_exp]
            expanded_mask = np.zeros_like(expanded_region, dtype=bool)

            # Translate object mask to expanded coordinates
            mask_y_offset = y1 - y1_exp
            mask_x_offset = x1 - x1_exp
            expanded_mask[
                mask_y_offset:mask_y_offset + (y2 - y1),
                mask_x_offset:mask_x_offset + (x2 - x1)
            ] = obj.mask[y1:y2, x1:x2]

            # Check neighbors (expanded region minus object itself)
            neighbor_region = expanded_region[~expanded_mask]

            return color in neighbor_region

        return Condition(
            name=f"touching_color_{color}",
            predicate=touches_specific_color,
            description=f"touching color {color}"
        )

    @staticmethod
    def between_objects() -> Condition:
        """Object is spatially between two other objects."""
        def is_between(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            if len(all_objs) < 3:
                return False

            obj_y, obj_x = obj.position

            # Check if object is between any pair of other objects
            other_objs = [o for o in all_objs if o is not obj]

            for i, obj1 in enumerate(other_objs):
                for obj2 in other_objs[i+1:]:
                    y1, x1 = obj1.position
                    y2, x2 = obj2.position

                    # Check if obj is between obj1 and obj2 on x-axis
                    if (min(x1, x2) < obj_x < max(x1, x2) and
                        min(y1, y2) <= obj_y <= max(y1, y2)):
                        return True

                    # Check if obj is between obj1 and obj2 on y-axis
                    if (min(y1, y2) < obj_y < max(y1, y2) and
                        min(x1, x2) <= obj_x <= max(x1, x2)):
                        return True

            return False

        return Condition(
            name="between_objects",
            predicate=is_between,
            description="between other objects"
        )

    @staticmethod
    def aligned_horizontally() -> Condition:
        """Object is horizontally aligned with another object."""
        def is_h_aligned(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            obj_y, obj_x = obj.position

            for other_obj in all_objs:
                if other_obj is obj:
                    continue

                other_y, other_x = other_obj.position

                # Check if y-coordinates are close (within 2 pixels)
                if abs(obj_y - other_y) <= 2:
                    return True

            return False

        return Condition(
            name="aligned_horizontally",
            predicate=is_h_aligned,
            description="horizontally aligned with another object"
        )

    @staticmethod
    def aligned_vertically() -> Condition:
        """Object is vertically aligned with another object."""
        def is_v_aligned(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            obj_y, obj_x = obj.position

            for other_obj in all_objs:
                if other_obj is obj:
                    continue

                other_y, other_x = other_obj.position

                # Check if x-coordinates are close (within 2 pixels)
                if abs(obj_x - other_x) <= 2:
                    return True

            return False

        return Condition(
            name="aligned_vertically",
            predicate=is_v_aligned,
            description="vertically aligned with another object"
        )

    @staticmethod
    def on_diagonal() -> Condition:
        """Object is on main diagonal or anti-diagonal."""
        def is_on_diag(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            h, w = grid.shape
            obj_y, obj_x = obj.position

            # Main diagonal: y ≈ x
            # Anti-diagonal: y ≈ (h - 1 - x)

            threshold = min(h, w) * 0.1  # 10% tolerance

            # Check main diagonal
            if abs(obj_y - obj_x) < threshold:
                return True

            # Check anti-diagonal
            if abs(obj_y - (h - 1 - obj_x)) < threshold:
                return True

            return False

        return Condition(
            name="on_diagonal",
            predicate=is_on_diag,
            description="on diagonal"
        )

    # === STRUCTURAL CONDITIONS ===

    @staticmethod
    def forms_pattern_with_others() -> Condition:
        """Objects form a regular pattern (grid, line, etc.)."""
        def forms_pattern(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            if len(all_objs) < 3:
                return False

            # Check if objects form a line or grid
            positions = [o.position for o in all_objs]
            ys = [p[0] for p in positions]
            xs = [p[1] for p in positions]

            # Check for line pattern (all x's same or all y's same)
            y_variance = np.var(ys)
            x_variance = np.var(xs)

            # Low variance means aligned
            if y_variance < 4 or x_variance < 4:
                return True

            # Check for grid pattern (regular spacing)
            y_diffs = sorted([ys[i+1] - ys[i] for i in range(len(ys)-1)])
            x_diffs = sorted([xs[i+1] - xs[i] for i in range(len(xs)-1)])

            # If diffs are regular, it's a grid
            if len(y_diffs) > 1 and len(x_diffs) > 1:
                y_regular = (max(y_diffs) - min(y_diffs)) < 2
                x_regular = (max(x_diffs) - min(x_diffs)) < 2
                if y_regular and x_regular:
                    return True

            return False

        return Condition(
            name="forms_pattern",
            predicate=forms_pattern,
            description="forms pattern with others"
        )

    @staticmethod
    def is_square_shaped() -> Condition:
        """Object has square shape (equal height and width)."""
        def is_square(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            height = y2 - y1
            width = x2 - x1

            # Check if height and width are equal (or very close)
            if height == 0 or width == 0:
                return False

            ratio = max(height, width) / min(height, width)
            return ratio < 1.2  # Within 20% of square

        return Condition(
            name="is_square",
            predicate=is_square,
            description="is square-shaped"
        )

    @staticmethod
    def is_compact() -> Condition:
        """Object is compact (high fill ratio in bounding box)."""
        def is_dense(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            bbox_area = (y2 - y1) * (x2 - x1)
            if bbox_area == 0:
                return False

            fill_ratio = obj.size / bbox_area
            return fill_ratio > 0.7  # >70% filled

        return Condition(
            name="is_compact",
            predicate=is_dense,
            description="is compact (dense)"
        )

    @staticmethod
    def is_sparse() -> Condition:
        """Object is sparse (low fill ratio in bounding box)."""
        def is_loose(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            y1, x1, y2, x2 = obj.bbox
            bbox_area = (y2 - y1) * (x2 - x1)
            if bbox_area == 0:
                return False

            fill_ratio = obj.size / bbox_area
            return fill_ratio < 0.4  # <40% filled

        return Condition(
            name="is_sparse",
            predicate=is_loose,
            description="is sparse (loose)"
        )

    @staticmethod
    def has_unique_color() -> Condition:
        """Object is the only one with its color."""
        def is_unique_color(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            obj_color = obj.dominant_color

            for other_obj in all_objs:
                if other_obj is obj:
                    continue
                if other_obj.dominant_color == obj_color:
                    return False

            return True

        return Condition(
            name="has_unique_color",
            predicate=is_unique_color,
            description="has unique color"
        )

    @staticmethod
    def same_color_as_largest() -> Condition:
        """Object has same color as the largest object."""
        def same_as_largest(obj: ArcObject, all_objs: List[ArcObject], grid: np.ndarray) -> bool:
            if not all_objs:
                return False

            largest = max(all_objs, key=lambda o: o.size)
            return obj.dominant_color == largest.dominant_color

        return Condition(
            name="same_color_as_largest",
            predicate=same_as_largest,
            description="same color as largest object"
        )


class ActionLibrary:
    """Library of reusable actions."""

    @staticmethod
    def recolor_to(color: int) -> ConditionalAction:
        """Recolor object to specific color."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()
            result[obj.mask] = color
            return result

        return ConditionalAction(
            name=f"recolor_{color}",
            apply_fn=apply,
            description=f"recolor to {color}",
            parameters={"color": color}
        )

    @staticmethod
    def move_by(dy: int, dx: int) -> ConditionalAction:
        """Move object by delta."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            # Clear old position
            result[obj.mask] = 0

            # Calculate new position
            y1, x1, y2, x2 = obj.bbox
            new_y1 = y1 + dy
            new_x1 = x1 + dx
            new_y2 = y2 + dy
            new_x2 = x2 + dx

            # Check bounds
            h, w = grid.shape
            if (new_y1 >= 0 and new_x1 >= 0 and
                new_y2 <= h and new_x2 <= w):

                # Extract object pixels
                obj_pixels = grid[y1:y2, x1:x2].copy()
                obj_pixels[~obj.mask[y1:y2, x1:x2]] = 0

                # Place at new position
                result[new_y1:new_y2, new_x1:new_x2] = np.maximum(
                    result[new_y1:new_y2, new_x1:new_x2],
                    obj_pixels
                )

            return result

        return ConditionalAction(
            name=f"move_{dy}_{dx}",
            apply_fn=apply,
            description=f"move by ({dy}, {dx})",
            parameters={"dy": dy, "dx": dx}
        )

    @staticmethod
    def move_to_edge(direction: str) -> ConditionalAction:
        """Move object to edge (top, bottom, left, right)."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            h, w = grid.shape
            y1, x1, y2, x2 = obj.bbox

            if direction == "top":
                dy, dx = -y1, 0
            elif direction == "bottom":
                dy, dx = h - y2, 0
            elif direction == "left":
                dy, dx = 0, -x1
            elif direction == "right":
                dy, dx = 0, w - x2
            else:
                return grid

            # Use move_by action
            return ActionLibrary.move_by(dy, dx).apply(grid, obj, all_objs)

        return ConditionalAction(
            name=f"move_to_{direction}",
            apply_fn=apply,
            description=f"move to {direction} edge",
            parameters={"direction": direction}
        )

    @staticmethod
    def move_to_center() -> ConditionalAction:
        """Move object to grid center."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            h, w = grid.shape
            y1, x1, y2, x2 = obj.bbox
            obj_h, obj_w = y2 - y1, x2 - x1

            # Calculate center position
            center_y = (h - obj_h) // 2
            center_x = (w - obj_w) // 2

            dy = center_y - y1
            dx = center_x - x1

            return ActionLibrary.move_by(dy, dx).apply(grid, obj, all_objs)

        return ConditionalAction(
            name="move_to_center",
            apply_fn=apply,
            description="move to center",
            parameters={}
        )

    @staticmethod
    def scale(factor: int) -> ConditionalAction:
        """Scale object by factor."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            y1, x1, y2, x2 = obj.bbox
            obj_grid = grid[y1:y2, x1:x2].copy()

            if factor > 1:
                # Upscale by repeating
                scaled = np.repeat(np.repeat(obj_grid, factor, axis=0), factor, axis=1)
            elif factor < 1 and factor > 0:
                # Downscale by sampling
                scaled = obj_grid[::int(1/factor), ::int(1/factor)]
            else:
                return result

            # Clear old position
            result[obj.mask] = 0

            # Place scaled version (if it fits)
            if (y1 + scaled.shape[0] <= result.shape[0] and
                x1 + scaled.shape[1] <= result.shape[1]):
                # Only overwrite non-zero pixels
                mask = scaled != 0
                result[y1:y1+scaled.shape[0], x1:x1+scaled.shape[1]][mask] = scaled[mask]

            return result

        return ConditionalAction(
            name=f"scale_{factor}",
            apply_fn=apply,
            description=f"scale {factor}x",
            parameters={"factor": factor}
        )

    @staticmethod
    def remove() -> ConditionalAction:
        """Remove/delete object."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()
            result[obj.mask] = 0
            return result

        return ConditionalAction(
            name="remove",
            apply_fn=apply,
            description="remove",
            parameters={}
        )

    @staticmethod
    def keep() -> ConditionalAction:
        """Keep object unchanged."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            return grid  # No change

        return ConditionalAction(
            name="keep",
            apply_fn=apply,
            description="keep",
            parameters={}
        )

    # ========== PHASE 5: COMPOSITE ACTIONS ==========

    # === GEOMETRIC TRANSFORMATIONS ===

    @staticmethod
    def rotate_90() -> ConditionalAction:
        """Rotate object 90 degrees clockwise."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            y1, x1, y2, x2 = obj.bbox
            obj_grid = grid[y1:y2, x1:x2].copy()
            obj_mask = obj.mask[y1:y2, x1:x2]

            # Extract object pixels only
            obj_grid[~obj_mask] = 0

            # Rotate 90 degrees clockwise
            rotated = np.rot90(obj_grid, k=-1)  # k=-1 for clockwise

            # Clear old position
            result[obj.mask] = 0

            # Place rotated version (if it fits)
            if (y1 + rotated.shape[0] <= result.shape[0] and
                x1 + rotated.shape[1] <= result.shape[1]):
                mask = rotated != 0
                result[y1:y1+rotated.shape[0], x1:x1+rotated.shape[1]][mask] = rotated[mask]

            return result

        return ConditionalAction(
            name="rotate_90",
            apply_fn=apply,
            description="rotate 90° clockwise",
            parameters={}
        )

    @staticmethod
    def rotate_180() -> ConditionalAction:
        """Rotate object 180 degrees."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            y1, x1, y2, x2 = obj.bbox
            obj_grid = grid[y1:y2, x1:x2].copy()
            obj_mask = obj.mask[y1:y2, x1:x2]

            obj_grid[~obj_mask] = 0

            # Rotate 180 degrees
            rotated = np.rot90(obj_grid, k=2)

            # Clear old position
            result[obj.mask] = 0

            # Place rotated version
            if (y1 + rotated.shape[0] <= result.shape[0] and
                x1 + rotated.shape[1] <= result.shape[1]):
                mask = rotated != 0
                result[y1:y1+rotated.shape[0], x1:x1+rotated.shape[1]][mask] = rotated[mask]

            return result

        return ConditionalAction(
            name="rotate_180",
            apply_fn=apply,
            description="rotate 180°",
            parameters={}
        )

    @staticmethod
    def rotate_270() -> ConditionalAction:
        """Rotate object 270 degrees clockwise (90 counter-clockwise)."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            y1, x1, y2, x2 = obj.bbox
            obj_grid = grid[y1:y2, x1:x2].copy()
            obj_mask = obj.mask[y1:y2, x1:x2]

            obj_grid[~obj_mask] = 0

            # Rotate 270 degrees clockwise (= 90 counter-clockwise)
            rotated = np.rot90(obj_grid, k=1)

            # Clear old position
            result[obj.mask] = 0

            # Place rotated version
            if (y1 + rotated.shape[0] <= result.shape[0] and
                x1 + rotated.shape[1] <= result.shape[1]):
                mask = rotated != 0
                result[y1:y1+rotated.shape[0], x1:x1+rotated.shape[1]][mask] = rotated[mask]

            return result

        return ConditionalAction(
            name="rotate_270",
            apply_fn=apply,
            description="rotate 270° clockwise",
            parameters={}
        )

    @staticmethod
    def reflect_horizontal() -> ConditionalAction:
        """Reflect object horizontally (flip left-right)."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            y1, x1, y2, x2 = obj.bbox
            obj_grid = grid[y1:y2, x1:x2].copy()
            obj_mask = obj.mask[y1:y2, x1:x2]

            obj_grid[~obj_mask] = 0

            # Flip horizontally
            reflected = np.fliplr(obj_grid)

            # Clear old position
            result[obj.mask] = 0

            # Place reflected version
            mask = reflected != 0
            result[y1:y2, x1:x2][mask] = reflected[mask]

            return result

        return ConditionalAction(
            name="reflect_h",
            apply_fn=apply,
            description="reflect horizontally",
            parameters={}
        )

    @staticmethod
    def reflect_vertical() -> ConditionalAction:
        """Reflect object vertically (flip up-down)."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            y1, x1, y2, x2 = obj.bbox
            obj_grid = grid[y1:y2, x1:x2].copy()
            obj_mask = obj.mask[y1:y2, x1:x2]

            obj_grid[~obj_mask] = 0

            # Flip vertically
            reflected = np.flipud(obj_grid)

            # Clear old position
            result[obj.mask] = 0

            # Place reflected version
            mask = reflected != 0
            result[y1:y2, x1:x2][mask] = reflected[mask]

            return result

        return ConditionalAction(
            name="reflect_v",
            apply_fn=apply,
            description="reflect vertically",
            parameters={}
        )

    # === GRID OPERATIONS ===

    @staticmethod
    def swap_colors(color1: int, color2: int) -> ConditionalAction:
        """Swap two colors in the object."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            # Get object region
            y1, x1, y2, x2 = obj.bbox
            obj_region = result[y1:y2, x1:x2].copy()
            obj_mask = obj.mask[y1:y2, x1:x2]

            # Swap colors within object
            temp = obj_region.copy()
            obj_region[(temp == color1) & obj_mask] = color2
            obj_region[(temp == color2) & obj_mask] = color1

            result[y1:y2, x1:x2] = obj_region

            return result

        return ConditionalAction(
            name=f"swap_{color1}_{color2}",
            apply_fn=apply,
            description=f"swap colors {color1} ↔ {color2}",
            parameters={"color1": color1, "color2": color2}
        )

    @staticmethod
    def fill_to_edge(direction: str, color: int) -> ConditionalAction:
        """Fill from object to edge in direction with color."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            h, w = grid.shape
            y1, x1, y2, x2 = obj.bbox

            if direction == "top":
                result[0:y1, x1:x2] = color
            elif direction == "bottom":
                result[y2:h, x1:x2] = color
            elif direction == "left":
                result[y1:y2, 0:x1] = color
            elif direction == "right":
                result[y1:y2, x2:w] = color

            return result

        return ConditionalAction(
            name=f"fill_to_{direction}_{color}",
            apply_fn=apply,
            description=f"fill to {direction} edge with {color}",
            parameters={"direction": direction, "color": color}
        )

    @staticmethod
    def replicate(dy: int, dx: int) -> ConditionalAction:
        """Replicate object at offset (dy, dx)."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            y1, x1, y2, x2 = obj.bbox
            new_y1 = y1 + dy
            new_x1 = x1 + dx
            new_y2 = y2 + dy
            new_x2 = x2 + dx

            # Check bounds
            h, w = grid.shape
            if (new_y1 >= 0 and new_x1 >= 0 and
                new_y2 <= h and new_x2 <= w):

                # Extract object pixels
                obj_pixels = grid[y1:y2, x1:x2].copy()
                obj_mask = obj.mask[y1:y2, x1:x2]
                obj_pixels[~obj_mask] = 0

                # Copy to new position (don't clear old position)
                mask = obj_pixels != 0
                result[new_y1:new_y2, new_x1:new_x2][mask] = obj_pixels[mask]

            return result

        return ConditionalAction(
            name=f"replicate_{dy}_{dx}",
            apply_fn=apply,
            description=f"replicate at offset ({dy}, {dx})",
            parameters={"dy": dy, "dx": dx}
        )

    @staticmethod
    def extend_to_edge(direction: str) -> ConditionalAction:
        """Extend object to edge in direction."""
        def apply(grid: np.ndarray, obj: ArcObject, all_objs: List[ArcObject]) -> np.ndarray:
            result = grid.copy()

            h, w = grid.shape
            y1, x1, y2, x2 = obj.bbox
            color = obj.dominant_color

            if direction == "top":
                result[0:y2, x1:x2][result[0:y2, x1:x2] == 0] = color
            elif direction == "bottom":
                result[y1:h, x1:x2][result[y1:h, x1:x2] == 0] = color
            elif direction == "left":
                result[y1:y2, 0:x2][result[y1:y2, 0:x2] == 0] = color
            elif direction == "right":
                result[y1:y2, x1:w][result[y1:y2, x1:w] == 0] = color

            return result

        return ConditionalAction(
            name=f"extend_to_{direction}",
            apply_fn=apply,
            description=f"extend to {direction} edge",
            parameters={"direction": direction}
        )


class ConditionalTransformBuilder:
    """
    Builder for creating conditional transformations from training examples.

    This is the key to unlocking higher solve rates - inferring conditional
    patterns from examples rather than just unconditional patterns.
    """

    def __init__(self):
        self.conditions = ConditionLibrary()
        self.actions = ActionLibrary()

    def build_from_observations(self, observations: List[Dict[str, Any]]) -> List[ConditionalTransform]:
        """
        Build conditional transforms from observations.

        Observations format:
        [
            {
                'obj': ArcObject (input),
                'obj_out': ArcObject (output),
                'grid': input grid,
                'grid_out': output grid,
                'all_objs': all input objects
            },
            ...
        ]

        Returns: List of conditional transforms that fit the data
        """
        transforms = []

        if not observations:
            return transforms

        # Try to find conditional patterns
        # Strategy: Look for cases where similar objects undergo different transformations

        # Group by transformation type
        color_changes = []
        position_changes = []
        size_changes = []
        removals = []

        for obs in observations:
            obj_in = obs['obj']
            obj_out = obs.get('obj_out')

            if obj_out is None:
                removals.append(obs)
                continue

            # Detect transformation type
            if obj_in.dominant_color != obj_out.dominant_color:
                color_changes.append(obs)

            dy = obj_out.position[0] - obj_in.position[0]
            dx = obj_out.position[1] - obj_in.position[1]
            if abs(dy) > 0.5 or abs(dx) > 0.5:
                obs['delta'] = (int(dy), int(dx))
                position_changes.append(obs)

            if abs(obj_in.size - obj_out.size) > 2:
                size_changes.append(obs)

        # Try to find conditional patterns for color changes
        if color_changes:
            transforms.extend(self._infer_color_conditionals(color_changes))

        # Try to find conditional patterns for position changes
        if position_changes:
            transforms.extend(self._infer_position_conditionals(position_changes))

        # Try to find conditional patterns for removals
        if removals and len(removals) < len(observations):
            transforms.extend(self._infer_removal_conditionals(removals, observations))

        return transforms

    def _infer_color_conditionals(self, observations: List[Dict[str, Any]]) -> List[ConditionalTransform]:
        """Infer conditional color transformations."""
        transforms = []

        # Check if color depends on size
        color_by_size = {}
        for obs in observations:
            size = obs['obj'].size
            color_out = obs['obj_out'].dominant_color
            color_by_size[size] = color_out

        # If different sizes get different colors, create conditional
        unique_colors = len(set(color_by_size.values()))
        if unique_colors > 1:
            # Find threshold
            sizes = sorted(color_by_size.keys())
            mid_size = sizes[len(sizes)//2]

            small_color = color_by_size[sizes[0]]
            large_color = color_by_size[sizes[-1]]

            if small_color != large_color:
                transforms.append(ConditionalTransform(
                    condition=self.conditions.size_greater_than(mid_size),
                    then_action=self.actions.recolor_to(large_color),
                    else_action=self.actions.recolor_to(small_color),
                    confidence=0.8
                ))

        # Check if color depends on position (edge vs center)
        colors_near_edge = []
        colors_near_center = []

        for obs in observations:
            obj = obs['obj']
            grid = obs['grid']
            color_out = obs['obj_out'].dominant_color

            y1, x1, y2, x2 = obj.bbox
            h, w = grid.shape
            near_edge = (y1 < 2 or x1 < 2 or y2 > h-2 or x2 > w-2)

            if near_edge:
                colors_near_edge.append(color_out)
            else:
                colors_near_center.append(color_out)

        if colors_near_edge and colors_near_center:
            edge_color = max(set(colors_near_edge), key=colors_near_edge.count)
            center_color = max(set(colors_near_center), key=colors_near_center.count)

            if edge_color != center_color:
                transforms.append(ConditionalTransform(
                    condition=self.conditions.near_edge(2),
                    then_action=self.actions.recolor_to(edge_color),
                    else_action=self.actions.recolor_to(center_color),
                    confidence=0.8
                ))

        return transforms

    def _infer_position_conditionals(self, observations: List[Dict[str, Any]]) -> List[ConditionalTransform]:
        """Infer conditional position transformations."""
        transforms = []

        # Check if movement depends on position
        movements_by_quadrant = {'top_left': [], 'top_right': [], 'bottom_left': [], 'bottom_right': []}

        for obs in observations:
            obj = obs['obj']
            grid = obs['grid']
            delta = obs.get('delta', (0, 0))

            h, w = grid.shape
            obj_y, obj_x = obj.position

            in_top = obj_y < h / 2
            in_left = obj_x < w / 2

            if in_top and in_left:
                quadrant = 'top_left'
            elif in_top:
                quadrant = 'top_right'
            elif in_left:
                quadrant = 'bottom_left'
            else:
                quadrant = 'bottom_right'

            movements_by_quadrant[quadrant].append(delta)

        # If different quadrants move differently, create conditionals
        # (simplified: just check top vs bottom)
        top_movements = movements_by_quadrant['top_left'] + movements_by_quadrant['top_right']
        bottom_movements = movements_by_quadrant['bottom_left'] + movements_by_quadrant['bottom_right']

        if top_movements and bottom_movements:
            top_avg = tuple(int(np.mean([m[i] for m in top_movements])) for i in range(2))
            bottom_avg = tuple(int(np.mean([m[i] for m in bottom_movements])) for i in range(2))

            if top_avg != bottom_avg:
                transforms.append(ConditionalTransform(
                    condition=self.conditions.in_quadrant('top_left'),  # Simplified
                    then_action=self.actions.move_by(*top_avg),
                    else_action=self.actions.move_by(*bottom_avg),
                    confidence=0.7
                ))

        return transforms

    def _infer_removal_conditionals(self, removals: List[Dict[str, Any]],
                                    all_obs: List[Dict[str, Any]]) -> List[ConditionalTransform]:
        """Infer conditional removal patterns."""
        transforms = []

        # Find what distinguishes removed objects from kept objects
        removed_sizes = [obs['obj'].size for obs in removals]
        kept_sizes = [obs['obj'].size for obs in all_obs if obs not in removals]

        if removed_sizes and kept_sizes:
            avg_removed = np.mean(removed_sizes)
            avg_kept = np.mean(kept_sizes)

            if avg_removed < avg_kept:
                # Small objects are removed
                threshold = int((avg_removed + avg_kept) / 2)
                transforms.append(ConditionalTransform(
                    condition=self.conditions.size_less_than(threshold),
                    then_action=self.actions.remove(),
                    else_action=self.actions.keep(),
                    confidence=0.8
                ))

        return transforms
