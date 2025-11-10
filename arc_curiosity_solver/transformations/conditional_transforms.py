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
