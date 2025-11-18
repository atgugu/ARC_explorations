"""
ARC Program Synthesis System
============================

Compositional program synthesis for ARC-AGI tasks.
Moves from fixed primitive selection to program generation.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time

from arc_active_inference_solver import (
    Grid, ARCTask, Hypothesis, PerceptionModule
)
from parameter_inference import ParameterInference, apply_color_mapping


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class Object:
    """A connected component in a grid"""
    mask: np.ndarray  # Boolean mask of object positions
    color: int        # Dominant color
    position: Tuple[int, int]  # (row, col) of top-left
    size: Tuple[int, int]      # (height, width)
    pixels: Set[Tuple[int, int]]  # Set of (row, col) positions

    @property
    def area(self) -> int:
        return len(self.pixels)

    @property
    def center(self) -> Tuple[float, float]:
        if not self.pixels:
            return (0.0, 0.0)
        rows = [r for r, c in self.pixels]
        cols = [c for r, c in self.pixels]
        return (np.mean(rows), np.mean(cols))


@dataclass
class ObjectSet:
    """Collection of objects"""
    objects: List[Object] = field(default_factory=list)

    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    def __getitem__(self, idx):
        return self.objects[idx]


# =============================================================================
# Predicates (for Conditionals)
# =============================================================================

class Predicate:
    """A boolean predicate over grids"""

    def __init__(self, name: str, func: Callable[[Grid], bool], params: Dict[str, Any] = None):
        self.name = name
        self.func = func
        self.params = params or {}

    def evaluate(self, grid: Grid) -> bool:
        """Evaluate predicate on grid"""
        try:
            return self.func(grid, **self.params)
        except Exception:
            return False

    def __str__(self):
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.name}({params_str})"
        return self.name

    def __repr__(self):
        return self.__str__()


def has_border_pred(grid: Grid, color: int) -> bool:
    """Check if grid has a border of given color"""
    h, w = grid.shape
    if h < 2 or w < 2:
        return False

    # Check if edges are all the same color
    top_border = grid.data[0, :]
    bottom_border = grid.data[h-1, :]
    left_border = grid.data[:, 0]
    right_border = grid.data[:, w-1]

    # Check if all border pixels match the color
    border_colors = np.concatenate([top_border, bottom_border, left_border, right_border])
    return np.all(border_colors == color)


def is_symmetric_pred(grid: Grid, axis: str = 'vertical') -> bool:
    """Check if grid is symmetric along axis"""
    if axis == 'vertical':
        return np.array_equal(grid.data, np.fliplr(grid.data))
    elif axis == 'horizontal':
        return np.array_equal(grid.data, np.flipud(grid.data))
    return False


def object_count_gt_pred(grid: Grid, count: int, bg_color: int = 0) -> bool:
    """Check if number of objects > count"""
    objects = detect_objects(grid, bg_color)
    return len(objects) > count


def object_count_eq_pred(grid: Grid, count: int, bg_color: int = 0) -> bool:
    """Check if number of objects == count"""
    objects = detect_objects(grid, bg_color)
    return len(objects) == count


def has_color_pred(grid: Grid, color: int) -> bool:
    """Check if grid contains color"""
    return color in grid.data


def size_matches_pred(grid: Grid, height: int, width: int) -> bool:
    """Check if grid size matches dimensions"""
    return grid.shape == (height, width)


def is_single_color_pred(grid: Grid) -> bool:
    """Check if grid is all one color"""
    return len(np.unique(grid.data)) == 1


def has_multiple_colors_pred(grid: Grid, min_colors: int = 2) -> bool:
    """Check if grid has at least min_colors different colors"""
    return len(np.unique(grid.data)) >= min_colors


# =============================================================================
# Program Representation
# =============================================================================

class Program:
    """A composable ARC program"""

    def __init__(self,
                 op_name: str,
                 op_func: Callable,
                 params: Dict[str, Any] = None,
                 children: List['Program'] = None):
        self.op_name = op_name
        self.op_func = op_func
        self.params = params or {}
        self.children = children or []

    def execute(self, input_grid: Grid) -> Grid:
        """Execute program on input grid"""
        try:
            if self.children:
                # Compositional operation
                return self.op_func(input_grid, self.children, self.params)
            else:
                # Primitive operation
                return self.op_func(input_grid, **self.params)
        except Exception as e:
            # On error, return input unchanged
            return input_grid.copy()

    def complexity(self) -> int:
        """MDL-style complexity score"""
        base_complexity = 1 + len(self.params)
        child_complexity = sum(child.complexity() for child in self.children)
        return base_complexity + child_complexity

    def __str__(self) -> str:
        """Human-readable representation"""
        if not self.children:
            if self.params:
                params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
                return f"{self.op_name}({params_str})"
            return self.op_name
        else:
            children_str = ", ".join(str(child) for child in self.children)
            return f"{self.op_name}({children_str})"

    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# Object Detection
# =============================================================================

def detect_objects(grid: Grid, bg_color: int = 0, min_size: int = 1) -> ObjectSet:
    """
    Detect connected components as objects

    Args:
        grid: Input grid
        bg_color: Background color to ignore
        min_size: Minimum object size (pixels)

    Returns:
        ObjectSet containing detected objects
    """
    from scipy.ndimage import label

    # Create binary mask (non-background)
    binary = (grid.data != bg_color).astype(np.int32)

    # Find connected components
    labeled, num_features = label(binary)

    objects = []
    for obj_id in range(1, num_features + 1):
        # Get mask for this object
        mask = (labeled == obj_id)

        # Get pixels
        pixels = set(zip(*np.where(mask)))

        if len(pixels) < min_size:
            continue

        # Get dominant color
        colors = grid.data[mask]
        unique, counts = np.unique(colors, return_counts=True)
        dominant_color = unique[np.argmax(counts)]

        # Get bounding box
        rows, cols = np.where(mask)
        position = (rows.min(), cols.min())
        size = (rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)

        obj = Object(
            mask=mask,
            color=int(dominant_color),
            position=position,
            size=size,
            pixels=pixels
        )
        objects.append(obj)

    return ObjectSet(objects)


# =============================================================================
# Object Operations
# =============================================================================

def largest_object(objects: ObjectSet) -> Optional[Object]:
    """Get largest object by area"""
    if len(objects) == 0:
        return None
    return max(objects.objects, key=lambda obj: obj.area)


def smallest_object(objects: ObjectSet) -> Optional[Object]:
    """Get smallest object by area"""
    if len(objects) == 0:
        return None
    return min(objects.objects, key=lambda obj: obj.area)


def filter_by_color(objects: ObjectSet, color: int) -> ObjectSet:
    """Filter objects by color"""
    filtered = [obj for obj in objects if obj.color == color]
    return ObjectSet(filtered)


def filter_by_size(objects: ObjectSet, min_area: int, max_area: int = float('inf')) -> ObjectSet:
    """Filter objects by area"""
    filtered = [obj for obj in objects if min_area <= obj.area <= max_area]
    return ObjectSet(filtered)


def recolor_object(obj: Object, grid: Grid, color: int) -> Grid:
    """Change object color in grid"""
    result = grid.copy()
    for r, c in obj.pixels:
        result.data[r, c] = color
    return result


def remove_object(obj: Object, grid: Grid, bg_color: int = 0) -> Grid:
    """Remove object from grid (fill with background)"""
    result = grid.copy()
    for r, c in obj.pixels:
        result.data[r, c] = bg_color
    return result


def keep_only_object(obj: Object, grid: Grid, bg_color: int = 0) -> Grid:
    """Keep only this object, remove everything else"""
    result = Grid(np.full_like(grid.data, bg_color))
    for r, c in obj.pixels:
        result.data[r, c] = grid.data[r, c]
    return result


# =============================================================================
# Size Inference
# =============================================================================

def infer_output_size(train_pairs: List[Tuple[Grid, Grid]], test_input: Grid) -> Tuple[int, int]:
    """
    Infer output size from training examples

    Strategies:
    1. Same as input (most common)
    2. Fixed scale factor (zoom 2x, 3x)
    3. Fixed size (all outputs same size)
    4. Related to input (2*input, 3*input, etc.)
    """

    if not train_pairs:
        return test_input.shape

    # Collect all input/output sizes
    sizes = []
    for inp, out in train_pairs:
        sizes.append((inp.shape, out.shape))

    # Strategy 1: Check if all outputs same as inputs
    if all(inp_shape == out_shape for inp_shape, out_shape in sizes):
        return test_input.shape

    # Strategy 2: Check for constant scale factor
    scales = []
    for inp_shape, out_shape in sizes:
        if inp_shape[0] > 0 and inp_shape[1] > 0:
            scale_h = out_shape[0] / inp_shape[0]
            scale_w = out_shape[1] / inp_shape[1]
            if scale_h == scale_w:  # Uniform scaling
                scales.append(scale_h)

    if scales and all(s == scales[0] for s in scales):
        # Constant scale factor
        scale = scales[0]
        return (int(test_input.shape[0] * scale), int(test_input.shape[1] * scale))

    # Strategy 3: Check if all outputs have same size
    output_sizes = [out_shape for _, out_shape in sizes]
    if all(s == output_sizes[0] for s in output_sizes):
        return output_sizes[0]

    # Strategy 4: Check for additive patterns (input + constant)
    deltas_h = [out_shape[0] - inp_shape[0] for inp_shape, out_shape in sizes]
    deltas_w = [out_shape[1] - inp_shape[1] for inp_shape, out_shape in sizes]

    if all(d == deltas_h[0] for d in deltas_h) and all(d == deltas_w[0] for d in deltas_w):
        return (test_input.shape[0] + deltas_h[0], test_input.shape[1] + deltas_w[0])

    # Default: same as input
    return test_input.shape


def resize_to_size(grid: Grid, target_size: Tuple[int, int], method: str = 'nearest') -> Grid:
    """Resize grid to target size"""
    if grid.shape == target_size:
        return grid.copy()

    target_h, target_w = target_size

    if method == 'nearest':
        # Nearest neighbor resizing
        scale_h = target_h / grid.shape[0]
        scale_w = target_w / grid.shape[1]

        new_data = np.zeros((target_h, target_w), dtype=grid.data.dtype)

        for i in range(target_h):
            for j in range(target_w):
                src_i = int(i / scale_h)
                src_j = int(j / scale_w)
                src_i = min(src_i, grid.shape[0] - 1)
                src_j = min(src_j, grid.shape[1] - 1)
                new_data[i, j] = grid.data[src_i, src_j]

        return Grid(new_data)

    return grid.copy()


# =============================================================================
# Composition Operations
# =============================================================================

def sequence_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Apply programs in sequence"""
    result = input_grid
    for program in children:
        result = program.execute(result)
    return result


def map_objects_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """
    Object-centric pipeline:
    1. Detect objects
    2. Transform each object using child program
    3. Compose back to grid
    """
    if len(children) == 0:
        return input_grid.copy()

    bg_color = params.get('bg_color', 0)
    transform_program = children[0]

    # Detect objects
    objects = detect_objects(input_grid, bg_color=bg_color)

    if len(objects) == 0:
        return input_grid.copy()

    # Start with background
    result = Grid(np.full_like(input_grid.data, bg_color))

    # Transform each object
    for obj in objects:
        # Extract object to its own grid
        obj_grid = keep_only_object(obj, input_grid, bg_color)

        # Apply transformation
        transformed = transform_program.execute(obj_grid)

        # Composite back (non-background pixels)
        mask = transformed.data != bg_color
        result.data[mask] = transformed.data[mask]

    return result


def keep_largest_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Keep only the largest object"""
    bg_color = params.get('bg_color', 0)

    objects = detect_objects(input_grid, bg_color=bg_color)
    if len(objects) == 0:
        return input_grid.copy()

    largest = largest_object(objects)
    return keep_only_object(largest, input_grid, bg_color)


def keep_smallest_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Keep only the smallest object"""
    bg_color = params.get('bg_color', 0)

    objects = detect_objects(input_grid, bg_color=bg_color)
    if len(objects) == 0:
        return input_grid.copy()

    smallest = smallest_object(objects)
    return keep_only_object(smallest, input_grid, bg_color)


def remove_largest_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Remove the largest object"""
    bg_color = params.get('bg_color', 0)

    objects = detect_objects(input_grid, bg_color=bg_color)
    if len(objects) == 0:
        return input_grid.copy()

    largest = largest_object(objects)
    return remove_object(largest, input_grid, bg_color)


def remove_smallest_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Remove the smallest object"""
    bg_color = params.get('bg_color', 0)

    objects = detect_objects(input_grid, bg_color=bg_color)
    if len(objects) == 0:
        return input_grid.copy()

    smallest = smallest_object(objects)
    return remove_object(smallest, input_grid, bg_color)


def recolor_largest_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Recolor the largest object"""
    bg_color = params.get('bg_color', 0)
    new_color = params.get('color', 1)

    objects = detect_objects(input_grid, bg_color=bg_color)
    if len(objects) == 0:
        return input_grid.copy()

    largest = largest_object(objects)
    return recolor_object(largest, input_grid, new_color)


def recolor_smallest_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Recolor the smallest object"""
    bg_color = params.get('bg_color', 0)
    new_color = params.get('color', 1)

    objects = detect_objects(input_grid, bg_color=bg_color)
    if len(objects) == 0:
        return input_grid.copy()

    smallest = smallest_object(objects)
    return recolor_object(smallest, input_grid, new_color)


def recolor_all_objects_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Recolor all objects"""
    bg_color = params.get('bg_color', 0)
    new_color = params.get('color', 1)

    objects = detect_objects(input_grid, bg_color=bg_color)
    if len(objects) == 0:
        return input_grid.copy()

    result = input_grid.copy()
    for obj in objects:
        result = recolor_object(obj, result, new_color)

    return result


# =============================================================================
# Phase 2: Conditional and Loop Operations
# =============================================================================

def conditional_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """
    Conditional operation: if predicate then then_op else else_op

    params['predicate']: Predicate to evaluate
    children[0]: then program (if predicate is true)
    children[1]: else program (if predicate is false)
    """
    predicate = params.get('predicate')
    if predicate is None or len(children) < 2:
        return input_grid.copy()

    then_prog = children[0]
    else_prog = children[1]

    # Evaluate predicate
    if predicate.evaluate(input_grid):
        return then_prog.execute(input_grid)
    else:
        return else_prog.execute(input_grid)


def for_each_object_v2_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """
    For-each loop: apply transform to each object independently

    children[0]: transform program to apply to each object
    params['bg_color']: background color (default 0)
    """
    bg_color = params.get('bg_color', 0)

    if len(children) == 0:
        return input_grid.copy()

    transform = children[0]

    # Detect objects
    objects = detect_objects(input_grid, bg_color=bg_color)

    if len(objects) == 0:
        return input_grid.copy()

    # Start with background
    result = Grid(np.full_like(input_grid.data, bg_color))

    # Transform each object
    for obj in objects:
        # Extract object to its own grid
        obj_grid = keep_only_object(obj, input_grid, bg_color)

        # Apply transformation
        try:
            transformed = transform.execute(obj_grid)

            # Compose back (non-background pixels)
            mask = transformed.data != bg_color
            result.data[mask] = transformed.data[mask]
        except Exception:
            # If transform fails, keep original object
            mask = obj_grid.data != bg_color
            result.data[mask] = obj_grid.data[mask]

    return result


def fill_interior_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """
    Fill interior of bordered region

    Finds rectangular border and fills interior with color
    """
    fill_color = params.get('color', 1)
    border_color = params.get('border_color', None)

    result = input_grid.copy()
    h, w = result.shape

    if h < 3 or w < 3:
        return result

    # Find border (edges of grid)
    # Simple version: fill everything except edges
    result.data[1:h-1, 1:w-1] = fill_color

    return result


def tile_nxm_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """Tile grid n×m times"""
    n = params.get('n', 2)
    m = params.get('m', 2)

    tiled = np.tile(input_grid.data, (n, m))
    return Grid(tiled)


# =============================================================================
# Primitive Wrappers
# =============================================================================

def wrap_primitive(name: str, func: Callable, params: Dict = None) -> Program:
    """Wrap a primitive operation as a Program"""
    return Program(name, func, params or {})


# Import primitives from existing solver
def get_geometric_primitives() -> List[Program]:
    """Get geometric transformation primitives"""
    primitives = []

    # Identity
    primitives.append(Program("identity", lambda g: g.copy(), {}))

    # Flips
    primitives.append(Program("flip_h", lambda g: Grid(np.fliplr(g.data)), {}))
    primitives.append(Program("flip_v", lambda g: Grid(np.flipud(g.data)), {}))

    # Rotations
    primitives.append(Program("rotate_90", lambda g: Grid(np.rot90(g.data, k=1)), {}))
    primitives.append(Program("rotate_180", lambda g: Grid(np.rot90(g.data, k=2)), {}))
    primitives.append(Program("rotate_270", lambda g: Grid(np.rot90(g.data, k=3)), {}))

    # Transpose
    primitives.append(Program("transpose", lambda g: Grid(g.data.T), {}))

    return primitives


def get_color_primitives() -> List[Program]:
    """Get color transformation primitives"""
    primitives = []

    # Color replacements (common colors)
    for old_c in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for new_c in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if old_c != new_c:
                primitives.append(Program(
                    f"replace_{old_c}_to_{new_c}",
                    lambda g, o=old_c, n=new_c: Grid(np.where(g.data == o, n, g.data)),
                    {"old_color": old_c, "new_color": new_c}
                ))

    # Limit to most common replacements to avoid explosion
    return primitives[:20]  # Keep top 20


def get_scaling_primitives() -> List[Program]:
    """Get scaling primitives"""
    primitives = []

    # Zoom 2x
    def zoom_2x(g):
        new_data = np.repeat(np.repeat(g.data, 2, axis=0), 2, axis=1)
        return Grid(new_data)

    primitives.append(Program("zoom_2x", zoom_2x, {}))

    # Zoom 3x
    def zoom_3x(g):
        new_data = np.repeat(np.repeat(g.data, 3, axis=0), 3, axis=1)
        return Grid(new_data)

    primitives.append(Program("zoom_3x", zoom_3x, {}))

    return primitives


# =============================================================================
# Program Synthesis Engine
# =============================================================================

class ProgramSynthesizer:
    """Synthesize programs via depth-bounded enumeration"""

    def __init__(self, max_depth: int = 2, max_programs: int = 100):
        self.max_depth = max_depth
        self.max_programs = max_programs

    def synthesize(self, task: ARCTask, verbose: bool = False) -> List[Program]:
        """
        Synthesize programs for task

        Phase 3: Uses parameter inference to generate targeted programs

        Returns list of Program objects, ranked by training performance
        """
        if verbose:
            print(f"\n=== Program Synthesis (Phase 3: Parameter Inference) ===")
            print(f"Max depth: {self.max_depth}, Max programs: {self.max_programs}")

        # Phase 3: Infer parameters from training examples
        param_inferrer = ParameterInference()
        inferred_params = param_inferrer.infer_all_parameters(task.train_pairs, verbose=verbose)

        if verbose and inferred_params:
            print(f"\nInferred {len(inferred_params)} parameter(s)")

        all_programs = []

        # Level 0: Primitives (with inferred parameters)
        level_0 = self._generate_primitives_with_params(inferred_params)
        if verbose:
            print(f"Level 0: Generated {len(level_0)} primitives (including {len([p for p in level_0 if 'learned' in p.op_name])} learned)")


        level_0_pruned = self._prune_by_training(level_0, task.train_pairs, keep_top=50)
        if verbose:
            print(f"Level 0: Pruned to {len(level_0_pruned)} programs")

        all_programs.extend(level_0_pruned)

        # Level 1: Object-centric operations
        if self.max_depth >= 1:
            level_1 = self._generate_object_programs(level_0_pruned)
            if verbose:
                print(f"Level 1: Generated {len(level_1)} object programs")

            level_1_pruned = self._prune_by_training(level_1, task.train_pairs, keep_top=30)
            if verbose:
                print(f"Level 1: Pruned to {len(level_1_pruned)} programs")

            all_programs.extend(level_1_pruned)

        # Level 2: Sequences
        if self.max_depth >= 2:
            level_2 = self._generate_sequences(level_0_pruned, level_1_pruned)
            if verbose:
                print(f"Level 2: Generated {len(level_2)} sequence programs")

            level_2_pruned = self._prune_by_training(level_2, task.train_pairs, keep_top=30)
            if verbose:
                print(f"Level 2: Pruned to {len(level_2_pruned)} programs")

            all_programs.extend(level_2_pruned)

        # Level 3: Conditionals, Loops, and Patterns (Phase 2)
        if self.max_depth >= 3:
            level_3 = []

            # Conditionals (if-then-else)
            conditionals = self._generate_conditionals(level_0_pruned + level_1_pruned)
            level_3.extend(conditionals)
            if verbose:
                print(f"Level 3: Generated {len(conditionals)} conditional programs")

            # Loops (for-each object)
            loops = self._generate_loops(level_0_pruned)
            level_3.extend(loops)
            if verbose:
                print(f"Level 3: Generated {len(loops)} loop programs")

            # Patterns (tile, fill)
            patterns = self._generate_patterns()
            level_3.extend(patterns)
            if verbose:
                print(f"Level 3: Generated {len(patterns)} pattern programs")

            # Prune level 3
            level_3_pruned = self._prune_by_training(level_3, task.train_pairs, keep_top=40)
            if verbose:
                print(f"Level 3: Pruned to {len(level_3_pruned)} programs")

            all_programs.extend(level_3_pruned)

        # Final ranking
        ranked = self._rank_programs(all_programs, task.train_pairs)

        if verbose:
            print(f"\nTotal programs: {len(ranked)}")
            if len(ranked) > 0:
                print(f"Best program: {ranked[0]} (score: {ranked[0].complexity()})")

        return ranked[:self.max_programs]

    def _generate_primitives(self) -> List[Program]:
        """Generate level-0 primitives"""
        primitives = []

        primitives.extend(get_geometric_primitives())
        primitives.extend(get_color_primitives())
        primitives.extend(get_scaling_primitives())

        # Object operations (non-compositional)
        primitives.append(Program("keep_largest", keep_largest_op, {}))
        primitives.append(Program("keep_smallest", keep_smallest_op, {}))
        primitives.append(Program("remove_largest", remove_largest_op, {}))
        primitives.append(Program("remove_smallest", remove_smallest_op, {}))

        return primitives

    def _generate_primitives_with_params(self, inferred_params: Dict) -> List[Program]:
        """
        Generate primitives using inferred parameters (Phase 3)

        Key innovation: Generate fewer, more targeted programs based on learned parameters
        """
        primitives = []

        # Always include geometric primitives (small set)
        primitives.extend(get_geometric_primitives())

        # Phase 3: Use inferred parameters when available
        if inferred_params.get('color_map'):
            # Generate program with learned color mapping
            color_map = inferred_params['color_map']
            primitives.append(Program(
                "replace_colors_learned",
                lambda g, cm=color_map: apply_color_mapping(g, cm),
                {"color_map": str(color_map)}
            ))

        else:
            # Fallback: Generate small set of common color operations
            primitives.extend(get_color_primitives()[:10])  # Only top 10

        if inferred_params.get('scale_factor'):
            # Generate program with learned scale
            scale = inferred_params['scale_factor']
            def zoom_learned(g, s=scale):
                new_data = np.repeat(np.repeat(g.data, s, axis=0), s, axis=1)
                return Grid(new_data)

            primitives.append(Program(
                f"zoom_{scale}x_learned",
                zoom_learned,
                {"scale": scale}
            ))
        else:
            # Fallback: Include 2x and 3x zooms
            primitives.extend(get_scaling_primitives())

        if inferred_params.get('tile_factor'):
            # Generate program with learned tiling
            n, m = inferred_params['tile_factor']
            primitives.append(Program(
                f"tile_{n}x{m}_learned",
                tile_nxm_op,
                {"n": n, "m": m}
            ))

        if inferred_params.get('rotation'):
            # Generate program with learned rotation
            rotation = inferred_params['rotation']
            k = rotation // 90
            primitives.append(Program(
                f"rotate_{rotation}_learned",
                lambda g, k=k: Grid(np.rot90(g.data, k=k)),
                {"rotation": rotation}
            ))

        if inferred_params.get('flip'):
            # Generate program with learned flip
            flip_dir = inferred_params['flip']
            if flip_dir == 'horizontal':
                primitives.append(Program(
                    "flip_h_learned",
                    lambda g: Grid(np.fliplr(g.data)),
                    {"flip": flip_dir}
                ))
            elif flip_dir == 'vertical':
                primitives.append(Program(
                    "flip_v_learned",
                    lambda g: Grid(np.flipud(g.data)),
                    {"flip": flip_dir}
                ))

        # Object operations (non-compositional)
        primitives.append(Program("keep_largest", keep_largest_op, {}))
        primitives.append(Program("keep_smallest", keep_smallest_op, {}))
        primitives.append(Program("remove_largest", remove_largest_op, {}))
        primitives.append(Program("remove_smallest", remove_smallest_op, {}))

        return primitives

    def _generate_object_programs(self, base_programs: List[Program]) -> List[Program]:
        """Generate object-centric programs"""
        object_programs = []

        # Recolor largest/smallest with different colors
        for color in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            object_programs.append(Program(
                f"recolor_largest_{color}",
                recolor_largest_op,
                {"color": color}
            ))
            object_programs.append(Program(
                f"recolor_smallest_{color}",
                recolor_smallest_op,
                {"color": color}
            ))

        # Recolor all objects
        for color in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            object_programs.append(Program(
                f"recolor_all_{color}",
                recolor_all_objects_op,
                {"color": color}
            ))

        return object_programs

    def _generate_sequences(self,
                           level_0: List[Program],
                           level_1: List[Program]) -> List[Program]:
        """Generate sequence compositions"""
        sequences = []

        # Top-5 from each level
        top_level_0 = level_0[:5]
        top_level_1 = level_1[:5] if level_1 else []

        # Sequences of 2 primitives
        for p1 in top_level_0:
            for p2 in top_level_0:
                if p1.op_name != p2.op_name:  # Avoid identity sequences
                    sequences.append(Program(
                        f"seq({p1.op_name},{p2.op_name})",
                        sequence_op,
                        {},
                        [p1, p2]
                    ))

        # Object operation + primitive
        for p1 in top_level_1:
            for p2 in top_level_0[:3]:
                sequences.append(Program(
                    f"seq({p1.op_name},{p2.op_name})",
                    sequence_op,
                    {},
                    [p1, p2]
                ))

        return sequences

    def _generate_conditionals(self,
                              base_programs: List[Program]) -> List[Program]:
        """Generate conditional programs (Phase 2)"""
        conditionals = []

        # Get identity program
        identity = Program("identity", lambda g: g.copy(), {})

        # Create predicates
        predicates = [
            Predicate("has_border_2", has_border_pred, {"color": 2}),
            Predicate("has_border_1", has_border_pred, {"color": 1}),
            Predicate("is_symmetric_v", is_symmetric_pred, {"axis": "vertical"}),
            Predicate("is_symmetric_h", is_symmetric_pred, {"axis": "horizontal"}),
            Predicate("obj_count_gt_0", object_count_gt_pred, {"count": 0}),
            Predicate("obj_count_gt_1", object_count_gt_pred, {"count": 1}),
            Predicate("obj_count_eq_1", object_count_eq_pred, {"count": 1}),
            Predicate("has_color_1", has_color_pred, {"color": 1}),
            Predicate("has_color_2", has_color_pred, {"color": 2}),
        ]

        # Generate conditionals: if predicate then operation else identity
        top_ops = base_programs[:5]

        for pred in predicates:
            for then_op in top_ops:
                conditionals.append(Program(
                    f"if({pred},{then_op.op_name})",
                    conditional_op,
                    {"predicate": pred},
                    [then_op, identity]
                ))

        return conditionals

    def _generate_loops(self,
                       base_programs: List[Program]) -> List[Program]:
        """Generate loop programs (Phase 2)"""
        loops = []

        # Select suitable programs for object transforms
        # (geometric transforms, color changes)
        top_transforms = base_programs[:8]

        for transform in top_transforms:
            loops.append(Program(
                f"for_each({transform.op_name})",
                for_each_object_v2_op,
                {"bg_color": 0},
                [transform]
            ))

        return loops

    def _generate_patterns(self) -> List[Program]:
        """Generate pattern programs (Phase 2)"""
        patterns = []

        # Tiling operations
        for n in [2, 3]:
            for m in [2, 3]:
                patterns.append(Program(
                    f"tile_{n}x{m}",
                    tile_nxm_op,
                    {"n": n, "m": m}
                ))

        # Fill interior operations
        for color in [1, 2, 3, 8]:
            patterns.append(Program(
                f"fill_interior_{color}",
                fill_interior_op,
                {"color": color}
            ))

        return patterns

    def _prune_by_training(self,
                          programs: List[Program],
                          train_pairs: List[Tuple[Grid, Grid]],
                          keep_top: int) -> List[Program]:
        """Prune programs by evaluating on training examples"""
        scored_programs = []

        for program in programs:
            scores = []

            for input_grid, output_grid in train_pairs:
                try:
                    prediction = program.execute(input_grid)

                    # Exact match
                    if np.array_equal(prediction.data, output_grid.data):
                        scores.append(1.0)
                    # Size mismatch
                    elif prediction.shape != output_grid.shape:
                        scores.append(0.0)
                    # Partial match
                    else:
                        accuracy = np.mean(prediction.data == output_grid.data)
                        scores.append(accuracy)

                except Exception:
                    scores.append(0.0)

            # Average score across training examples
            avg_score = np.mean(scores) if scores else 0.0

            # Complexity penalty (favor simpler programs)
            complexity = program.complexity()
            final_score = avg_score / np.sqrt(complexity) if complexity > 0 else 0.0

            # Always add program to scored list (even with score 0)
            scored_programs.append((program, final_score))

        # Sort by score
        scored_programs.sort(key=lambda x: x[1], reverse=True)

        # Keep top-k, but ensure minimum diversity
        # Always keep at least 5 programs for diversity (even if they score poorly)
        min_keep = min(5, len(programs))
        actual_keep = max(min_keep, min(keep_top, len(scored_programs)))

        return [p for p, score in scored_programs[:actual_keep]]

    def _rank_programs(self,
                      programs: List[Program],
                      train_pairs: List[Tuple[Grid, Grid]]) -> List[Program]:
        """Final ranking of all programs"""
        return self._prune_by_training(programs, train_pairs, keep_top=len(programs))


# =============================================================================
# Integration with Active Inference
# =============================================================================

class ProgramSynthesisHypothesisGenerator:
    """Generate hypotheses via program synthesis"""

    def __init__(self, max_depth: int = 2, max_programs: int = 100):
        self.synthesizer = ProgramSynthesizer(max_depth, max_programs)
        self.perception = PerceptionModule()

    def generate_hypotheses(self, task: ARCTask, features: Dict, verbose: bool = False) -> List[Hypothesis]:
        """
        Generate hypotheses via program synthesis

        Returns list of Hypothesis objects (same interface as before)
        """
        # Synthesize programs
        programs = self.synthesizer.synthesize(task, verbose=verbose)

        # Infer output size
        target_size = infer_output_size(task.train_pairs, task.test_input)

        # Convert to Hypothesis objects
        hypotheses = []
        for i, program in enumerate(programs):
            # Wrap program execution with size adjustment
            def make_program_func(prog, size):
                def func(g):
                    result = prog.execute(g)
                    # Resize if needed
                    if result.shape != size:
                        result = resize_to_size(result, size)
                    return result
                return func

            hypothesis = Hypothesis(
                program=make_program_func(program, target_size),
                name=str(program),
                complexity=float(program.complexity()),
                parameters=program.params
            )
            hypotheses.append(hypothesis)

        if verbose and len(hypotheses) > 0:
            print(f"\nGenerated {len(hypotheses)} hypotheses")
            print(f"Top-3:")
            for i, h in enumerate(hypotheses[:3]):
                print(f"  {i+1}. {h.name} (complexity: {h.complexity})")

        return hypotheses


if __name__ == "__main__":
    # Test program synthesis
    print("Testing Program Synthesis System")
    print("=" * 60)

    # Create simple test task (flip horizontal)
    train_pairs = [
        (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
        (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
    ]
    test_input = Grid([[9, 0], [1, 2]])
    task = ARCTask(train_pairs, test_input)

    # Synthesize programs
    synthesizer = ProgramSynthesizer(max_depth=2, max_programs=10)
    programs = synthesizer.synthesize(task, verbose=True)

    print(f"\n\nTop-5 programs:")
    for i, prog in enumerate(programs[:5]):
        print(f"{i+1}. {prog}")

        # Test on training examples
        print(f"   Training accuracy:")
        for j, (inp, out) in enumerate(task.train_pairs):
            pred = prog.execute(inp)
            match = np.array_equal(pred.data, out.data)
            print(f"     Example {j+1}: {'✓' if match else '✗'}")

    print("\n✓ Program synthesis test complete")
