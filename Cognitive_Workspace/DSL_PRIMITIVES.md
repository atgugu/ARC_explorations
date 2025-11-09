# Comprehensive DSL for ARC-AGI
## 65 Non-Redundant Primitives Covering All Task Types

This document defines a complete Domain-Specific Language for ARC tasks, carefully designed to be:
- **Comprehensive**: Covers all major ARC task patterns
- **Non-redundant**: Each primitive has a unique purpose
- **Composable**: Primitives combine to solve complex tasks
- **Type-safe**: Clear input/output types

---

## Table of Contents
1. [Selection & Filtering](#1-selection--filtering-12-primitives)
2. [Spatial Transformations](#2-spatial-transformations-10-primitives)
3. [Color Operations](#3-color-operations-8-primitives)
4. [Pattern Operations](#4-pattern-operations-9-primitives)
5. [Grid Operations](#5-grid-operations-7-primitives)
6. [Topological Operations](#6-topological-operations-6-primitives)
7. [Arithmetic & Logic](#7-arithmetic--logic-5-primitives)
8. [Line & Path Operations](#8-line--path-operations-8-primitives)

**Total: 65 primitives**

---

## Type System

```python
Grid = np.ndarray              # 2D integer array (colors 0-9)
Object = List[Tuple[int, int]] # List of (row, col) coordinates
ObjectSet = List[Object]       # Collection of objects
Color = int                    # 0-9 (0=black/background)
Direction = Enum               # UP, DOWN, LEFT, RIGHT, DIAG_*
Axis = Enum                    # HORIZONTAL, VERTICAL, BOTH
Point = Tuple[int, int]        # (row, col)
Shape = str                    # "rectangle", "line", "L", "T", etc.
```

---

## 1. Selection & Filtering (12 primitives)

### 1.1 `select_by_color(grid, color) -> ObjectSet`
**Purpose:** Extract all objects of a specific color
**Example:** `select_by_color(grid, 2)` → all blue objects

### 1.2 `select_by_size(objects, size, comparator='==') -> ObjectSet`
**Purpose:** Filter objects by size (exact, >, <, etc.)
**Example:** `select_by_size(objects, 4, '>')` → objects with >4 pixels

### 1.3 `select_largest(objects, k=1) -> ObjectSet`
**Purpose:** Select k largest objects
**Example:** `select_largest(objects, 3)` → 3 largest objects

### 1.4 `select_smallest(objects, k=1) -> ObjectSet`
**Purpose:** Select k smallest objects
**Example:** `select_smallest(objects, 1)` → smallest object

### 1.5 `select_by_shape(objects, shape) -> ObjectSet`
**Purpose:** Filter by shape type (rectangle, line, L-shape, T-shape, plus, etc.)
**Example:** `select_by_shape(objects, "rectangle")` → rectangular objects

### 1.6 `select_by_position(objects, position) -> ObjectSet`
**Purpose:** Filter by grid position (corner, edge, center, top, bottom, left, right)
**Example:** `select_by_position(objects, "corner")` → objects in corners

### 1.7 `select_by_property(objects, property, value) -> ObjectSet`
**Purpose:** Filter by computed property (has_hole, is_symmetric, is_convex, etc.)
**Example:** `select_by_property(objects, "is_symmetric", True)`

### 1.8 `select_unique_color(objects) -> ObjectSet`
**Purpose:** Select objects that are the only one of their color
**Example:** Objects that appear once in the grid

### 1.9 `select_touching(object, objects) -> ObjectSet`
**Purpose:** Select objects that touch a given object
**Example:** `select_touching(obj1, all_objects)` → neighbors

### 1.10 `select_aligned(objects, axis) -> ObjectSet`
**Purpose:** Select objects aligned on an axis
**Example:** `select_aligned(objects, HORIZONTAL)` → objects in same row

### 1.11 `select_by_distance(object, objects, distance, comparator='==') -> ObjectSet`
**Purpose:** Select objects at specific distance from reference object
**Example:** `select_by_distance(obj, others, 5, '<')` → objects within 5 units

### 1.12 `select_background() -> Grid`
**Purpose:** Extract the background pattern (most common non-object color/pattern)
**Example:** Useful for tasks with textured backgrounds

---

## 2. Spatial Transformations (10 primitives)

### 2.1 `rotate(object, angle) -> Object`
**Purpose:** Rotate object by angle (90, 180, 270, or arbitrary)
**Example:** `rotate(obj, 90)` → rotate 90° clockwise

### 2.2 `reflect(object, axis) -> Object`
**Purpose:** Reflect across axis (horizontal, vertical, diagonal)
**Example:** `reflect(obj, VERTICAL)` → mirror vertically

### 2.3 `translate(object, delta_row, delta_col) -> Object`
**Purpose:** Move object by offset
**Example:** `translate(obj, 2, -3)` → move down 2, left 3

### 2.4 `scale(object, factor) -> Object`
**Purpose:** Scale object by integer factor
**Example:** `scale(obj, 2)` → double size

### 2.5 `scale_to_fit(object, width, height) -> Object`
**Purpose:** Scale to specific dimensions
**Example:** `scale_to_fit(obj, 5, 5)` → resize to 5×5

### 2.6 `move_to(object, position) -> Object`
**Purpose:** Move to absolute position (corner, center, edge, specific coords)
**Example:** `move_to(obj, "top_left")` → move to top-left corner

### 2.7 `align(objects, axis, spacing=0) -> ObjectSet`
**Purpose:** Align multiple objects along axis with spacing
**Example:** `align(objects, HORIZONTAL, 1)` → arrange in row with gap=1

### 2.8 `center(object, grid_size) -> Object`
**Purpose:** Center object in grid
**Example:** `center(obj, (10, 10))` → center in 10×10 grid

### 2.9 `gravity(objects, direction, until='edge') -> ObjectSet`
**Purpose:** Apply gravity - move objects until collision
**Example:** `gravity(objects, DOWN, 'edge')` → drop to bottom

### 2.10 `orbit(object, center, angle) -> Object`
**Purpose:** Rotate object around a point
**Example:** `orbit(obj, (5, 5), 90)` → rotate around center

---

## 3. Color Operations (8 primitives)

### 3.1 `recolor(object, new_color) -> Object`
**Purpose:** Change object to single color
**Example:** `recolor(obj, 3)` → make green

### 3.2 `recolor_by_rule(objects, rule) -> ObjectSet`
**Purpose:** Recolor based on rule (size→color, position→color, shape→color)
**Example:** `recolor_by_rule(objects, "size_ascending")` → smallest=1, largest=9

### 3.3 `swap_colors(grid, color1, color2) -> Grid`
**Purpose:** Swap two colors globally
**Example:** `swap_colors(grid, 1, 2)` → blue↔red

### 3.4 `gradient_color(objects, start_color, end_color, axis) -> ObjectSet`
**Purpose:** Apply color gradient along axis
**Example:** `gradient_color(objects, 1, 9, HORIZONTAL)` → left=1, right=9

### 3.5 `recolor_by_neighbor(objects, rule) -> ObjectSet`
**Purpose:** Recolor based on neighboring colors
**Example:** `recolor_by_neighbor(objects, "majority")` → take most common neighbor color

### 3.6 `palette_reduce(grid, num_colors) -> Grid`
**Purpose:** Reduce to top-k most common colors
**Example:** `palette_reduce(grid, 3)` → keep only 3 colors

### 3.7 `color_cycle(objects, colors, start=0) -> ObjectSet`
**Purpose:** Assign colors from list cyclically
**Example:** `color_cycle(objects, [1,2,3])` → obj1=1, obj2=2, obj3=3, obj4=1...

### 3.8 `invert_colors(grid, palette) -> Grid`
**Purpose:** Invert color mapping
**Example:** `invert_colors(grid, [0,9])` → 0→9, 9→0

---

## 4. Pattern Operations (9 primitives)

### 4.1 `tile(object, rows, cols) -> Grid`
**Purpose:** Tile object in rows×cols pattern
**Example:** `tile(obj, 3, 3)` → 3×3 repetition

### 4.2 `tile_with_spacing(object, rows, cols, spacing) -> Grid`
**Purpose:** Tile with gap between copies
**Example:** `tile_with_spacing(obj, 2, 2, 1)` → 2×2 with 1-pixel gaps

### 4.3 `copy_to_positions(object, positions) -> Grid`
**Purpose:** Copy object to specific positions
**Example:** `copy_to_positions(obj, [(0,0), (5,5), (10,10)])`

### 4.4 `copy_to_pattern(object, pattern_object) -> Grid`
**Purpose:** Copy object to all positions where pattern exists
**Example:** Copy X to all positions where Y appears

### 4.5 `symmetrize(grid, axis) -> Grid`
**Purpose:** Make grid symmetric by mirroring
**Example:** `symmetrize(grid, VERTICAL)` → mirror bottom from top

### 4.6 `extend_pattern(grid, direction, steps) -> Grid`
**Purpose:** Detect and extend repeating pattern
**Example:** `extend_pattern(grid, RIGHT, 3)` → continue pattern 3 steps

### 4.7 `rotate_pattern(objects, center, angles) -> ObjectSet`
**Purpose:** Create rotational copies around center
**Example:** `rotate_pattern(obj, center, [90,180,270])` → 4-fold rotation

### 4.8 `kaleidoscope(object, order) -> Grid`
**Purpose:** Create kaleidoscope pattern with n-fold symmetry
**Example:** `kaleidoscope(obj, 4)` → 4-way mirror symmetry

### 4.9 `tessellate(objects, pattern) -> Grid`
**Purpose:** Arrange objects in tessellation pattern (square, hex, brick, etc.)
**Example:** `tessellate(objects, "hexagonal")`

---

## 5. Grid Operations (7 primitives)

### 5.1 `overlay(grid1, grid2, mode='replace') -> Grid`
**Purpose:** Combine grids (modes: replace, blend, add, or, and, xor)
**Example:** `overlay(base, pattern, 'or')` → logical OR of grids

### 5.2 `crop(grid, top, left, height, width) -> Grid`
**Purpose:** Extract rectangular region
**Example:** `crop(grid, 0, 0, 5, 5)` → top-left 5×5

### 5.3 `crop_to_content(grid) -> Grid`
**Purpose:** Crop to bounding box of non-background pixels
**Example:** Remove empty borders

### 5.4 `pad(grid, padding, color=0) -> Grid`
**Purpose:** Add border around grid
**Example:** `pad(grid, 2, 1)` → 2-pixel blue border

### 5.5 `resize_grid(grid, new_height, new_width) -> Grid`
**Purpose:** Resize with nearest-neighbor or replication
**Example:** `resize_grid(grid, 20, 20)` → scale to 20×20

### 5.6 `split_grid(grid, rows, cols) -> List[Grid]`
**Purpose:** Divide grid into subgrids
**Example:** `split_grid(grid, 2, 2)` → 4 quadrants

### 5.7 `merge_grids(grids, rows, cols) -> Grid`
**Purpose:** Combine subgrids into single grid
**Example:** `merge_grids([g1,g2,g3,g4], 2, 2)` → combine 2×2

---

## 6. Topological Operations (6 primitives)

### 6.1 `fill_holes(object) -> Object`
**Purpose:** Fill interior holes in object
**Example:** Hollow square → filled square

### 6.2 `hollow(object, thickness=1) -> Object`
**Purpose:** Keep only outline
**Example:** Filled rectangle → frame

### 6.3 `grow(object, amount=1) -> Object`
**Purpose:** Dilate object by pixels (morphological dilation)
**Example:** `grow(obj, 1)` → expand by 1 pixel in all directions

### 6.4 `shrink(object, amount=1) -> Object`
**Purpose:** Erode object (morphological erosion)
**Example:** `shrink(obj, 1)` → contract by 1 pixel

### 6.5 `convex_hull(object) -> Object`
**Purpose:** Compute convex hull
**Example:** L-shape → filled rectangle enclosing it

### 6.6 `skeleton(object) -> Object`
**Purpose:** Extract medial axis (thinning)
**Example:** Thick line → 1-pixel line

---

## 7. Arithmetic & Logic (5 primitives)

### 7.1 `count(objects, property=None) -> int`
**Purpose:** Count objects or objects with property
**Example:** `count(objects, "is_red")` → number of red objects

### 7.2 `measure(object, metric) -> float`
**Purpose:** Measure property (area, perimeter, width, height, aspect_ratio, etc.)
**Example:** `measure(obj, "aspect_ratio")` → width/height

### 7.3 `majority_vote(objects, property) -> value`
**Purpose:** Find most common value of property
**Example:** `majority_vote(objects, "color")` → most common color

### 7.4 `distribute_evenly(objects, grid_size) -> ObjectSet`
**Purpose:** Space objects evenly in grid
**Example:** `distribute_evenly(objects, (10,10))` → equal spacing

### 7.5 `sort_objects(objects, key, order='ascending') -> ObjectSet`
**Purpose:** Sort objects by property
**Example:** `sort_objects(objects, "size", "descending")` → largest first

---

## 8. Line & Path Operations (8 primitives)

### 8.1 `connect(obj1, obj2, pattern='line', color=1) -> Object`
**Purpose:** Draw connection between objects
**Example:** `connect(obj1, obj2, 'line', 2)` → red line between them

### 8.2 `draw_line(start, end, color=1, thickness=1) -> Object`
**Purpose:** Draw line from point to point
**Example:** `draw_line((0,0), (9,9), 3, 1)` → green diagonal

### 8.3 `draw_rectangle(top_left, bottom_right, fill=False, color=1) -> Object`
**Purpose:** Draw rectangle
**Example:** `draw_rectangle((2,2), (8,8), fill=True, color=4)` → yellow square

### 8.4 `extend_line(line_object, direction, length='edge') -> Object`
**Purpose:** Extend line in direction
**Example:** `extend_line(line, UP, 'edge')` → extend to top edge

### 8.5 `trace_boundary(object, color=1) -> Object`
**Purpose:** Draw outline around object
**Example:** `trace_boundary(obj, 2)` → red border

### 8.6 `shortest_path(start, end, obstacles, color=1) -> Object`
**Purpose:** Draw shortest path avoiding obstacles
**Example:** Pathfinding between two points

### 8.7 `fill_region(grid, seed_point, new_color) -> Grid`
**Purpose:** Flood fill from point
**Example:** `fill_region(grid, (5,5), 3)` → flood fill with green

### 8.8 `detect_lines(grid, direction=None) -> ObjectSet`
**Purpose:** Extract all lines from grid (horizontal, vertical, or diagonal)
**Example:** `detect_lines(grid, HORIZONTAL)` → all horizontal lines

---

## Composition & Control Flow

### 9.1 `compose(operations) -> Operation`
**Purpose:** Chain operations sequentially
**Example:** `compose([rotate(90), reflect(VERTICAL), translate(2,2)])`

### 9.2 `map_objects(operation, objects) -> ObjectSet`
**Purpose:** Apply operation to each object
**Example:** `map_objects(rotate(90), objects)` → rotate all

### 9.3 `filter_objects(predicate, objects) -> ObjectSet`
**Purpose:** Keep objects satisfying predicate
**Example:** `filter_objects(lambda o: measure(o, "area") > 5, objects)`

### 9.4 `conditional(predicate, if_true, if_false) -> Operation`
**Purpose:** Conditional execution
**Example:** `conditional(is_symmetric, reflect(VERTICAL), rotate(90))`

### 9.5 `foreach_color(operation, grid) -> Grid`
**Purpose:** Apply operation to each color separately
**Example:** `foreach_color(rotate(90), grid)` → rotate each color's objects

---

## Advanced Composites (Built from primitives)

These are **not** new primitives but show how primitives compose:

### Example 1: Copy largest to corners
```python
largest = select_largest(select_by_color(grid, 2), 1)
corners = [(0,0), (0,W-1), (H-1,0), (H-1,W-1)]
result = copy_to_positions(largest, corners)
```

### Example 2: Color by size rank
```python
objects = select_by_color(grid, 2)
sorted_objs = sort_objects(objects, "size", "ascending")
colored = color_cycle(sorted_objs, [1,2,3,4,5])
```

### Example 3: Mirror and tile
```python
base = select_by_color(grid, 3)
mirrored = reflect(base, VERTICAL)
tiled = tile(mirrored, 3, 3)
```

### Example 4: Gravity simulation
```python
objects = select_by_color(grid, 5)
fallen = gravity(objects, DOWN, 'edge')
stacked = align(fallen, VERTICAL, spacing=0)
```

---

## Primitive Coverage Analysis

### By ARC Task Type:

| Task Category | Covered By | Key Primitives |
|---------------|------------|----------------|
| **Rotation/Reflection** | Group 2 | rotate, reflect, orbit |
| **Tiling/Patterns** | Group 4 | tile, tessellate, extend_pattern |
| **Color Mapping** | Group 3 | recolor_by_rule, gradient_color |
| **Size/Scale** | Group 2 | scale, scale_to_fit |
| **Gravity/Physics** | Group 2 | gravity, align |
| **Filling/Holes** | Group 6 | fill_holes, hollow, fill_region |
| **Lines/Connections** | Group 8 | connect, draw_line, extend_line |
| **Cropping/Splitting** | Group 5 | crop, split_grid, merge_grids |
| **Selection Logic** | Group 1 | All 12 selection primitives |
| **Counting/Measurement** | Group 7 | count, measure, majority_vote |
| **Growth/Morphology** | Group 6 | grow, shrink, skeleton |
| **Symmetry** | Groups 2,4 | reflect, symmetrize, kaleidoscope |

### Coverage Estimate:
- **Simple tasks** (1-2 primitives): ~70-80% coverage
- **Medium tasks** (3-5 primitives): ~50-60% coverage
- **Complex tasks** (6+ primitives): ~30-40% coverage

---

## Implementation Priority

### Phase 1: Core (20 primitives)
Must-have for MVP:
1. select_by_color, select_largest, select_by_position
2. rotate, reflect, translate, scale
3. recolor, swap_colors
4. tile, copy_to_positions
5. overlay, crop, resize_grid
6. fill_holes, grow, shrink
7. count, measure
8. connect, draw_line, fill_region

### Phase 2: Extended (25 primitives)
Add for 80% coverage:
- All remaining selection primitives
- Pattern operations (tessellate, extend_pattern)
- Color operations (gradient, recolor_by_rule)
- Advanced spatial (gravity, orbit, align)
- Line operations (extend_line, detect_lines)

### Phase 3: Advanced (20 primitives)
Specialized cases:
- Sophisticated topology (convex_hull, skeleton)
- Complex patterns (kaleidoscope)
- Path operations (shortest_path)
- Advanced composition

---

## Type Signatures (Python Implementation)

```python
from typing import List, Tuple, Union, Callable, Literal
from enum import Enum
import numpy as np

# Type aliases
Grid = np.ndarray
Object = List[Tuple[int, int]]
ObjectSet = List[Object]
Color = int
Point = Tuple[int, int]

class Direction(Enum):
    UP = (−1, 0)
    DOWN = (1, 0)
    LEFT = (0, −1)
    RIGHT = (0, 1)
    DIAG_NE = (−1, 1)
    DIAG_NW = (−1, −1)
    DIAG_SE = (1, 1)
    DIAG_SW = (1, −1)

class Axis(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2
    BOTH = 3

# Example signatures
def select_by_color(grid: Grid, color: Color) -> ObjectSet: ...
def rotate(obj: Object, angle: int) -> Object: ...
def tile(obj: Object, rows: int, cols: int) -> Grid: ...
def overlay(g1: Grid, g2: Grid, mode: str = 'replace') -> Grid: ...
def grow(obj: Object, amount: int = 1) -> Object: ...
def connect(obj1: Object, obj2: Object, pattern: str = 'line', color: Color = 1) -> Object: ...
```

---

## Testing Strategy

Each primitive needs:
1. **Unit test**: Verify correct behavior on simple inputs
2. **Property test**: Check invariants (e.g., rotate(rotate(obj, 90), 270) ≈ original)
3. **Integration test**: Compose with other primitives
4. **ARC task test**: Solve at least 1 real task using this primitive

Example test structure:
```python
def test_rotate():
    # Simple square
    obj = [(0,0), (0,1), (1,0), (1,1)]
    rotated = rotate(obj, 90)
    assert rotated == [(0,1), (1,1), (0,0), (1,0)]

    # Idempotence
    obj2 = [(2,3), (2,4), (3,3)]
    assert rotate(rotate(obj2, 90), 270) == obj2

    # Solve task 00d62c1b (example)
    grid = load_task("00d62c1b")
    objects = select_by_color(grid, 5)
    rotated = map_objects(rotate(90), objects)
    assert check_solution(rotated, ground_truth)
```

---

## Redundancy Check

**Why these 65 and not more?**

Each primitive satisfies one of:
1. **Unique operation**: No other primitive does this (e.g., `skeleton`)
2. **Essential parameter**: Different enough to warrant separate function (e.g., `select_largest` vs `select_smallest` - could merge but less clear)
3. **Fundamental building block**: Needed for composability (e.g., all 4 basic rotations could be one function with parameter, but having separate `rotate` is clearer)

**Avoided redundancies:**
- ❌ Separate `rotate_90`, `rotate_180`, `rotate_270` → ✅ `rotate(angle)`
- ❌ `reflect_horizontal`, `reflect_vertical` → ✅ `reflect(axis)`
- ❌ `tile_2x2`, `tile_3x3` → ✅ `tile(rows, cols)`
- ❌ `select_red`, `select_blue` → ✅ `select_by_color(color)`

**Why keep separate:**
- ✅ `select_largest` and `select_by_size`: Different use cases (top-k vs threshold)
- ✅ `fill_holes` and `fill_region`: Different algorithms (topology vs flood-fill)
- ✅ `grow` and `scale`: Different semantics (morphology vs geometric)
- ✅ `tile` and `copy_to_positions`: Different patterns (regular vs arbitrary)

---

## Extension Mechanism

For tasks not covered by 65 primitives:

1. **Analyze failure**: What primitive is missing?
2. **Check composability**: Can it be composed from existing primitives?
   - If yes: Add to examples, not as primitive
   - If no: Consider adding
3. **Test uniqueness**: Does it overlap with existing primitives?
4. **Add to appropriate group**: Maintain organization
5. **Update coverage analysis**: Track new task types

**Example:**
- Missing: "remove smallest object"
- Compose: `filter_objects(lambda o: o not in select_smallest(objects, 1), objects)`
- OR add as primitive if very common: `remove_smallest(objects, k=1)`

---

## Summary

This DSL provides:
- ✅ **65 non-redundant primitives**
- ✅ **8 organized categories**
- ✅ **70-80% coverage** of simple ARC tasks
- ✅ **Composable** for complex tasks
- ✅ **Type-safe** with clear signatures
- ✅ **Interpretable** - each operation has clear semantics
- ✅ **Extensible** - can add more as needed

**Next steps:**
1. Implement Phase 1 primitives (20 core operations)
2. Test on 50 simple ARC tasks
3. Analyze coverage gaps
4. Implement Phase 2 primitives (25 extended)
5. Achieve 60%+ solve rate on easy tasks

The DSL strikes a balance between coverage and complexity - rich enough to express most ARC solutions, but not so large that it becomes unwieldy.
