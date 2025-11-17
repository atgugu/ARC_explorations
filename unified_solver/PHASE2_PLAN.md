# Phase 2 Implementation Plan: Advanced DSL

## Goal

Push success rate from **1.0% → 3-5%** by adding:
1. Conditional operations (if-then-else)
2. Loop constructs (for-each)
3. Pattern operations
4. Enhanced predicates

---

## Phase 2 Features

### 1. Conditional Operations

**Syntax**:
```python
conditional(predicate, then_program, else_program)
```

**Example Programs**:
```python
# If grid has a border, fill interior
conditional(
    has_border(color=2),
    fill_interior(color=8),
    identity()
)

# If object count > 1, keep largest; else identity
conditional(
    object_count_gt(1),
    keep_largest(),
    identity()
)
```

**Predicates Needed**:
- `has_border(color)`: Grid has border of given color
- `is_symmetric(axis)`: Grid is symmetric
- `object_count_gt(n)`: Number of objects > n
- `object_count_eq(n)`: Number of objects == n
- `has_color(color)`: Grid contains color
- `size_matches(h, w)`: Grid size matches dimensions

### 2. Loop Constructs

**Syntax**:
```python
for_each_object(transform_program, bg_color=0)
```

**Example Programs**:
```python
# Recolor all objects to color 5
for_each_object(recolor(5))

# Flip each object horizontally
for_each_object(flip_h())

# Scale each object by 2x
for_each_object(zoom_2x())
```

**Semantics**:
1. Detect objects in grid
2. Apply transform to each object independently
3. Compose back to grid

### 3. Pattern Operations

**Operations**:
- `tile_nxm(n, m)`: Tile grid n×m times
- `detect_symmetry()`: Find symmetry axis
- `complete_symmetry(axis)`: Complete symmetric pattern
- `extend_horizontally(factor)`: Extend pattern horizontally
- `extend_vertically(factor)`: Extend pattern vertically

### 4. Advanced Object Operations

**Operations**:
- `count_objects()`: Return number of objects as metadata
- `sort_by_size()`: Sort objects by size
- `align_objects(axis)`: Align all objects
- `stack_objects(direction)`: Stack objects in direction

---

## Implementation Strategy

### Step 1: Add Predicates

```python
class Predicate:
    """A boolean predicate over grids"""
    def __init__(self, name: str, func: Callable[[Grid], bool], params: Dict = None):
        self.name = name
        self.func = func
        self.params = params or {}

    def evaluate(self, grid: Grid) -> bool:
        return self.func(grid, **self.params)
```

### Step 2: Add Conditional Programs

```python
def conditional_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """
    Conditional operation: if predicate then then_op else else_op

    children[0]: predicate program (returns bool metadata)
    children[1]: then program
    children[2]: else program
    """
    predicate = params['predicate']
    then_prog = children[0]
    else_prog = children[1]

    if predicate.evaluate(input_grid):
        return then_prog.execute(input_grid)
    else:
        return else_prog.execute(input_grid)
```

### Step 3: Add Loop Programs

```python
def for_each_object_op(input_grid: Grid, children: List[Program], params: Dict) -> Grid:
    """
    For-each loop: apply transform to each object

    children[0]: transform program
    """
    bg_color = params.get('bg_color', 0)
    transform = children[0]

    # Detect objects
    objects = detect_objects(input_grid, bg_color)

    # Start with background
    result = Grid(np.full_like(input_grid.data, bg_color))

    # Transform each object
    for obj in objects:
        obj_grid = keep_only_object(obj, input_grid, bg_color)
        transformed = transform.execute(obj_grid)

        # Compose back
        mask = transformed.data != bg_color
        result.data[mask] = transformed.data[mask]

    return result
```

### Step 4: Update Synthesis

Add conditional and loop generation at Level 2:

```python
# Level 2: Conditionals and loops
if self.max_depth >= 2:
    level_2 = []

    # Conditionals: if predicate then top_op else identity
    for pred in predicates:
        for then_op in level_1[:5]:
            level_2.append(conditional(pred, then_op, identity))

    # Loops: for each object do transform
    for transform in level_0[:10]:
        level_2.append(for_each_object(transform))

    level_2_pruned = prune(level_2)
    all_programs.extend(level_2_pruned)
```

---

## Expected Impact

### Success Rate Prediction

**Conservative**: 1% → 2%
- Conditionals solve ~5 tasks
- Loops solve ~5 tasks
- Total: +5 tasks

**Realistic**: 1% → 3%
- Conditionals solve ~10 tasks
- Loops solve ~10 tasks
- Overlap: -6 tasks
- Total: +14 tasks

**Optimistic**: 1% → 5%
- Conditionals solve ~15 tasks
- Loops solve ~15 tasks
- Compositions: +10 tasks
- Total: +25 tasks

### Diversity Impact

**Expected**: 36% → 50%
- More programs pass training filter
- Better variety of program types
- Conditionals create branching paths

### Failure Modes

**Size Mismatch**: 13% → 10%
- Pattern operations help with size inference

**Wrong Transform**: 87% → 85%
- Conditionals and loops cover more task types

---

## Testing Plan

1. **Unit Tests**: Test each new operation
2. **Integration Tests**: Test conditional/loop programs on synthetic tasks
3. **Regression Tests**: Ensure Phase 1 performance maintained
4. **Evaluation**: Run on 200 tasks, compare to Phase 1

---

## Success Criteria

### Minimum (MVP)
- ✓ Conditionals working (at least 1 predicate)
- ✓ Loops working (for-each objects)
- ✓ Success rate ≥ 1.5% (+50%)

### Target
- ✓ 5+ predicates implemented
- ✓ Conditional + loop compositions
- ✓ Success rate ≥ 2-3% (+100-200%)
- ✓ Diversity ≥ 40%

### Stretch
- ✓ Pattern operations
- ✓ Success rate ≥ 3-5% (+200-400%)
- ✓ Diversity ≥ 50%

---

## Risk Mitigation

**Risk 1**: Conditionals slow down synthesis
- Mitigation: Generate only top-5 combinations

**Risk 2**: Loops cause execution errors
- Mitigation: Robust exception handling

**Risk 3**: No improvement in success rate
- Mitigation: Detailed failure analysis to identify gaps

---

## Timeline

**Day 1**: Implement predicates and conditionals
**Day 2**: Implement loops
**Day 3**: Integrate into synthesis
**Day 4**: Test and evaluate
**Day 5**: Analyze and iterate

---

**Status**: Ready to implement
**Target**: 1% → 3-5% success rate
**Key Features**: Conditionals, loops, patterns
