# Program Synthesis Implementation Plan

## Executive Summary

**Goal**: Move from fixed primitive selection to compositional program synthesis to improve ARC-AGI success rate from 0.5% to 10-30%.

**Current**: Select from 50 fixed primitives → 0.5% success
**Target**: Synthesize compositional programs → 10-30% success

**Timeline**: Implement core system, test, iterate

---

## Phase 1: DSL Design

### Core Design Principles

1. **Compositional**: Programs built from smaller programs
2. **Typed**: Operations have type signatures (Grid→Grid, Object→Object, etc.)
3. **Parameterized**: Operations can be parameterized (color, count, direction, etc.)
4. **Executable**: All programs can be evaluated on grids

### DSL Components

#### 1. Data Types

```python
Grid          # 2D numpy array (existing)
Object        # Connected component with properties
ObjectSet     # Collection of objects
Color         # Integer 0-9
Position      # (row, col) tuple
Direction     # Enum: UP, DOWN, LEFT, RIGHT
Size          # (height, width) tuple
Pattern       # Repeating structure
Region        # Set of positions
```

#### 2. Primitive Operations (Level 0)

**Grid Transforms** (keep existing 50 primitives):
- Geometric: flip_h, flip_v, rotate_90, rotate_180, rotate_270, transpose
- Color: replace_color, swap_colors, map_colors
- Morphological: dilate, erode, open, close
- Scaling: zoom_2x, zoom_3x, tile

**Object Operations** (new):
```python
detect_objects(grid: Grid, bg_color: Color = 0) → ObjectSet
largest_object(objects: ObjectSet) → Object
smallest_object(objects: ObjectSet) → Object
filter_by_color(objects: ObjectSet, color: Color) → ObjectSet
filter_by_size(objects: ObjectSet, min_size: int, max_size: int) → ObjectSet
count_objects(objects: ObjectSet) → int
```

**Object Transforms** (new):
```python
move_object(obj: Object, direction: Direction, distance: int) → Object
move_to_position(obj: Object, pos: Position) → Object
recolor_object(obj: Object, color: Color) → Object
scale_object(obj: Object, factor: int) → Object
rotate_object(obj: Object, degrees: int) → Object
```

**Spatial Operations** (new):
```python
get_position(obj: Object) → Position
get_bounding_box(obj: Object) → (Position, Size)
distance_between(obj1: Object, obj2: Object) → int
is_adjacent(obj1: Object, obj2: Object) → bool
align_objects(obj1: Object, obj2: Object, axis: str) → Object
```

**Region Operations** (new):
```python
detect_border(grid: Grid, color: Color) → Region
get_interior(region: Region) → Region
fill_region(grid: Grid, region: Region, color: Color) → Grid
detect_connected(grid: Grid, color: Color) → Region
```

**Pattern Operations** (new):
```python
detect_pattern(grid: Grid) → Pattern
extend_pattern(pattern: Pattern, direction: Direction) → Grid
tile_pattern(pattern: Pattern, rows: int, cols: int) → Grid
infer_sequence(examples: List[Grid]) → Pattern
```

**Size Operations** (new):
```python
infer_output_size(train_pairs: List[Tuple[Grid, Grid]], test_input: Grid) → Size
resize_to(grid: Grid, size: Size, method: str) → Grid
crop_to_content(grid: Grid, bg_color: Color = 0) → Grid
pad_to_size(grid: Grid, size: Size, color: Color = 0) → Grid
```

#### 3. Composition Operations (Level 1)

```python
sequence(op1: Program, op2: Program) → Program
  # Apply op1 then op2

conditional(predicate: Predicate, then_op: Program, else_op: Program) → Program
  # If predicate then then_op else else_op

for_each_object(objects: ObjectSet, op: ObjectProgram) → Grid
  # Apply op to each object

map_and_compose(grid: Grid, obj_op: ObjectProgram) → Grid
  # Detect objects, transform each, compose back

compose_objects(background: Grid, objects: ObjectSet) → Grid
  # Place objects on background
```

#### 4. Predicates (for conditionals)

```python
has_border(grid: Grid, color: Color) → bool
is_symmetric(grid: Grid, axis: str) → bool
has_object_count(grid: Grid, count: int) → bool
color_exists(grid: Grid, color: Color) → bool
size_matches(grid: Grid, size: Size) → bool
```

#### 5. Parameter Inference

```python
infer_color_mapping(train_pairs: List[Tuple[Grid, Grid]]) → Dict[Color, Color]
infer_scale_factor(train_pairs: List[Tuple[Grid, Grid]]) → int
infer_pattern_rule(train_pairs: List[Tuple[Grid, Grid]]) → Pattern
infer_movement_vector(train_pairs: List[Tuple[Grid, Grid]]) → (int, int)
```

---

## Phase 2: Program Representation

### Program Structure

```python
class Program:
    """A composable ARC program"""

    def __init__(self, op_name: str, params: Dict, children: List[Program] = None):
        self.op_name = op_name        # Operation name
        self.params = params           # Parameters
        self.children = children or [] # Sub-programs
        self.type_signature = None     # Grid→Grid, Object→Object, etc.

    def execute(self, input: Any) → Any:
        """Execute program on input"""
        pass

    def complexity(self) → int:
        """MDL-style complexity score"""
        return 1 + sum(child.complexity() for child in self.children)

    def __repr__(self) → str:
        """Human-readable representation"""
        pass
```

### Example Programs

**Simple (current approach)**:
```python
Program("flip_h", params={})
Program("zoom_2x", params={})
```

**Compositional**:
```python
Program("sequence", params={}, children=[
    Program("detect_objects", params={"bg_color": 0}),
    Program("for_each_object", params={}, children=[
        Program("recolor_object", params={"color": 2})
    ])
])
```

**Conditional**:
```python
Program("conditional", params={}, children=[
    Program("has_border", params={"color": 2}),
    Program("fill_interior", params={"color": 8}),
    Program("identity", params={})
])
```

---

## Phase 3: Synthesis Strategy

### Approach: Depth-Bounded Enumeration with Pruning

#### Algorithm

```python
def synthesize_programs(task: ARCTask, max_depth: int = 3, max_programs: int = 100) → List[Program]:
    """
    Synthesize programs up to max_depth

    Strategy:
    1. Level 0: Generate all primitive programs
    2. Level 1: Compose level-0 programs (sequence, conditional)
    3. Level 2: Compose level-1 programs
    4. At each level: Prune based on training examples
    """

    programs = []

    # Level 0: Primitives
    level_0 = generate_primitives()
    level_0_pruned = prune_by_training(level_0, task.train_pairs, keep_top=50)
    programs.extend(level_0_pruned)

    # Level 1: Simple compositions
    if max_depth >= 2:
        level_1 = []

        # Sequences of 2 primitives
        for p1 in level_0_pruned[:10]:  # Only top-10
            for p2 in level_0_pruned[:10]:
                level_1.append(sequence(p1, p2))

        # Object-centric programs
        for obj_op in object_operations:
            level_1.append(
                Program("map_and_compose", children=[obj_op])
            )

        level_1_pruned = prune_by_training(level_1, task.train_pairs, keep_top=30)
        programs.extend(level_1_pruned)

    # Level 2: Complex compositions
    if max_depth >= 3:
        level_2 = []

        # Sequences of level-1 + level-0
        for p1 in level_1_pruned[:5]:
            for p2 in level_0_pruned[:5]:
                level_2.append(sequence(p1, p2))

        # Conditionals
        for pred in predicates:
            for then_op in level_1_pruned[:5]:
                for else_op in level_0_pruned[:5]:
                    level_2.append(conditional(pred, then_op, else_op))

        level_2_pruned = prune_by_training(level_2, task.train_pairs, keep_top=20)
        programs.extend(level_2_pruned)

    return programs[:max_programs]
```

#### Pruning Strategy

```python
def prune_by_training(programs: List[Program],
                     train_pairs: List[Tuple[Grid, Grid]],
                     keep_top: int) → List[Program]:
    """
    Prune programs by evaluating on training examples

    Scoring:
    - Exact match: score = 1.0
    - Partial match: score = 1.0 - normalized_distance
    - Exception/fail: score = 0.0
    - Penalize complexity: final_score = score / sqrt(complexity)
    """

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
                # Partial match (pixel-wise accuracy)
                else:
                    accuracy = np.mean(prediction.data == output_grid.data)
                    scores.append(accuracy)

            except Exception:
                scores.append(0.0)

        # Average score across training examples
        avg_score = np.mean(scores) if scores else 0.0

        # Complexity penalty (MDL principle)
        complexity = program.complexity()
        final_score = avg_score / np.sqrt(complexity)

        scored_programs.append((program, final_score))

    # Sort by score and keep top-k
    scored_programs.sort(key=lambda x: x[1], reverse=True)
    return [p for p, score in scored_programs[:keep_top] if score > 0.0]
```

### Computational Budget

- Max depth: 3 (primitives → simple composition → complex composition)
- Max programs per level: 50-30-20 = 100 total
- Max programs per task: 100
- Estimated time per task: ~1-5 seconds (vs current 0.01s)

This is acceptable given current system runs at 0.01s per task.

---

## Phase 4: Integration with Active Inference

### Modified Hypothesis Generator

```python
class ProgramSynthesisHypothesisGenerator:
    """Generate programs via synthesis instead of fixed primitives"""

    def __init__(self, max_depth: int = 3, max_programs: int = 100):
        self.max_depth = max_depth
        self.max_programs = max_programs
        self.primitive_ops = self._build_primitives()
        self.object_ops = self._build_object_ops()
        self.composition_ops = self._build_composition_ops()

    def generate_hypotheses(self, task: ARCTask, features: Dict) → List[Hypothesis]:
        """
        Generate hypotheses via program synthesis

        Returns list of Hypothesis objects (same interface as before)
        Each hypothesis wraps a synthesized Program
        """

        # Synthesize programs
        programs = synthesize_programs(
            task,
            max_depth=self.max_depth,
            max_programs=self.max_programs
        )

        # Convert to Hypothesis objects
        hypotheses = []
        for program in programs:
            hypothesis = Hypothesis(
                program=lambda g, p=program: p.execute(g),
                name=str(program),
                complexity=program.complexity(),
                parameters={}
            )
            hypotheses.append(hypothesis)

        return hypotheses
```

### Keep Existing Components

- ✓ **PerceptionModule**: Still useful for feature extraction
- ✓ **ActiveInferenceEngine**: Still manages belief distributions
- ✓ **StabilityFilter**: Still assesses robustness
- ✓ **WorkspaceController**: Still manages attention
- ✓ **Diversity selection**: Still ensures different outputs

**Only change**: HypothesisGenerator → ProgramSynthesisHypothesisGenerator

---

## Phase 5: Implementation Priorities

### Must-Have (Core System)

1. **Program representation** (Program class)
2. **Primitive operations** (reuse existing + add object detection)
3. **Sequence composition** (op1 → op2)
4. **Object-centric operations** (detect, transform, compose)
5. **Synthesis engine** (depth-bounded enumeration)
6. **Pruning** (evaluation on training examples)
7. **Integration** (replace HypothesisGenerator)

### Should-Have (Enhanced System)

8. **Size inference** (infer output dimensions)
9. **Parameter inference** (infer colors, scales)
10. **Conditional operations** (if-then-else)
11. **Pattern operations** (tile, extend)
12. **Region operations** (border, interior, fill)

### Nice-to-Have (Advanced System)

13. **Loop constructs** (for-each)
14. **Spatial reasoning** (relative positioning)
15. **Pattern inference** (from examples)
16. **Type system** (prevent invalid compositions)

---

## Phase 6: Testing Strategy

### Test Suite

1. **Unit tests** for each new operation
2. **Integration tests** for program synthesis
3. **Regression tests** on existing 50 synthetic tasks
4. **Main evaluation** on 200 ARC-AGI tasks

### Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| **Success Rate** | 0.5% | 5-10% | 10-30% |
| **Size Mismatch** | 26% | <15% | <10% |
| **Diversity** | 100% | 100% | 100% |
| **Time per Task** | 0.01s | <5s | <2s |

### Comparison Points

1. **Before vs After**: 0.5% → ?%
2. **By failure mode**: Size mismatch, wrong transform
3. **By task complexity**: Train size, grid size, object count
4. **By task type**: Geometric, object-based, pattern, etc.

---

## Phase 7: Implementation Plan

### Week 1: Core Infrastructure

**Day 1-2**: DSL Foundation
- [ ] Program class with execute()
- [ ] Type system (basic)
- [ ] Primitive operations (wrap existing)
- [ ] Unit tests

**Day 3-4**: Object Operations
- [ ] Object detection (connected components)
- [ ] Object properties (position, size, color)
- [ ] Object transforms (move, recolor, scale)
- [ ] Unit tests

**Day 5-6**: Composition
- [ ] Sequence composition
- [ ] Object-centric pipeline (detect → transform → compose)
- [ ] Integration tests

**Day 7**: Integration
- [ ] ProgramSynthesisHypothesisGenerator
- [ ] Integration with Active Inference
- [ ] End-to-end test

### Week 2: Synthesis & Evaluation

**Day 1-2**: Synthesis Engine
- [ ] Depth-bounded enumeration
- [ ] Pruning strategy
- [ ] Complexity scoring
- [ ] Performance optimization

**Day 3-4**: Enhanced Operations
- [ ] Size inference
- [ ] Parameter inference
- [ ] Region operations
- [ ] Pattern operations

**Day 5**: Testing
- [ ] Run on 50 synthetic tasks (regression)
- [ ] Run on 200 evaluation tasks
- [ ] Performance profiling

**Day 6-7**: Analysis & Iteration
- [ ] Compare results
- [ ] Identify failure modes
- [ ] Implement fixes
- [ ] Re-test

---

## Phase 8: Expected Improvements

### Success Rate Predictions

**Optimistic (10-30%)**:
- Good object detection
- Effective composition
- Size inference works
- Pruning is effective

**Realistic (5-10%)**:
- Basic object operations work
- Sequence composition helps
- Some size inference
- Some tasks need deeper reasoning

**Pessimistic (2-5%)**:
- Object detection has issues
- Composition doesn't help much
- Still missing key capabilities
- Need more sophisticated synthesis

### Risk Mitigation

**Risk 1**: Synthesis too slow
- Mitigation: Aggressive pruning, limit depth to 2

**Risk 2**: Object detection fails
- Mitigation: Use multiple detection strategies

**Risk 3**: Composition doesn't help
- Mitigation: Start with object-centric pipeline only

**Risk 4**: Still missing key primitives
- Mitigation: Iterative - add operations based on failures

---

## Implementation Milestones

### Milestone 1: Basic Program Synthesis (2-3 days)
- ✓ Program class
- ✓ Sequence composition
- ✓ Basic synthesis
- **Target**: 1-2% success (prove system works)

### Milestone 2: Object Operations (2-3 days)
- ✓ Object detection
- ✓ Object transforms
- ✓ Object pipeline
- **Target**: 3-5% success

### Milestone 3: Size Inference (1-2 days)
- ✓ Size inference
- ✓ Resize operations
- **Target**: 5-8% success (reduce size mismatch)

### Milestone 4: Full System (1-2 days)
- ✓ All operations
- ✓ Full synthesis
- ✓ Optimization
- **Target**: 5-10% success

---

## Success Criteria

### Minimum Viable Product (MVP)

- [ ] Program synthesis working
- [ ] Object detection + transforms
- [ ] Integration with Active Inference
- [ ] 2-5% success rate on evaluation (4-10x improvement)

### Full Success

- [ ] Compositional programs
- [ ] Object-centric reasoning
- [ ] Size inference
- [ ] 5-10% success rate (10-20x improvement)

### Stretch Goals

- [ ] Conditional operations
- [ ] Pattern inference
- [ ] Parameter learning
- [ ] 10-30% success rate (20-60x improvement)

---

## Conclusion

This plan provides a pragmatic path from 0.5% to 5-10% success rate by:

1. **Adding composition**: Move beyond single primitives
2. **Adding object reasoning**: Detect and manipulate objects
3. **Adding size inference**: Reduce 26% size mismatch failures
4. **Keeping what works**: Active Inference, diversity, architecture

The key insight: Generate better hypotheses (programs), not just select better primitives.

**Next Step**: Begin implementation with Milestone 1 (Basic Program Synthesis).
