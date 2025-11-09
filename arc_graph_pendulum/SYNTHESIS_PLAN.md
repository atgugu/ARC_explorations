# Plan: Advanced Program Synthesis for ARC Graph Pendulum

## Problem Statement

**Current State:**
- Only 3-6 simple programs synthesized per task
- 90% of performance gap attributed to limited program synthesis
- 10% solve rate on real ARC tasks

**Root Causes:**
1. Limited primitive operations (only rotate, flip, identity, tile)
2. No compositional reasoning (can't combine operations)
3. Rich features extracted but NOT used in synthesis
4. No object-level operations
5. No pattern-based transformations
6. No conditional logic

**Target State:**
- 50-100 diverse programs per task
- Feature-driven synthesis (use extracted information)
- Compositional programs (2-3 step sequences)
- Object-aware operations
- Pattern-based transformations
- Expected: 40-50% solve rate (4-5x improvement)

---

## Implementation Plan

### Phase 1: Rich DSL Primitives (30+ operations)

#### 1.1 Object Operations
```python
- extract_objects(grid, color=None) → list of objects
- filter_objects(objects, predicate) → filtered list
- move_object(object, dx, dy) → moved object
- copy_object(object, positions) → multiple copies
- delete_object(object) → None
- scale_object(object, factor) → scaled object
- rotate_object(object, degrees) → rotated object
- color_object(object, new_color) → recolored object
```

#### 1.2 Grid Operations
```python
- crop(grid, bbox) → cropped grid
- resize(grid, new_shape) → resized grid
- tile(grid, repeat_x, repeat_y) → tiled grid
- overlay(base, overlay, position) → combined grid
- fill_background(grid, color) → filled grid
- extract_region(grid, bbox) → region
- paste_region(grid, region, position) → modified grid
```

#### 1.3 Color Operations
```python
- remap_colors(grid, mapping) → remapped grid
- swap_colors(grid, c1, c2) → swapped grid
- replace_color(grid, old, new) → replaced grid
- fill_color(grid, color, mask) → filled grid
- dominant_color_remap(grid) → remapped grid
```

#### 1.4 Spatial Operations
```python
- translate(grid, dx, dy) → translated grid
- rotate(grid, degrees) → rotated grid
- flip_horizontal(grid) → flipped grid
- flip_vertical(grid) → flipped grid
- transpose(grid) → transposed grid
- scale(grid, factor_x, factor_y) → scaled grid
```

#### 1.5 Pattern Operations
```python
- detect_pattern(grid) → pattern
- apply_pattern(grid, pattern) → transformed grid
- complete_pattern(grid) → completed grid
- extend_pattern(grid, direction) → extended grid
- mirror_pattern(grid, axis) → mirrored grid
```

#### 1.6 Composition Operators
```python
- sequence(op1, op2, ...) → composed operation
- conditional(predicate, op_true, op_false) → conditional op
- map_objects(op, objects) → apply op to each object
- reduce_objects(op, objects) → combine objects with op
```

### Phase 2: Feature-Driven Synthesis

#### 2.1 Object-Driven Programs
```python
IF objects detected:
  - Generate object extraction programs
  - Generate object movement programs (all directions)
  - Generate object copying programs
  - Generate object filtering programs (by size, color, position)
  - Generate object transformation programs
```

#### 2.2 Symmetry-Driven Programs
```python
IF vertical_symmetry detected:
  - Generate vertical flip/mirror programs

IF horizontal_symmetry detected:
  - Generate horizontal flip/mirror programs

IF rotational_symmetry detected:
  - Generate rotation programs (90°, 180°, 270°)
```

#### 2.3 Color-Driven Programs
```python
IF color_changes detected:
  - Infer color mapping from train examples
  - Generate color remap programs
  - Generate color swap programs
  - Generate dominant color programs
```

#### 2.4 Pattern-Driven Programs
```python
IF periodicity detected:
  - Generate tiling programs
  - Generate pattern extension programs

IF pattern detected:
  - Generate pattern completion programs
  - Generate pattern replication programs
```

#### 2.5 Shape-Driven Programs
```python
IF shape_change detected:
  - Generate resize programs (to common ratios)
  - Generate crop programs (to output size)
  - Generate scale programs (1x, 2x, 3x, 0.5x)
```

### Phase 3: Compositional Program Generation

#### 3.1 Single-Step Programs (20-30 programs)
- Each primitive operation
- Parameterized variants

#### 3.2 Two-Step Compositions (20-30 programs)
```
Examples:
- extract_objects → move_objects → compose_grid
- detect_pattern → extend_pattern → apply
- flip → color_remap → output
- rotate → translate → output
```

#### 3.3 Three-Step Compositions (10-20 programs)
```
Examples:
- extract_objects → filter_by_size → move_to_positions → compose
- detect_symmetry → flip → overlay → compose
- crop → resize → tile → output
```

#### 3.4 Smart Composition
- Only compose compatible operations
- Use features to guide composition
- Prioritize likely combinations

### Phase 4: Implementation Strategy

#### 4.1 DSL Engine
```python
class Operation:
    def __call__(self, grid, **kwargs) -> grid

class CompositeOperation:
    def __init__(self, operations):
        self.ops = operations

    def __call__(self, grid):
        result = grid
        for op in self.ops:
            result = op(result)
        return result
```

#### 4.2 Program Generator
```python
class AdvancedProgramSynthesizer:
    def __init__(self):
        self.primitives = [all_operations]

    def synthesize(self, task_data, facts):
        programs = []

        # Generate feature-driven programs
        programs += self.object_driven_synthesis(facts)
        programs += self.symmetry_driven_synthesis(facts)
        programs += self.color_driven_synthesis(facts)
        programs += self.pattern_driven_synthesis(facts)

        # Generate compositional programs
        programs += self.compose_programs(programs, max_depth=2)

        return programs[:100]  # Top 100
```

#### 4.3 Integration
- Replace current program_synthesizer node
- Keep compatibility with existing pipeline
- Add new primitive nodes as needed

### Phase 5: Testing & Validation

#### 5.1 Unit Tests
- Test each primitive operation
- Test composition mechanics
- Test feature-driven synthesis logic

#### 5.2 Integration Tests
- Test on synthetic tasks
- Verify program diversity
- Check execution performance

#### 5.3 Comprehensive Evaluation
- Re-run on same 10 ARC tasks
- Measure improvement in:
  - Perfect solve rate
  - High quality rate (>0.8 IoU)
  - Average IoU
  - Program diversity

#### 5.4 Success Metrics
```
Current:
- Perfect solves: 10% (1/10)
- High quality: 40% (4/10)
- Avg IoU: 0.496
- Programs per task: 3-6

Target:
- Perfect solves: 30-40% (3-4/10)
- High quality: 60-70% (6-7/10)
- Avg IoU: 0.650-0.700
- Programs per task: 50-100
```

---

## Implementation Phases

### Week 1: Core DSL (Phase 1)
- Implement all primitive operations
- Unit test each operation
- Document API

### Week 2: Feature-Driven Synthesis (Phase 2)
- Implement synthesis logic for each feature type
- Integration with existing feature extractors
- Test on synthetic examples

### Week 3: Composition Engine (Phase 3)
- Implement composition mechanics
- Smart composition heuristics
- Performance optimization

### Week 4: Integration & Testing (Phases 4-5)
- Integrate into solver
- Comprehensive testing
- Performance analysis

---

## Risk Mitigation

### Risk 1: Combinatorial Explosion
- **Mitigation:** Limit compositions to depth 3
- **Mitigation:** Use smart heuristics to filter
- **Mitigation:** Parallel execution where possible

### Risk 2: Execution Time
- **Mitigation:** Cache operation results
- **Mitigation:** Early termination on bad scores
- **Mitigation:** Optimize hot paths

### Risk 3: Quality vs Quantity
- **Mitigation:** Score programs during synthesis
- **Mitigation:** Keep only top 100 programs
- **Mitigation:** Use beam search in program space

---

## Expected Outcomes

### Quantitative
- 3-4x increase in solve rate (10% → 30-40%)
- 1.5x increase in high quality results (40% → 60%)
- 40% increase in average IoU (0.50 → 0.70)
- 15x increase in program diversity (6 → 100)

### Qualitative
- Can solve compositional tasks
- Can handle object manipulation
- Can complete patterns
- Can infer and apply transformations
- More robust across task types

---

## Next Steps

1. ✅ Document plan (this file)
2. ⏭ Implement core DSL primitives
3. ⏭ Implement feature-driven synthesis
4. ⏭ Implement composition engine
5. ⏭ Integration testing
6. ⏭ Comprehensive evaluation
7. ⏭ Analysis and iteration

---

## Success Criteria

**Minimum Viable:**
- 20% solve rate (2x current)
- 50 diverse programs per task
- At least one compositional solve

**Target:**
- 30-40% solve rate (3-4x current)
- 80-100 programs per task
- Compositional solves on 50% of tasks

**Stretch:**
- 40-50% solve rate (4-5x current)
- Feature-driven synthesis working on all task types
- Competitive with mid-tier ARC systems

---

This plan addresses the critical 90% performance gap through systematic expansion of program synthesis capabilities, feature-driven generation, and compositional reasoning.
