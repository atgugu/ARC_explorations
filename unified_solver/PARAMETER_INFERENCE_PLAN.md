# Parameter Inference Implementation Plan

## Goal

Push success rate from **1.0% → 3-5%** by learning parameters from training examples instead of enumerating fixed values.

**Key Insight from Phase 2**: Generic operations don't help. Need task-specific parameters learned from examples.

---

## Problem Statement

### Current Approach (Fails)
```python
# Generate all possible color replacements
for old_c in [1,2,3,4,5,6,7,8,9]:
    for new_c in [1,2,3,4,5,6,7,8,9]:
        if old_c != new_c:
            replace_color(old_c, new_c)
# Result: 72 programs, but only 1 relevant
```

### New Approach (Parameter Inference)
```python
# Analyze training examples
color_map = infer_color_mapping(train_pairs)  # {1:5, 2:6}

# Generate single relevant program
replace_colors(color_map)  # Only 1 program, but correct!
```

**Advantage**: Generate 1 relevant program instead of 72 irrelevant ones.

---

## Parameters to Infer

### 1. Color Mappings (Highest Priority)

**What to infer**:
- Which input colors map to which output colors
- Example: {1→5, 2→6, 0→0} (replace 1 with 5, 2 with 6, keep 0)

**Detection strategy**:
```python
def infer_color_mapping(train_pairs):
    """
    Analyze training examples to find color mappings

    Returns: Dict[int, int] - input_color → output_color
    """
    # For each training pair
    # Track which input colors become which output colors
    # Return most consistent mapping
```

**Example tasks this solves**:
- "Replace all 1s with 5s, all 2s with 6s"
- "Swap colors 1 and 2"
- "Map sparse color set to dense"

**Expected impact**: +2-3% (4-6 tasks)

### 2. Scale Factors (High Priority)

**What to infer**:
- Zoom factors (2x, 3x, 4x)
- Tile factors (NxM tiling)

**Detection strategy**:
```python
def infer_scale_factor(train_pairs):
    """
    Detect if output is scaled version of input

    Returns: (scale_h, scale_w) or None
    """
    # Check if output_size = input_size * constant
    # Return scale factor
```

**Example tasks this solves**:
- "Zoom input by 3x"
- "Tile input 2x3"

**Expected impact**: +1-2% (2-4 tasks)

### 3. Movement Vectors (Medium Priority)

**What to infer**:
- Shift directions and distances
- Translation patterns

**Detection strategy**:
```python
def infer_movement_vector(train_pairs):
    """
    Detect object movement patterns

    Returns: (delta_row, delta_col) or None
    """
    # Track object positions across train examples
    # Return consistent movement vector
```

**Example tasks this solves**:
- "Shift grid right by 2"
- "Move all objects down by 1"

**Expected impact**: +0-1% (0-2 tasks)

### 4. Pattern Frequencies (Low Priority)

**What to infer**:
- Repetition counts
- Grid extensions

**Detection strategy**:
```python
def infer_pattern_frequency(train_pairs):
    """
    Detect repetition patterns

    Returns: (repeat_h, repeat_w) or None
    """
```

**Expected impact**: +0-1% (0-2 tasks)

---

## Implementation Strategy

### Phase 3 Architecture

```python
class ParameterInference:
    """Infer parameters from training examples"""

    def infer_all_parameters(self, train_pairs):
        """Analyze training pairs and extract all parameters"""
        return {
            'color_map': self.infer_color_mapping(train_pairs),
            'scale_factor': self.infer_scale_factor(train_pairs),
            'movement': self.infer_movement_vector(train_pairs),
            'tile_factor': self.infer_tile_factor(train_pairs),
        }

    def infer_color_mapping(self, train_pairs):
        """Infer color mapping from training examples"""
        # Implementation

    def infer_scale_factor(self, train_pairs):
        """Infer zoom/scale factor"""
        # Implementation

    # ... more inference methods

class ProgramSynthesizer:
    """Extended with parameter inference"""

    def synthesize(self, task):
        # 1. Infer parameters from training
        params = ParameterInference().infer_all_parameters(task.train_pairs)

        # 2. Generate level 0 with inferred params
        level_0 = self._generate_primitives(params)

        # 3. Continue with composition
        # ...
```

### Integration Points

**Modify**: `ProgramSynthesizer.synthesize()`
1. Before generating primitives, infer parameters
2. Pass inferred parameters to generators
3. Generate parameterized programs

**Add**: `ParameterInference` class
- Standalone module for parameter extraction
- Multiple inference strategies
- Returns structured parameter dictionary

**Extend**: `_generate_primitives()`, `_generate_object_programs()`
- Accept inferred parameters
- Generate programs using learned values
- Skip irrelevant parameter combinations

---

## Expected Results

### Success Rate Prediction

**Conservative**: 1% → 2% (+1%)
- Color mapping helps on ~2 tasks
- Scale inference helps on ~2 tasks
- Overlap: -2 tasks
- Net: +2 tasks

**Realistic**: 1% → 3% (+2%)
- Color mapping: ~4 tasks
- Scale inference: ~2 tasks
- Overlap: -2 tasks
- Net: +4 tasks

**Optimistic**: 1% → 5% (+4%)
- Color mapping: ~6 tasks
- Scale inference: ~4 tasks
- Movement: ~2 tasks
- Overlap: -4 tasks
- Net: +8 tasks

### Program Count Impact

**Before**: 130 programs generated (98.5% irrelevant)

**After**: 40-60 programs generated (90% relevant)

**Improvement**:
- Fewer programs (faster)
- Higher quality programs (more accurate)
- Better diversity (more viable options)

---

## Implementation Steps

### Step 1: Implement Color Mapping Inference

```python
def infer_color_mapping(train_pairs):
    """
    Infer consistent color mapping from training examples

    Strategy:
    1. For each training pair, track color correspondences
    2. Build frequency map: input_color → {output_colors: counts}
    3. Return most frequent mapping for each input color
    """
    color_freq = defaultdict(lambda: defaultdict(int))

    for input_grid, output_grid in train_pairs:
        # Only works if same size
        if input_grid.shape != output_grid.shape:
            continue

        # Track color correspondences
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = input_grid.data[i, j]
                out_color = output_grid.data[i, j]
                color_freq[in_color][out_color] += 1

    # Extract most frequent mapping for each color
    color_map = {}
    for in_color, out_colors in color_freq.items():
        if out_colors:
            best_out_color = max(out_colors.items(), key=lambda x: x[1])[0]
            # Only include if mapping is consistent (>80% of pixels)
            total = sum(out_colors.values())
            if out_colors[best_out_color] / total > 0.8:
                color_map[in_color] = best_out_color

    return color_map if color_map else None
```

### Step 2: Generate Parameterized Color Programs

```python
def _generate_color_programs_with_inference(self, params):
    """Generate color programs using inferred color mapping"""
    programs = []

    # Use inferred color mapping if available
    if params.get('color_map'):
        color_map = params['color_map']

        # Single program with learned mapping
        programs.append(Program(
            f"replace_colors_learned",
            lambda g, cm=color_map: apply_color_mapping(g, cm),
            {"color_map": color_map}
        ))

    # Also generate some fixed mappings (for robustness)
    # But far fewer than before (only common ones)
    for old, new in [(1, 2), (2, 1), (0, 1)]:
        programs.append(Program(
            f"replace_{old}_to_{new}",
            lambda g, o=old, n=new: replace_color(g, o, n),
            {"old": old, "new": new}
        ))

    return programs
```

### Step 3: Implement Scale Factor Inference

```python
def infer_scale_factor(train_pairs):
    """Infer zoom/scale factor if consistent"""
    scale_factors = []

    for input_grid, output_grid in train_pairs:
        if input_grid.shape[0] == 0 or input_grid.shape[1] == 0:
            continue

        scale_h = output_grid.shape[0] / input_grid.shape[0]
        scale_w = output_grid.shape[1] / input_grid.shape[1]

        # Check for integer uniform scaling
        if scale_h == scale_w and scale_h == int(scale_h):
            scale_factors.append(int(scale_h))

    # Return if all consistent
    if scale_factors and all(s == scale_factors[0] for s in scale_factors):
        return scale_factors[0]

    return None
```

### Step 4: Integration

Modify `ProgramSynthesizer.synthesize()`:

```python
def synthesize(self, task, verbose=False):
    # NEW: Infer parameters first
    param_inferrer = ParameterInference()
    inferred_params = param_inferrer.infer_all_parameters(task.train_pairs)

    if verbose:
        print(f"\nInferred Parameters:")
        if inferred_params.get('color_map'):
            print(f"  Color map: {inferred_params['color_map']}")
        if inferred_params.get('scale_factor'):
            print(f"  Scale factor: {inferred_params['scale_factor']}")

    # Generate programs with inferred parameters
    level_0 = self._generate_primitives_with_params(inferred_params)
    # ... rest of synthesis
```

---

## Testing Strategy

### Test Suite

1. **Unit Tests**: Test each inference function
2. **Integration Tests**: Test parameterized program generation
3. **Regression Tests**: Ensure Phase 2 performance maintained (1%)
4. **Main Evaluation**: Test on 200 tasks

### Comparison

| Metric | Phase 2 | Phase 3 (Target) |
|--------|---------|------------------|
| Success Rate | 1.0% | 3-5% |
| Programs Generated | ~130 | ~40-60 |
| Diversity | 64% identical | >50% |
| Speed | 0.149s | <0.15s |

---

## Success Criteria

### Minimum (MVP)
- ✓ Color mapping inference working
- ✓ Generate programs with inferred colors
- ✓ Success rate ≥ 1.5% (+50%)

### Target
- ✓ Color + scale inference
- ✓ Parameterized program generation
- ✓ Success rate ≥ 2-3% (+100-200%)
- ✓ Fewer but better programs

### Stretch
- ✓ Movement + pattern inference
- ✓ Success rate ≥ 3-5% (+200-400%)
- ✓ Diversity ≥ 50%

---

## Risk Mitigation

**Risk 1**: Inference too specific, overfits
- Mitigation: Keep some generic programs as fallback

**Risk 2**: Inference fails on many tasks
- Mitigation: Graceful degradation (use fixed params if inference fails)

**Risk 3**: No improvement in success rate
- Mitigation: Detailed analysis of which tasks benefit

---

## Timeline

**Day 1**: Implement color mapping inference + integration
**Day 2**: Implement scale factor + testing
**Day 3**: Evaluate on 200 tasks + analysis

---

**Status**: Ready to implement
**Target**: 1% → 3-5% success rate
**Key Innovation**: Learn from examples instead of enumerate possibilities
