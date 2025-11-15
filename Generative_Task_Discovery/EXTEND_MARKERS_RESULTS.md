# extend_markers Primitive: Implementation and Initial Results

**Date**: 2025-11-13
**Status**: Implemented but needs refinement
**Current Success**: 0/3 near-miss tasks solved

---

## Summary

Implemented the `extend_markers()` primitive based on near-miss analysis, which suggested that 60% of 95-99% near-miss tasks needed pattern extrapolation. However, initial testing shows the pattern is more complex than simple adjacent marker extension.

## Implementation

### Files Created/Modified:
1. **extend_markers_primitive.py** (new, 350 lines)
   - Core primitive implementation
   - Parameter inference functions
   - Auto-detection of markers, base color, directions

2. **advanced_solver.py** (modified)
   - Added extend_markers execution logic
   - Added 10 candidate variations (different directions, distances)

3. **parameter_inference.py** (modified)
   - Added extension_params to InferredParameters dataclass
   - Integrated infer_extension_parameters()

4. **inferred_solver.py** (modified)
   - Generate extend_markers candidates with inferred parameters
   - Create variations with different distances

### Primitive Logic:
```python
def extend_markers(grid, marker_colors, base_color, directions, distance):
    """
    Extend special colored pixels (markers) to nearby base-colored pixels

    For each marker:
        For each direction:
            Extend up to 'distance' pixels
            Change base_color pixels to marker_color
            Stop if hit non-base-color
    """
```

## Test Results on Near-Miss Tasks

### Task fd096ab6 (was 97.0%)

**Baseline (identity)**: 97.73% (input and output are 97.73% similar)

**extend_markers configurations tested**:
- All directions, distance=1: **90.91%** ❌ (worse than baseline!)
- All directions, distance=2: 81.61% ❌
- Auto-detect: 91.74% ❌
- **Primary directions (up/down/left/right), distance=1**: **94.21%** (best)

**Current solver choice**: `identity → color_remap` at **97.00%** ✓ (better than extend_markers)

**Analysis**:
- Only 11 cells change out of 484 (2.3%)
- 9/11 cells have a marker directly adjacent ✓
- **2/11 cells have NO adjacent marker** ❌
  - (8,13): 1 → 6, no nearby markers
  - (10,5): 1 → 8, no nearby markers

**Why extend_markers fails**:
The primitive only extends to ADJACENT cells, but some cells that need to change are NOT adjacent to any marker. They might be:
- Part of a diagonal ray from a marker
- Connected through a pattern/line
- Based on a different rule entirely

### Task 42918530 (was 95.5%)

**extend_markers accuracy**: 89.07% on training example 1
**Current solver**: `identity → color_remap` at 95.5% (better)

### Task e681b708 (was 94.8%)

**extend_markers accuracy**: 71.50% on training example 1
**Current solver**: `identity → color_remap` at 94.8% (much better)

---

## Key Findings

### 1. Misdiagnosed the Pattern

**Initial hypothesis** (from NEAR_MISS_ANALYSIS.md):
> "Tasks fd096ab6, 42918530, e681b708 need marker extension - special colored pixels extending to nearby base pixels"

**Reality**:
- These tasks are 95-98% identity (input ≈ output)
- Only 2-5% of cells change
- Current `identity → color_remap` is BETTER than extend_markers
- The pattern is more subtle than simple adjacent extension

### 2. extend_markers Works, But Not for These Tasks

**Primitive correctness**: ✓
- Successfully extends markers to adjacent cells
- Parameter inference works
- Integration with solver works
- Candidates are generated

**Task fit**: ❌
- Designed for tasks where markers expand to fill regions
- These near-miss tasks have a different pattern
- extend_markers achieves 71-94% vs baseline 95-98%

### 3. Direction Inference Issues

**Inferred directions**:
- fd096ab6: ['left', 'down-right'] → 87% accuracy
- Better: ['up', 'down', 'left', 'right'] → 94% accuracy

**Problem**: Direction inference sometimes picks wrong directions. Primary 4 directions are safer default.

### 4. The Real Pattern (Hypothesis)

Looking at fd096ab6 in detail:
- Most changed cells (9/11) ARE adjacent to markers
- But 2/11 are NOT adjacent
- Suggests pattern might be:
  - **Ray/line extension** from markers (not just adjacent)
  - **Pattern continuation** (markers start lines that continue)
  - **Context-dependent** rules (some markers extend, some don't)

This is MORE complex than the extend_markers primitive handles.

---

## Why Current Solver Wins

The solver choosing `identity → color_remap` at 97% is actually SMART:

1. **High identity**: Input and output are 97.73% identical
2. **Few changes**: Only 11 cells out of 484 change (2.3%)
3. **Best strategy**: Start with identity, make minimal adjustments
4. **color_remap attempts**: Try to learn which specific pixels change

The problem is that color_remap can't learn POSITION-DEPENDENT transformations. It learns "color X always maps to color Y", but these tasks need "color X at THESE specific positions maps to Y".

---

## Next Steps

### Option A: Fix extend_markers for These Tasks ❌ (Not Recommended)

Modify extend_markers to handle:
- Non-adjacent extension (rays/lines)
- Pattern continuation
- Context-dependent rules

**Problem**: This makes the primitive too complex and task-specific.

### Option B: Create Different Primitives ⚠️

These tasks might actually need:
- **propagate_along_line()**: Extend markers along straight lines
- **pattern_continuation()**: Continue patterns started by markers
- **conditional_pixel_change()**: Change specific pixels based on neighbors

### Option C: Try extend_markers on OTHER Tasks ✓ (Recommended)

**Hypothesis**: extend_markers might work better on tasks where:
- Markers genuinely expand to fill regions
- Most of the grid changes (not just 2-3%)
- Extension is simple (adjacent cells, not rays)

**Action**: Run full evaluation on 100 tasks and see which tasks benefit.

### Option D: Improve Position-Aware Transformations 🎯 (Best)

The real gap: Current primitives can't do "change these specific 11 pixels".

**Ideas**:
1. **object_aware_color_remap()**: Apply different color maps to different detected objects
2. **region_based_transform()**: Divide grid into regions, transform each differently
3. **template_matching()**: Find patterns in training, apply to test
4. **learned_pixel_rules()**: Use ML to learn which pixels should change

---

## Conclusion

**extend_markers primitive**: ✓ Implemented correctly

**Near-miss diagnosis**: ❌ Incorrect for these specific tasks
- fd096ab6, 42918530, e681b708 are NOT pure "marker extension" tasks
- They need more sophisticated position-aware or pattern-continuation logic

**Recommended path forward**:
1. **Run full evaluation** with extend_markers to find tasks it DOES help
2. **Re-analyze** the 95%+ near-misses to find their TRUE pattern
3. **Design** new primitives for position-aware or pattern-based transformations

**Expected impact**: Likely +0.5-1pp instead of projected +3pp
- Some tasks will benefit from extend_markers
- But the top near-misses need different primitives

---

**Files**:
- extend_markers_primitive.py
- test_extend_markers.py
- debug_extend_markers.py
- test_extend_markers.log

**Next action**: Run full 100-task evaluation to see real-world impact.
