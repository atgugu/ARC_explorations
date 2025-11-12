# Phase 6: Action Learning from Training Data

## Overview

**Goal**: Learn which transformations occur in training data instead of trying all 18 hardcoded actions for every task.

**Status**: ⚠️ IMPLEMENTED BUT NEEDS REFINEMENT - Improved efficiency (-7.6% hypotheses) but reduced accuracy (-2.5%)

**Key Innovation**: `ActionInference` module that analyzes training pairs to detect which geometric transformations, color mappings, and object manipulations actually occur, enabling focused hypothesis generation.

---

## Implementation

### 1. Core Module: `action_inference.py`

Created comprehensive action detection system with two main components:

#### `ActionInference` Class

Analyzes training pairs to detect:
- **Rotations**: 90°, 180°, 270° using `np.rot90()` comparison
- **Reflections**: Horizontal and vertical using `np.fliplr()` and `np.flipud()`
- **Color mappings**: Which input colors map to which output colors
- **Size changes**: Object scaling detection
- **Position changes**: Movement, centering, edge alignment
- **Extensions**: Direction-specific growth (top, bottom, left, right)
- **Replications**: Object duplication detection
- **Removals**: Object deletion detection

**Detection Method** (example - rotation):
```python
def _detect_rotation(self, inp: np.ndarray, out: np.ndarray) -> Set[int]:
    """Detect if output is a rotation of input."""
    detected_angles = set()

    for angle, k in [(90, -1), (180, 2), (270, 1)]:
        rotated = np.rot90(inp, k=k)

        if rotated.shape == out.shape:
            similarity = (rotated == out).mean()
            if similarity > 0.8:  # 80% match threshold
                detected_angles.add(angle)

    return detected_angles
```

#### `ActionFocusedGenerator` Class

Uses detected actions to guide hypothesis generation:
- `should_generate_rotation()` - Returns which angles to try
- `should_generate_reflection()` - Returns which axes to try
- `should_generate_color_swap()` - Returns whether to try swaps
- `get_priority_actions()` - Orders actions by detection confidence

### 2. Solver Integration

Modified `solver_conditional.py` to use action inference:

```python
# PHASE 6: Detect which actions occur in training data
detected_actions = None
if self.use_action_learning:
    try:
        detected_actions = self.action_inference.analyze_training_pairs(train_pairs)
    except Exception as e:
        detected_actions = None  # Fallback: try all actions

# Only generate if detected or detection disabled
should_try_rotation = (detected_actions is None or
                      (detected_actions and detected_actions.get('rotations')))

if should_try_rotation:
    # Generate rotation hypotheses...
```

Applied to all composite action types:
- ✅ Rotations
- ✅ Reflections
- ✅ Color swaps
- ✅ Extensions
- ✅ Replications

---

## Results

### Quantitative Performance (30 tasks)

| Metric | Phase 5 | Phase 6 | Change |
|--------|---------|---------|--------|
| **Exact Solves** | 0/30 (0.0%) | 0/30 (0.0%) | +0 |
| **Average Accuracy** | 54.8% | 52.2% | **-2.5%** ⚠️ |
| **Hypotheses/Task** | 35.3 | 32.6 | **-2.7 (-7.6%)** ✅ |
| **Tasks w/ Fewer Hyps** | - | 15/30 (50%) | - |

### Detection Statistics

Action detection rates across 30 tasks:
- **Rotations**: 9/30 (30.0%)
- **Reflections**: 11/30 (36.7%)
- **Color swaps**: 5/30 (16.7%)
- **Extensions**: 15/30 (50.0%)
- **Replications**: 5/30 (16.7%)
- **Size changes**: 24/30 (80.0%)
- **Position changes**: 15/30 (50.0%)
- **Removals**: 15/30 (50.0%)

### Critical Finding

**Task 0dfd9992** (Phase 5's biggest win: +76.4%): **NO ACTIONS DETECTED** ❌

This task showed:
- Same grid dimensions (21×21)
- Background removal (color 0 → removed)
- Likely involves **object-level** rotations/reflections
- Grid-level detection failed to detect transformations

---

## Why Accuracy Decreased

### Root Cause: Grid-Level vs Object-Level Detection

**The Problem**: Current detection compares entire grids, but many ARC tasks involve transformations on **small objects within large grids**.

**Example**: Task 0dfd9992
- Grid size: 21×21 (441 cells)
- Objects: Small colored shapes (maybe 3×3 or 5×5)
- Transformation: Rotate individual objects
- Detection result: ❌ False negative

When we rotate the entire grid and compare:
- Background cells: No change (most of grid)
- Object cells: Rotated positions (small fraction)
- Similarity score: ~40-60% (below 80% threshold)
- Decision: "No rotation detected" → Skip rotation hypotheses → Miss solution

### Why 80% Threshold?

Chosen to avoid false positives, but:
- ✅ Good for grid-level transformations (whole grid rotates)
- ❌ Bad for object-level transformations (small objects within grid)
- ❌ Bad for partial transformations (some objects transform, others don't)

### Impact

By filtering out "undetected" actions:
- ✅ Reduced wasted hypotheses (-7.6%)
- ❌ Also filtered out valid hypotheses (task 0dfd9992 and others)
- ❌ Net effect: -2.5% accuracy

---

## Diagnostic Example: Task 0dfd9992

```
Training Example 1:
  Input:  21×21 grid with colored objects on background (color 0)
  Output: 21×21 grid, background removed, objects transformed

Detection Results:
  Rotations:   ❌ (0.8 threshold not met - objects too small)
  Reflections: ❌ (same reason)
  Extensions:  ❌ (no bbox changes at grid level)

Phase 5 Result: 76.4% accuracy (composite rotation hypothesis worked!)
Phase 6 Result: ~0% accuracy (rotation hypothesis not generated)
```

---

## Path Forward

### Option 1: Object-Aware Detection (Recommended)

Modify `ActionInference` to detect at **two levels**:

```python
def analyze_training_pairs(self, train_pairs):
    detected = {
        'grid_level': self._detect_grid_level_actions(train_pairs),
        'object_level': self._detect_object_level_actions(train_pairs)
    }
    return detected

def _detect_object_level_actions(self, train_pairs):
    """Detect transformations on individual objects."""
    for inp, out in train_pairs:
        inp_objs = self.detector.detect_objects(inp)
        out_objs = self.detector.detect_objects(out)

        # Match objects and compare their transformations
        for inp_obj in inp_objs:
            for out_obj in out_objs:
                if self._same_object(inp_obj, out_obj):
                    # Check if this object was rotated
                    if self._object_rotated(inp_obj, out_obj):
                        detected['rotations'].add(angle)
```

**Pros**: More accurate detection, handles both grid and object-level tasks
**Cons**: More complex, requires robust object matching

### Option 2: Lower Threshold + Confidence Scores

```python
if similarity > 0.5:  # More permissive
    confidence = similarity
    detected_angles.add((angle, confidence))
```

Then prioritize but don't exclude:
```python
if should_try_rotation:
    # Try detected angles first with higher priority
    for angle in detected_angles:
        generate_with_priority(angle, priority=2.0)
else:
    # Still try rotations, but with lower priority
    for angle in [90, 180, 270]:
        generate_with_priority(angle, priority=0.5)
```

**Pros**: Simpler, doesn't completely exclude actions
**Cons**: Might not reduce hypotheses as much

### Option 3: Hybrid Approach (Best of Both)

1. Use object-level detection for object-heavy tasks
2. Use grid-level detection for grid transformations
3. If uncertain, try both with different priorities

```python
if task_has_many_objects:
    use_object_level_detection()
else:
    use_grid_level_detection()

# Always fall back to trying all actions with lower priority
```

**Pros**: Flexible, adaptive to task type
**Cons**: Most complex, requires task classification

---

## Lessons Learned

### ✅ What Worked

1. **Efficiency gain**: 7.6% fewer hypotheses generated
2. **Detection infrastructure**: Solid foundation for future improvements
3. **Modular design**: Easy to add new detection methods
4. **Fallback safety**: When detection fails (None), try all actions

### ❌ What Didn't Work

1. **Grid-level only**: Missed object-level transformations
2. **Binary decision**: Either try action or don't (no middle ground)
3. **Fixed threshold**: 80% too strict for diverse task types
4. **No confidence scores**: Couldn't prioritize detected vs undetected actions

### 🔑 Key Insight

**The fundamental assumption was wrong**: I assumed transformations happen at the grid level, but ARC tasks often transform **objects within grids**. The detection needs to be **multi-level** and **object-aware**.

---

## Recommendation

**Implement Option 1 (Object-Aware Detection)** with fallback to Option 2 (confidence scores):

### Phase 6.1 Plan

1. **Add object-level rotation detection**:
   - Match input/output objects by color and size
   - Compare their orientations using bounding box analysis
   - Detect rotations on matched object pairs

2. **Add confidence-based prioritization**:
   - Don't exclude undetected actions
   - Boost priority for detected actions (×2.0)
   - Lower priority for undetected actions (×0.5)
   - Still validate all, just in different order

3. **Add task classification**:
   - Detect if task is object-heavy or grid-heavy
   - Use appropriate detection strategy
   - Fall back gracefully if uncertain

**Expected Impact**:
- Keep efficiency gain (-7.6% hypotheses or better)
- Recover accuracy loss (+2.5% or better)
- Net positive: Better performance with fewer hypotheses

---

## Files Modified

- ✅ `arc_curiosity_solver/core/action_inference.py` (NEW - 337 lines)
- ✅ `arc_curiosity_solver/solver_conditional.py` (modified)
- ✅ `test_phase6_solver.py` (NEW)
- ✅ `diagnose_action_detection.py` (NEW)

---

## Verdict

**Phase 6: Partial Success** 🟡

The action learning concept is **sound and necessary**, but the implementation needs refinement:
- ✅ Proof of concept works (efficiency gain)
- ⚠️ Detection strategy needs improvement (object-awareness)
- 📊 Expected final gain: +3-5% accuracy with object-level detection

**Status**: Ready for Phase 6.1 (Object-Aware Detection) or move to Phase 6 Priority 2 (Predicate Learning) and revisit action learning later.

---

## Statistics Summary

```
PHASE 6: ACTION LEARNING
├─ Accuracy:        54.8% → 52.2% (-2.5%)
├─ Hypotheses:      35.3 → 32.6 (-7.6%)
├─ Efficiency:      Improved ✅
└─ Overall:         Needs refinement ⚠️

KEY ISSUE: Grid-level detection misses object-level transformations
SOLUTION: Add object-aware detection (Phase 6.1)
```
