# Phase 6.1: Object-Aware Action Learning + Confidence Prioritization

## Overview

**Goal**: Fix Phase 6's accuracy regression (-2.5%) by adding object-level detection and confidence-based prioritization.

**Status**: ✅ **SUCCESSFUL** - Fully recovered from Phase 6 regression, back to Phase 5 performance

**Key Innovations**:
1. Object-level rotation/reflection detection (not just grid-level)
2. Confidence-based prioritization (not binary yes/no)
3. Always try actions, but boost detected ones with priority multipliers

---

## Problem Statement

Phase 6 had a critical flaw:
- **Binary decision**: Either generate hypotheses or don't (no middle ground)
- **Grid-level only**: Missed object transformations within grids
- **Result**: -2.5% accuracy (54.8% → 52.2%)

**Example failure**: Task 0dfd9992 (Phase 5's +76.4% win)
- Has small colored objects within 21×21 grid
- Objects are rotated, not entire grid
- Grid-level detection: 40-60% similarity (below 80% threshold)
- Decision: "No rotation" → Skip hypotheses → Miss solution

---

## Phase 6.1 Implementation

### 1. Object-Level Detection

Added two new methods to `ActionInference`:

#### `_detect_object_rotation(inp_objs, out_objs)`

Detects rotations on individual objects, not just entire grids:

```python
def _detect_object_rotation(self, inp_objs: List[ArcObject], out_objs: List[ArcObject]) -> Set[int]:
    """Detect if objects were rotated."""
    detected_angles = set()

    for inp_obj in inp_objs:
        inp_grid = inp_obj.grid  # Individual object grid

        for out_obj in out_objs:
            if inp_obj.dominant_color != out_obj.dominant_color:
                continue  # Different objects

            out_grid = out_obj.grid

            # Try rotations
            for angle, k in [(90, -1), (180, 2), (270, 1)]:
                rotated = np.rot90(inp_grid, k=k)

                if rotated.shape == out_grid.shape:
                    similarity = (rotated == out_grid).mean()
                    if similarity > 0.6:  # More lenient for objects
                        detected_angles.add(angle)

    return detected_angles
```

**Key differences from grid-level**:
- Compares individual `obj.grid` instead of entire grid
- Lower threshold (60% vs 80%) for object-level noise
- Matches objects by color before comparing

#### `_detect_object_reflection(inp_objs, out_objs)`

Similar logic for reflections on individual objects.

### 2. Dual-Level Detection

Modified `analyze_training_pairs()` to detect at **both levels**:

```python
for inp, out in train_pairs:
    # Grid-level detection (whole-grid transformations)
    grid_rotations = self._detect_rotation(inp, out)
    detected['rotations'].update(grid_rotations)

    # Object-level detection (object transformations)
    inp_objs = self.detector.detect_objects(inp)
    out_objs = self.detector.detect_objects(out)

    obj_rotations = self._detect_object_rotation(inp_objs, out_objs)
    detected['rotations'].update(obj_rotations)
```

### 3. Confidence Scoring

Calculate confidence based on detection frequency:

```python
# Count detections per pair
rotation_counts = {90: 0, 180: 0, 270: 0}

for inp, out in train_pairs:
    rotations = detect_all_rotations(inp, out)  # Grid + object
    for angle in rotations:
        rotation_counts[angle] += 1

# Calculate confidence
if detected['rotations']:
    max_count = max(rotation_counts[angle] for angle in detected['rotations'])
    detected['confidence']['rotations'] = max_count / total_pairs
```

**Confidence range**: 0.0 (not detected) to 1.0 (detected in all pairs)

### 4. Confidence-Based Prioritization

Modified `solver_conditional.py` to **always generate** but with priority multipliers:

```python
def get_priority_multiplier(action_type):
    """Get priority multiplier based on detection confidence."""
    if detected_actions is None:
        return 1.0  # No detection, equal priority

    confidence = detected_actions.get('confidence', {}).get(action_type, 0.0)

    if detected_actions.get(action_type):
        # Detected: boost priority based on confidence
        return 1.5 + confidence  # Range: 1.5-2.5
    else:
        # Not detected: lower but non-zero priority
        return 0.3  # Still try, but with lower priority

rotation_priority = get_priority_multiplier('rotations')

# Apply to accuracy for sorting
prioritized_accuracy = accuracy * rotation_priority

composite_hyps.append((transform_obj, description, prioritized_accuracy))
```

**Key change**: Instead of `if should_try_rotation:`, now `if True:` (always try)

**Priority multipliers**:
- **Detected (high confidence)**: 1.5-2.5× boost
- **Detected (low confidence)**: 1.5× boost
- **Not detected**: 0.3× (lower priority but still tried)
- **No detection data**: 1.0× (equal priority)

Applied to all composite actions:
- ✅ Rotations
- ✅ Reflections
- ✅ Color swaps
- ✅ Extensions
- ✅ Replications

---

## Results

### Quantitative Performance (30 tasks)

| Metric | Phase 5 | Phase 6 | Phase 6.1 | Change (6.1 vs 5) |
|--------|---------|---------|-----------|-------------------|
| **Exact Solves** | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | +0 |
| **Average Accuracy** | 54.8% | 52.2% | **54.8%** | **+0.0%** ✅ |
| **Hypotheses/Task** | 35.3 | 32.6 | 35.3 | +0.0 |

### Key Findings

1. **Full recovery**: Phase 6.1 recovered the -2.5% loss from Phase 6
2. **No efficiency gain**: Hypothesis count same as Phase 5
3. **No regression**: No tasks performed worse than Phase 5
4. **Stable baseline**: Back to Phase 5 performance level

### Why No Further Improvement?

Investigation of task 0dfd9992 (the problematic task) revealed:

```
Input objects:  1 (size=395, shape=(21, 21))
Output objects: 1 (size=441, shape=(21, 21))
```

**Root cause**: `ObjectDetector` treats entire grids as single objects, not individual colored regions.

**Implications**:
- Object-level detection still operates at grid level
- No true object segmentation happening
- Detection threshold (60% vs 80%) doesn't help if comparing full grids

**Why Phase 6.1 = Phase 5**:
- Always trying all actions (not skipping) = same as Phase 5
- Prioritization affects sorting, but top 12 hypotheses are similar
- Validation threshold filters most anyway
- Net effect: identical behavior to Phase 5

---

## Technical Details

### Files Modified

1. **`arc_curiosity_solver/core/action_inference.py`** (enhanced):
   - Added `_detect_object_rotation()` method (34 lines)
   - Added `_detect_object_reflection()` method (34 lines)
   - Enhanced `analyze_training_pairs()` with confidence scoring
   - Dual-level detection (grid + object)

2. **`arc_curiosity_solver/solver_conditional.py`** (refactored):
   - Added `get_priority_multiplier()` function
   - Removed all binary `if should_try_X:` checks
   - Changed to `if True:` (always generate)
   - Applied priority multipliers to all composite actions
   - Multiplied accuracy by priority for sorting

3. **New test file**: `test_phase6_1_solver.py` (270 lines)

### Code Changes Summary

**Before (Phase 6)**:
```python
should_try_rotation = (detected_actions is None or
                      (detected_actions and detected_actions.get('rotations')))

if should_try_rotation:
    # Generate rotation hypotheses
    composite_hyps.append((transform, desc, accuracy))
```

**After (Phase 6.1)**:
```python
rotation_priority = get_priority_multiplier('rotations')  # 0.3-2.5

# Always generate rotation hypotheses
prioritized_accuracy = accuracy * rotation_priority
composite_hyps.append((transform, desc, prioritized_accuracy))
```

---

## Lessons Learned

### ✅ What Worked

1. **Confidence-based prioritization**: Eliminated regression by always trying actions
2. **Safety net**: Lower priority (0.3) still allows undetected actions to be tried
3. **No information loss**: All actions considered, just with different priorities
4. **Stable behavior**: Matched Phase 5 performance exactly

### ❌ What Didn't Work (Yet)

1. **Object segmentation**: `ObjectDetector` treats grids as single objects
2. **No efficiency gain**: Same hypothesis count as Phase 5
3. **No accuracy improvement**: 54.8% ceiling unchanged

### 🔑 Key Insights

1. **Binary decisions are risky**: Phase 6's binary skip caused -2.5% regression
2. **Prioritization safer than exclusion**: Better to de-prioritize than skip
3. **Object detection needs work**: Current detector insufficient for fine-grained analysis
4. **Threshold tuning insufficient**: 60% vs 80% doesn't matter if comparing wrong granularity

---

## Path Forward

### Option A: Improve Object Segmentation (Recommended)

Fix `ObjectDetector` to detect individual colored regions, not entire grids:

```python
def detect_objects(self, grid: np.ndarray) -> List[ArcObject]:
    """Detect individual colored regions as separate objects."""
    objects = []

    for color in np.unique(grid):
        if color == 0:  # Skip background
            continue

        # Label connected components of this color
        mask = (grid == color)
        labeled, num_features = ndimage.label(mask)

        for label_id in range(1, num_features + 1):
            obj_mask = (labeled == label_id)
            # Create ArcObject for this connected component
            objects.append(ArcObject(grid[obj_mask], ...))

    return objects
```

**Expected impact**: Enable true object-level rotation/reflection detection

### Option B: Grid Pattern Detection

Instead of object-level, detect transformation patterns at grid level with more sophistication:
- Compare subgrids of different sizes (3×3, 5×5, 7×7)
- Detect partial transformations (some regions rotate, others don't)
- Use structural similarity (SSIM) instead of exact match

### Option C: Move to Next Priority

Accept Phase 6.1 as baseline and implement Phase 6 Priority 2 (Predicate Learning):
- Learn conditions from training data
- Reduce hardcoded predicates
- Potentially more impactful than action learning

---

## Recommendation

**Implement Option C**: Move to Phase 6 Priority 2 (Predicate Learning)

**Rationale**:
1. Phase 6.1 successfully recovered from regression (main goal achieved)
2. Object segmentation is a deep problem requiring significant rework
3. Phase 6 showed that action learning has limited impact (only +2% theoretical if perfect)
4. Predicate learning potentially more impactful (conditions are more variable across tasks)

**Phase 6.1 verdict**: Mission accomplished - regression fixed, stable baseline restored.

---

## Statistics Summary

```
PHASE 6 SERIES PROGRESSION:
├─ Phase 5:   54.8% (baseline)
├─ Phase 6:   52.2% (-2.5% regression due to binary skipping)
└─ Phase 6.1: 54.8% (✅ full recovery via confidence prioritization)

KEY ACHIEVEMENT: Fixed Phase 6 regression
HYPOTHESIS COUNT: No efficiency gain (same as Phase 5)
OVERALL STATUS: Stable baseline restored ✅
```

---

## Conclusion

Phase 6.1 achieved its primary objective: **fully recover from Phase 6's -2.5% regression**.

By replacing binary yes/no decisions with confidence-based prioritization, we eliminated the risk of skipping valid hypotheses while maintaining the action detection infrastructure.

Although object-level detection didn't provide additional gains due to limitations in object segmentation, the confidence prioritization framework is sound and could be leveraged if object detection improves in the future.

**Status**: Ready to proceed to Phase 6 Priority 2 (Predicate Learning) or Phase 7 (Multi-stage pipelines).
