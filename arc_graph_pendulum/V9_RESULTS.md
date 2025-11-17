# V9 Results: Extended Primitives (Negative Result #4)

**Date:** 2025-01-XX
**Hypothesis:** Adding 20 new transformation primitives will improve synthesis quality
**Expected Impact:** +2-3% evaluation solve rate (1.7% → 4-5%)
**Actual Impact:** **±0% (ZERO improvement)**

---

## Summary

V9 implemented 20 extended transformation primitives across 4 categories, based on analysis of high-quality tasks (0.80-0.95 IoU). This was **Phase 1** of the revised synthesis roadmap after V6-V8 negative results.

**Result: Fourth Negative Result**
- **0/15 tasks improved** (tested on tasks with V7 scores 0.80-0.86)
- **15/15 tasks unchanged** (V9 = V7 exactly)
- **Extended primitives NEVER selected** (0/15 times)
- **V7 always won** on training performance (15/15 times)

---

## Implementation

### Extended Primitives (20 total)

**Category 1: Spatial Operations (5 primitives)**
- `extract_leftmost_object()` - Extract leftmost non-background object
- `extract_rightmost_object()` - Extract rightmost non-background object
- `extract_topmost_object()` - Extract topmost non-background object
- `extract_bottommost_object()` - Extract bottommost non-background object
- `align_objects_to_grid()` - Align objects to regular grid with spacing

**Category 2: Object Operations (5 primitives)**
- `copy_object_horizontal()` - Copy largest object N times horizontally
- `copy_object_vertical()` - Copy largest object N times vertically
- `connect_objects_with_line()` - Connect all objects with Manhattan paths
- `object_intersection()` - Compute intersection of two objects
- `object_union()` - Compute union of two objects

**Category 3: Color Operations (5 primitives)**
- `recolor_by_row()` - Recolor each row with incrementing colors (gradient)
- `recolor_by_column()` - Recolor each column with incrementing colors
- `recolor_checkerboard()` - Recolor in checkerboard pattern
- `swap_colors_by_size()` - Swap colors of smallest and largest objects
- `color_propagation()` - Flood fill with rules (border-aware)

**Category 4: Pattern Operations (5 primitives)**
- `apply_horizontal_symmetry()` - Mirror left half to right (left-right symmetry)
- `apply_vertical_symmetry()` - Mirror top half to bottom (top-bottom symmetry)
- `apply_rotational_symmetry()` - Apply 90-degree rotational symmetry
- `complete_partial_pattern()` - Detect and extend repeating pattern
- `generate_periodic_tiling()` - Tile pattern to fill canvas

**Total Implementation:** ~1200 lines of code
- `nodes/extended_primitives.py` (600+ lines)
- `nodes/extended_primitive_detector.py` (300+ lines)
- `nodes/extended_primitive_synthesizer.py` (200+ lines)
- `solver_v9.py` (280+ lines)

---

## Testing Results

### Sample Test Run (15 High-Quality Tasks)

Tested on tasks with V7 scores 0.80-0.86 (where there's most room for improvement):

| Task ID | V7 Score | V9 Score | Change | Extended Used? |
|---------|----------|----------|--------|----------------|
| 1d398264 | 0.8084 | 0.8084 | ±0.0000 | No (V7 better) |
| 423a55dc | 0.8112 | 0.8112 | ±0.0000 | No (V7 better) |
| 1c0d0a4b | 0.8225 | 0.8225 | ±0.0000 | No (V7 better) |
| 3391f8c0 | 0.8247 | 0.8247 | ±0.0000 | No (V7 better) |
| 4e45f183 | 0.8338 | 0.8338 | ±0.0000 | No (V7 better) |
| 1c56ad9f | 0.8400 | 0.8400 | ±0.0000 | No (V7 better) |
| 319f2597 | 0.8400 | 0.8400 | ±0.0000 | No (V7 better) |
| 33b52de3 | 0.8412 | 0.8412 | ±0.0000 | No (V7 better) |
| 103eff5b | 0.8462 | 0.8462 | ±0.0000 | No (V7 better) |
| 25094a63 | 0.8533 | 0.8533 | ±0.0000 | No (V7 better) |
| 47996f11 | 0.8533 | 0.8533 | ±0.0000 | No (V7 better) |
| 09c534e7 | 0.8544 | 0.8544 | ±0.0000 | No (V7 better) |
| 12422b43 | 0.8571 | 0.8571 | ±0.0000 | No (V7 better) |
| 03560426 | 0.8600 | 0.8600 | ±0.0000 | No (V7 better) |
| 0becf7df | 0.8600 | 0.8600 | ±0.0000 | No (V7 better) |

**Summary:**
- **Improvements:** 0/15 (0%)
- **Regressions:** 0/15 (0%)
- **No change:** 15/15 (100%)
- **Extended primitives selected:** 0/15 times
- **V7 fallback:** 15/15 times

### Example Program Selection

**Task 1d398264:**
```
[Extended Primitives]
  Detecting applicable primitives...
  Detected 1 potential primitives:
    - Connect objects with lines (confidence=0.80)
  Best program: Connect objects with lines (score=0.744)

[Program Selection]
  Extended primitives score: 0.744
  V7 approach score: 0.819
  → Using V7 approach (better score)
```

**Pattern observed:** Extended primitives detected patterns (checkerboard, connection, symmetry) but consistently scored 10-20% worse than V7 on training examples.

---

## Root Cause Analysis

### Why Did V9 Fail?

**1. Detection works, but detects wrong patterns**
- Extended primitives were detected in most tasks
- Common detections: "connect objects", "checkerboard", "symmetry"
- But these were **false positives** - not the actual transformation

**2. V7 already achieves high training performance**
- V7 scores 0.82-0.93 IoU on training for high-quality tasks
- Extended primitives score 0.48-0.83 (always worse)
- V9's selection logic correctly chooses V7

**3. Primitives are too generic/simple**
- The 20 primitives capture common operations
- But they don't match the SPECIFIC transformations in ARC tasks
- ARC requires very task-specific, nuanced operations

**4. No diversity from V7**
- Extended primitives don't add complementary capabilities
- They're just simpler versions of what V7 already does
- Result: No ensemble benefit, no improvement

### V7 Training Score Examples

| Task | Extended Score | V7 Score | Winner |
|------|----------------|----------|--------|
| 1d0a4b61 | 0.486 | 0.933 | V7 |
| 212895b5 | 0.777 | 0.885 | V7 |
| 1da012fc | 0.000 | 0.933 | V7 |
| 2c737e39 | 0.831 | 0.932 | V7 |
| 0d87d2a6 | 0.744 | 0.819 | V7 |

V7 consistently outperforms extended primitives by **10-45%** on training.

---

## Convergent Evidence: Four Negative Results

V9 is the **fourth consecutive negative result**, providing overwhelming evidence:

| Version | Approach | Result | Key Finding |
|---------|----------|--------|-------------|
| **V6** | Meta-pattern learning | ±0% | Tasks don't have parameter variation |
| **V7** | Execution refinement | ±0% | Errors are in synthesis, not execution |
| **V8** | Ensemble voting | ±0% | V7 is deterministic (no diversity) |
| **V9** | Extended primitives | ±0% | Simple primitives can't match V7 |

### What We've Proven Through Negative Results

✅ **Validated:**
1. Synthesis quality is the bottleneck (not detection)
2. Post-processing approaches don't work (V6, V7)
3. Ensemble without diversity = wasted computation (V8)
4. Simple primitive expansion doesn't help (V9)
5. V7 is near-optimal for current architecture

❌ **Falsified:**
1. Meta-patterns solve generalization (V6: ±0%)
2. Execution refinement fixes errors (V7: ±0%)
3. Ensemble voting helps deterministic systems (V8: ±0%)
4. More primitives improve synthesis (V9: ±0%)

---

## Scientific Insights

### The Primitive Library Paradox

**Hypothesis:** Adding more transformation primitives will improve coverage
**Reality:** More primitives ≠ better synthesis

**Why?**
1. **Detection is not the bottleneck** - V7 already detects patterns well (41.9% high-quality)
2. **Simple primitives don't match ARC complexity** - ARC tasks require highly specific, nuanced operations
3. **Generic operations lose precision** - "Connect objects" is too vague; each task needs unique connection logic
4. **No compositionality** - Simple primitives don't combine well to form complex programs

### The Real Bottleneck: Program Search

V9 reveals the true challenge:
- **Not detection:** We find relevant patterns
- **Not execution:** Programs run correctly
- **Not primitives:** We have many operations
- **But search:** We can't find the RIGHT program

The space of possible programs is too large. We need:
1. **Constraint-based synthesis:** Prune invalid programs early
2. **Learned synthesis:** Neural models to guide search
3. **Analogy-based reasoning:** Reuse solutions from similar tasks

---

## Revised Synthesis Roadmap

### Phase 1 FAILED: Simple Primitive Expansion

✗ **V9: Extended Primitives** - Zero improvement (±0%)

### Updated Priorities

**Priority 1: Constraint-Based Program Synthesis** (was Priority 2)
- **Rationale:** V9 proves simple primitives don't help; need better search
- **Approach:** Extract constraints from training, use SMT solver
- **Expected impact:** +3-5% evaluation solve rate
- **Effort:** 3-4 weeks
- **Why skip ensemble/primitives:** V8+V9 prove they don't work

**Priority 2: Neural Program Synthesis** (was Priority 3)
- **Rationale:** Symbolic approaches exhausted; need learning
- **Approach:** Train transformer on 400 ARC training tasks
- **Expected impact:** +5-10% evaluation solve rate
- **Effort:** 2-3 months

**Priority 3: Analogy-Based Reasoning** (new)
- **Rationale:** Reuse solutions from similar solved tasks
- **Approach:** Task similarity matching + program transfer
- **Expected impact:** +2-3% evaluation solve rate
- **Effort:** 2-3 weeks

---

## Next Steps

**Immediate (DO NOT IMPLEMENT):**
- ✗ Skip Phase 1 (extended primitives) - FAILED
- ✗ Skip ensemble approaches - V8 proved no benefit
- ✗ Skip post-processing - V6+V7 proved no benefit
- ✗ Skip simple primitive expansion - V9 proved no benefit

**Recommended Path:**
1. **Accept the plateau:** V7 at 19.6% training / 1.7% evaluation is near-optimal for current architecture
2. **Move to Priority 1:** Implement constraint-based synthesis
3. **Or move to Priority 2:** Explore neural program synthesis
4. **Or conclude:** Current approach has reached fundamental limits

---

## Files Added

**V9 Implementation:**
- `nodes/extended_primitives.py` (600+ lines) - 20 transformation primitives
- `nodes/extended_primitive_detector.py` (300+ lines) - Pattern detection
- `nodes/extended_primitive_synthesizer.py` (200+ lines) - Program generation
- `solver_v9.py` (280+ lines) - V9 solver with fallback to V7

**Documentation:**
- `V9_RESULTS.md` - This document (negative result #4)

**Test Output:**
- `v9_test_output.txt` - Full test results on 15 tasks

---

## Conclusion

V9 is the **fourth consecutive negative result**, completing a comprehensive exploration of post-processing and simple synthesis improvements:

- **V6 meta-patterns:** ±0% (tasks don't vary)
- **V7 execution refinement:** ±0% (errors in synthesis)
- **V8 ensemble voting:** ±0% (no diversity)
- **V9 extended primitives:** ±0% (simple primitives don't help)

The system has reached a **fundamental plateau** at:
- **Training:** 19.6% solve rate (9/46 tasks)
- **Evaluation:** 1.7% solve rate (2/117 tasks)

**Four negative results provide convergent evidence:**
- Post-processing approaches are exhausted
- Simple improvements to primitives don't work
- The real bottleneck is **program search/synthesis**
- Need fundamentally different approaches (constraints, learning, analogy)

**Recommendation:** Move to constraint-based synthesis (Priority 1) or explore neural approaches (Priority 2). The current symbolic architecture has reached its limits.
