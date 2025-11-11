# 3-Step Composition Evaluation: Analysis and Comparison

**Date**: 2025-11-11
**Evaluation**: 100 real ARC tasks, max_depth=3 vs max_depth=2
**Goal**: Determine if allowing 3-step compositions improves success rate

---

## Executive Summary

Increasing max_depth from 2 to 3 **did NOT improve performance**. While we found one task that benefits from 3-step composition (e41c6fd3), overall metrics **degraded**:

- **Success rate**: 2.0% (unchanged)
- **Average accuracy**: 55.9% (**-6.8pp** vs basic inference)
- **Median accuracy**: 73.1% (**-6.5pp** vs enhanced inference)
- **Average time**: 6.56s (**+162%** vs 2-step at ~2.5s)
- **Near-misses (70-95%)**: 54 (similar to 50-63 in 2-step)

**Key Finding**: Task e41c6fd3 succeeded with a 3-step composition: `color_remap → align_horizontal → translation`. However, this single success doesn't outweigh the overall performance degradation.

**Recommendation**: **Revert to max_depth=2**. Focus on improving beam search quality and candidate generation rather than increasing search depth.

---

## Detailed Comparison

### Performance Metrics

| Metric | Basic Inference (depth=2) | Enhanced Inference (depth=2) | 3-Step (depth=3) | Change vs Enhanced |
|--------|---------------------------|------------------------------|------------------|-------------------|
| Success Rate | 2.0% (2/100) | 2.0% (2/100) | 2.0% (2/100) | 0pp |
| Avg Accuracy | 62.7% | 62.4% | **55.9%** | **-6.5pp** ⚠️ |
| Median Accuracy | 77.0% | 79.6% | **73.1%** | **-6.5pp** ⚠️ |
| Near-Misses (70-95%) | 63 | 61 | 54 | -7 |
| High Range (80-95%) | 32 (32%) | 40 (40%) | 30 (30%) | -10pp ⚠️ |
| 95%+ Accuracy | 10 | 9 | 6 | -3 ⚠️ |
| Avg Time/Task | ~2.5s | ~2.5s | **6.56s** | **+162%** ⚠️ |
| Median Time/Task | ~1.5s | ~1.5s | **4.20s** | **+180%** ⚠️ |

### Successful Tasks by Configuration

**Basic Inference (max_depth=2)**:
- 68b67ca3: repeat_vertical (single-step)
- aa18de87: identity → color_remap (2-step)

**Enhanced Inference (max_depth=2)**:
- e41c6fd3: (program not captured in summary)
- 32e9702f: (program not captured in summary)

**3-Step Composition (max_depth=3)**:
- **e41c6fd3**: `color_remap → align_horizontal → translation` (**3-step!** ✓)
- aa18de87: `connect_objects` (single-step)

**Key Observation**: Task e41c6fd3 appears in both enhanced inference (depth=2) and 3-step (depth=3), but only the depth=3 run shows it used a 3-step composition. This suggests:
1. The 3-step composition was necessary for this task
2. Enhanced inference at depth=2 may have succeeded with a different approach (need to verify)

### Accuracy Distribution Comparison

| Range | Basic (depth=2) | Enhanced (depth=2) | 3-Step (depth=3) | Change |
|-------|----------------|--------------------|--------------------|--------|
| 0-20% | 23% | 24% | **29%** | +5pp ⚠️ |
| 20-40% | 2% | 1% | 2% | +1pp |
| 40-60% | 2% | 2% | 6% | +4pp ⚠️ |
| 60-80% | 31% | 24% | 27% | +3pp |
| 80-95% | 32% | **40%** | 30% | **-10pp** ⚠️ |
| 95-100% | 10% | 9% | 6% | **-3pp** ⚠️ |

**Analysis**: Distribution shifted **downward** with depth=3:
- More tasks in 0-20% range (complete failures)
- Fewer tasks in 80-95% range (high near-misses)
- Fewer tasks in 95-100% range (very close solutions)

This suggests the larger search space is **diluting** beam search quality.

### Top Near-Misses Comparison

#### Basic Inference (depth=2)
1. 0b17323b - 99.1% (identity → color_remap)
2. 27a77e38 - 98.8%
3. a096bf4d - 98.4% (reflection → color_remap)
4. 14754a24 - 95.8% (identity → extract_largest)

#### Enhanced Inference (depth=2)
1. 0b17323b - 99.1% (identity → color_remap)
2. 27a77e38 - 98.8%
3. a096bf4d - 98.4% (reflection → color_remap)
4. 18419cfa - 97.9%

#### 3-Step Composition (depth=3)
1. **fd096ab6 - 97.0%** (identity → color_remap → color_remap) **NEW TOP!**
2. 14754a24 - 95.8% (identity → erode → erode)
3. 42918530 - 95.5% (identity → color_remap → color_remap)
4. 575b1a71 - 95.0% (fill_enclosed → color_remap → connect_objects)

**Observation**: Top near-miss accuracy **dropped** from 99.1% to 97.0%. This indicates that even the best attempts are getting worse with depth=3.

---

## Why Performance Degraded

### 1. Search Space Explosion

With max_depth=3, the number of possible programs grows exponentially:
- **1-step**: ~50 primitives
- **2-step**: ~50 × 50 = 2,500 programs
- **3-step**: ~50 × 50 × 50 = 125,000 programs

Even with beam_width=15, we're only exploring a tiny fraction of the search space.

### 2. Beam Search Dilution

With 3 depth levels, beam search has to make more decisions:
- At depth=1: Keep top 15 1-step programs
- At depth=2: Keep top 15 2-step programs
- At depth=3: Keep top 15 3-step programs

Each decision compounds errors. A good 3-step program might be discarded early if its first 1-2 steps don't score well on training examples.

### 3. Evaluation Noise

Training example evaluation isn't perfect:
- Programs might score high on training but fail on test
- Complex multi-step programs harder to evaluate accurately
- More steps = more opportunities for compounding errors

### 4. Time Budget Exhaustion

Average time increased from 2.5s to 6.56s:
- Spending more time searching doesn't guarantee better results
- Longer search ≠ better search
- May be hitting diminishing returns

### 5. Rare 3-Step Solutions

Only **1 task** (e41c6fd3) clearly benefited from 3-step composition:
- 99 other tasks didn't need 3 steps
- Cost of searching 3-step space outweighs benefit
- Most tasks are either solvable in 1-2 steps or not at all

---

## Analysis of 3-Step Success: Task e41c6fd3

**Successful 3-step composition**: `color_remap → align_horizontal → translation`

**Grid size**: 17x30 (large)
**Training examples**: 3

**Why 3 steps were needed**:
1. **color_remap**: Change input colors
2. **align_horizontal**: Align objects horizontally
3. **translation**: Shift objects to correct position

This is a rare case where multiple independent transformations need to be chained. However, this single success doesn't justify the overall performance degradation.

**Question**: Did enhanced inference (depth=2) also solve this task? If so, with what program?
- Need to check previous evaluation logs

---

## 3-Step Composition Adoption

**Total**: **61 out of 100 tasks** (61%) selected 3-step compositions.

However, pattern analysis reveals a critical insight:

### Composition Type Breakdown

| Type | Count | Percentage | Success Rate |
|------|-------|------------|--------------|
| **Repetitive operations** (same primitive 3×) | 50 | **82%** | 0% (0/50) |
| **Diverse operations** (different primitives) | 11 | **18%** | 9.1% (1/11) ✓ |

### Repetitive Operations (82% of 3-step programs)

**Top patterns**:
1. `color_remap × 3`: 13 tasks, avg 81.4% accuracy, **0 successes**
2. `erode × 3`: 12 tasks, avg 80.7% accuracy, **0 successes**
3. `identity × 3`: 8 tasks, avg 90.6% accuracy, **0 successes**
4. `fill_enclosed × 3`: 3 tasks, avg 75.5% accuracy, **0 successes**

**Insight**: These should be handled by **parameterized primitives** instead of composition:
- `erode_n(iterations=3)` instead of `erode → erode → erode`
- `dilate_n(iterations=2)` instead of `dilate → dilate`
- This would reduce search space while maintaining functionality

### Diverse 3-Step Compositions (18% of 3-step programs)

**All 11 diverse compositions** (sorted by accuracy):
1. ✓ `color_remap → align_horizontal → translation` (100.0%) - **e41c6fd3**
2. ✗ `fill_enclosed → color_remap → connect_objects` (95.0%)
3. ✗ `translation → align_horizontal → dilate` (90.6%)
4. ✗ `composite → extract_largest → color_remap` (87.0%)
5. ✗ `connect_objects → fill_enclosed → erode` (84.0%)
6. ✗ `connect_objects → connect_objects → erode` (83.0%)
7. ✗ `color_remap → color_swap → color_remap` (79.5%)
8. ✗ `connect_objects → connect_objects → erode` (79.5%)
9. ✗ `duplicate_object → duplicate_object → connect_objects` (71.2%)
10. ✗ `align_vertical → dilate → gravity` (60.4%)
11. ✗ `downscale → fill_all_background → tile_pattern` (0.0%)

**Key Finding**: Only **1 out of 11** diverse compositions succeeded (9.1% success rate). This is still better than 0/50 for repetitive operations, but indicates that even with diverse 3-step compositions, the search space is too large to find correct programs consistently.

---

## Recommendations

### 1. Revert to max_depth=2 (RECOMMENDED)

**Rationale**:
- Depth=2 achieves same 2% success rate
- Better accuracy distribution (40% in 80-95% range vs 30%)
- 2.6× faster (2.5s vs 6.56s per task)
- Only 1 task clearly benefited from depth=3

**Action**: Update evaluate_real_arc.py to use max_depth=2

### 2. Add Parameterized Iteration Primitives

Instead of depth=3, add primitives that repeat operations:
- `erode_n(iterations=3)` instead of `erode → erode → erode`
- `dilate_n(iterations=2)` instead of `dilate → dilate`
- `color_remap_multi(map1, map2)` for sequential color changes

**Expected impact**: +1-2pp success rate, maintain speed

### 3. Improve Beam Search Heuristics

Current beam search treats all primitives equally. Implement:
- **Priority-based search**: Prefer high-scoring primitive types
- **Early pruning**: Discard programs that score <30% on training
- **Diversity preservation**: Keep at least N programs of each schema type

**Expected impact**: +2-3pp success rate at depth=2

### 4. Targeted 3-Step Search

Instead of enabling depth=3 for all tasks, use it selectively:
- If top depth=2 program scores <95%, try depth=3
- Only for large tasks (>15×15)
- Only if time budget allows

**Expected impact**: +1pp success rate, minimal time overhead

### 5. Focus on Near-Misses (HIGHEST PRIORITY)

54-63 tasks consistently achieve 70-95% accuracy. Analyze failure modes:
- **fd096ab6** (97.0%): What's missing?
- **14754a24** (95.8%): identity → erode → erode - close!
- **42918530** (95.5%): identity → color_remap → color_remap

**Action**: Manually inspect top 10 near-misses, identify patterns
**Expected impact**: +3-5pp success rate (convert near-misses to successes)

---

## Computational Cost Analysis

### Resource Usage Comparison

| Metric | depth=2 | depth=3 | Increase |
|--------|---------|---------|----------|
| Avg time/task | 2.5s | 6.56s | **+162%** |
| Median time/task | 1.5s | 4.20s | **+180%** |
| Total time (100 tasks) | 4.2 min | 10.9 min | **+160%** |
| Total time (400 tasks) | ~17 min | **~44 min** | **+160%** |

### Cost-Benefit Analysis

**depth=3 costs**:
- 2.6× slower
- More complex code
- Lower accuracy distribution

**depth=3 benefits**:
- Found 1 task needing 3-step composition
- Validated that most tasks don't need >2 steps

**Verdict**: Costs outweigh benefits. Revert to depth=2.

---

## Comparison with Theoretical Expectations

**Initial expectation** (from Phase 1 plan):
> "Compositional reasoning with 2-3 step chains: Expected impact 1% → 10-15% success rate"

**Reality**:
- Depth=2: 1% → 2% success rate (**+100% relative, +1pp absolute**)
- Depth=3: 2% → 2% success rate (**0% improvement**)

**Revised expectation**:
- Compositional reasoning alone: +1-2pp
- Need to combine with other improvements:
  - Better primitives: +2-3pp
  - Improved search: +2-3pp
  - Near-miss analysis: +3-5pp
  - **Combined**: 2% → 10-15% ✓

---

## Next Steps

### Immediate Actions

1. **Revert to max_depth=2** ✓
   - Update evaluate_real_arc.py
   - Document that depth=3 was tested and rejected

2. **Analyze Near-Misses** (PRIORITY)
   - Manually inspect fd096ab6 (97.0%)
   - Manually inspect 14754a24 (95.8%)
   - Manually inspect 42918530 (95.5%)
   - Identify common failure patterns

3. **Commit Results**
   - Add this analysis document
   - Update evaluation logs
   - Push to branch

### Future Exploration

1. **Parameterized Primitives**
   - Add iteration parameters to morphology operations
   - Test on same 100 tasks

2. **Improved Beam Search**
   - Implement priority-based search
   - Add diversity preservation
   - Benchmark against current

3. **Selective 3-Step Search**
   - Only for near-misses (>90% at depth=2)
   - Only for large tasks
   - Evaluate cost/benefit

4. **New Primitive Categories**
   - Pattern extrapolation (for tasks like 0b17323b)
   - Context-aware transformations
   - Object-level operations

---

## Conclusion

Increasing max_depth from 2 to 3 **degraded performance** despite finding one task that benefits from 3-step composition. The exponential search space explosion dilutes beam search quality, leading to:

- Lower accuracy distribution
- Fewer high near-misses
- 2.6× slower execution
- No success rate improvement

**Recommendation**: **Revert to max_depth=2** and focus on:
1. Analyzing near-miss failure modes (highest priority)
2. Adding parameterized iteration primitives
3. Improving beam search heuristics
4. Creating new primitive types for patterns we're missing

The path to 10-15% success rate lies in **smarter search** and **better primitives**, not deeper search.

---

**Generated**: 2025-11-11
**Evaluation run**: depth3_evaluation.log
**Configuration**: max_depth=3, beam_width=15, max_candidates=200
