# Plan to Fix Second Attempt Diversity Problem

## Problem Statement

**Current State:**
- 28% of tasks produce identical top-2 predictions
- 0% of tasks are solved by attempt 2 (all solved by attempt 1)
- Second attempt provides **zero value** in competition setting

**Impact:**
- Wasting the 2-attempt advantage in ARC-AGI competition
- Missing potential 10-20% performance boost

---

## Root Cause Analysis

### 1. Edge Cases (71% of identical predictions)

**Issue**: For trivial inputs, different transforms produce same output

**Example**: 1x1 grid `[[3]]`
- identity → `[[3]]`
- rotate_90 → `[[3]]`
- rotate_180 → `[[3]]`
- flip_h → `[[3]]`

**Why it happens**: These ARE mathematically equivalent for this input

**Solution**: This is actually correct behavior. For 1x1 grids, there's no meaningful second attempt.

### 2. Winner-Takes-All Scoring (Main Issue)

**Issue**: When one hypothesis strongly dominates, all others score near-zero

**Example**: flip_vertical task
```
Top-1: flip_vertical (score=0.9999)
Top-2: rotate_90    (score=0.0000227)  ← 44,000x smaller!
```

**Result**: Top-2 selection is essentially: "best hypothesis + random noise"

**Solution**: Need diversity-aware selection that actively chooses different hypotheses

### 3. No Output-Space Diversity Enforcement

**Issue**: We rank by score without checking if predictions are different

**Current algorithm**:
```python
ranked = sort_by_score(hypotheses)
return [ranked[0], ranked[1]]  # May produce same output!
```

**Solution**: Ensure top-2 produce different outputs

---

## Proposed Solutions

### Strategy 1: Output-Space Diversity (Highest Priority)

**Approach**: Actively select hypotheses that produce different outputs

**Implementation**:
```python
def select_diverse_top_2(hypotheses, scores, test_input):
    # Get best hypothesis
    top_1 = argmax(scores)
    output_1 = top_1.apply(test_input)

    # Find best hypothesis that produces DIFFERENT output
    candidates = []
    for h in hypotheses:
        output_h = h.apply(test_input)
        if not equal(output_h, output_1):
            candidates.append((h, scores[h]))

    if candidates:
        top_2 = argmax(candidates, key=score)
    else:
        # Fallback: all produce same output (edge case)
        top_2 = second_best_by_score(hypotheses)

    return [top_1, top_2]
```

**Pros**:
- Directly addresses the problem
- Guarantees different outputs (when possible)
- Simple to implement

**Cons**:
- Requires applying all hypotheses to test input
- May select lower-quality hypothesis just for diversity

**Expected Impact**:
- Identical predictions: 28% → <5% (only true edge cases)
- Attempt 2 value: 0% → 10-15%

### Strategy 2: Hypothesis-Space Diversity

**Approach**: Select hypotheses from different "families"

**Implementation**:
```python
def cluster_hypotheses(hypotheses):
    families = {
        'geometric': ['rotate', 'flip', 'transpose'],
        'color': ['replace', 'swap', 'invert'],
        'scaling': ['zoom', 'tile', 'crop'],
        'object': ['largest', 'smallest', 'filter'],
    }

    # Cluster by name patterns
    clusters = group_by_family(hypotheses, families)
    return clusters

def select_diverse_top_2(hypotheses, scores):
    top_1 = argmax(scores)
    family_1 = get_family(top_1)

    # Find best from different family
    other_families = [h for h in hypotheses if get_family(h) != family_1]
    if other_families:
        top_2 = argmax(other_families, key=scores.get)
    else:
        top_2 = second_best(hypotheses, key=scores.get)

    return [top_1, top_2]
```

**Pros**:
- Encourages exploration of different approaches
- Doesn't require test execution

**Cons**:
- Families may still produce same output
- Requires manual family definitions

**Expected Impact**:
- Modest improvement (5-10%)
- May miss best alternative if it's in same family

### Strategy 3: Uncertainty-Aware Selection

**Approach**: Use different strategies based on confidence

**Implementation**:
```python
def select_top_2(hypotheses, scores, belief):
    top_1 = argmax(scores)
    top_score = scores[top_1]

    if top_score > 0.9:
        # High confidence: stick with top-2 by score
        # (second attempt unlikely to help anyway)
        strategy = "standard"

    elif top_score < 0.3:
        # Low confidence: maximize diversity
        strategy = "max_diversity"

    else:
        # Medium confidence: balance score and diversity
        strategy = "balanced"

    return select_by_strategy(strategy, hypotheses, scores)
```

**Pros**:
- Adapts to confidence level
- Makes sense from Bayesian perspective

**Cons**:
- More complex
- Thresholds need tuning

**Expected Impact**:
- Attempt 2 value: 0% → 5-10%

### Strategy 4: Diversity Penalty in Scoring

**Approach**: Modify final score to penalize similarity

**Implementation**:
```python
def compute_final_scores_with_diversity(hypotheses, belief, test_input):
    base_scores = {h: posterior[h] * stability[h] for h in hypotheses}

    # Apply diversity bonus
    diverse_scores = {}
    top_1 = argmax(base_scores)
    output_1 = top_1.apply(test_input)

    for h in hypotheses:
        score = base_scores[h]
        output_h = h.apply(test_input)

        # Penalize if same as top-1
        if equal(output_h, output_1) and h != top_1:
            score *= 0.1  # Heavy penalty

        diverse_scores[h] = score

    return diverse_scores
```

**Pros**:
- Integrates into existing scoring
- Natural extension of current approach

**Cons**:
- Requires test input execution during ranking
- May interfere with correct ranking

**Expected Impact**:
- Identical predictions: 28% → 5%
- Attempt 2 value: 0% → 8-12%

---

## Recommended Implementation Plan

### Phase 1: Quick Fix (Output-Space Diversity)

**Priority: HIGHEST**

**What**: Implement Strategy 1 (output-space diversity)

**Why**:
- Directly solves the problem
- Simple to implement
- Guaranteed to reduce identical predictions

**Implementation**:
1. Modify `ARCActiveInferenceSolver.solve()` final selection
2. Apply top-k hypotheses to test input
3. Select best + best-different
4. Fallback to standard top-2 for edge cases

**Code location**: `arc_active_inference_solver.py`, lines 930-960

**Expected time**: 30 minutes

**Expected impact**:
- Identical predictions: 28% → 5%
- Attempt 2 success: 0% → 8-12%

### Phase 2: Enhanced Selection (Hybrid Approach)

**Priority: MEDIUM**

**What**: Combine Strategies 1 + 3 (output diversity + uncertainty-aware)

**Implementation**:
1. Check top score confidence
2. If high (>0.9): Use output diversity (Strategy 1)
3. If medium (0.3-0.9): Use output diversity with larger search
4. If low (<0.3): Maximize diversity + exploration

**Expected time**: 1 hour

**Expected impact**:
- Attempt 2 success: 8-12% → 12-15%

### Phase 3: Family-Based Diversity (Optional)

**Priority: LOW**

**What**: Add Strategy 2 as additional signal

**Implementation**:
1. Define hypothesis families
2. Use as tiebreaker when multiple candidates
3. Prefer different families when scores are close

**Expected time**: 1 hour

**Expected impact**:
- Marginal improvement (1-2%)

---

## Implementation Details

### Code Changes Required

**File**: `arc_active_inference_solver.py`

**Function**: `solve()` method, around line 930

**Current code**:
```python
# Step 5: Rank hypotheses by posterior probability × stability
final_scores = {}
for h in hypotheses:
    posterior = belief.probabilities.get(h, 0.0)
    stability = belief.stability_scores.get(h, None)
    if stability is None:
        stability = 0.0
    final_scores[h] = posterior * stability

# Get top-2 hypotheses
ranked = sorted(hypotheses, key=lambda h: final_scores.get(h, 0.0), reverse=True)
top_2_hypotheses = ranked[:2]
```

**New code**:
```python
# Step 5: Rank hypotheses and select diverse top-2
final_scores = {}
for h in hypotheses:
    posterior = belief.probabilities.get(h, 0.0)
    stability = belief.stability_scores.get(h, None)
    if stability is None:
        stability = 0.0
    final_scores[h] = posterior * stability

# Select top-2 with output diversity
top_2_hypotheses = self._select_diverse_top_2(
    hypotheses, final_scores, task.test_input
)
```

**New method to add**:
```python
def _select_diverse_top_2(self, hypotheses, scores, test_input):
    """
    Select top-2 hypotheses ensuring different outputs

    Strategy:
    1. Select best hypothesis by score
    2. Find best hypothesis that produces different output
    3. Fallback to second-best if all produce same output
    """
    if len(hypotheses) == 0:
        return []
    if len(hypotheses) == 1:
        return [hypotheses[0], hypotheses[0]]

    # Get top-1
    ranked = sorted(hypotheses, key=lambda h: scores.get(h, 0.0), reverse=True)
    top_1 = ranked[0]
    output_1 = top_1.apply(test_input)

    # Find best with different output
    best_different = None
    best_different_score = -1

    for h in ranked[1:]:  # Skip top_1
        output_h = h.apply(test_input)
        if not np.array_equal(output_h.data, output_1.data):
            score_h = scores.get(h, 0.0)
            if score_h > best_different_score:
                best_different = h
                best_different_score = score_h

    # Select top-2
    if best_different is not None:
        top_2 = best_different
    else:
        # All hypotheses produce same output (edge case)
        # Fall back to second-best by score
        top_2 = ranked[1] if len(ranked) > 1 else ranked[0]

    return [top_1, top_2]
```

---

## Testing Plan

### Phase 1: Verify Fix Works

**Test 1**: Run on known "same prediction" tasks
- `edge_single_pixel` (currently identical)
- `pattern_fill_bg` (currently identical)
- Verify outputs are now different (when possible)

**Test 2**: Run on full 50-task suite
- Measure: % identical predictions
- Measure: % solved by attempt 2
- Compare to baseline

**Success Criteria**:
- Identical predictions: <10% (down from 28%)
- Attempt 2 success: >5% (up from 0%)

### Phase 2: Performance Impact

**Test 3**: Measure overall success rate
- Ensure we didn't hurt performance on solved tasks
- Check if new selections help on previously failed tasks

**Test 4**: Analyze edge cases
- Verify 1x1 grids still work (may be identical, that's OK)
- Check uniform grids
- Ensure no crashes

**Success Criteria**:
- Overall success: ≥50% (no regression)
- Attempt 2 value: >0% (improvement)

### Phase 3: Comprehensive Re-test

**Test 5**: Run extended test suite (100+ tasks if possible)
- Broader coverage
- Statistical significance
- Category breakdown

**Success Criteria**:
- Consistent improvement across categories
- Attempt 2 contributes meaningfully (5-15% of solves)

---

## Expected Outcomes

### Baseline (Current)
- Success rate: 50% (25/50)
- Attempt 1: 50% (25/50)
- Attempt 2: 0% (0/50)
- Identical: 28% (14/50)

### After Phase 1 (Output Diversity)
- Success rate: 52-55% (26-27/50)
- Attempt 1: 45% (22-23/50)
- Attempt 2: 7-10% (3-5/50)
- Identical: 5-8% (2-4/50)

### After Phase 2 (Enhanced)
- Success rate: 55-58% (27-29/50)
- Attempt 1: 42% (21/50)
- Attempt 2: 12-15% (6-8/50)
- Identical: <5% (1-2/50)

### Interpretation

**Realistic expectation**:
- Second attempt will help on tasks where:
  - Top score is moderate (0.3-0.7)
  - Multiple plausible hypotheses exist
  - Diversity reveals alternative solution

**Won't help on**:
- Clear wins (top score >0.9) - first attempt gets it
- No good hypotheses (all score <0.1) - both attempts fail
- True edge cases (1x1 grids) - mathematically identical

**Net impact**:
- 5-8% absolute performance gain
- Meaningful use of 2-attempt advantage
- Better exploration of hypothesis space

---

## Risk Assessment

### Low Risk
✅ Output diversity selection
✅ Fallback to standard top-2
✅ No changes to core algorithms

### Medium Risk
⚠️ May select lower-quality hypothesis for diversity
⚠️ Test execution cost (need to apply all hypotheses)

### Mitigation
- Keep standard top-2 as fallback
- Only execute hypotheses with score >threshold
- Monitor performance on previously-solved tasks

---

## Success Metrics

### Primary
- **Identical predictions**: <10% (target: 5%)
- **Attempt 2 success**: >5% (target: 8-12%)

### Secondary
- Overall success rate: ≥50% (no regression)
- Diversity on non-edge cases: >90%

### Tertiary
- Execution time: <2x increase
- Code complexity: Minimal increase

---

## Timeline

- **Phase 1 Implementation**: 30 minutes
- **Phase 1 Testing**: 15 minutes
- **Phase 2 Implementation**: 1 hour
- **Phase 2 Testing**: 30 minutes
- **Analysis & Documentation**: 30 minutes

**Total**: ~3 hours for complete solution

---

## Conclusion

The "worthless second attempt" problem is **solvable** with a straightforward fix:

1. **Root cause**: Not enforcing output diversity
2. **Solution**: Actively select different outputs
3. **Expected impact**: 0% → 8-12% value from attempt 2
4. **Implementation**: Low risk, high reward

This will significantly improve competition performance by properly utilizing the 2-attempt advantage.

---

**Status**: Ready to implement
**Priority**: HIGH - directly impacts competition performance
**Complexity**: Low (Phase 1), Medium (Phase 2)
**Expected ROI**: High (5-8% absolute performance gain)
