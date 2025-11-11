# Ensemble Analysis: Current Architecture IS an Ensemble

**Date**: 2025-11-10
**Key Discovery**: The current InferredCompositionalSolver is already an ensemble!

---

## The Ensemble Was Already Built

When analyzing why different tasks were solved in different evaluation runs:
- **Basic inference run**: Solved aa18de87, 68b67ca3
- **Enhanced inference run**: Solved e41c6fd3, 32e9702f

I initially thought we needed to build an ensemble. But upon inspection, **the current architecture already IS an ensemble!**

### How parameter_inference.py Works

```python
for train_ex in task['train']:
    input_grid = np.array(train_ex['input'])
    output_grid = np.array(train_ex['output'])

    # Strategy 1: Basic inference
    color_maps = ParameterInference.infer_color_mapping(input_grid, output_grid)
    all_color_maps.extend(color_maps)

    # Strategy 2: Enhanced inference
    enhanced_maps = EnhancedColorInference.infer_color_mappings_enhanced(input_grid, output_grid)
    all_color_maps.extend(enhanced_maps)

    # Strategy 3: Identity-based inference
    id_maps = EnhancedColorInference.infer_color_mappings_with_identity(input_grid, output_grid)
    all_color_maps.extend(id_maps)
```

**All three strategies** run for every training example, and all candidates are merged!

### Why Different Tasks Were Solved

The difference between "basic" and "enhanced" runs wasn't about which inference method was used - it was about:

1. **Different random task samples**: Each run used a different random selection of 100 tasks from the 400-task dataset
2. **Candidate ordering**: With more candidates from enhanced inference, the ordering and prioritization changed
3. **Beam search dynamics**: More candidates means different paths through the compositional search space

### The Current Architecture

```
InferredCompositionalSolver
    ↓
InferredProgramGenerator
    ↓
parameter_inference.infer_all_parameters()
    ↓
┌─────────────┬────────────────────┬───────────────────┐
│ Basic       │ Enhanced           │ Identity-Based    │
│ Inference   │ Color Inference    │ Inference         │
└─────────────┴────────────────────┴───────────────────┘
    ↓              ↓                     ↓
         ALL CANDIDATES MERGED
              ↓
    Compositional Solver
     (beam search, scoring)
              ↓
         Best Solution
```

**This IS an ensemble!** We're already combining multiple strategies.

---

## What We Learned

### ✅ Ensemble Architecture Validated

The current design naturally creates an ensemble by:
- Generating candidates from multiple inference strategies
- Merging all candidates without prejudice
- Letting the compositional solver score and select the best

### ✅ Strategy Diversity Helps

Having multiple inference strategies increases the candidate pool and allows the solver to find solutions that any single strategy might miss.

### ✅ Success Rate Plateau at 2%

Both "basic" and "enhanced" evaluations achieved 2% success rate, but solved different tasks. This suggests:
- The 2% represents the current capability ceiling
- Task variance explains which specific 2 tasks succeed
- To exceed 2%, we need architectural improvements beyond just more candidates

---

## Why We're Stuck at 2%

### Candidate Pool Is Not The Bottleneck

We're generating:
- Basic inference: ~5-10 color mapping candidates
- Enhanced inference: ~15-20 color mapping candidates
- Identity-based: ~3-5 color mapping candidates
- Plus translations, rotations, scaling, etc.

**Total**: 150+ candidates with inferred parameters

This is plenty! The issue isn't lack of candidates.

### Compositional Search Is Limited

Current constraints:
- max_depth=2: Only 1-2 step compositions
- composition_beam_width=10: Limited search breadth
- No lookahead: Greedy selection at each step

Many 95%+ near-misses likely need:
- 3+ step compositions
- Non-greedy search
- Better heuristics for pruning

### Missing Primitive Types

Near-misses like 0b17323b (99.1%) need:
- Pattern extrapolation (extend diagonal sequences)
- Context-aware operations (change based on position/neighbors)
- Geometric reasoning (understand spatial relationships)

These can't be fixed by better parameter inference - they need new primitive types.

---

## What Actually Would Help

### 1. 3-Step Compositions (HIGH IMPACT)

**Current**: max_depth=2
**Proposed**: max_depth=3

**Why**: Many 80-95% tasks might be:
```
Input → [primitive1] → [primitive2] → [primitive3] → Output
```

**Expected Impact**: +2-3pp → 4-5% success rate

**Implementation**:
```python
CompositionalARCSolver(
    max_depth=3,  # Allow 3-step sequences
    composition_beam_width=15,  # Increase beam for larger search
    max_candidates=200  # More candidates to explore
)
```

### 2. Better Search Heuristics (MEDIUM IMPACT)

**Current**: Uniform exploration of all candidates
**Proposed**: Priority-based search

**Why**: Some primitive sequences are more likely than others

**Examples**:
- `identity → X` is high-priority for high-identity tasks
- `extract_object → transform → place` is common pattern
- `upscale → X` likely pairs with size-changing operations

**Expected Impact**: +1-2pp

### 3. New Primitive Categories (HIGH IMPACT - LONG TERM)

**Missing**:
- **Pattern extrapolation**: `extend_sequence()`, `complete_pattern()`
- **Context-aware**: `change_color_where()`, `fill_if_condition()`
- **Geometric**: `detect_line()`, `extend_diagonal()`, `mirror_about()`

**Expected Impact**: +5-10pp → 7-12% success rate

---

## Recommendations

### Immediate (This Session)

**Document that current architecture is already an ensemble**, clarify the architecture diagram, and explain why success rate is stable at 2%.

### Next Priority: 3-Step Compositions

**Goal**: Allow max_depth=3
**Expected**: 2% → 4-5% success rate
**Effort**: Low (just change parameter and tune beam width)
**Risk**: Low (can fall back to 2-step if performance degrades)

### Medium Term: Better Search

**Goal**: Implement priority-based beam search
**Expected**: +1-2pp
**Effort**: Medium (requires heuristic design)
**Risk**: Medium (heuristics might not generalize)

### Long Term: New Primitives

**Goal**: Add pattern extrapolation, context-aware, geometric primitives
**Expected**: +5-10pp → 10-15% success rate
**Effort**: High (each primitive needs careful design and testing)
**Risk**: High (new primitives might not help or might hurt performance)

---

## Conclusion

**Key Insight**: We've been using an ensemble all along!

The current InferredCompositionalSolver combines:
- Basic color inference
- Enhanced color inference
- Identity-based inference
- Plus translations, rotations, scaling, morphology

**This IS the ensemble approach** we wanted to implement.

The 2% success rate plateau indicates we've maximized what parameter inference can achieve. To progress further, we need:

1. **Deeper search**: 3-step compositions (quick win)
2. **Smarter search**: Priority-based heuristics (medium effort)
3. **New capabilities**: Pattern extrapolation primitives (long term)

**Recommendation**: Proceed with 3-step compositions as the next immediate improvement.

---

## Architecture Clarification

### What We Have (Current)

```
Task → InferredProgramGenerator → [150+ candidates from 3 strategies]
         ↓
    CompositionalSolver (max_depth=2, beam_width=10)
         ↓
    [Evaluate 2-step sequences]
         ↓
    Best 1-2 step solution
```

### What We Need (Next)

```
Task → InferredProgramGenerator → [150+ candidates]
         ↓
    CompositionalSolver (max_depth=3, beam_width=15)
         ↓
    [Evaluate 3-step sequences]
         ↓
    Best 1-3 step solution
```

Simple change, potentially significant impact!

---

**End of Analysis**
