"""
Analysis: How to Improve Program Synthesis Quality

Based on V4-V7 results, we've validated:
✓ Detection capabilities are strong (41.9% high-quality on evaluation)
✗ Post-processing approaches don't work (V6 meta-patterns, V7 refinement both ±0%)
✓ Synthesis is the bottleneck (wrong programs selected, not execution errors)

This document analyzes concrete approaches to improve synthesis.
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter

def analyze_synthesis_bottleneck():
    """Analyze where synthesis fails to identify improvement opportunities."""

    print("="*80)
    print("SYNTHESIS BOTTLENECK ANALYSIS")
    print("="*80)

    # Load V7 results
    with open('v7_evaluation_results.json', 'r') as f:
        v7_results = json.load(f)

    results = v7_results['results']

    # Categorize by performance
    perfect = [r for r in results if r['avg_score'] >= 0.99]
    near_miss = [r for r in results if 0.95 <= r['avg_score'] < 0.99]
    high_quality = [r for r in results if 0.80 <= r['avg_score'] < 0.95]
    medium = [r for r in results if 0.50 <= r['avg_score'] < 0.80]
    low = [r for r in results if 0.20 <= r['avg_score'] < 0.50]
    failures = [r for r in results if r['avg_score'] < 0.20]

    print(f"\nPerformance Distribution:")
    print(f"  Perfect (≥0.99):     {len(perfect):3d} tasks (1.7%)")
    print(f"  Near-miss (0.95-99): {len(near_miss):3d} tasks (10.3%) ← Just 1-5% wrong")
    print(f"  High (0.80-0.95):    {len(high_quality):3d} tasks (31.6%) ← Detection works")
    print(f"  Medium (0.50-0.79):  {len(medium):3d} tasks (22.2%)")
    print(f"  Low (0.20-0.49):     {len(low):3d} tasks (6.8%)")
    print(f"  Failure (<0.20):     {len(failures):3d} tasks (29.1%) ← Coverage gaps")

    print(f"\n{'='*80}")
    print("KEY INSIGHT: Three Distinct Synthesis Problems")
    print(f"{'='*80}")

    print(f"\n1. NEAR-MISS PROBLEM (12 tasks, 0.95-0.99 IoU):")
    print(f"   Issue: Program is 95-99% correct but fails at threshold")
    print(f"   Root cause: Slightly wrong program selected (not execution error)")
    print(f"   Examples: 27a77e38 (0.9877), 1e81d6f9 (0.9822)")
    print(f"   ")
    print(f"   What doesn't work:")
    print(f"   ✗ Post-processing refinement (V7 proved this)")
    print(f"   ✗ Better execution (errors are in program logic)")
    print(f"   ")
    print(f"   What could work:")
    print(f"   ✓ Better program selection (try more candidates)")
    print(f"   ✓ Ensemble methods (combine multiple programs)")
    print(f"   ✓ More granular primitives (finer-grained operations)")

    print(f"\n2. HIGH-QUALITY PROBLEM (37 tasks, 0.80-0.95 IoU):")
    print(f"   Issue: Detection correct, but program has systematic errors")
    print(f"   Root cause: Available programs don't match exact transformation")
    print(f"   Examples: Detection finds 'rotate + flip' but program has edge case bug")
    print(f"   ")
    print(f"   What doesn't work:")
    print(f"   ✗ Meta-pattern learning (V6 proved tasks don't vary)")
    print(f"   ✗ Conditional rules (no parameter variation to learn)")
    print(f"   ")
    print(f"   What could work:")
    print(f"   ✓ Richer primitive library (more transformation types)")
    print(f"   ✓ Better program composition (combine primitives differently)")
    print(f"   ✓ Constraint-based synthesis (enforce correctness from examples)")

    print(f"\n3. FAILURE PROBLEM (34 tasks, <0.20 IoU):")
    print(f"   Issue: Detection fails completely or no matching primitives")
    print(f"   Root cause: Coverage gaps - transformation type not in library")
    print(f"   Examples: Multi-object reasoning, spatial relationships")
    print(f"   ")
    print(f"   What could work:")
    print(f"   ✓ More diverse primitives (spatial, relational, counting)")
    print(f"   ✓ Neural program synthesis (learn new patterns)")
    print(f"   ✓ Analogy-based reasoning (find similar solved tasks)")

    return {
        'near_miss': near_miss,
        'high_quality': high_quality,
        'failures': failures
    }


def propose_concrete_improvements():
    """Propose concrete, prioritized improvements."""

    print(f"\n{'='*80}")
    print("PROPOSED IMPROVEMENTS (Prioritized by Evidence)")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("PRIORITY 1: Ensemble Program Selection (High Impact, Low Effort)")
    print(f"{'='*80}")
    print(f"""
Why: V7 proved that single-program selection is the bottleneck
Evidence: Near-miss tasks (0.95+ IoU) have slightly wrong programs
Expected gain: +10-15% solve rate (convert near-miss + some high-quality)

Approach:
1. Generate top-K programs (K=5-10) instead of selecting best
2. Evaluate all on training examples
3. Use voting/consensus for test prediction
4. Weight by training performance

Implementation:
- Modify _evaluate_programs() to return top-K instead of best
- Add ensemble predictor that combines outputs
- Use majority voting for same-shape tasks
- Use IoU-weighted averaging for predictions

Estimated effort: 2-3 days
Expected improvement: +1-2% evaluation solve rate
Target tasks: 12 near-miss → potentially 4-6 solves

Code sketch:
```python
def ensemble_predict(programs, input_grid, train_examples):
    predictions = []
    weights = []

    for prog in programs:
        pred = prog['function'](input_grid)
        train_score = evaluate_on_training(prog, train_examples)
        predictions.append(pred)
        weights.append(train_score)

    # Weighted voting for each pixel
    return weighted_vote(predictions, weights)
```
""")

    print(f"\n{'='*80}")
    print("PRIORITY 2: Richer Primitive Library (High Impact, Medium Effort)")
    print(f"{'='*80}")
    print(f"""
Why: High-quality tasks (0.80-0.95) need finer-grained operations
Evidence: 37 tasks detect correctly but lack exact transformation
Expected gain: +5-10% solve rate

Missing primitives identified:
1. Spatial operations:
   - "Extract leftmost/rightmost object" (not just largest/smallest)
   - "Align objects to grid" (spacing, positioning)
   - "Fill region between objects"

2. Object operations:
   - "Copy object N times in pattern"
   - "Connect objects with line/path"
   - "Object intersection/union/difference"

3. Color operations:
   - "Recolor by position" (gradient, checkerboard)
   - "Swap colors based on property"
   - "Color propagation/flood fill with rules"

4. Pattern operations:
   - "Detect and apply symmetry"
   - "Complete partial pattern"
   - "Generate periodic tiling"

Implementation:
- Add 15-20 new primitive functions
- Update analyzers to detect new patterns
- Add synthesis strategies for combinations

Estimated effort: 1-2 weeks
Expected improvement: +2-3% evaluation solve rate
Target tasks: 37 high-quality → potentially 8-12 solves
""")

    print(f"\n{'='*80}")
    print("PRIORITY 3: Constraint-Based Program Synthesis (High Impact, High Effort)")
    print(f"{'='*80}")
    print(f"""
Why: Generate programs that provably satisfy training examples
Evidence: Current synthesis generates good programs but doesn't guarantee correctness
Expected gain: +10-20% solve rate (more reliable on high-quality tasks)

Approach:
1. Extract constraints from training examples:
   - Input-output size relationships
   - Color preservation/transformation rules
   - Object count changes
   - Spatial relationship invariants

2. Use constraint solver to:
   - Filter invalid programs early
   - Guide program search
   - Verify candidate correctness

3. SMT-based verification:
   - Encode grid transformations as logical formulas
   - Use Z3 or similar to check satisfiability
   - Generate counterexamples for debugging

Implementation:
```python
def extract_constraints(train_examples):
    constraints = []

    # Size constraints
    for inp, out in train_examples:
        if inp.shape == out.shape:
            constraints.append(SizeEqual())
        elif out.shape[0] < inp.shape[0]:
            constraints.append(SizeReduction('height'))

    # Color constraints
    inp_colors = set(np.unique(inp))
    out_colors = set(np.unique(out))
    if out_colors.issubset(inp_colors):
        constraints.append(ColorPreserving())

    return constraints

def synthesize_with_constraints(patterns, constraints):
    candidates = generate_programs(patterns)

    # Filter by constraints
    valid = [p for p in candidates
             if all(c.check(p) for c in constraints)]

    return valid
```

Estimated effort: 3-4 weeks
Expected improvement: +3-5% evaluation solve rate
Target tasks: All high-quality (37 + 12 near-miss) → 15-20 solves
""")

    print(f"\n{'='*80}")
    print("PRIORITY 4: Neural Program Synthesis (Very High Impact, Very High Effort)")
    print(f"{'='*80}")
    print(f"""
Why: Learn to generate programs for novel patterns
Evidence: 34 failures (29.1%) have no matching primitives
Expected gain: +15-30% solve rate (address coverage gaps)

Approach:
1. Train neural model to predict programs from examples
2. Use large ARC training set (400 tasks)
3. Combine with symbolic synthesis (hybrid)

Architecture options:
A. Transformer-based program synthesis:
   - Encode input/output grids as sequences
   - Decode to domain-specific language (DSL)
   - Beam search for program candidates

B. Graph neural network:
   - Represent grids as graphs (pixels as nodes)
   - Learn transformation as graph edit operations
   - Decode to symbolic program

C. Neuro-symbolic hybrid:
   - Neural detection (learned)
   - Symbolic synthesis (interpretable)
   - Best of both worlds

Implementation challenges:
- Need large training set (400 ARC tasks)
- Data augmentation critical
- May lose interpretability
- Generalization to novel patterns uncertain

Estimated effort: 2-3 months
Expected improvement: +5-10% evaluation solve rate
Target tasks: 34 failures → potentially 10-15 solves
""")

    print(f"\n{'='*80}")
    print("RECOMMENDED PATH FORWARD")
    print(f"{'='*80}")
    print(f"""
Phase 1 (2-3 weeks):
1. ✓ Implement ensemble program selection (Priority 1)
   - Quick win, low effort
   - Target: +1-2% solve rate
   - Validate approach on near-miss tasks

2. ✓ Add 20 new primitives (Priority 2 - partial)
   - Focus on most common missing patterns
   - Target: +2-3% solve rate
   - Iterative: add, test, refine

Expected after Phase 1: 1.7% → 4-6% evaluation solve rate

Phase 2 (4-6 weeks):
3. ✓ Implement constraint-based synthesis (Priority 3)
   - More reliable program generation
   - Target: +3-5% solve rate
   - Build on Phase 1 ensemble

Expected after Phase 2: 6% → 9-11% evaluation solve rate

Phase 3 (2-3 months):
4. ✓ Explore neural program synthesis (Priority 4)
   - Research project, not guaranteed
   - Target: +5-10% solve rate if successful
   - May require significant resources

Potential after Phase 3: 11% → 16-21% evaluation solve rate

Alternative: Focus on Phases 1-2 for reliable ~10% evaluation solve rate
(This would be 6x improvement from current 1.7%)
""")


def main():
    """Main analysis."""
    categories = analyze_synthesis_bottleneck()
    propose_concrete_improvements()

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print(f"""
Based on V4-V7 evidence:

✓ VALIDATED: Synthesis quality is the bottleneck
✓ FALSIFIED: Post-processing approaches (V6, V7)
✓ CLEAR PATH: Three-tiered approach

Immediate (weeks): Ensemble + primitives → +3-5% solve rate
Medium-term (months): Constraint-based → +6-10% solve rate
Long-term (months): Neural synthesis → +10-20% solve rate

The system has strong foundations (detection works well).
Now we need better program generation, not better detection.

Next recommended action: Implement Priority 1 (Ensemble Selection)
""")


if __name__ == "__main__":
    main()
