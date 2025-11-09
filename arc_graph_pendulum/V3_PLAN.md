# V3 Plan: Example-Driven Rule Inference (Evidence-Based Approach)

## Critical Lessons from V2

**What V2 Taught Us:**
- ❌ 64 diverse programs → 0% solve rate, 0.415 IoU
- ❌ More programs ≠ better performance
- ❌ Compositional programs accumulate errors
- ❌ Feature-driven synthesis without understanding = random search
- ✅ Simple programs + repairs worked (V1: identity + placement → 1.000)
- ✅ Repair loops are critical
- ✅ Quality >> Quantity

**Quote from V2_RESULTS.md:**
> "We need 1 CORRECT program, not 100 wrong ones"

---

## The Real Problem

### What ARC Tasks Require
```
1. Understanding the RULE from examples
   "Move each colored object to opposite quadrant"

2. Implementing that SPECIFIC rule
   Not: "try 100 transformations and see what sticks"
   But: "understand the pattern, apply it"
```

### What V2 Did Wrong
```
1. Generate 100 syntactic transformations
   rotate, flip, scale, tile, etc.

2. Test all of them
   Hope one happens to match

Result: NONE match because we don't understand the rule
```

---

## V3 Approach: Rule Inference → Targeted Synthesis

### Philosophy Shift

**V1/V2 (Bottom-Up):** Generate programs → Test → Hope one works
**V3 (Top-Down):** Understand rule → Synthesize correct program → Refine

### Evidence from V1 Perfect Solve (Task 25ff71a9)
```
Strategy:
1. Try simplest program (identity)
2. Measure error
3. Apply targeted repair (placement offset)
4. Achieve perfection ✓

Why it worked:
- Simple baseline
- Specific error analysis
- Targeted fix
- Not random exploration
```

---

## V3 Architecture

### Phase 1: Differential Analysis
**Analyze what changed between input and output**

```python
def analyze_transformation(input_grid, output_grid):
    """
    Extract the actual transformation that occurred.

    Returns:
        - transformation_type: 'geometric', 'color', 'object', 'pattern', etc.
        - parameters: Specific details
        - confidence: How certain we are
    """

    # Check 1: Is it a geometric transformation?
    if same_colors(input, output):
        if is_rotation(input, output):
            return {'type': 'rotate', 'angle': detect_angle(), 'conf': 0.9}
        if is_flip(input, output):
            return {'type': 'flip', 'axis': detect_axis(), 'conf': 0.9}
        if is_translation(input, output):
            return {'type': 'translate', 'offset': detect_offset(), 'conf': 0.9}

    # Check 2: Is it a color transformation?
    if same_structure(input, output):
        mapping = learn_color_mapping(input, output)
        return {'type': 'color_remap', 'mapping': mapping, 'conf': 0.8}

    # Check 3: Is it an object operation?
    if different_object_count(input, output):
        return analyze_object_transformation(input, output)

    # Check 4: Is it a pattern operation?
    if is_tiling(input, output):
        return {'type': 'tile', 'factor': detect_tile_factor(), 'conf': 0.8}

    # Check 5: Is it a composite transformation?
    return analyze_composite(input, output)
```

### Phase 2: Rule Abstraction
**Generalize from multiple examples**

```python
def infer_rule(train_examples):
    """
    Infer the general rule from all training examples.

    Returns:
        - rule_description: Human-readable description
        - rule_type: Category
        - consistency: How consistent across examples
        - parameters: Learned parameters
    """

    # Analyze each example
    transformations = []
    for input_grid, output_grid in train_examples:
        trans = analyze_transformation(input_grid, output_grid)
        transformations.append(trans)

    # Find consensus
    rule_type = most_common([t['type'] for t in transformations])

    # Check consistency
    consistency = count(t['type'] == rule_type) / len(transformations)

    if consistency >= 0.8:
        # High consistency - simple rule
        return {
            'description': generate_description(transformations),
            'type': rule_type,
            'parameters': merge_parameters(transformations),
            'consistency': consistency
        }
    else:
        # Low consistency - composite or conditional rule
        return infer_composite_rule(transformations)
```

### Phase 3: Targeted Program Synthesis
**Generate the ONE correct program**

```python
def synthesize_from_rule(rule, dsl):
    """
    Generate a program that implements the inferred rule.

    Returns:
        - programs: 1-3 targeted programs (not 100 random ones!)
    """

    programs = []

    if rule['type'] == 'rotate':
        angle = rule['parameters']['angle']
        if angle == 90:
            programs.append(dsl.get('rotate_90'))
        elif angle == 180:
            programs.append(dsl.get('rotate_180'))
        # ...

    elif rule['type'] == 'color_remap':
        mapping = rule['parameters']['mapping']
        # Create custom color remap function
        def color_remap(grid):
            result = grid.copy()
            for old_color, new_color in mapping.items():
                result[grid == old_color] = new_color
            return result
        programs.append(Operation('learned_color_remap', color_remap))

    elif rule['type'] == 'translate':
        offset = rule['parameters']['offset']
        # Create custom translation
        def translate(grid):
            return translate_grid(grid, offset[0], offset[1])
        programs.append(Operation('learned_translate', translate))

    # Always include identity as baseline
    programs.insert(0, dsl.get('identity'))

    return programs
```

### Phase 4: Iterative Refinement
**Refine the program using repairs**

```python
def refine_program(program, train_examples, repairers):
    """
    Iteratively refine the program using repair loops.

    This is the V1 approach that WORKED.
    """

    best_program = program
    best_score = evaluate(program, train_examples)

    for iteration in range(10):  # Max 10 refinement iterations
        # Try each repairer
        for repairer in repairers:
            refined = repairer(best_program, train_examples)
            score = evaluate(refined, train_examples)

            if score > best_score:
                best_program = refined
                best_score = score

                if score >= 0.99:
                    return best_program, best_score  # Good enough!

        # No improvement this round - stop
        if score <= best_score:
            break

    return best_program, best_score
```

---

## Implementation Details

### New Nodes

**`rule_inferencer`** (reasoner)
- Analyzes all training examples
- Infers transformation rule
- Outputs: rule_description, rule_type, parameters, consistency

**`differential_analyzer`** (extractor)
- Compares input vs output for single example
- Detects: rotations, flips, translations, color remaps, object changes
- Outputs: transformation_type, parameters, confidence

**`targeted_synthesizer`** (reasoner)
- Takes inferred rule
- Generates 1-3 specific programs (not 100 random ones!)
- Uses learned parameters to create custom operations

**`iterative_refiner`** (repairer)
- Takes program + training examples
- Applies repair loops iteratively
- Stops when score plateaus or reaches 0.99

### Modified Pipeline

```
Phase 1: Feature Extraction (keep existing)
  ↓
Phase 2: Differential Analysis (NEW)
  - Analyze each input→output transformation
  - Extract concrete changes
  ↓
Phase 3: Rule Inference (NEW)
  - Find consensus across examples
  - Abstract to general rule
  ↓
Phase 4: Targeted Synthesis (REPLACE advanced_synthesizer)
  - Generate 1-3 programs implementing the rule
  - Include identity as baseline
  ↓
Phase 5: Iterative Refinement (ENHANCE repair loops)
  - Start with best program
  - Refine 10 times using repairs
  - Stop when converged or perfect
  ↓
Phase 6: Test Prediction
```

---

## Expected Improvements

### Quantitative Predictions

| Metric | V1 | V2 | V3 (Expected) |
|--------|----|----|---------------|
| Perfect Solves | 10% | 0% | **20-30%** |
| High Quality (>0.8) | 40% | 10% | **50-60%** |
| Avg IoU | 0.496 | 0.415 | **0.600-0.650** |
| Programs per task | 3-6 | 64 | **2-5** |

### Why V3 Will Work

1. **Learns from examples** - Understands what transformation is needed
2. **Targeted synthesis** - Generates the RIGHT program, not random ones
3. **Keeps V1's wins** - Identity + repairs worked, keep them!
4. **Iterative refinement** - Refine 1 program 10 times > Try 100 programs once
5. **Evidence-based** - Built on what actually worked (V1 perfect solve)

### Tasks V3 Should Solve

**Task 25ff71a9** (V1: ✓ 1.000, V2: ✗ 0.667)
- Differential analysis: detects translation offset (1, 1)
- Rule: "translate by (1, 1)"
- Targeted synthesis: custom translate function
- Expected: ✓ 1.000

**Task 025d127b** (V1: 0.980, V2: 0.740)
- Differential analysis: small placement offset
- Rule: "identity + translate"
- Refinement: placement repairer finds offset
- Expected: ✓ 0.990+

**Color remap tasks**
- Differential analysis: detects exact color mapping
- Rule: specific color_remap with learned mapping
- Expected: High success rate

---

## Implementation Priority

### Week 1: Core Analysis
1. ✅ Document V3 plan (this file)
2. Implement `differential_analyzer` node
3. Test on synthetic examples
4. Implement `rule_inferencer` node
5. Test rule inference on 3-4 simple tasks

### Week 2: Synthesis & Refinement
1. Implement `targeted_synthesizer` node
2. Implement `iterative_refiner` node
3. Integration testing
4. Create V3 solver

### Week 3: Evaluation
1. Test on same 10 ARC tasks
2. Compare V1 vs V2 vs V3
3. Analyze improvements and failures
4. Document results

---

## Success Criteria

### Minimum Viable
- Solves task 25ff71a9 perfectly (regression fix from V2)
- Improves on V1 (>10% solve rate)
- Uses <10 programs per task (targeted, not random)

### Target
- 20-30% solve rate (2-3x V1)
- 50-60% high quality (1.5x V1)
- Demonstrates rule learning on at least 3 tasks

### Stretch
- 30-40% solve rate (3-4x V1)
- Can verbalize learned rules
- Transfers rules across similar tasks

---

## Risk Mitigation

### Risk 1: Rule inference fails
**Mitigation:** Fallback to identity + repairs (V1 baseline)

### Risk 2: Can't generalize from few examples
**Mitigation:** Use consistency thresholding, flag low-confidence

### Risk 3: Complex composite rules
**Mitigation:** Start with simple rule types, expand incrementally

---

## Bottom Line

**V2 taught us:** More programs ≠ better (evidence: 64 programs → worse)

**V3 hypothesis:** Understanding rule → 1 correct program > 100 wrong ones

**Evidence for V3:** V1's perfect solve came from understanding (identity + offset)

**Approach:** Top-down (infer rule → synthesize) not bottom-up (generate → hope)

This is the evidence-based path forward.
