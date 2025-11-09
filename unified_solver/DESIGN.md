# ARC Active Inference Solver - Design Document

## Executive Summary

The **ARC Active Inference Solver (AAIS)** is a unified system that synthesizes five major theoretical frameworks from the ARC_explorations repository into a single, coherent architecture. The system solves ARC-AGI tasks by maintaining a probabilistic belief distribution over transformation hypotheses and updating these beliefs through **active inference** as it observes training examples.

**Key Innovation**: Active Inference serves as the natural unifying principle that connects curiosity-driven search, stability-aware selection, workspace attention, and program synthesis into a simple, elegant process.

---

## Design Philosophy

### Three Core Principles

1. **Simplicity Through Unification**
   - Instead of implementing five separate systems, we identified that **active inference** (Bayesian belief updating) is the common thread
   - All other mechanisms (curiosity, stability, workspace) naturally emerge as components of the inference process

2. **Learning During Inference**
   - No separate training phase
   - Pure few-shot learning from 2-5 examples
   - Beliefs evolve continuously with each observation

3. **Guaranteed Outputs**
   - Always produces exactly 2 predictions
   - Graceful degradation (never fails to produce output)
   - Interpretable reasoning traces

---

## Theoretical Integration

### How Five Frameworks Become One

#### 1. Active Inference (Core Engine)

**From**: Probabilistic Program Spaces framework

**Role**: The central belief updating mechanism

```python
# Bayesian update at each training example
P(h | D_new) ∝ P(D_new | h) × P(h | D_old)
```

**Key Insight**: The inference process IS the reasoning process. We don't search for solutions; we update beliefs about solutions.

#### 2. Curiosity Signals (Exploration Guide)

**From**: Curiosity-Driven Neurosymbolic Framework

**Role**: Guide which hypotheses deserve computational resources

```python
curiosity(h) = information_gain(h) + epistemic_uncertainty(h)
```

**Integration**: Curiosity signals are computed as byproducts of belief updates:
- **Information Gain**: KL divergence between successive belief states
- **Epistemic Uncertainty**: Entropy of current belief distribution
- **Learning Progress**: Change in entropy over time

**Why It Works**: Active inference naturally quantifies how much each hypothesis reduces uncertainty, which is exactly what curiosity measures.

#### 3. Stability Analysis (Robustness Filter)

**From**: Graph Pendulum / Dynamical Systems framework

**Role**: Filter out chaotic, unreliable hypotheses

```python
stability(h) = mean_accuracy(h) × exp(-variance_accuracy(h))
```

**Integration**: Stability is assessed by testing hypothesis consistency across all training examples. This fits naturally into the inference loop.

**Why It Works**: A hypothesis with high posterior probability but high variance is unreliable. The product `P(h) × stability(h)` selects robust solutions.

#### 4. Workspace Controller (Attention Mechanism)

**From**: Global Workspace / Cognitive Workspace framework

**Role**: Limited capacity attention over hypotheses

```python
workspace = top_k(hypotheses, score=α·P(h) + β·curiosity + γ·stability)
```

**Integration**: The workspace implements selective attention by maintaining only the top-k most promising hypotheses.

**Why It Works**: Computational constraints force prioritization. The workspace ensures we spend resources on hypotheses that are:
- Probable (high P(h))
- Informative (high curiosity)
- Robust (high stability)

#### 5. Program Synthesis (Hypothesis Generator)

**From**: Generative Task Discovery framework

**Role**: Generate transformation candidates from a DSL

**Integration**: The DSL provides the hypothesis space over which we perform inference. Each primitive becomes a hypothesis family.

**Why It Works**: Program synthesis gives us a structured, compositional hypothesis space. The typed DSL ensures hypotheses are valid transformations.

---

## System Architecture

### Information Flow

```
Input: Training pairs + Test input
  │
  ├─→ Perception Module
  │     └─→ Features: {objects, colors, symmetries, patterns}
  │
  ├─→ Hypothesis Generator
  │     └─→ Hypotheses: H = {h₁, h₂, ..., hₙ}
  │
  ├─→ Belief Initialization
  │     └─→ P(h) = exp(-complexity(h)) / Z
  │
  └─→ Active Inference Loop (for each training example):
        │
        ├─→ Likelihood Computation
        │     └─→ P(obs | h) = exp(accuracy(h) / T)
        │
        ├─→ Bayesian Update
        │     └─→ P(h | D_new) ∝ P(obs | h) × P(h | D_old)
        │
        ├─→ Curiosity Computation
        │     ├─→ Information Gain: KL(P_new || P_old)
        │     ├─→ Epistemic Uncertainty: H[P(h)]
        │     └─→ Learning Progress: ΔH
        │
        ├─→ Stability Assessment
        │     └─→ stability(h) = μ(acc) × exp(-σ(acc))
        │
        └─→ Workspace Selection
              └─→ Keep top-k by combined score

Final Ranking: score(h) = P(h | D) × stability(h)

Output: Top-2 hypotheses applied to test input
```

---

## Mathematical Foundations

### Bayesian Framework

**Prior** (initialized with MDL bias):
```
P(h) = exp(-λ · complexity(h)) / Z
```

**Likelihood** (pixel-wise accuracy with temperature):
```
P(y | h, x) = exp(accuracy(h(x), y) / T)
```

**Posterior** (Bayes rule):
```
P(h | x, y) = P(y | h, x) · P(h) / P(y | x)
```

### Curiosity Signals

**Information Gain** (KL divergence):
```
IG(h) = P_t(h) · log(P_t(h) / P_{t-1}(h))
```

**Epistemic Uncertainty** (Shannon entropy):
```
EU = -Σ P(h) log P(h)
```

**Learning Progress** (entropy reduction):
```
LP = H[P_{t-1}] - H[P_t]
```

### Stability Metric

**Consistency across examples**:
```
acc_i(h) = matches(h(x_i), y_i) / total_pixels

stability(h) = mean(acc) · exp(-std(acc))
```

High mean + low variance = stable hypothesis

### Final Selection

**Combined score**:
```
score(h) = P(h | D) × stability(h)
```

**Top-2 selection**:
```
predictions = [h₁(x_test), h₂(x_test)]
where h₁, h₂ = argmax_{h₁≠h₂} score(h)
```

---

## Design Decisions

### Why Active Inference?

**Alternative Considered**: Separate search algorithm + heuristic evaluation

**Why Active Inference Wins**:
1. **Principled**: Bayesian framework is mathematically sound
2. **Unified**: Single mechanism instead of multiple ad-hoc components
3. **Natural Uncertainty Quantification**: Probabilities encode confidence
4. **Curiosity Falls Out**: Information-theoretic signals emerge naturally
5. **Online Learning**: Updates with each observation

### Why DSL-based Programs?

**Alternative Considered**: Neural end-to-end learning

**Why DSL Wins**:
1. **Interpretable**: Programs are human-readable
2. **Compositional**: Combine primitives to build complex transformations
3. **Sample Efficient**: No training data needed
4. **Generalizable**: Symbolic rules transfer across tasks
5. **Debuggable**: Can inspect and modify programs

### Why Stability Filtering?

**Alternative Considered**: Pure probability ranking

**Why Stability Wins**:
1. **Robustness**: Filters overfitted, brittle solutions
2. **Generalization**: Consistent hypotheses transfer better
3. **Variance Matters**: High probability but high variance = unreliable
4. **Empirical**: Directly tests hypothesis behavior

### Why Limited Workspace?

**Alternative Considered**: Consider all hypotheses always

**Why Workspace Wins**:
1. **Computational Efficiency**: Focus on promising hypotheses
2. **Biologically Plausible**: Matches human working memory limits
3. **Quality over Quantity**: Better to explore few good options deeply
4. **Coherence**: Prevents fragmentation across too many hypotheses

---

## Implementation Highlights

### Modular Design

Each component is a separate class with clear responsibilities:

```python
# Perception
perception = PerceptionModule()
features = perception.perceive(grid)

# Hypothesis Generation
generator = HypothesisGenerator(perception)
hypotheses = generator.generate_hypotheses(task, features)

# Active Inference
engine = ActiveInferenceEngine()
belief = engine.update_beliefs(belief, observation, hypotheses)

# Stability
filter = StabilityFilter()
stability = filter.assess_stability(hypothesis, task, belief)

# Workspace
workspace = WorkspaceController(capacity=20)
selected = workspace.select_hypotheses(hypotheses, belief)
```

### Extensibility

**Adding New Primitives**:
```python
def my_transform(grid: Grid, **kwargs) -> Grid:
    # Custom transformation
    return transformed_grid

# Register in library
generator.primitive_library['my_transform'] = my_transform
```

**Custom Curiosity Signals**:
```python
def my_curiosity(hypothesis, belief):
    # Custom metric
    return score

# Override in engine
engine.compute_curiosity_score = my_curiosity
```

### Performance Optimizations

1. **Hypothesis Caching**: Avoid redundant computations
2. **Early Termination**: Stop if entropy near zero
3. **Lazy Evaluation**: Only compute stability for top hypotheses
4. **Vectorization**: Use NumPy for grid operations

---

## Complexity Analysis

### Time Complexity

**Per Task**:
```
T(n, h, k) = O(h·n)
where:
  n = number of training examples (typically 2-5)
  h = number of hypotheses (typically 50-200)
  k = workspace capacity (typically 20)
```

**Breakdown**:
- Perception: O(grid_size) per example
- Hypothesis Generation: O(primitives) = O(1) typically
- Belief Update: O(h·n) total across all examples
- Stability: O(h·n) worst case, O(k·n) with workspace
- Workspace Selection: O(h log k) per update

**Practical Performance**: ~1-10 seconds per task on CPU

### Space Complexity

```
S(h, n) = O(h + n·grid_size)
where:
  h = number of hypotheses
  n = number of training examples
  grid_size = typically < 900 (30×30 max)
```

**Memory Usage**: ~10-100 MB per task

---

## Evaluation Metrics

### Success Criteria

1. **Exact Match**: Prediction exactly matches ground truth
2. **Pixel Accuracy**: Fraction of correctly predicted pixels
3. **IoU**: Intersection-over-Union for non-zero pixels

### System Metrics

1. **Solve Rate**: Fraction of tasks with exact match
2. **Top-2 Coverage**: Fraction of tasks where either prediction is correct
3. **Average Accuracy**: Mean pixel accuracy across tasks
4. **Stability**: Variance in hypothesis predictions

### Interpretability Metrics

1. **Hypothesis Diversity**: Entropy of selected hypotheses
2. **Learning Progress**: Average ΔH per training example
3. **Convergence Rate**: Speed of belief concentration

---

## Strengths and Limitations

### Strengths

✅ **Few-Shot Learning**: Works with 2-5 examples
✅ **Interpretable**: Symbolic programs, clear reasoning traces
✅ **Principled**: Bayesian foundations, information theory
✅ **No Training Required**: Pure inference, no pre-training
✅ **Guaranteed Output**: Always produces 2 predictions
✅ **Compositional**: Handles novel combinations of primitives
✅ **Uncertainty-Aware**: Probabilities encode confidence
✅ **Robust**: Stability filtering prevents fragile solutions

### Limitations

⚠️ **DSL Coverage**: Limited to pre-defined primitives
⚠️ **Perception**: Heuristic-based object detection
⚠️ **Scalability**: Evaluates each hypothesis sequentially
⚠️ **Complex Compositions**: May miss deeply nested patterns
⚠️ **Search Space**: Exponential in composition depth

### Failure Modes

1. **Missing Primitive**: If required transformation not in DSL
2. **Perception Failure**: If objects not detected correctly
3. **Ambiguous Tasks**: If multiple hypotheses equally likely
4. **Complex Compositions**: If requires >2 primitive compositions

---

## Future Extensions

### Near-Term (Already Designed)

1. **Richer DSL**: Add more primitives
   - Path-based transformations
   - Graph operations
   - Relational reasoning

2. **Better Perception**:
   - Neural object detector
   - Relation extraction
   - Abstract pattern recognition

3. **Meta-Learning**:
   - Learn primitive priors from solved tasks
   - Transfer knowledge across tasks
   - Build task families

### Medium-Term (Framework Exists)

4. **Generative Task Discovery**:
   - Learn transformation priors
   - Generate synthetic training tasks
   - Self-curriculum learning

5. **Hierarchical Composition**:
   - Multi-level program synthesis
   - Subroutine discovery
   - Abstraction learning

6. **Parallel Evaluation**:
   - GPU acceleration
   - Distributed hypothesis testing
   - Beam search variants

### Long-Term (Research Questions)

7. **Neural-Symbolic Hybrid**:
   - Learn new primitives
   - Continuous program spaces
   - Differentiable execution

8. **Causal Reasoning**:
   - Counterfactual inference
   - Intervention testing
   - Structural causal models

9. **Multi-Modal Integration**:
   - Language descriptions
   - Interactive querying
   - Human-in-the-loop

---

## Comparison to State-of-the-Art

### vs. DreamCoder (Lake et al.)
- **Similar**: DSL-based program synthesis
- **Different**: We use active inference instead of search
- **Advantage**: Online learning, uncertainty quantification

### vs. Neural Program Synthesis
- **Similar**: Learn transformations from examples
- **Different**: Symbolic vs. neural representations
- **Advantage**: Interpretability, sample efficiency

### vs. GPT-4 / LLMs
- **Similar**: Few-shot learning
- **Different**: Probabilistic inference vs. language modeling
- **Advantage**: Guaranteed symbolic output, no hallucinations

### vs. RL-based Solvers
- **Similar**: Learn from rewards
- **Different**: Bayesian inference vs. policy learning
- **Advantage**: No reward engineering, pure supervision

---

## Conclusion

The ARC Active Inference Solver demonstrates that **active inference** provides a natural, elegant framework for unifying multiple theoretical approaches to abstract reasoning. By maintaining probabilistic beliefs and updating them through Bayesian inference, the system achieves:

1. **Simplicity**: Single coherent process, not multiple subsystems
2. **Elegance**: Information-theoretic principles throughout
3. **Practicality**: Works on real ARC tasks without training
4. **Interpretability**: Symbolic programs with reasoning traces

The key insight is that **active inference is not just another component**—it's the organizing principle that makes curiosity, stability, attention, and synthesis work together seamlessly.

This design shows that sometimes the best way to unify complex frameworks is to find the right abstraction that makes them all special cases of a single, principled process.

---

## References

### Theoretical Frameworks (ARC_explorations)

1. **Curiosity-Driven Neurosymbolic Framework**
   - Bayesian surprise, information gain, epistemic uncertainty
   - Hierarchical curiosity signals
   - Resource allocation via curiosity budget

2. **Global Workspace Theory**
   - Limited capacity attention
   - Winner-take-most dynamics
   - Hypothesis broadcasting

3. **Graph Pendulum / Dynamical Systems**
   - Stability analysis
   - Lyapunov-like indicators
   - Basin discovery

4. **Probabilistic Program Spaces**
   - Continuous belief dynamics
   - Information geometry
   - Phase transitions

5. **Generative Task Discovery**
   - Typed DSL
   - Self-curriculum learning
   - Program synthesis

### Foundational Papers

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Chollet, F. (2019). On the Measure of Intelligence
- Baars, B. (1988). A Cognitive Theory of Consciousness
- Lake, B. et al. (2017). Building machines that learn and think like people

---

**Document Version**: 1.0
**Last Updated**: 2025
**Author**: ARC Explorations Project
