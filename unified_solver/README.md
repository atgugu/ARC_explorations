# ARC Active Inference Solver (AAIS)

## A Unified System for ARC-AGI Challenge

**The ARC Active Inference Solver** is an elegant, principled system that blends the best ideas from multiple theoretical frameworks to solve diverse ARC-AGI tasks. It always produces **two best predictions** and **learns during inference** using active inference principles.

---

## Core Philosophy

The system is built on a simple but powerful insight: **Active Inference** naturally unifies all the key ideas needed for ARC solving:

1. **Active Inference**: Continuously update beliefs over transformation hypotheses as we observe each training example
2. **Curiosity-Driven Exploration**: Prioritize hypotheses with high information gain and epistemic uncertainty
3. **Stability-Aware Selection**: Prefer robust, low-variance solutions that generalize well
4. **Workspace Attention**: Limited capacity focus on the most promising hypotheses
5. **Program Synthesis**: DSL-based compositional transformations

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ARC Active Inference Solver               │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐         ┌──────▼──────┐      ┌──────▼──────┐
   │Perception│         │  Hypothesis  │      │   Active    │
   │  Module  │────────▶│  Generator   │────▶│  Inference  │
   └──────────┘         │   (DSL)      │      │   Engine    │
        │               └──────────────┘      └──────┬──────┘
        │                                            │
        │                                            │
   ┌────▼────────┐                           ┌──────▼──────┐
   │  Feature    │                           │   Belief    │
   │ Extraction  │                           │Distribution │
   └─────────────┘                           │  P(h|data)  │
                                             └──────┬──────┘
                                                    │
                    ┌───────────────────────────────┤
                    │                               │
             ┌──────▼──────┐               ┌────────▼────────┐
             │  Stability  │               │   Workspace     │
             │   Filter    │──────────────▶│  Controller     │
             └─────────────┘               │  (Top-K)        │
                                           └────────┬────────┘
                                                    │
                                            ┌───────▼────────┐
                                            │   Top-2        │
                                            │  Predictions   │
                                            └────────────────┘
```

---

## Theoretical Foundations

This system synthesizes ideas from five major frameworks developed in this repository:

### 1. **Curiosity-Driven Neurosymbolic Framework**
- **Bayesian Surprise**: KL divergence measuring belief changes
- **Epistemic Uncertainty**: Entropy of belief distribution
- **Learning Progress**: Reduction in uncertainty over time
- **Information Gain**: Expected belief refinement from observations

### 2. **Global Workspace Theory**
- **Limited Capacity**: Workspace holds only top-k hypotheses
- **Winner-Take-Most**: Best hypotheses dominate attention
- **Broadcasting**: Selected hypotheses are shared across all modules
- **Recurrent Refinement**: Iterative improvement through feedback

### 3. **Dynamical Systems / Graph Pendulum**
- **Stability Analysis**: Test hypothesis robustness via perturbations
- **Lyapunov-like Indicators**: Prefer low-variance solution trajectories
- **Basin Discovery**: Identify stable regions in hypothesis space
- **Chaos Avoidance**: Filter out unstable, chaotic hypotheses

### 4. **Probabilistic Program Spaces**
- **Continuous Belief Dynamics**: P(h|D_t) evolves continuously
- **Information Geometry**: Natural gradients in belief space
- **Bayesian Updates**: Posterior ∝ Likelihood × Prior
- **Phase Transitions**: Belief convergence and oscillations

### 5. **Generative Task Discovery**
- **Typed DSL**: Compositional transformation language
- **Program Synthesis**: Generate hypotheses from primitives
- **Schema Composition**: Combine basic operations
- **MDL Bias**: Prefer simpler programs (Occam's Razor)

---

## How It Works

### Step-by-Step Process

```python
# 1. Initialize
solver = ARCActiveInferenceSolver()

# 2. Solve task (always returns 2 predictions)
predictions = solver.solve(task)
```

**Internal Process:**

1. **Perception Phase**
   - Extract objects, colors, symmetries, patterns from training examples
   - Build feature representation of the task

2. **Hypothesis Generation**
   - Generate transformation candidates from DSL primitives
   - Include geometric, color, object-based, and compositional transforms
   - Bias toward simpler programs (MDL principle)

3. **Belief Initialization**
   - Initialize uniform prior: P(h) ∝ exp(-complexity)
   - Compute initial epistemic uncertainty (entropy)

4. **Active Inference Loop** (for each training example)
   ```python
   for input, output in training_examples:
       # Bayesian update
       P(h | new_data) ∝ P(new_data | h) × P(h | old_data)

       # Compute curiosity signals
       - Information Gain: KL(P_new || P_old)
       - Epistemic Uncertainty: H[P(h)]
       - Learning Progress: ΔH

       # Assess stability
       - Test consistency across examples
       - Compute variance in predictions

       # Workspace selection
       - Select top-k by: αP(h) + β·curiosity + γ·stability
   ```

5. **Final Selection**
   - Rank hypotheses by: `posterior_probability × stability`
   - Select top-2 for prediction
   - Apply to test input

6. **Return Predictions**
   - Always return exactly 2 predictions
   - First = highest ranked, Second = second highest

---

## Key Features

### ✅ Always Learns During Inference
- **Active Inference**: Updates beliefs with every training example
- **No Pre-training Required**: Works with zero prior knowledge
- **Few-Shot Learning**: Effective with 2-5 training examples

### ✅ Always Returns 2 Predictions
- Guaranteed to produce exactly 2 predictions
- First prediction = most probable × most stable
- Second prediction = next best alternative

### ✅ Curiosity-Driven Exploration
- **Information Gain**: Prioritize observations that reduce uncertainty
- **Epistemic Uncertainty**: Explore hypotheses with high knowledge gaps
- **Learning Progress**: Track improvement velocity

### ✅ Stability-Aware
- **Robustness Testing**: Filter chaotic, unstable hypotheses
- **Consistency**: Prefer hypotheses that work across all training examples
- **Low Variance**: Select solutions with predictable behavior

### ✅ Interpretable
- **Symbolic Programs**: Human-readable transformations
- **Reasoning Traces**: Track belief evolution
- **Hypothesis Names**: Clear descriptions (e.g., "rotate_90", "flip_vertical")

### ✅ Compositional
- **DSL Primitives**: Geometric, color, object, spatial operations
- **Schema Composition**: Combine basic operations
- **Hierarchical**: Build complex transformations from simple ones

---

## DSL Primitives

The system uses a rich Domain-Specific Language (DSL) for transformations:

### Geometric Primitives
- `rotate_90`, `rotate_180`, `rotate_270`
- `flip_horizontal`, `flip_vertical`
- `transpose`

### Color Primitives
- `replace_color(old, new)`
- `invert_colors`
- `swap_colors(c1, c2)`

### Morphological Primitives
- `dilate`, `erode`
- `fill_background`

### Object-Based Primitives
- `keep_largest_object`
- `keep_smallest_object`
- Object detection and segmentation

### Spatial Primitives
- `zoom(scale)` - Enlarge by repeating pixels
- `tile(nx, ny)` - Tile pattern
- `crop` - Crop to bounding box
- `extend(padding)` - Add padding

### Compositional
- Any combination of the above
- Examples: `rotate_90_then_flip_h`, `flip_v_then_transpose`

---

## Mathematical Foundations

### Active Inference

**Belief Update (Bayes Rule):**
```
P(h | D_new) = P(D_new | h) · P(h | D_old) / P(D_new)
```

**Likelihood Function:**
```
P(observation | hypothesis) = exp(accuracy / temperature)
```
where accuracy = fraction of correctly predicted pixels

### Curiosity Signals

**Information Gain:**
```
IG(h) = P_new(h) · log(P_new(h) / P_old(h))
```

**Epistemic Uncertainty (Entropy):**
```
H[P(h)] = -Σ P(h) log P(h)
```

**Learning Progress:**
```
LP = H[P_old] - H[P_new]
```

### Stability Score

```
stability(h) = mean_accuracy(h) · exp(-std_accuracy(h))
```

High mean + low variance = stable hypothesis

### Final Ranking

```
score(h) = P(h | data) × stability(h)
```

Top-2 hypotheses selected by this score.

---

## Usage Examples

### Basic Usage

```python
from arc_active_inference_solver import (
    ARCActiveInferenceSolver,
    ARCTask,
    Grid
)

# Create task
task = ARCTask(
    train_pairs=[
        (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),  # flip vertical
        (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
    ],
    test_input=Grid([[9, 0], [1, 2]])
)

# Solve
solver = ARCActiveInferenceSolver()
predictions = solver.solve(task, verbose=True)

# Results
print("Prediction 1:", predictions[0].data)
print("Prediction 2:", predictions[1].data)
```

### Verbose Mode

```python
# Enable detailed output
predictions = solver.solve(task, verbose=True)
```

Prints:
- Number of training examples
- Perceived features (objects, colors, symmetries)
- Generated hypotheses count
- Belief entropy after each observation
- Learning progress
- Stable vs. chaotic hypotheses
- Top-5 hypotheses with probabilities
- Final top-2 selections with scores

---

## Configuration

### Solver Parameters

```python
solver = ARCActiveInferenceSolver(
    workspace_capacity=20,     # Max hypotheses in workspace (Global Workspace)
    n_perturbations=5,         # Perturbations for stability testing
)
```

### Workspace Controller

```python
workspace = WorkspaceController(capacity=10)  # Top-k selection

selected = workspace.select_hypotheses(
    hypotheses,
    belief,
    curiosity_weight=0.3,      # Weight for curiosity signals
    stability_weight=0.3,      # Weight for stability
)
```

### Active Inference Engine

```python
engine = ActiveInferenceEngine(
    prior_weight=1.0           # MDL bias strength
)
```

---

## Implementation Status

### ✅ Completed
- [x] Perception Module (object detection, symmetry, patterns)
- [x] Hypothesis Generator (DSL with 30+ primitives)
- [x] Active Inference Engine (Bayesian updating)
- [x] Curiosity Signals (IG, epistemic uncertainty, learning progress)
- [x] Stability Filter (robustness testing)
- [x] Workspace Controller (top-k selection)
- [x] Unified Solver (end-to-end pipeline)
- [x] Always returns 2 predictions
- [x] Always learns during inference

### 🚧 Future Enhancements
- [ ] More sophisticated object relation detection
- [ ] Learned DSL primitives (neural modules)
- [ ] Meta-learning across tasks
- [ ] Compositional grammar learning
- [ ] Self-curriculum generation (Generative Task Discovery)
- [ ] Parallel hypothesis evaluation
- [ ] GPU acceleration for large-scale search

---

## Design Principles

### 1. **Simplicity**
- Clean, modular architecture
- Each component has a single responsibility
- Easy to understand and extend

### 2. **Elegance**
- Active Inference naturally unifies all frameworks
- Bayesian principles throughout
- Information theory provides principled foundations

### 3. **Practicality**
- Works out-of-the-box
- No pre-training required
- Handles diverse task types

### 4. **Interpretability**
- Symbolic programs (not neural black boxes)
- Reasoning traces
- Human-readable hypothesis names

### 5. **Robustness**
- Stability filtering prevents brittle solutions
- Handles edge cases (empty grids, mismatched shapes)
- Graceful degradation

---

## Theoretical Contributions

This implementation demonstrates how to:

1. **Unify Multiple Frameworks**: Active Inference serves as the natural glue
2. **Learn During Inference**: No training phase, pure few-shot learning
3. **Balance Exploration/Exploitation**: Curiosity guides search efficiently
4. **Ensure Robustness**: Stability analysis filters fragile hypotheses
5. **Maintain Interpretability**: Symbolic programs remain human-readable

---

## Comparison to Other Approaches

| Approach | AAIS | Pure Neural | Pure Symbolic | Hybrid |
|----------|------|-------------|---------------|--------|
| Interpretable | ✅ | ❌ | ✅ | ⚠️ |
| Few-Shot | ✅ | ❌ | ✅ | ⚠️ |
| Learns During Inference | ✅ | ❌ | ❌ | ⚠️ |
| Active Inference | ✅ | ❌ | ❌ | ❌ |
| Curiosity-Driven | ✅ | ⚠️ | ❌ | ⚠️ |
| Stability-Aware | ✅ | ❌ | ❌ | ❌ |
| Always 2 Predictions | ✅ | ⚠️ | ⚠️ | ⚠️ |
| No Pre-training | ✅ | ❌ | ✅ | ❌ |

---

## Performance Characteristics

### Strengths
- **Few-shot learning**: Effective with 2-5 examples
- **Compositional generalization**: Handles novel combinations
- **Robust**: Filters unstable hypotheses
- **Interpretable**: Clear reasoning traces
- **Principled**: Bayesian foundations

### Limitations
- **DSL Coverage**: Limited to pre-defined primitives
- **Computational Cost**: Evaluates many hypotheses
- **Complex Compositions**: May miss deeply nested rules
- **Perceptual Limitations**: Object detection is heuristic-based

### Computational Complexity
- **Hypothesis Generation**: O(k) where k = number of primitives
- **Belief Update**: O(n·h) where n = training examples, h = hypotheses
- **Stability Testing**: O(h·n) for each hypothesis
- **Total**: O(h·n) per task

---

## Extension Points

The system is designed for easy extension:

### 1. Add New Primitives
```python
# In HypothesisGenerator._build_primitive_library()
def my_custom_transform(grid: Grid, **kwargs) -> Grid:
    # Your transformation logic
    return transformed_grid

library['my_transform'] = my_custom_transform
```

### 2. Custom Curiosity Signals
```python
# In ActiveInferenceEngine.compute_curiosity_score()
def custom_curiosity(hypothesis, belief):
    # Your curiosity metric
    return score
```

### 3. Enhanced Stability Testing
```python
# In StabilityFilter.assess_stability()
def advanced_stability(hypothesis, task):
    # Your stability metric
    return stability_score
```

### 4. Alternative Workspace Selection
```python
# In WorkspaceController.select_hypotheses()
def custom_selection(hypotheses, belief):
    # Your selection strategy
    return selected_hypotheses
```

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{arc_active_inference_solver,
  title={ARC Active Inference Solver: A Unified System for Abstract Reasoning},
  author={ARC Explorations Project},
  year={2025},
  url={https://github.com/your-repo/ARC_explorations}
}
```

---

## License

This project is part of the ARC Explorations repository. See main repository for license details.

---

## Acknowledgments

This system synthesizes ideas from:
- **Curiosity Framework**: Bayesian surprise, information gain, learning progress
- **Global Workspace Theory**: Limited capacity attention, broadcasting
- **Graph Pendulum System**: Stability analysis, dynamical systems perspective
- **Probabilistic Program Spaces**: Continuous belief dynamics
- **Generative Task Discovery**: DSL design, schema composition

All theoretical frameworks developed in the ARC_explorations repository.

---

## Contact & Contributions

For questions, issues, or contributions, please see the main repository README.

**The ARC Active Inference Solver** - *Simple, Elegant, Powerful* ✨
