# ARC Active Inference Solver

A unified system for ARC-AGI tasks that combines Active Inference with program synthesis. It produces two candidate predictions per task and updates beliefs during inference — no pre-training required.

> **Note:** Tested on small synthetic examples only. Performance on the full ARC-AGI benchmark has not been evaluated.

## Architecture

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
                        └──────────────┘      └──────┬──────┘
                                                     │
                    ┌────────────────────────────────┤
                    │                                │
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

## How It Works

1. **Perception** — extract objects, colors, symmetries, and patterns from training examples
2. **Hypothesis Generation** — build transformation candidates from a DSL of 50+ composable primitives
3. **Belief Initialization** — set priors proportional to `exp(-complexity)` (MDL / Occam bias)
4. **Active Inference Loop** — for each training example:
   - Bayesian update: `P(h | data) ∝ P(output | h, input) × P(h)`
   - Curiosity signals: information gain, epistemic uncertainty, learning progress
   - Stability assessment: filter inconsistent hypotheses
   - Workspace selection: attend to top-k hypotheses
5. **Final Selection** — rank by `posterior × stability`, return top-2 diverse predictions

## Theoretical Foundations

The solver draws on five frameworks:

| Framework | Contribution |
|-----------|-------------|
| Curiosity-Driven Neurosymbolic | Information gain, Bayesian surprise, learning progress |
| Global Workspace Theory | Limited-capacity attention, hypothesis broadcasting |
| Dynamical Systems / Graph Pendulum | Stability analysis, Lyapunov indicators, chaos filtering |
| Probabilistic Program Spaces | Continuous belief dynamics, information geometry |
| Generative Task Discovery | Typed DSL, compositional transformations, MDL bias |

## DSL Primitives

**Geometric:** `rotate_90`, `rotate_180`, `rotate_270`, `flip_horizontal`, `flip_vertical`, `transpose`

**Color:** `replace_color(old, new)`, `invert_colors`, `swap_colors(c1, c2)`

**Morphological:** `dilate`, `erode`, `fill_background`

**Object-based:** `keep_largest_object`, `keep_smallest_object`, detection and segmentation

**Spatial:** `zoom(scale)`, `tile(nx, ny)`, `crop`, `extend(padding)`

**Compositional:** any combination of the above (e.g. `rotate_90` then `flip_h`)

## Usage

```python
from arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid

task = ARCTask(
    train_pairs=[
        (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
        (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
    ],
    test_input=Grid([[9, 0], [1, 2]])
)

solver = ARCActiveInferenceSolver()
predictions = solver.solve(task, verbose=True)

print("Prediction 1:", predictions[0].data)
print("Prediction 2:", predictions[1].data)
```

### Running Examples

```bash
python examples.py 1    # Flip vertical
python examples.py 7    # Active inference demo
python examples.py 9    # Evaluation suite
python examples.py a    # Run all
```

### Configuration

```python
solver = ARCActiveInferenceSolver(
    workspace_capacity=20,   # Max hypotheses in workspace
    n_perturbations=5,       # Perturbations for stability testing
)
```

## Mathematical Foundation

**Belief Update:**
```
P(h | D) = P(D | h) · P(h) / P(D)
```

**Likelihood:**
```
P(observation | h) = exp(accuracy / temperature)
```

**Curiosity Signals:**
```
Information Gain:       IG(h) = P_new(h) · log(P_new(h) / P_old(h))
Epistemic Uncertainty:  H[P] = −Σ P(h) log P(h)
Learning Progress:      LP = H[P_old] − H[P_new]
```

**Stability:**
```
stability(h) = mean_accuracy(h) · exp(−std_accuracy(h))
```

**Final Ranking:**
```
score(h) = P(h | data) × stability(h)
```

## Extension Points

```python
# Add a new primitive
def my_transform(grid: Grid, **kwargs) -> Grid:
    return transformed_grid

library['my_transform'] = my_transform

# Custom curiosity signal
def custom_curiosity(hypothesis, belief):
    return score

# Custom stability metric
def custom_stability(hypothesis, task):
    return stability_score
```

## Limitations

- **DSL coverage:** Limited to pre-defined primitives
- **Perceptual:** Object detection is heuristic-based
- **Compositional depth:** May miss deeply nested rules
- **Validation:** Not yet tested on the full ARC-AGI benchmark

## Files

| File | Description |
|------|-------------|
| `arc_active_inference_solver.py` | Core solver implementation (~1,100 lines) |
| `arc_loader.py` | Data loading, evaluation, task utilities |
| `examples.py` | 8 demonstrations + evaluation suite |
| `DESIGN.md` | Detailed design rationale and comparisons |
| `IMPLEMENTATION_SUMMARY.md` | Implementation notes |

## License

Part of [ARC Active Inference Explorations](../README.md). MIT License.
