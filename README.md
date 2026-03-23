# ARC Active Inference Solver

A program synthesis approach to the [ARC-AGI Challenge](https://arcprize.org/) using [Active Inference](https://en.wikipedia.org/wiki/Active_inference) as a unifying framework for abstract reasoning.

The core idea: multiple cognitive frameworks — curiosity-driven exploration, stability analysis, workspace attention, and program synthesis — are unified as components of a single Bayesian belief updating process.

## How It Works

The solver maintains a probability distribution over transformation hypotheses (symbolic programs) and updates beliefs as it observes training examples:

```
1. Perceive → Extract features (objects, colors, symmetries, patterns)
2. Hypothesize → Generate candidate programs from a DSL of 50+ primitives
3. Infer → Bayesian update: P(h|data) ∝ P(data|h) × P(h)
4. Select → Rank by posterior × stability, return top-2 predictions
```

**Key properties:**
- Always produces exactly 2 predictions per task
- No training phase — learns from 2–5 examples at inference time
- Interpretable: outputs symbolic programs, not opaque predictions
- Minimal dependencies (Python + NumPy)

## Repository Structure

```
├── unified_solver/                 # Primary solver implementation
│   ├── arc_active_inference_solver.py   # Core: perception, inference, synthesis (~1000 lines)
│   ├── arc_loader.py                    # Data loading and evaluation metrics
│   └── examples.py                      # Demonstrations and benchmarks
│
├── Generative_Task_Discovery/      # Compositional solver with parameter inference
│   ├── arc_generative_solver.py         # Base generative solver (TRG + active inference)
│   ├── compositional_solver.py          # Beam search over primitive compositions
│   ├── inferred_solver.py               # Combined compositional + parameter inference
│   ├── parameter_inference.py           # Learn parameters from training examples
│   ├── enhanced_color_inference.py      # Multi-strategy color mapping
│   ├── evaluate_real_arc.py             # Evaluation on official ARC dataset
│   └── ...                              # Additional primitives and evaluation tools
│
├── solvers/                        # Modular experimental solvers
│   ├── curiosity_solver/           # Curiosity-driven active inference
│   ├── graph_pendulum/             # Stability-aware dynamical systems approach
│   └── generative_solver/          # Typed rule grammar + program synthesis
│
├── docs/theory/                    # Theoretical foundations
│   ├── curiosity/                  # Information gain, Bayesian surprise, learning progress
│   ├── dynamical_systems/          # Graph pendulum, Lyapunov stability analysis
│   ├── workspace/                  # Global workspace theory, attention mechanisms
│   └── generative/                 # Generative task discovery, typed DSL
│
├── data/arc_samples/               # Sample ARC tasks for testing
└── tests/                          # Test suites
```

## Quick Start

```bash
pip install numpy
```

```python
from unified_solver.arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid

task = ARCTask(
    train_pairs=[
        (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),  # flip columns
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
cd unified_solver

python examples.py 1    # Flip vertical
python examples.py 7    # Active inference dynamics demo
python examples.py 9    # Evaluation suite
python examples.py a    # Run all examples
```

### Evaluating on Real ARC

```bash
cd Generative_Task_Discovery

# Download the ARC dataset
python download_real_arc.py

# Run evaluation
python evaluate_real_arc.py
```

### Running Tests

```bash
python -m tests.test_generative_solver
python -m tests.test_curiosity_solver
```

## Architecture

### Unified Solver

The primary implementation (`unified_solver/`) combines five theoretical frameworks into a single Active Inference architecture:

| Component | Role | Inspired By |
|-----------|------|-------------|
| `PerceptionModule` | Feature extraction (objects, colors, symmetries) | Neurosymbolic perception |
| `HypothesisGenerator` | DSL-based program synthesis (50+ primitives) | Generative task discovery |
| `ActiveInferenceEngine` | Bayesian belief updating over hypotheses | Active inference / free energy |
| `StabilityFilter` | Robustness testing via perturbations | Dynamical systems / Lyapunov analysis |
| `WorkspaceController` | Limited-capacity attention (7±2 hypotheses) | Global workspace theory |

### Generative Task Discovery

The `Generative_Task_Discovery/` module extends the base solver with:

- **Compositional reasoning**: Beam search over 2-step primitive chains (66% adoption)
- **Parameter inference**: Learn transformation parameters from training examples
- **Enhanced color mapping**: Multiple inference strategies (basic, enhanced, identity-aware)
- **Real ARC evaluation**: Tested on official ARC-AGI dataset (2% success rate, work in progress)

### Modular Solvers

Three standalone solvers explore different theoretical angles:

- **Curiosity Solver** — Bayesian belief updating with curiosity signals (information gain, epistemic uncertainty, learning progress) and hierarchical reasoning with a multi-armed bandit strategy layer.

- **Graph Pendulum Solver** — Models reasoning as a dynamical system. Uses behavior vectors, stability metrics (Lyapunov exponents), and basin discovery to navigate hypothesis space with stability-aware beam search.

- **Generative Solver** — Typed rule grammar (TRG) with active inference. Defines a typed DSL (Grid, Object, Mask, Color) and uses beta-annealed belief updating for program synthesis.

## Mathematical Foundation

**Bayesian Belief Updating:**
```
Prior:      P(h) = exp(-λ · complexity(h)) / Z
Likelihood: P(y|h,x) = exp(accuracy(h(x), y) / T)
Posterior:  P(h|x,y) ∝ P(y|h,x) · P(h)
```

**Curiosity Signals:**
```
Information Gain:       IG(h) = P_t(h) · log(P_t(h) / P_{t-1}(h))
Epistemic Uncertainty:  EU = -Σ P(h) log P(h)
Learning Progress:      LP = H[P_{t-1}] - H[P_t]
```

**Stability Metric:**
```
stability(h) = mean_accuracy(h) · exp(-std_accuracy(h))
```

**Final Selection:**
```
score(h) = P(h|data) × stability(h)
predictions = top_2(hypotheses, by=score)
```

## DSL Primitives

The solver's transformation library includes 50+ primitives across several categories:

| Category | Examples |
|----------|----------|
| Geometric | rotate (90/180/270), reflect (h/v/diagonal), translate, scale |
| Color | remap, fill, swap, conditional recolor |
| Morphological | erode, dilate, open, close, flood fill |
| Object | extract, count, align, connect, group by size/color |
| Spatial | tile, crop, pad, overlay, bounding box |
| Pattern | detect periodicity, symmetry completion |

## Requirements

- Python 3.7+
- NumPy ≥ 1.20
- Optional: SciPy (morphological operations, some curiosity metrics)

## Status

This is a research exploration. The solvers have been tested on synthetic examples and a subset of ARC tasks. The theoretical frameworks are documented in `docs/theory/` and serve as the conceptual foundation for the implementations.

## License

MIT License — see [LICENSE](LICENSE).

## References

- [ARC-AGI Challenge](https://arcprize.org/) — Francois Chollet
- [Active Inference](https://en.wikipedia.org/wiki/Active_inference) — Karl Friston
- [The Free Energy Principle](https://doi.org/10.1038/nrn2787) — Friston, 2010
- [ARC-AGI Dataset](https://github.com/fchollet/ARC-AGI) — GitHub
