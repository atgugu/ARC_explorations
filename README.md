# ARC Active Inference Explorations

Exploring [Active Inference](https://en.wikipedia.org/wiki/Active_inference) as a unifying framework for abstract reasoning, applied to the [ARC-AGI Challenge](https://arcprize.org/).

The central idea: curiosity-driven exploration, stability analysis, workspace attention, and program synthesis can all be viewed as components of a single Bayesian belief-updating process.

> **Note:** This is a research exploration. The solvers have been tested on small synthetic tasks only; performance on the full ARC-AGI benchmark has not been evaluated.

## Repository Structure

```
├── unified_solver/          # Unified Active Inference solver (primary)
│   ├── arc_active_inference_solver.py   # Core solver (~1,100 lines)
│   ├── arc_loader.py                    # Data loading and evaluation
│   ├── examples.py                      # Usage demonstrations
│   ├── DESIGN.md                        # Architecture and design rationale
│   └── IMPLEMENTATION_SUMMARY.md        # Implementation details
│
├── solvers/                 # Standalone experimental solvers
│   ├── curiosity_solver/    # Curiosity-driven active inference
│   ├── graph_pendulum/      # Stability-aware dynamical system
│   └── generative_solver/   # Typed rule grammar + program synthesis
│
├── docs/theory/             # Theoretical foundations
│   ├── curiosity/           # Bayesian surprise, information gain, learning progress
│   ├── dynamical_systems/   # Graph pendulum, stability analysis
│   ├── workspace/           # Cognitive workspace theory
│   └── generative/          # Generative task discovery, typed DSL
│
├── data/arc_samples/        # Sample ARC training tasks
└── tests/                   # Test suite
```

## Unified Solver

The primary implementation combines five theoretical frameworks into a single Active Inference architecture:

1. **Curiosity-Driven Exploration** — information gain and epistemic uncertainty guide hypothesis search
2. **Global Workspace Theory** — limited-capacity attention selects the most promising hypotheses
3. **Dynamical Systems / Stability** — Lyapunov-like indicators filter unreliable hypotheses
4. **Probabilistic Program Spaces** — continuous Bayesian belief dynamics over transformations
5. **Program Synthesis** — typed DSL with 50+ composable primitives

The solver maintains a probability distribution over transformation hypotheses, updates beliefs as it observes each training example, and returns the two highest-scoring predictions.

See [unified_solver/README.md](unified_solver/README.md) for details.

## Standalone Solvers

Three modular solvers, each exploring a different angle:

- **Curiosity Solver** — Bayesian belief updating with curiosity signals (information gain, epistemic uncertainty, learning progress) and hierarchical reasoning.
- **Graph Pendulum Solver** — Models reasoning as a dynamical system with stability analysis, basin discovery, and chaos filtering.
- **Generative Solver** — Typed rule grammar for program synthesis with active inference-based belief updates.

## Quick Start

```bash
pip install numpy
cd unified_solver
python examples.py 1
```

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
```

## Requirements

- Python 3.7+
- NumPy
- Optional: SciPy, scikit-learn, matplotlib (graph pendulum solver only)

## Tests

```bash
python -m pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [ARC-AGI Challenge](https://arcprize.org/) by Francois Chollet
- [Active Inference](https://en.wikipedia.org/wiki/Active_inference) framework by Karl Friston
