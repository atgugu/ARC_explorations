# ARC Active Inference Explorations

> Theoretical explorations of Active Inference approaches to the ARC-AGI Challenge.
> These are experimental ideas requiring empirical validation on the full ARC benchmark.

## Overview

This repository explores how [Active Inference](https://en.wikipedia.org/wiki/Active_inference) (Bayesian belief updating) can serve as a unifying principle for abstract reasoning, applied to the [ARC-AGI Challenge](https://arcprize.org/).

The central idea: multiple cognitive frameworks -- curiosity-driven exploration, stability analysis, workspace attention, and program synthesis -- can be viewed as components of a single Active Inference process.

## Repository Structure

```
ARC_explorations/
├── README.md
├── LICENSE
├── requirements.txt
├── UNIFIED_SYSTEM_OVERVIEW.md
│
├── unified_solver/                    # Primary: Unified Active Inference Solver
│   ├── arc_active_inference_solver.py
│   ├── arc_loader.py
│   ├── examples.py
│   ├── README.md
│   ├── DESIGN.md
│   └── IMPLEMENTATION_SUMMARY.md
│
├── solvers/                           # Modular experimental solvers
│   ├── curiosity_solver/              # Curiosity-driven active inference
│   ├── graph_pendulum/                # Stability-aware dynamical system
│   └── generative_solver/             # Typed rule grammar + program synthesis
│
├── docs/                              # Theoretical documentation
│   └── theory/
│       ├── curiosity/                 # Curiosity-driven neurosymbolic framework
│       ├── dynamical_systems/         # Graph pendulum / dynamical systems
│       ├── workspace/                 # Cognitive workspace theory
│       └── generative/               # Generative task discovery
│
├── data/                              # Sample ARC tasks
│   └── arc_samples/training/
│
└── tests/                             # Test files
    ├── test_curiosity_solver.py
    └── test_generative_solver.py
```

## Unified Solver (Primary)

The unified solver combines ideas from five theoretical frameworks into a single Active Inference architecture. It uses Bayesian belief updating over a DSL of transformation hypotheses, guided by curiosity signals and stability filtering.

See [unified_solver/README.md](unified_solver/README.md) for details.

## Modular Solvers

Three standalone solvers, each exploring a different theoretical angle:

- **Curiosity Solver** (`solvers/curiosity_solver/`) -- Bayesian belief updating with curiosity signals (information gain, epistemic uncertainty, learning progress) and hierarchical reasoning.

- **Graph Pendulum Solver** (`solvers/graph_pendulum/`) -- Models reasoning as a dynamical system with stability analysis, basin discovery, and chaos filtering.

- **Generative Solver** (`solvers/generative_solver/`) -- Typed rule grammar (TRG) for program synthesis with active inference-based belief updates.

## Theoretical Foundations

The solvers are grounded in five theoretical frameworks, documented in `docs/theory/`:

1. **Curiosity-Driven Neurosymbolic Framework** -- Information gain, Bayesian surprise, learning progress
2. **Global Workspace Theory** -- Limited capacity attention, hypothesis broadcasting
3. **Graph Pendulum / Dynamical Systems** -- Stability analysis, Lyapunov indicators, basin discovery
4. **Probabilistic Program Spaces** -- Continuous belief dynamics, information geometry
5. **Generative Task Discovery** -- Typed DSL, compositional transformations, program synthesis

## Quick Start

```bash
pip install numpy

# Run a basic example
cd unified_solver
python examples.py 1
```

```python
from unified_solver.arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid

task = ARCTask(
    train_pairs=[
        (Grid([[1,2],[3,4]]), Grid([[2,1],[4,3]])),
        (Grid([[5,6],[7,8]]), Grid([[6,5],[8,7]])),
    ],
    test_input=Grid([[9,0],[1,2]])
)

solver = ARCActiveInferenceSolver()
predictions = solver.solve(task, verbose=True)
```

## Requirements

- Python 3.7+
- NumPy (core dependency)
- Optional: SciPy, scikit-learn, matplotlib (for graph pendulum solver)

## Status

This is a **research exploration**, not a production solver. Key caveats:

- The solvers have been tested on small synthetic examples only
- No validation has been performed on the full ARC-AGI benchmark
- The theoretical frameworks are speculative and require empirical testing
- Performance on the actual ARC challenge is unknown

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- [ARC-AGI Challenge](https://arcprize.org/) by Francois Chollet
- [Active Inference](https://en.wikipedia.org/wiki/Active_inference) framework by Karl Friston
