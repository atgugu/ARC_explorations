# Cognitive Workspace for ARC-AGI

A cognitive-inspired architecture for solving the Abstraction and Reasoning Corpus (ARC-AGI) challenge, based on Global Workspace Theory.

## Overview

This project implements a **Global Workspace-inspired controller** that coordinates specialized modules to solve ARC tasks through:

- **Perception:** Object detection and relation extraction
- **Rule Library:** Domain-specific language of transformations
- **Hypothesis Generation:** Proposing candidate programs
- **Workspace Controller:** Attention-based selection and broadcasting
- **Evaluation:** Scoring and verification of solutions

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download ARC dataset
mkdir -p data && cd data
git clone https://github.com/fchollet/ARC-AGI.git
cd ..

# 3. Run first example
python examples/simple_dsl_example.py

# 4. Solve a task
python scripts/run_solver.py --task-id "00d62c1b"
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## Documentation

- **[Cognitive_Workspace.md](Cognitive_Workspace.md)** - Theoretical framework and motivation
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Complete implementation roadmap
- **[QUICKSTART.md](QUICKSTART.md)** - Getting started guide

## Architecture

```
┌─────────────────────────────────────────────────┐
│          COGNITIVE WORKSPACE (Controller)        │
│  ┌──────────────────────────────────────────┐  │
│  │  Admitted Hypotheses (Top-k)             │  │
│  │  • rotate_largest(90°)                   │  │
│  │  • recolor_by_size(...)                  │  │
│  │  • copy_to_corners(...)                  │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         ↑                           ↓
    SELECT (Top-k)            BROADCAST (Global)
         ↑                           ↓
┌────────┴────────┬──────────────────┴─────────┐
│   PROPOSER      │    EVALUATOR    │  CRITIC  │
│ Generate        │ Score           │ Stop?    │
│ Hypotheses      │ Hypotheses      │ Branch?  │
└─────────────────┴─────────────────┴──────────┘
         ↑
         │ Objects, Relations
         ↓
┌────────────────────────────────────────────────┐
│              PERCEPTION MODULE                  │
│  • Object Detection                            │
│  • Feature Extraction                          │
│  • Relation Graphs                             │
└────────────────────────────────────────────────┘
         ↑
         │ Grids
         ↓
┌────────────────────────────────────────────────┐
│                 ARC TASK                       │
│  Input → Output pairs + Test input             │
└────────────────────────────────────────────────┘
```

## Implementation Status

- [x] **Phase 0:** Foundation & Infrastructure
  - [ ] Data pipeline (in progress)
  - [ ] Domain-specific language (planned)
- [ ] **Phase 1:** Perception Module
- [ ] **Phase 2:** Rule Library & Hypothesis Generation
- [ ] **Phase 3:** Evaluator & Executor
- [ ] **Phase 4:** Global Workspace Controller
- [ ] **Phase 5:** Integration & Baseline Evaluation

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for full roadmap.

## Project Structure

```
Cognitive_Workspace/
├── src/                  # Source code
│   ├── data/            # Dataset loading and utilities
│   ├── dsl/             # Domain-specific language
│   ├── perception/      # Object detection & relations
│   ├── proposer/        # Hypothesis generation
│   ├── evaluator/       # Hypothesis scoring
│   ├── workspace/       # Controller & reasoning loop
│   └── main.py          # Main entry point
├── tests/               # Unit & integration tests
├── scripts/             # Utility scripts
├── examples/            # Example usage
├── configs/             # Configuration files
├── results/             # Experiment outputs
└── docs/                # Additional documentation
```

## Key Concepts

### Global Workspace Theory (GWT)

The architecture is inspired by Baars' Global Workspace Theory and implements:

1. **Limited Capacity:** Workspace admits only top-k hypotheses (k=3-10)
2. **Global Broadcasting:** Admitted hypotheses are visible to all modules
3. **Competition:** Hypotheses compete for workspace admission
4. **Coherence:** Recurrent broadcasts stabilize solutions

### Reasoning Cycle

```python
for iteration in range(max_iterations):
    # 1. PROPOSE: Generate candidate hypotheses
    hypotheses = proposer.propose(state)

    # 2. SELECT: Controller picks top-k
    admitted = controller.select(hypotheses, k=5)

    # 3. BROADCAST: Make globally available
    controller.broadcast(admitted, state)

    # 4. EVALUATE: Score hypotheses
    scores = evaluator.score(admitted, task)

    # 5. UPDATE: Revise state
    state.update(admitted, scores)

    # 6. CHECK: Stop if solved
    if solved(state):
        return state.best_program
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_perception.py -v

# With coverage
pytest --cov=src --cov-report=html
```

### Code Style

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Experiments

```bash
# Evaluate on tasks
python scripts/evaluate.py --num-tasks 50 --difficulty easy

# Visualize workspace dynamics
python scripts/visualize_workspace.py --task-id "00d62c1b"

# Compare configurations
python scripts/compare_configs.py configs/baseline.yaml configs/advanced.yaml
```

## Performance Targets

| Milestone | Target | Status |
|-----------|--------|--------|
| MVP (Phase 5) | 10% training set | 🔄 In Progress |
| Neural (Phase 6) | 20% training set | ⏳ Planned |
| Optimized (Phase 8) | 25% training set | ⏳ Planned |
| Final (Phase 10) | 25% training + 12% eval | ⏳ Planned |

## Contributing

This is an experimental research project. Contributions are welcome!

Areas for contribution:
- New DSL primitives
- Better perception heuristics
- Novel hypothesis generation strategies
- Improved scoring functions
- Optimization and efficiency improvements

## License

MIT License - see LICENSE file for details

## Citations

If you use this work, please cite:

```bibtex
@article{cognitive_workspace_arc,
  title={Cognitive-Inspired Attention and Workspace for ARC},
  author={ARC Explorations Team},
  year={2025},
  note={From Baars' Global Workspace to an Attention-Based Controller}
}
```

## References

- Baars, B. J. (1988). A Cognitive Theory of Consciousness
- Dehaene, S., & Changeux, J. P. (2011). Experimental and theoretical approaches to conscious processing
- Chollet, F. (2019). The Measure of Intelligence (ARC-AGI)

## Contact

For questions and discussions, see the main repository: [ARC_explorations](../)
