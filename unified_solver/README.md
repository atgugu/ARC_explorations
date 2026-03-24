# Unified Solver

The primary ARC solver implementation. Combines five theoretical frameworks into a single Active Inference architecture.

## Files

| File | Description |
|------|-------------|
| `arc_active_inference_solver.py` | Core solver: perception, hypothesis generation, belief updating, stability filtering (~1000 lines) |
| `arc_loader.py` | Data loading (JSON, programmatic) and evaluation metrics (pixel accuracy, exact match, IoU) |
| `examples.py` | 8 demonstration tasks + active inference dynamics demo + evaluation suite |

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
```

## Configuration

```python
solver = ARCActiveInferenceSolver(
    workspace_capacity=20,   # Max hypotheses in workspace
    n_perturbations=5,       # Perturbations for stability testing
)
```

## Components

- **PerceptionModule** — Extracts objects, colors, symmetries, and patterns from training grids
- **HypothesisGenerator** — Generates candidate programs from a DSL of 50+ primitives
- **ActiveInferenceEngine** — Bayesian belief updating with MDL-weighted priors
- **StabilityFilter** — Tests hypothesis robustness via perturbations, filters chaotic solutions
- **WorkspaceController** — Limited-capacity attention mechanism (top-k selection)

## Adding Primitives

```python
# In HypothesisGenerator._build_primitive_library()
def my_transform(grid: Grid, **kwargs) -> Grid:
    # transformation logic
    return transformed_grid

library['my_transform'] = my_transform
```
