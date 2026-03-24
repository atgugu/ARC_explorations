# Generative Task Discovery

Compositional ARC solver with parameter inference, tested on the official ARC-AGI dataset.

This module builds on the base generative solver (`solvers/generative_solver/`) and adds compositional reasoning (beam search over primitive chains), smart parameter inference from training examples, and evaluation on real ARC tasks.

## Architecture

```
arc_generative_solver.py          Base solver: TRG primitives + active inference
    ↓
compositional_solver.py           Beam search over 2-step primitive compositions
    ↓
inferred_solver.py                Combined compositional + parameter inference
    ↓
evaluate_real_arc.py              Evaluation on official ARC dataset
```

### Core Modules

| File | Description |
|------|-------------|
| `arc_generative_solver.py` | Base solver with typed rule grammar, active inference, dual predictions |
| `advanced_solver.py` | Extended solver with advanced primitives and execution engine |
| `advanced_primitives.py` | Extended primitive library (morphological ops, object operations) |
| `compositional_solver.py` | Beam search over 2-step primitive compositions |
| `parameter_inference.py` | Learn color mappings, translations, rotations from training examples |
| `enhanced_color_inference.py` | Multi-strategy color mapping (basic, enhanced, identity-aware) |
| `inferred_solver.py` | Full solver combining composition + inference |
| `extend_markers_primitive.py` | Pattern extrapolation primitive (extend marker pixels) |
| `near_miss_primitives.py` | Primitives targeting near-miss failure patterns |
| `task_generation.py` | Reverse inference: generate tasks from programs |

### Evaluation & Testing

| File | Description |
|------|-------------|
| `evaluate_real_arc.py` | Evaluate on official ARC-AGI dataset (400 tasks) |
| `arc_test_suite.py` | Test suite for solver validation |
| `test_system.py` | Basic transformation tests |
| `download_real_arc.py` | Download official ARC dataset |

### Other

| File | Description |
|------|-------------|
| `example_usage.py` | Demonstrations and benchmarks |
| `demo_task_generation.py` | Task generation demos |
| `diverse_solver.py` | Diverse dual prediction strategy |
| `enhanced_solver.py` | Enhanced solver with near-miss primitives |
| `ensemble_solver.py` | Ensemble combining multiple strategies |

## Quick Start

```bash
pip install numpy

# Download ARC dataset
python download_real_arc.py

# Run evaluation on real ARC
python evaluate_real_arc.py
```

### Basic Usage

```python
from arc_generative_solver import ARCGenerativeSolver

solver = ARCGenerativeSolver(
    max_candidates=100,
    beam_width=15,
    active_inference_steps=5
)

task = {
    "train": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[3, 2, 1], [6, 5, 4]]}],
    "test": [{"input": [[7, 8, 9]], "output": [[9, 8, 7]]}]
}

pred1, pred2, metadata = solver.solve(task)
```

### With Compositional Reasoning + Parameter Inference

```python
from inferred_solver import InferredCompositionalSolver

solver = InferredCompositionalSolver(
    max_candidates=150,
    beam_width=20,
    active_inference_steps=3,
    max_depth=2,
    composition_beam_width=10
)

pred1, pred2, metadata = solver.solve(task)
```

## Results on ARC-AGI

Evaluated on 100 tasks from the official ARC training set:

| Metric | Value |
|--------|-------|
| Success rate (exact match) | 2% |
| Average pixel accuracy | 55.9% |
| Median pixel accuracy | 74.5% |
| Tasks with >90% accuracy | 15% |
| Tasks with >70% accuracy | 52% |

### Key Findings

- **Compositional reasoning**: 66% of tasks select multi-step compositions
- **Parameter inference**: Doubled success rate from 1% to 2%
- **Bimodal performance**: Tasks tend to be either nearly solved (>90%) or poorly matched (<50%)
- **Main gap**: Position-aware transformations (changing specific pixels based on location rather than global color rules)
