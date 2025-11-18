# ARC Conditional Reasoning Solver

A high-performance solver for ARC-AGI tasks using conditional transformations, spatial reasoning, and multi-stage pipelines.

## Current Performance

**57.75% average accuracy** on 100 ARC training tasks
**2% exact solve rate** (first exact solves achieved!)
**0.24s per task** (very fast inference)

## Architecture

### Multi-Phase System

The solver evolved through 7 phases, each adding new capabilities:

| Phase | Innovation | Accuracy Gain |
|-------|-----------|---------------|
| **1-2** | Active inference + curiosity | Baseline (28%) |
| **3** | Nested conditionals (AND/OR/NOT) | +7% |
| **4** | Richer spatial predicates | +7% |
| **5** | Geometric transformations | +13% |
| **6.1** | Action learning + confidence | +2.6% |
| **7** | Multi-stage pipelines | +2 exact solves ✅ |

**Total: 28% → 57.75% (+29.75 points)**

### Core Components

**1. Conditional Transformations** (`transformations/conditional_transforms.py`)
- IF-THEN-ELSE logic on object properties
- Spatial predicates (near_edge, touching, symmetric, etc.)
- Composite conditions (AND/OR/NOT combinations)

**2. Action Inference** (`core/action_inference.py`)
- Learns which transformations occur from training data
- Grid-level and object-level detection
- Confidence-based prioritization

**3. Multi-Stage Pipelines** (`core/pipeline_transform.py`)
- Chains transformations sequentially
- Greedy beam search for pipeline discovery
- Handles complex sequential reasoning tasks

**4. Composite Actions** (`solver_conditional.py`)
- Geometric transformations (rotate, reflect)
- Grid operations (extend, replicate, swap colors)
- Context-aware application

## Usage

### Basic Example

```python
from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver
import numpy as np

# Create solver
solver = ConditionalARCCuriositySolver()

# Prepare task data
train_pairs = [
    (input1, output1),
    (input2, output2),
]
test_input = np.array([[...]])

# Generate hypotheses
hypotheses = solver._generate_hypotheses(train_pairs, test_input)

# Test top hypothesis
best = hypotheses[0]
prediction = best.program.function(test_input)
```

### Load ARC Task

```python
import json

# Load task
with open('ARC-AGI/data/training/task.json', 'r') as f:
    task_data = json.load(f)

# Extract training pairs
train_pairs = [(np.array(ex['input']), np.array(ex['output']))
               for ex in task_data['train']]
test_input = np.array(task_data['test'][0]['input'])

# Solve
hypotheses = solver._generate_hypotheses(train_pairs, test_input)
```

## Key Features

### 1. Conditional Logic

Transforms based on object properties:
```python
IF size > median THEN rotate_90
ELSE keep

IF near_edge AND color=red THEN extend_to_edge
ELSE recolor(blue)
```

### 2. Action Learning (Phase 6)

Learns from training data instead of trying all possibilities:
- Detects which rotations/reflections occur
- Detects color mappings
- Prioritizes detected actions (2.5× boost)
- Still tries undetected actions (0.3× priority)

### 3. Multi-Stage Pipelines (Phase 7)

Chains transformations for sequential reasoning:
```python
Stage 1: IF size > median THEN recolor(1 → 2)
Stage 2: Rotate 90° clockwise
```

Achieved **first exact solves** (tasks 25ff71a9, 3c9b0459)!

### 4. Confidence-Based Prioritization

All hypotheses generated with priority scores:
- Detected actions: 1.5-2.5× boost
- Composite actions: 3.0× boost (highly expressive)
- Pipelines: 2.5× boost (sequential reasoning)
- Undetected: 0.3× (still considered)

## Test Scripts

```bash
# Run Phase 7 evaluation (100 tasks)
python test_phase7_pipelines.py

# Run Phase 6.1 evaluation
python test_phase6_1_solver.py

# Quick 100-task benchmark
python test_100_tasks.py
```

## Performance Analysis

### Accuracy Distribution (100 tasks)

- **Perfect (100%)**: 2 tasks (2%)
- **Near-perfect (95-99%)**: 11 tasks (11%)
- **High (80-94%)**: 38 tasks (38%)
- **Medium (50-79%)**: 19 tasks (19%)
- **Low (<50%)**: 30 tasks (30%)

### Bimodal Pattern

- **Success mode**: 50% of tasks achieve 78%+ accuracy
- **Failure mode**: 28% require different approaches
- **Median**: 78.3% (very high for successful tasks)

### What Works Well

✅ Conditional geometric transformations
✅ Color-based conditionals
✅ Spatial reasoning (near edge, alignment, etc.)
✅ Composite patterns (multiple conditions)
✅ Sequential transformations (via pipelines)

### What Doesn't Work

❌ Abstract pattern completion
❌ Arithmetic/counting operations
❌ Complex spatial constraints
❌ Tasks requiring domain knowledge

## Implementation Details

### Hypothesis Generation Priority

1. **Validated conditionals** (highest priority)
2. **Nested conditionals** (AND/OR/NOT logic)
3. **Composite actions** (geometric + conditionals)
4. **Multi-stage pipelines** (sequential reasoning)
5. **Spatial variations**
6. **Simple transforms** (baseline)

### Validation Strategy

- **Threshold**: 0.15 (optimal from Phase 4 testing)
- **Method**: Test on all training pairs
- **Scoring**: Exact match = 1.0, partial = proportion correct
- **Filtering**: Only keep hypotheses above threshold

### Optimization

- **Fast**: 0.24s per task average
- **Efficient**: Generates ~36 hypotheses per task
- **Discriminative**: More hypotheses for pattern-rich tasks
- **Adaptive**: Learns which actions to prioritize

## Theoretical Foundation

Built on active inference and curiosity-driven learning principles:

- **Bayesian belief updating** over hypothesis space
- **Free energy minimization** for coherent explanations
- **Curiosity signals** guide exploration
- **Hierarchical reasoning** (strategic/tactical/operational)

See detailed theoretical foundations in the original README sections above.

## Module Structure

```
arc_curiosity_solver/
├── core/
│   ├── action_inference.py          # Phase 6: Action learning
│   ├── pipeline_transform.py        # Phase 7: Multi-stage pipelines
│   ├── object_reasoning.py          # Object detection
│   ├── conditional_pattern_inference.py  # Pattern detection
│   └── improved_conditional_inference.py # Enhanced validation
├── transformations/
│   ├── conditional_transforms.py    # IF-THEN-ELSE logic
│   ├── nested_conditionals.py       # AND/OR/NOT composition
│   └── arc_primitives.py            # Basic transformations
├── solver_conditional.py            # Main solver (all phases)
└── solver_diverse.py                # Parent solver
```

## Citation

```bibtex
@software{arc_conditional_solver,
  title = {ARC Conditional Reasoning Solver},
  year = {2025},
  note = {Multi-phase conditional transformation system for ARC-AGI}
}
```

## License

MIT License

## Acknowledgments

- **ARC Challenge**: François Chollet
- **Active Inference Framework**: Karl Friston
- **Curiosity Research**: Oudeyer, Kaplan, et al.
