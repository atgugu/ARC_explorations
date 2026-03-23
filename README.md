# ARC Graph Pendulum

A stability-aware dynamical system for solving [ARC-AGI](https://arcprize.org/) (Abstraction and Reasoning Corpus) tasks through graph-based program synthesis.

## Overview

ARC Graph Pendulum treats abstract reasoning as a controlled dynamical system. It builds a graph of computational nodes (feature extractors, analyzers, synthesizers) connected by edges with geometric properties (angles, distances, utility scores). A stability-aware beam search navigates this graph to find programs that transform input grids into output grids.

The system achieves **19.6% solve rate on ARC training** (9/46 tasks) and **1.7% on evaluation** (2/117 tasks) through a pipeline of increasingly sophisticated synthesis strategies.

## Architecture

```
ARC Task (input/output grid pairs)
         |
         v
  ┌──────────────────────────────────────────┐
  │           Feature Extraction              │
  │  Color histograms, object detection,      │
  │  symmetry, periodicity, shape analysis    │
  └──────────────┬───────────────────────────┘
                 v
  ┌──────────────────────────────────────────┐
  │        Differential Analysis              │
  │  What changed between input and output?   │
  │  Geometric, color, structural transforms  │
  └──────────────┬───────────────────────────┘
                 v
  ┌──────────────────────────────────────────┐
  │        Rule Inference & Synthesis         │
  │  Infer transformation rules, generate     │
  │  candidate programs, verify on training   │
  └──────────────┬───────────────────────────┘
                 v
  ┌──────────────────────────────────────────┐
  │     Stability-Aware Beam Search           │
  │  Navigate the graph, penalize unstable    │
  │  paths, prefer generalizable solutions    │
  └──────────────┬───────────────────────────┘
                 v
         Best Program
         (applied to test inputs)
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **Node System** (`core/node.py`) | Typed computational units with caching and contracts |
| **Edge System** (`core/edge.py`) | Directed edges with geometric properties and causal credit |
| **Behavior Vectors** (`core/behavior.py`) | Profile nodes on probe cases to compute similarity |
| **Stability Meter** (`core/stability.py`) | Lyapunov-inspired sensitivity and variance analysis |
| **Controller** (`core/controller.py`) | Stability-aware beam search with trajectory tracking |
| **Trajectory & Basin** (`core/trajectory.py`, `core/basin.py`) | Solution paths and stable attractor regions |

### Solver Versions

The solver evolved through iterative refinement. Each version inherits from the previous and adds new capabilities:

| Version | Strategy | Key Innovation | Training Solve Rate |
|---------|----------|----------------|---------------------|
| **V3** | Example-driven rule inference | Differential analysis + targeted synthesis | 2.2% |
| **V3+** | Enhanced pattern detection | Pattern tiling, extraction, object operations | 6.5% |
| **V4** | Shape transformations | Object extraction, cropping, region selection | 17.4% |
| **V5** | Compositional transforms | 2-step and 3-step program composition | 19.6% |
| **V6** | Meta-pattern learning | Conditional rules, test-time adaptation | 19.6% |
| **V7** | Execution refinement | Post-processing corrections for near-misses | 19.6% |
| **V8** | Ensemble voting | Weighted majority voting across predictions | 19.6% |
| **V9** | Extended primitives | 20 new transformation primitives | 19.6% |
| **V10** | Constraint-based synthesis | Formal constraint extraction to guide search | 19.6% |

**Key finding:** V5 achieved the performance ceiling at 19.6%. Versions V6-V10 explored five distinct improvement strategies (meta-patterns, execution refinement, ensemble voting, primitive expansion, constraint-based synthesis) -- all yielded zero improvement, establishing that the symbolic synthesis approach has reached a fundamental architectural limit.

## Installation

```bash
pip install numpy scipy scikit-learn requests
```

Optional (for landscape visualization):
```bash
pip install umap-learn matplotlib
```

## Quick Start

```python
from arc_graph_pendulum.utils.arc_loader import ARCLoader
from arc_graph_pendulum.solver_v7 import ARCGraphPendulumSolverV7

# Download and load ARC tasks
loader = ARCLoader(cache_dir="./arc_data")
loader.download_dataset("training")
tasks = loader.load_all_tasks("training")

# Create solver (V7 is the recommended version)
solver = ARCGraphPendulumSolverV7(
    beam_width=5,
    use_stability=True,
    enable_refinement=True
)

# Solve a task
task = list(tasks.values())[0]
result = solver.evaluate_on_task(task, verbose=True)

print(f"Task: {result['task_id']}")
print(f"Solved: {result['solved']}")
print(f"IoU: {result['avg_score']:.4f}")
```

### Evaluate on multiple tasks

```python
results = solver.evaluate_multiple_tasks(
    list(tasks.values()),
    verbose=True
)

solved = sum(1 for r in results if r['solved'])
total = len(results)
print(f"Solved: {solved}/{total} ({100*solved/total:.1f}%)")
```

### Run from command line

```bash
cd arc_graph_pendulum
python solver.py          # Run base solver demo
python solver_v7.py       # Run V7 (recommended)
```

## Project Structure

```
arc_graph_pendulum/
├── solver.py                 # Base solver with graph infrastructure
├── solver_v3.py              # Example-driven rule inference
├── solver_v3_plus.py         # Enhanced differential analysis
├── solver_v4.py              # Shape transformation support
├── solver_v5.py              # Compositional transformations
├── solver_v6.py              # Meta-pattern learning
├── solver_v7.py              # Execution refinement (recommended)
├── solver_v8.py              # Ensemble voting (experimental)
├── solver_v9.py              # Extended primitives (experimental)
├── solver_v10.py             # Constraint-based synthesis (experimental)
│
├── core/                     # Graph infrastructure
│   ├── node.py               # Computational nodes with caching
│   ├── edge.py               # Edges with geometry and utility
│   ├── behavior.py           # Behavior vector computation
│   ├── stability.py          # Lyapunov-inspired stability analysis
│   ├── controller.py         # Stability-aware beam search
│   ├── trajectory.py         # Solution path tracking
│   ├── basin.py              # Stable attractor regions
│   ├── landscape.py          # UMAP-based landscape analytics
│   └── dsl.py                # Domain-specific language primitives
│
├── nodes/                    # Computational components
│   ├── extractors.py         # Feature extraction (color, objects, symmetry)
│   ├── differential_analyzer.py       # Input-output diff analysis
│   ├── rule_inferencer.py             # Transformation rule inference
│   ├── targeted_synthesizer.py        # Program synthesis
│   ├── shape_transformation_*.py      # Shape-changing operations
│   ├── compositional_*.py             # Multi-step composition
│   ├── execution_refiner.py           # Post-processing corrections
│   ├── critics.py                     # IoU scoring and failure analysis
│   └── ...                            # Additional analysis/synthesis nodes
│
└── utils/
    ├── arc_loader.py         # ARC dataset download and loading
    └── grid_utils.py         # Grid operations and IoU computation
```

## How It Works

### 1. Graph Construction

The solver builds a computational graph where:
- **Nodes** are typed functions (extractors, analyzers, synthesizers, critics)
- **Edges** connect compatible nodes with geometric properties derived from behavior vectors
- **Behavior vectors** profile each node on a bank of small test cases

### 2. Differential Analysis

For each training example pair (input, output), the system analyzes:
- **What changed**: Pixel-level differences, added/removed objects
- **How it changed**: Geometric transforms, color remappings, structural operations
- **Patterns across examples**: Consistent rules that generalize

### 3. Program Synthesis

Based on the analysis, the system generates candidate programs:
- **V3**: Single-step transformations (identity, geometric, color remap)
- **V4**: Shape-changing operations (extraction, cropping, selection)
- **V5**: Multi-step compositions (2-3 operations chained together)

Each candidate is verified against all training examples using IoU (Intersection over Union).

### 4. Stability-Aware Search

The beam search prefers stable solutions:
- **Sensitivity analysis**: How much does the output change with small input perturbations?
- **Variance penalties**: High-variance trajectories are penalized
- **Basin discovery**: Cluster similar trajectories to find reliable solution strategies

### 5. Best Program Selection

Programs are ranked by:
1. Training IoU (must be >= 0.99 to count as "solved")
2. Stability score (prefer low-sensitivity solutions)
3. Simplicity (prefer shorter programs)

The best program is applied to test inputs to generate predictions.

## Results

### Training Set (46 tasks)

| Version | Solved | Solve Rate |
|---------|--------|------------|
| V3 | 1 | 2.2% |
| V3+ | 3 | 6.5% |
| V4 | 8 | 17.4% |
| V5 | 9 | 19.6% |
| V7 | 9 | 19.6% |

### Evaluation Set (117 tasks)

| Version | Solved | Solve Rate |
|---------|--------|------------|
| V7 | 2 | 1.7% |

### Performance Distribution (V7, 117 evaluation tasks)

| IoU Range | Count | Description |
|-----------|-------|-------------|
| >= 0.99 | 2 | Solved |
| 0.95 - 0.99 | 12 | Near-miss (1-5% pixel errors) |
| 0.80 - 0.95 | 37 | High-quality (close but not solved) |
| 0.20 - 0.80 | 32 | Partial |
| < 0.20 | 34 | Failed |

## Key Insights

### What Works
- **Differential analysis** effectively identifies transformation types
- **Shape transformation detection** (V4) was the biggest single improvement (+11%)
- **Compositional synthesis** (V5) captures multi-step transformations
- **Stability-aware search** reliably selects generalizable programs

### What Doesn't Work (Established Through Systematic Experimentation)

Five independent strategies (V6-V10) all yielded zero improvement, providing convergent evidence:

1. **Meta-pattern learning (V6)**: ARC tasks don't exhibit parameter variation across training examples
2. **Execution refinement (V7)**: Errors are in program synthesis, not execution
3. **Ensemble voting (V8)**: The solver is deterministic, so voting provides no diversity
4. **Extended primitives (V9)**: Generic primitives can't match the specificity ARC tasks require
5. **Constraint-based synthesis (V10)**: Constraints either over-prune or under-constrain the search

### The Fundamental Limit

ARC tasks require **highly specific, ad-hoc transformations** that don't decompose into generic primitives or constraint-satisfying programs. Each task is essentially unique, requiring custom logic that symbolic synthesis alone cannot discover at scale.

## Dependencies

- **numpy** >= 1.21.0
- **scipy** >= 1.7.0
- **scikit-learn** >= 1.0.0
- **requests** >= 2.25.0

Optional:
- **umap-learn** >= 0.5.0 (landscape visualization)
- **matplotlib** >= 3.5.0 (plotting)

## License

MIT
