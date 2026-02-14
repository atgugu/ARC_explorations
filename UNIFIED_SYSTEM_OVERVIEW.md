# ARC Active Inference Solver - Unified System Overview

> **Disclaimer**: This document describes theoretical explorations and experimental
> implementations. The ideas presented here have not been empirically validated on
> the full ARC-AGI benchmark and require further testing to assess their practical
> effectiveness.

## Overview

This project developed a unified ARC-AGI solving system that combines five
theoretical frameworks from this repository into a single architecture based on
Active Inference.

**Location**: `unified_solver/`

## The Unifying Idea

Instead of implementing five separate systems, this project uses Active Inference
(Bayesian belief updating) as a common framework that connects all five approaches:

```
┌─────────────────────────────────────────────┐
│        Active Inference (Core Engine)        │
│     Bayesian Belief Updating: P(h|data)     │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
   ┌───▼────┐      ┌────▼───┐
   │Curiosity│      │Stability│
   │ Signals │      │ Filter  │
   └───┬────┘      └────┬───┘
       │                │
       └───────┬────────┘
               │
        ┌──────▼──────┐
        │  Workspace  │
        │ Controller  │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │   Program   │
        │  Synthesis  │
        └─────────────┘
```

## Frameworks Combined

### 1. Curiosity-Driven Neurosymbolic Framework
   Provides: Information gain, epistemic uncertainty, learning progress

### 2. Global Workspace Theory
   Provides: Limited capacity attention, hypothesis broadcasting

### 3. Graph Pendulum / Dynamical Systems
   Provides: Stability analysis, chaos filtering, basin discovery

### 4. Probabilistic Program Spaces
   Provides: Continuous belief dynamics, information geometry

### 5. Generative Task Discovery
   Provides: Typed DSL, program synthesis, compositional transformations

## Design Goals

- **Two predictions per task**: The solver is designed to always produce exactly 2 candidate predictions
- **Learning during inference**: Uses active inference to update beliefs without a separate training phase
- **Curiosity-driven exploration**: Information gain guides hypothesis search
- **Stability-aware selection**: Filters inconsistent or unreliable hypotheses
- **Interpretable reasoning**: Symbolic programs rather than opaque models
- **Minimal dependencies**: Implemented in approximately 1,700 lines of Python with NumPy

## System Architecture

### Core Process

```python
# 1. Perceive patterns from training examples
features = perception.perceive(training_examples)

# 2. Generate hypotheses from DSL
hypotheses = generator.generate(features)

# 3. Initialize beliefs
belief = P(h) ∝ exp(-complexity(h))

# 4. Active Inference Loop (for each training example)
for input, output in training_examples:
    # Bayesian update
    P(h | data) ∝ P(output | h, input) × P(h)

    # Compute curiosity signals
    information_gain = KL(P_new || P_old)
    epistemic_uncertainty = H[P(h)]

    # Assess stability
    stability(h) = consistency_across_examples(h)

    # Workspace selection
    workspace = top_k(hypotheses, by=P(h)×curiosity×stability)

# 5. Final selection
score(h) = P(h | all_data) × stability(h)
predictions = top_2_hypotheses.apply(test_input)
```

## Implementation

### File Structure

```
unified_solver/
├── arc_active_inference_solver.py  # Core implementation (1,100 lines)
│   ├── PerceptionModule           # Feature extraction
│   ├── HypothesisGenerator         # DSL-based program synthesis
│   ├── ActiveInferenceEngine       # Bayesian belief updating
│   ├── StabilityFilter             # Robustness testing
│   └── ARCActiveInferenceSolver    # Main solver
│
├── arc_loader.py                   # Data loading & evaluation (350 lines)
│   ├── ARCDataLoader              # Load tasks from JSON
│   └── ARCEvaluator               # Evaluation metrics
│
├── examples.py                     # Demonstrations (250 lines)
│   ├── 8 diverse examples
│   ├── Active inference demo
│   └── Evaluation suite
│
└── Documentation (3 files)
    ├── README.md                   # User guide
    ├── DESIGN.md                   # Design document
    └── IMPLEMENTATION_SUMMARY.md   # Implementation summary
```

## Quick Start

### Installation

```bash
cd ARC_explorations/unified_solver
pip install numpy
```

### Basic Usage

```python
from arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid

# Create task
task = ARCTask(
    train_pairs=[
        (Grid([[1,2],[3,4]]), Grid([[2,1],[4,3]])),  # flip vertical
        (Grid([[5,6],[7,8]]), Grid([[6,5],[8,7]])),
    ],
    test_input=Grid([[9,0],[1,2]])
)

# Solve (returns 2 predictions)
solver = ARCActiveInferenceSolver()
predictions = solver.solve(task, verbose=True)

print("Prediction 1:", predictions[0].data)
print("Prediction 2:", predictions[1].data)
```

### Run Examples

```bash
# Run specific example
python examples.py 1    # Flip vertical
python examples.py 7    # Active inference demo
python examples.py 9    # Evaluation suite

# Run all examples
python examples.py a
```

## Mathematical Foundation

**Active Inference (Bayesian Framework)**:
```
Prior:      P(h) = exp(-λ·complexity(h)) / Z
Likelihood: P(y|h,x) = exp(accuracy(h(x),y) / T)
Posterior:  P(h|x,y) ∝ P(y|h,x) · P(h)
```

**Curiosity Signals**:
```
Information Gain:         IG(h) = P_t(h)·log(P_t(h)/P_{t-1}(h))
Epistemic Uncertainty:    EU = -Σ P(h)log P(h)
Learning Progress:        LP = H[P_{t-1}] - H[P_t]
```

**Stability Metric**:
```
stability(h) = mean_accuracy(h) · exp(-std_accuracy(h))
```

**Final Selection**:
```
score(h) = P(h|data) × stability(h)
top_2 = argmax_{h₁≠h₂} score(h)
```

## Key Ideas

### 1. Unification Through Abstraction
- Single principle (active inference) instead of five separate systems
- Curiosity, stability, and attention arise as components of the inference loop

### 2. Guaranteed Outputs
- Designed to always produce exactly 2 predictions
- Graceful degradation when no strong hypothesis is found

### 3. Learning During Inference
- No separate training phase
- Few-shot learning from 2-5 examples

### 4. Interpretability
- Symbolic programs (DSL-based)
- Reasoning traces visible
- Probability distributions explicit

## Computational Complexity

- **Time**: O(h*n) where h=hypotheses, n=training examples
- **Space**: O(h + n*grid_size)
- **Typical**: ~1-10 seconds per task on CPU

### DSL Coverage

- **50+ primitives**: Geometric, color, morphological, object-based, spatial
- **Compositional**: Can combine primitives
- **Extensible**: New transformations can be added

## Future Directions

### Near-Term
- [ ] Enhanced DSL primitives (path-based, graph operations)
- [ ] Neural object detection
- [ ] Parallel hypothesis evaluation
- [ ] GPU acceleration

### Medium-Term
- [ ] Meta-learning across tasks
- [ ] Self-curriculum generation
- [ ] Hierarchical composition
- [ ] Learned primitives

### Long-Term
- [ ] Neural-symbolic hybrid
- [ ] Causal reasoning
- [ ] Interactive querying
- [ ] Human-in-the-loop

## Documentation

- **README.md**: User guide and API reference
- **DESIGN.md**: Design document with mathematical foundations
- **IMPLEMENTATION_SUMMARY.md**: Implementation details
- **This file**: High-level overview

## Repository Structure

This unified system is part of the ARC_explorations repository:

```
ARC_explorations/
├── docs/theory/          # Theoretical frameworks
├── solvers/              # Modular experimental solvers
└── unified_solver/       # Unified implementation
```

## Central Idea

The main theoretical contribution of this project is exploring whether Active
Inference can serve as a unifying principle for diverse cognitive frameworks
applied to abstract reasoning. By framing the problem as Bayesian belief updating,
curiosity signals, stability analysis, attention mechanisms, and program synthesis
can be expressed as components of a single process.

This remains a hypothesis that requires empirical validation on the full ARC-AGI
benchmark.

## Acknowledgments

This work builds upon five theoretical frameworks developed in this repository.
The unified system explores how these frameworks might relate as different
perspectives on a single underlying process.
