# ARC Curiosity-Driven Active Inference Solver

A comprehensive implementation of curiosity-driven active inference for solving ARC-AGI tasks, based on the theoretical frameworks in `/ARC_Curiosity`.

## Overview

This system combines:
- **Curiosity-driven exploration** - Multiple information-theoretic signals guide exploration
- **Belief dynamics** - Continuous evolution of beliefs over hypothesis space
- **Active inference** - Learning during inference via free energy minimization
- **Hierarchical reasoning** - Three-tier architecture (Generator, Workspace, Navigator)
- **Always produces 2 predictions** - Ensures diverse hypothesis coverage

## Architecture

### Core Components

#### 1. Curiosity Signals (`curiosity/signals.py`)

Implements all curiosity metrics from the theoretical framework:

- **Bayesian Surprise**: `KL[p(θ|D∪{e}) || p(θ|D)]` - Measures belief update magnitude
- **Epistemic Uncertainty**: `Var_epistemic[score(h)]` - Model disagreement/uncertainty
- **Learning Progress**: `LP(t) = m(t) - m(t-Δ)` - Recent performance improvement
- **Information Gain**: `IG = H[X] - E[H[X|o]]` - Expected uncertainty reduction
- **Empowerment**: `I(A; S' | S)` - Action-outcome controllability

```python
from arc_curiosity_solver.curiosity.signals import CuriositySignals

curiosity = CuriositySignals()
surprise = curiosity.bayesian_surprise(prior_params, posterior_params)
uncertainty = curiosity.epistemic_uncertainty(predictions)
progress = curiosity.learning_progress(current_performance)
info_gain = curiosity.information_gain(prior_entropy, expected_posterior_entropy)
```

#### 2. Belief Dynamics (`belief_dynamics/belief_space.py`)

Probabilistic program space with continuous dynamics:

- **Bayesian updating**: `P(h|D_{t+1}) = P(e|h) · P(h|D_t) / P(e)`
- **Continuous flow**: `dP(h)/dt = P(h) · [log P(e|h) - ⟨log P(e|h')⟩]`
- **Hierarchical beliefs**: Multi-level (meta, compositional, primitive)
- **Information geometry**: KL divergence, entropy tracking

```python
from arc_curiosity_solver.belief_dynamics.belief_space import BeliefSpace, Hypothesis

space = BeliefSpace(hypotheses)
space.bayesian_update(evidence, likelihood_fn)
top_hyps = space.top_k_hypotheses(k=2)
```

#### 3. Active Inference Engine (`active_inference/engine.py`)

Implements active inference principles:

- **Free energy minimization**: `F = prediction_error + complexity`
- **Belief updating**: Gradient-based or Bayesian
- **Active sampling**: Select informative observations
- **Predictive coding**: Hierarchical prediction and error propagation

```python
from arc_curiosity_solver.active_inference.engine import ActiveInferenceEngine

engine = ActiveInferenceEngine(learning_rate=0.1)
state = engine.initialize_state(n_hypotheses=50)
updated_beliefs = engine.update_beliefs(current_beliefs, observation, predictions)
```

#### 4. Hierarchical Solver (`core/hierarchical_solver.py`)

Three-tier cognitive architecture:

**Generator (Strategic)**: What to explore?
- Task selection via multi-armed bandit (UCB)
- Schema family exploration
- Learning progress tracking per family

**Workspace (Tactical)**: Which hypotheses to consider?
- Working memory with capacity constraint (7±2)
- Hypothesis competition via priority queue
- Lateral inhibition between similar hypotheses

**Navigator (Operational)**: Where to explore?
- Basin curiosity: `C_basin = exp(-Var_stab) · Novelty · LP`
- Edge curiosity: `C_edge = ω₁·IG + ω₂·Surprise + ω₃·CoverageGap`
- Stability-aware exploration

```python
from arc_curiosity_solver.core.hierarchical_solver import HierarchicalSolver

solver = HierarchicalSolver(workspace_capacity=7)
task = solver.select_and_load_task(available_tasks)
solver.add_hypothesis_to_workspace(hyp, fit, curiosity, stability)
predictions = solver.get_top_predictions(k=2)
```

#### 5. ARC Transformations (`transformations/arc_primitives.py`)

Comprehensive transformation library:

**Spatial**: translate, rotate, reflect, scale
**Color**: recolor, swap, filter
**Logical**: AND, OR, XOR
**Topological**: fill, detect objects, bounding boxes
**Compositional**: Combine multiple primitives

```python
from arc_curiosity_solver.transformations.arc_primitives import (
    ARCPrimitives, TransformLibrary, CompositeTransform
)

library = TransformLibrary()
transform = library.get_transform('rotate_90')
composite = library.create_composite(['translate_right', 'rotate_90'])
```

## Usage

### Basic Solving

```python
from arc_curiosity_solver.solver import ARCCuriositySolver
import numpy as np

# Create solver
solver = ARCCuriositySolver(
    workspace_capacity=7,
    learning_rate=0.1,
    exploration_bonus=1.0
)

# Prepare task
train_pairs = [
    (input1, output1),
    (input2, output2),
]
test_input = np.array([[...]])

# Solve - always returns 2 predictions
pred1, pred2 = solver.solve(train_pairs, test_input, verbose=True)
```

### With Real ARC Tasks

```python
from arc_curiosity_solver.solver import load_arc_task

# Load from JSON
train_pairs, test_input = load_arc_task('path/to/task.json')

# Solve
pred1, pred2 = solver.solve(train_pairs, test_input)
```

## Key Features

### 1. Curiosity-Driven Exploration

The system uses multiple curiosity signals to guide exploration:
- Prioritizes **learnable novelty** over random exploration
- Balances exploitation (known good approaches) with exploration (new patterns)
- Adapts exploration strategy based on learning progress

### 2. Active Inference During Inference

Unlike static systems, this solver:
- **Updates beliefs dynamically** as it sees training examples
- **Learns from evidence** accumulated during the inference process
- **Minimizes free energy** to find coherent explanations

### 3. Hierarchical Reasoning

Three-tier architecture mirrors human cognition:
- **Strategic** (Generator): Long-term learning and curriculum
- **Tactical** (Workspace): Working memory and attention
- **Operational** (Navigator): Moment-to-moment exploration

### 4. Always Two Predictions

The system guarantees 2 predictions:
- **Prediction 1**: Highest belief hypothesis
- **Prediction 2**: Second highest belief hypothesis

This provides:
- Diversity in solutions
- Fallback if top prediction is wrong
- Insight into competing interpretations

## Test Results

```bash
python test_curiosity_solver.py
```

**Test Results (4/4 passed):**

| Task | Prediction 1 Accuracy | Prediction 2 Accuracy |
|------|----------------------|----------------------|
| Translate Right | 100.0% | 77.8% |
| Rotate 90 | 100.0% | 75.0% |
| Reflect Horizontal | 100.0% | 100.0% |
| Scale 2x | 100.0% | 75.0% |

## Implementation Details

### Belief Updating Algorithm

1. **Generate hypotheses**: Create candidate transformations
2. **Initialize beliefs**: Uniform distribution over hypotheses
3. **For each training example**:
   - Apply each hypothesis to input
   - Compute likelihood based on match with output
   - Bayesian update: `P(h|D) ∝ P(D|h) · P(h)`
   - Track surprise and learning progress
4. **Select top-k**: Return k highest belief hypotheses

### Curiosity Integration

Curiosity influences:
- **Task selection** (Generator): UCB with learning progress bonus
- **Hypothesis evaluation** (Workspace): Priority combines fit, curiosity, stability
- **Exploration strategy** (Navigator): Balance stable vs. novel regions

### Computational Complexity

- **Hypothesis generation**: O(n) for n transforms
- **Belief update per example**: O(n·m) for n hypotheses, m grid cells
- **Top-k selection**: O(n log k)
- **Overall per task**: O(n·m·t) for t training examples

Optimizations:
- Hypothesis pruning (remove low-belief)
- Early stopping (high-confidence convergence)
- Cached transformations

## Theoretical Foundations

Based on papers in `/ARC_Curiosity`:

1. **ARC_Curiosity_Blueprint.md**
   - Core curiosity signals and decision bonuses
   - UCB task selection, hypothesis scoring

2. **ARC_Curiosity_Blueprint_Enhanced.md**
   - Cognitive grounding and human alignment
   - Information-theoretic formalization

3. **Probabilistic_Program_Spaces_ARC.md**
   - Continuous dynamics in belief space
   - Information geometry and flow equations

4. **Temporal_Solver_Dynamics_ARC.md**
   - Cognitive state evolution
   - Attention and working memory dynamics

## Extensions and Future Work

### Planned Extensions

1. **Neural ODEs**: Replace discrete updates with continuous-time dynamics
2. **Meta-learning**: Learn curiosity weights from experience
3. **Multi-task transfer**: Share learned patterns across tasks
4. **Compositional discovery**: Automatic composition of primitives

### Research Directions

1. **Optimal curiosity**: Provably optimal curiosity functions
2. **Human alignment**: Match human problem-solving trajectories
3. **Emergent strategies**: Discovery of novel solution patterns
4. **Scalability**: Handle larger, more complex ARC tasks

## Citation

If you use this solver in your research, please cite:

```bibtex
@software{arc_curiosity_solver,
  title = {ARC Curiosity-Driven Active Inference Solver},
  year = {2025},
  note = {Implementation of curiosity-driven active inference for ARC-AGI}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **ARC Challenge**: François Chollet's Abstraction and Reasoning Corpus
- **Active Inference**: Karl Friston's free energy principle
- **Curiosity Research**: Oudeyer, Kaplan, Kidd, and colleagues
