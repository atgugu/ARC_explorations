# ARC-AGI Challenge Explorations

Active-inference and conditional reasoning approaches for solving the [ARC-AGI Challenge](https://arcprize.org/).

## Current Performance

**57.75% average accuracy** on 100 ARC training tasks with **2% exact solve rate**.

### Phase Progression

| Phase | Accuracy | Exact Solves | Key Innovation |
|-------|----------|--------------|----------------|
| Baseline | 28% | 0% | Pattern matching |
| Phase 3 | 35% | 0% | Nested conditionals (AND/OR/NOT) |
| Phase 4 | 42% | 0% | Richer spatial predicates |
| Phase 5 | 55% | 0% | Geometric transformations |
| Phase 6.1 | 57.75% | 0% | Action learning + confidence |
| **Phase 7** | **57.75%** | **2%** ✅ | **Multi-stage pipelines** |

**Total improvement**: +29.75 percentage points from baseline.

## Solver Architecture

The **Conditional ARC Curiosity Solver** combines:
- **Conditional transformations** (IF-THEN-ELSE logic)
- **Spatial predicates** (near_edge, touching, symmetric, etc.)
- **Composite actions** (rotations, reflections, color swaps)
- **Action inference** (learns from training data)
- **Multi-stage pipelines** (sequential reasoning)

## Key Features

- ⚡ **Fast**: 0.24s per task
- 🎯 **Accurate**: 57.75% average accuracy
- 🔧 **Modular**: Extensible architecture
- 🧠 **Adaptive**: Learns actions from training data
- 🔗 **Sequential**: Multi-stage transformation pipelines

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on 100 tasks
python test_phase7_pipelines.py

# Test specific phase
python test_phase6_1_solver.py
```

## Repository Structure

```
arc_curiosity_solver/      # Main solver package
├── core/                  # Core reasoning modules
│   ├── action_inference.py       # Phase 6: Action learning
│   └── pipeline_transform.py     # Phase 7: Multi-stage pipelines
├── transformations/       # Transformation libraries
└── solver_conditional.py  # Main solver (all phases integrated)

PHASE*.md                  # Detailed phase documentation
test_*.py                  # Evaluation scripts
```

## Documentation

- **PHASE6_ACTION_LEARNING.md** - Action inference from training
- **PHASE6_1_OBJECT_AWARE_LEARNING.md** - Confidence prioritization
- **PHASE6_1_100TASK_EVALUATION.md** - Comprehensive 100-task results
- **PHASE7_MULTI_STAGE_PIPELINES.md** - Sequential reasoning breakthrough

## Highlights

### First Exact Solves (Phase 7)
- **Task 25ff71a9**: 100% via 2-stage pipeline
- **Task 3c9b0459**: 100% via 2-stage pipeline
- Both impossible for single-stage approaches (0% → 100%)

### High-Quality Results
- **49% of tasks**: ≥80% accuracy
- **11% of tasks**: ≥95% accuracy (near-perfect)
- **Median accuracy**: 78.3%

### Bimodal Performance
- **Success mode**: 50% of tasks achieve 78%+ accuracy
- **Failure mode**: 28% of tasks require different approaches
- **Clear pattern**: Either works very well or needs alternative strategy

## Future Directions

1. **Optimize pipelines**: Increase beam width, add caching
2. **3-stage pipelines**: Handle more complex sequential tasks
3. **Task classification**: Route to appropriate solver strategy
4. **Ensemble methods**: Combine multiple approaches

## About ARC-AGI

The Abstraction and Reasoning Corpus (ARC) is a benchmark for measuring general intelligence through visual reasoning tasks requiring pattern understanding and abstract thinking.

## License

MIT
