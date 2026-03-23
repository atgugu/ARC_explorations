# ARC Active Inference Solver

A neurosymbolic system for solving [ARC-AGI](https://arcprize.org/) tasks using Active Inference and Global Workspace Theory. The solver combines a 65-primitive DSL with Bayesian hypothesis generation, curiosity-driven search, and stability filtering to reason about abstract visual patterns — no pre-training required.

## How It Works

```
Input/Output Pairs → Pattern Analysis → Hypothesis Generation → Bayesian Ranking → Prediction
```

1. **Pattern Analysis** — Detects transformation patterns (rotation, tiling, color mapping, symmetry, etc.) across training pairs
2. **Hypothesis Generation** — Proposes candidate programs from a composable DSL of 65 primitives
3. **Bayesian Scoring** — Ranks hypotheses by how well they explain all training examples
4. **Composition Search** — Tries multi-step compositions (2-step and 3-step pipelines)
5. **Prediction** — Applies the best-scoring hypothesis to the test input

## Quick Start

```bash
pip install numpy scipy

# Run unit tests
python -m pytest tests/test_core_primitives.py -v

# Run a demo of DSL primitives
python demo_primitives.py

# Test on real ARC tasks (requires ARC dataset)
mkdir -p data && cd data && git clone https://github.com/fchollet/ARC-AGI.git && cd ..
python solve_arc_improved.py
```

### Basic Usage

```python
from src.hypothesis_proposer import HypothesisProposer, PatternAnalyzer
import numpy as np

# Analyze a pattern
analyzer = PatternAnalyzer()
input_grid = np.array([[1, 0], [0, 0]])
output_grid = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])

patterns = analyzer.analyze_pair(input_grid, output_grid)
for p in patterns:
    print(f"{p.name}: {p.confidence:.2f} — {p.description}")
```

```python
from src.dsl.core_primitives import *
import numpy as np

# Use DSL primitives directly
grid = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

rotated = rotate(grid, 90)                    # Rotate 90°
reflected = reflect(grid, Axis.HORIZONTAL)     # Horizontal mirror
objects = select_by_color(grid, 1)             # Extract color-1 objects
tiled = tile(grid, 2, 2)                      # Tile 2x2
```

## Repository Structure

```
├── src/
│   ├── dsl/
│   │   ├── core_primitives.py        # 65 DSL primitives (3000+ lines)
│   │   └── primitives.py             # Additional primitive definitions
│   └── hypothesis_proposer.py        # Pattern analysis & hypothesis generation
│
├── tests/
│   ├── test_core_primitives.py       # Unit tests for all primitives
│   ├── test_all_65_primitives.py     # Comprehensive primitive coverage
│   ├── test_hypothesis_proposer.py   # Hypothesis proposer tests
│   └── test_real_arc.py             # Tests on real ARC tasks
│
├── docs/
│   ├── DSL_PRIMITIVES.md            # Complete DSL reference (65 primitives)
│   └── theory/                      # Theoretical foundations
│       ├── curiosity/               # Curiosity-driven exploration
│       ├── dynamical_systems/       # Stability analysis (Graph Pendulum)
│       ├── generative/              # Program synthesis & task generation
│       └── workspace/               # Global Workspace Theory
│
├── configs/default.yaml             # Solver configuration
├── demo_primitives.py               # Interactive DSL demo
├── solve_arc_improved.py            # ARC solver script
├── setup.py
└── requirements.txt
```

## DSL Primitives

The solver uses 65 composable primitives organized into 8 categories:

| Category | Count | Examples |
|----------|-------|---------|
| **Selection & Filtering** | 12 | `select_by_color`, `select_largest`, `select_by_shape`, `select_by_position` |
| **Spatial Transforms** | 10 | `rotate`, `reflect`, `translate`, `scale`, `gravity`, `align` |
| **Color Operations** | 8 | `recolor`, `swap_colors`, `map_colors`, `most_common_color` |
| **Pattern Operations** | 9 | `tile`, `repeat`, `mirror_extend`, `copy_to_positions` |
| **Grid Operations** | 7 | `overlay`, `crop`, `crop_to_content`, `resize`, `pad` |
| **Topological** | 6 | `fill_holes`, `grow`, `shrink`, `flood_fill`, `connected_components` |
| **Arithmetic & Logic** | 5 | `count`, `measure`, `sort_objects`, `grid_and`, `grid_or` |
| **Line & Path** | 8 | `draw_line`, `draw_rectangle`, `extend_line`, `trace_boundary`, `connect` |

See [docs/DSL_PRIMITIVES.md](docs/DSL_PRIMITIVES.md) for the complete reference.

## Pattern Detection

The hypothesis proposer detects 18 pattern types from input-output pairs:

- **Geometric**: rotation, reflection, tiling, scaling, cropping
- **Color**: color replacement, color mapping, inversion, color-by-size
- **Spatial**: translation, alignment (horizontal/vertical), symmetrization
- **Morphological**: fill holes, fill background, grow/shrink
- **Line**: vertical/horizontal line drawing, line extension
- **Cleanup**: noise removal (small object filtering)

Detected patterns are converted into executable programs and ranked by confidence. Multi-step compositions allow solving tasks that require chaining multiple transformations.

## Theoretical Foundations

The solver is grounded in five cognitive-inspired frameworks (see `docs/theory/`):

1. **Global Workspace Theory** — Limited-capacity attention mechanism that selects the most promising hypotheses for evaluation. Based on Baars' cognitive architecture.

2. **Active Inference** — Bayesian belief updating over transformation hypotheses. Each training example refines the posterior distribution: `P(h|D) ∝ P(D|h) · P(h)`.

3. **Curiosity-Driven Exploration** — Information gain, epistemic uncertainty, and learning progress guide which hypotheses to explore, avoiding redundant search.

4. **Stability Filtering** — Hypotheses are tested for robustness across perturbations. Unstable/chaotic candidates are filtered using Lyapunov-inspired indicators.

5. **Program Synthesis** — A typed DSL with MDL (Minimum Description Length) priors generates candidate programs. Simpler compositions are preferred (Occam's razor).

## Testing

```bash
# Core primitives (unit tests)
python -m pytest tests/test_core_primitives.py -v

# All 65 primitives
python tests/test_all_65_primitives.py

# Hypothesis proposer
python tests/test_hypothesis_proposer.py

# Real ARC tasks (needs ARC dataset in data/)
python tests/test_real_arc.py
```

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.10.0

## License

MIT License — see [LICENSE](LICENSE).

## References

- Chollet, F. (2019). [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547). arXiv.
- Friston, K. (2010). [The free-energy principle: a unified brain theory?](https://doi.org/10.1038/nrn2787) Nature Reviews Neuroscience.
- Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.
- ARC-AGI Challenge: https://arcprize.org/
