# ARC Program Synthesis Solver

A hybrid system combining **Program Synthesis** with **Active Inference** for solving ARC-AGI tasks.

## Current Performance

**Success Rate**: 1.0% (2/200 evaluation tasks)

**Solved Tasks**: `60c09cac`, `68b67ca3`

## Architecture

```
Input Task
    ↓
[1] Perception → Extract features (objects, colors, symmetries)
    ↓
[2] Program Synthesis → Generate compositional programs (100+ candidates)
    ↓
[3] Active Inference → Bayesian belief updating over programs
    ↓
[4] Stability Filter → Remove unstable/chaotic programs
    ↓
[5] Workspace → Select top-20 programs (attention mechanism)
    ↓
[6] Ranking → Score by posterior × stability
    ↓
[7] Selection → Choose diverse top-2 predictions
    ↓
Output: 2 Predictions
```

## Key Features

**Compositional DSL** (90+ operations)
- Geometric: rotate, flip, transpose
- Color: replace, recolor
- Objects: detect, extract, transform
- Spatial: zoom, tile, crop
- Advanced: conditionals, loops, patterns

**Active Inference**
- Bayesian belief updating
- Learns during inference (no pre-training)
- Few-shot learning (2-5 examples)

**Stability-Aware Selection**
- Filters chaotic/unstable programs
- Prefers consistent solutions

**Always Returns 2 Predictions**
- Top-1: highest posterior × stability
- Top-2: next best alternative

## Development Journey

### Phase 1: Compositional Synthesis ✓
- Added compositional program synthesis (3 levels)
- Implemented size inference and auto-resizing
- **Result**: 0.5% → 1.0% (+100% relative improvement)

### Phase 2: Conditionals + Loops ✗
- Added 9 predicates (has_border, is_symmetric, etc.)
- Implemented conditionals and for-each loops
- Added pattern operations (tile, fill_interior)
- **Result**: 1.0% → 1.0% (no improvement)
- **Lesson**: Generic operations don't help without matching task semantics

### Phase 3: Parameter Inference ✗
- Implemented color mapping, scale factor, rotation inference
- Generated task-specific programs from learned parameters
- **Result**: 1.0% → 1.0% (no improvement)
- **Root Cause**: Existing size inference made learned parameters redundant
- **Lesson**: Wrong abstraction level; ARC needs semantic understanding

## Next Direction: LLM Integration

**Phase 4 Plan**: Hybrid LLM-Guided Synthesis

**Approach**:
1. LLM analyzes task semantics: "This is 2× zoom with color swap"
2. Program synthesis generates targeted programs based on LLM guidance
3. Active Inference evaluates and selects best programs

**Expected Impact**: 1% → 15-30% success rate

**See**: `LLM_INTEGRATION_DESIGN.md` for technical specification

## Usage

```python
from arc_program_solver import ARCProgramSolver
from arc_loader import ARCDataLoader

# Load task
task = ARCDataLoader.load_task_from_file("path/to/task.json")

# Solve
solver = ARCProgramSolver(verbose=True)
predictions = solver.solve(task)

print("Prediction 1:", predictions[0].data)
print("Prediction 2:", predictions[1].data)
```

## Evaluation

```bash
# Test on 200 evaluation tasks
python test_phase3_200.py

# Test single task with verbose output
python test_single_task.py
```

## Key Files

**Core Implementation**:
- `arc_program_synthesis.py` (1100+ lines) - Program synthesis engine
- `arc_program_solver.py` (300 lines) - Main solver with Active Inference
- `arc_active_inference_solver.py` (800 lines) - Original baseline solver
- `parameter_inference.py` (354 lines) - Parameter learning (Phase 3)

**Documentation**:
- `COMPLETE_JOURNEY.md` - Full development journey (Phases 1-3)
- `LLM_INTEGRATION_DESIGN.md` - Phase 4 technical design
- `PHASE3_RESULTS.md` - Latest evaluation and analysis
- `PROGRAM_SYNTHESIS_RESULTS.md` - Phase 1 results
- `PARAMETER_INFERENCE_PLAN.md` - Phase 3 implementation plan

**Evaluation**:
- `test_program_synthesis_200.py` - Phase 1 evaluation
- `test_phase2_200.py` - Phase 2 evaluation
- `test_phase3_200.py` - Phase 3 evaluation
- `test_single_task.py` - Single task debugging

## Configuration

```python
solver = ARCProgramSolver(
    workspace_capacity=20,      # Top-k attention
    n_perturbations=5,          # Stability testing
    max_synthesis_depth=3,      # Composition depth (0-3)
    max_programs=150,           # Max programs generated
    verbose=True                # Debug output
)
```

## Current Limitations

**Success Rate**: Only 1.0% (2/200 tasks)

**Main Issues**:
1. **Missing Semantic Understanding**: System tries operations blindly
2. **Wrong Hypothesis Space**: Most generated programs irrelevant
3. **Complexity Penalty Too Strong**: Favors simple programs over correct ones
4. **Limited DSL Coverage**: Missing arithmetic, advanced spatial reasoning

**Why Phases 2-3 Failed**:
- Generic operations (conditionals, loops) don't match task semantics
- Parameter learning addresses wrong abstraction level
- Size inference already handles scaling (made learned params redundant)

## Key Insights

1. **Composition Helps**: Phase 1 doubled success rate (0.5% → 1.0%)
2. **Semantics Matter**: Adding more operations doesn't help if they don't match tasks
3. **Redundancy Check**: Always check for existing capabilities before adding new ones
4. **ARC Requires Understanding**: Syntactic composition insufficient; need semantic reasoning

## Next Steps

**Phase 4a** (1 day): Minimal LLM integration
- Basic task analysis with Claude/GPT-4
- Keyword-based program prioritization
- Test on 10 diverse tasks

**Phase 4b** (2 days): Full LLM-guided synthesis
- Structured semantic analysis
- Priority-based belief initialization
- Full 200-task evaluation

**Target**: 15-30% success rate with LLM integration

## Citation

```bibtex
@software{arc_program_synthesis_solver,
  title={ARC Program Synthesis Solver: Active Inference with Compositional Programs},
  year={2025},
  url={https://github.com/your-repo/ARC_explorations}
}
```

## License

Part of ARC Explorations repository. See main repository for license.

---

**Status**: Phase 3 complete (1.0% success rate) | **Next**: Phase 4 (LLM integration)
