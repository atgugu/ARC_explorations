# ARC-AGI Challenge Explorations

Experimental approaches to solving the [ARC-AGI Challenge](https://arcprize.org/) using hybrid neurosymbolic systems.

## Current System

**Unified Solver** (`unified_solver/`) - Program synthesis with Active Inference

**Latest Results**: 1.0% success rate (2/200 evaluation tasks)

**Journey**:
- Baseline (primitives): 0.5%
- Phase 1 (composition): 1.0% ✓
- Phase 2 (conditionals/loops): 1.0% (no improvement)
- Phase 3 (parameter inference): 1.0% (no improvement)

**Next**: LLM integration for semantic understanding (target: 15-30%)

## Quick Start

```bash
cd unified_solver
python arc_program_solver.py  # Run solver
python test_phase3_200.py     # Evaluate on 200 tasks
```

## Documentation

- `unified_solver/COMPLETE_JOURNEY.md` - Full 3-phase development journey
- `unified_solver/LLM_INTEGRATION_DESIGN.md` - Next phase technical design
- `unified_solver/PHASE3_RESULTS.md` - Latest evaluation results

## Key Insight

ARC requires **semantic understanding**, not just syntactic composition. LLM-guided program synthesis is the promising next direction.

## Repository Structure

```
ARC_explorations/
├── unified_solver/          # Main solver (Active Inference + Program Synthesis)
├── temp_arc_data/          # ARC-AGI dataset (400 training, 400 evaluation)
└── data/                   # Symlink to dataset
```
