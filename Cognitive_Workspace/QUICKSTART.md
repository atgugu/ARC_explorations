# Quick Start Guide
## Getting Started with Cognitive Workspace Implementation

This guide will get you running your first ARC solver in < 30 minutes.

---

## Prerequisites

```bash
- Python 3.10+
- Git
- 4GB+ RAM
- (Optional) GPU for neural components
```

---

## Step 1: Environment Setup (5 min)

```bash
# Navigate to project
cd /home/user/ARC_explorations/Cognitive_Workspace

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 2: Download ARC Dataset (2 min)

```bash
# Create data directory
mkdir -p data
cd data

# Clone ARC dataset
git clone https://github.com/fchollet/ARC-AGI.git

# Verify download
ls ARC-AGI/data/training/*.json | wc -l  # Should show 400
```

---

## Step 3: Run Your First Example (5 min)

```bash
# Return to project root
cd ..

# Run the data loader test
python src/data/arc_loader.py

# You should see:
# ✓ Loaded 400 training tasks
# ✓ Loaded 400 evaluation tasks
# ✓ Sample task visualization saved to outputs/sample_task.png
```

---

## Step 4: Visualize a Task (5 min)

```bash
# Visualize a specific task
python scripts/visualize_task.py --task-id "00d62c1b"

# This will display:
# - Training input/output pairs
# - Test input (output to predict)
```

---

## Step 5: Test DSL Primitives (5 min)

```bash
# Run DSL tests
pytest tests/test_dsl.py -v

# Try executing a simple transformation
python examples/simple_dsl_example.py
```

Expected output:
```
Original grid:
[[0 0 1 0]
 [0 1 1 0]
 [0 0 1 0]
 [0 0 0 0]]

After rotate_90:
[[0 0 0 0]
 [1 1 1 0]
 [1 1 1 0]
 [0 0 0 0]]
```

---

## Step 6: Run Baseline Solver (10 min)

```bash
# Run on a simple task
python scripts/run_solver.py \
    --task-id "00d62c1b" \
    --mode heuristic \
    --max-iterations 50 \
    --verbose

# You should see:
# Iteration 1: Proposed 15 hypotheses
# Iteration 2: Best score: 0.75
# ...
# ✓ Solved! Program: [rotate_90, recolor(blue->red)]
```

---

## Understanding the Output

The solver will show you:

1. **Perception Results:**
   ```
   Found 3 objects:
   - Object 0: blue rectangle at (1,1), size=4
   - Object 1: red square at (5,5), size=1
   - Object 2: green line at (0,3), size=3
   ```

2. **Hypothesis Generation:**
   ```
   Generated 20 hypotheses:
   1. rotate_largest(90°) - score: 0.65
   2. recolor_by_size(size->color) - score: 0.82
   3. reflect_vertical() - score: 0.41
   ...
   ```

3. **Workspace Selection:**
   ```
   Workspace (k=5):
   ✓ recolor_by_size(size->color)
   ✓ copy_to_corners(largest)
   ✓ rotate_90 + translate(2,2)
   ✓ tile(object_0, 2x2)
   ✓ reflect_horizontal()
   ```

4. **Final Solution:**
   ```
   ✓ Task solved in 12 iterations
   Solution: [select_largest, rotate(90), copy_to(corners)]
   Accuracy: 100% on training pairs
   ```

---

## Development Workflow

### Run Tests
```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_perception.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Run Experiments
```bash
# Evaluate on multiple tasks
python scripts/evaluate.py \
    --num-tasks 50 \
    --difficulty easy \
    --output results/experiment_1.json

# Compare different configurations
python scripts/compare_configs.py \
    --configs configs/baseline.yaml configs/advanced.yaml
```

### Visualize Workspace Dynamics
```bash
# Generate workspace attention heatmaps
python scripts/visualize_workspace.py \
    --task-id "00d62c1b" \
    --output outputs/workspace_dynamics.gif
```

---

## Project Structure Overview

```
Cognitive_Workspace/
├── src/
│   ├── data/          # Dataset loading
│   ├── dsl/           # Domain-specific language
│   ├── perception/    # Object detection & relations
│   ├── proposer/      # Hypothesis generation
│   ├── evaluator/     # Hypothesis scoring
│   ├── workspace/     # Controller & reasoning loop
│   └── main.py
├── tests/             # Unit & integration tests
├── scripts/           # Utility scripts
├── examples/          # Example usage
├── configs/           # Configuration files
├── data/              # ARC dataset
└── results/           # Experiment outputs
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'arc_loader'"
**Solution:** Make sure you're in the venv and installed requirements:
```bash
source venv/bin/activate
pip install -e .
```

### Issue: "FileNotFoundError: ARC data not found"
**Solution:** Download the dataset:
```bash
cd data && git clone https://github.com/fchollet/ARC-AGI.git
```

### Issue: Solver takes too long
**Solution:** Reduce search space:
```bash
python scripts/run_solver.py --max-iterations 20 --workspace-size 3
```

### Issue: Low solve rate
**Solution:** This is expected! ARC is extremely hard. Start with:
- Easy tasks (< 5 objects, simple transformations)
- Expand the DSL for failing tasks
- Analyze error logs to improve components

---

## Next Steps

1. **Explore the codebase:**
   - Read `src/workspace/cognitive_workspace.py` - main reasoning loop
   - Check `src/dsl/primitives.py` - available operations
   - Review `src/proposer/heuristic_proposer.py` - how hypotheses are generated

2. **Solve your first task manually:**
   - Pick a simple task: `python scripts/pick_easy_task.py`
   - Understand the pattern
   - Write a DSL program: `python examples/manual_solve.py`

3. **Improve the system:**
   - Add new DSL primitives for failing tasks
   - Tune workspace size and search parameters
   - Implement better heuristics

4. **Follow the implementation plan:**
   - See `IMPLEMENTATION_PLAN.md` for full roadmap
   - We're at Phase 0-1 (Foundation)
   - Next: improve perception and rule library

---

## Getting Help

- **Documentation:** See `docs/` folder
- **Examples:** Check `examples/` for usage patterns
- **Implementation Plan:** `IMPLEMENTATION_PLAN.md` for architecture details
- **Issues:** If something breaks, check logs in `results/logs/`

---

## Tips for Success

1. **Start Small:** Don't try to solve all 400 tasks at once
2. **Iterate Fast:** Test each component independently
3. **Visualize Everything:** Use visualization scripts to debug
4. **Log Extensively:** Enable verbose logging to understand failures
5. **Analyze Errors:** Failed tasks teach you what's missing

---

## Example Session

```bash
# Activate environment
source venv/bin/activate

# Pick 10 easy tasks
python scripts/pick_easy_tasks.py --num 10 --output easy_tasks.json

# Run solver on them
python scripts/evaluate.py --tasks easy_tasks.json --verbose

# Analyze results
python scripts/analyze_results.py results/latest.json

# Visualize failures
python scripts/visualize_failures.py results/latest.json

# Add missing primitives to DSL
vim src/dsl/primitives.py

# Re-run
python scripts/evaluate.py --tasks easy_tasks.json
```

---

## Success Metrics

After completing setup, you should be able to:

- [x] Load and visualize ARC tasks
- [x] Execute DSL primitives
- [x] Run the baseline solver
- [x] See workspace dynamics
- [x] Understand solver output

**Congratulations! You're ready to start implementing the Cognitive Workspace architecture! 🎉**

Next: Read `IMPLEMENTATION_PLAN.md` Phase 0 and start building the foundation.
