# Cognitive Workspace Implementation Plan
## From Theory to Working ARC-AGI Solver

**Goal:** Build a working implementation that solves real ARC-AGI tasks using the Cognitive Workspace architecture.

---

## Phase 0: Foundation & Infrastructure (Week 1-2)

### 0.1 Setup & Data Pipeline
**Deliverable:** Load and visualize ARC tasks

```python
# Core components:
- ARC dataset loader (training/evaluation/test)
- Grid visualization utilities
- Input/output pair representation
- Task metadata extraction
```

**Success Criteria:**
- [ ] Load all 400 training tasks
- [ ] Load 400 evaluation tasks
- [ ] Visualize tasks with matplotlib
- [ ] Parse grid -> numpy arrays
- [ ] Handle variable grid sizes

**Files to Create:**
- `src/data/arc_loader.py`
- `src/data/grid_utils.py`
- `src/visualization/grid_viz.py`
- `tests/test_data_loading.py`

---

### 0.2 Domain-Specific Language (DSL)
**Deliverable:** A minimal DSL for expressing ARC transformations

**Core Primitives:**
```python
# Object Operations
- select_by_color(grid, color)
- select_by_size(grid, size)
- select_largest(objects)
- select_smallest(objects)

# Spatial Operations
- translate(obj, dx, dy)
- rotate(obj, angle)
- reflect(obj, axis)
- scale(obj, factor)

# Color Operations
- recolor(obj, color)
- recolor_by_position(grid, rule)
- swap_colors(grid, c1, c2)

# Composition Operations
- copy_to_positions(obj, positions)
- tile(obj, nx, ny)
- overlay(grid1, grid2)
- mask(grid, condition)

# Relational Operations
- connect(obj1, obj2, pattern)
- group_by(objects, property)
- extend_line(obj, direction)
- fill_region(grid, condition)
```

**Success Criteria:**
- [ ] 20-30 primitive operations implemented
- [ ] Each primitive has unit tests
- [ ] Primitives compose correctly
- [ ] Can express 10+ simple ARC solutions manually

**Files to Create:**
- `src/dsl/primitives.py`
- `src/dsl/operations.py`
- `src/dsl/executor.py`
- `tests/test_dsl.py`

---

## Phase 1: Perception Module (Week 3-4)

### 1.1 Object Detection & Segmentation
**Deliverable:** Extract objects from ARC grids

**Approaches (implement all, compare):**
1. **Connected Components:**
   - Color-based 4/8-connectivity
   - Shape-based grouping

2. **Rule-Based Segmentation:**
   - Bounding box detection
   - Symmetry detection
   - Pattern matching (rectangles, lines, etc.)

3. **Neural Approach (optional for Phase 1):**
   - Small CNN trained on synthetic object masks
   - Use if rule-based insufficient

**Features to Extract:**
```python
class GridObject:
    pixels: List[Tuple[int, int]]
    color: int
    bounding_box: Tuple[int, int, int, int]
    size: int
    shape_type: str  # rectangle, line, L-shape, etc.
    symmetries: List[str]  # vertical, horizontal, rotational
    centroid: Tuple[float, float]
```

**Success Criteria:**
- [ ] Correctly segment objects in 80%+ of training tasks
- [ ] Extract features for each object
- [ ] Handle overlapping objects
- [ ] Detect background vs foreground

**Files to Create:**
- `src/perception/object_detector.py`
- `src/perception/feature_extractor.py`
- `src/perception/segmentation.py`
- `tests/test_perception.py`

---

### 1.2 Relation Extraction
**Deliverable:** Build a relational graph between objects

**Relations to Detect:**
```python
# Spatial Relations
- above(obj1, obj2)
- below, left_of, right_of
- contains(obj1, obj2)
- aligned(obj1, obj2, axis)
- same_row(obj1, obj2)
- same_column(obj1, obj2)

# Property Relations
- same_color(obj1, obj2)
- same_size(obj1, obj2)
- same_shape(obj1, obj2)

# Topological Relations
- adjacent(obj1, obj2)
- connected(obj1, obj2)
- surrounds(obj1, obj2)
```

**Success Criteria:**
- [ ] Build relation graph for any grid
- [ ] Extract all pairwise relations
- [ ] Handle multi-object scenes (10+ objects)

**Files to Create:**
- `src/perception/relations.py`
- `tests/test_relations.py`

---

## Phase 2: Rule Library & Hypothesis Generation (Week 5-6)

### 2.1 Rule Templates
**Deliverable:** Parameterized rule templates that can be instantiated

**Template Structure:**
```python
class RuleTemplate:
    name: str
    selectors: List[Selector]  # how to select objects
    operations: List[Operation]  # what to do with them
    parameters: Dict[str, Any]  # free parameters to bind

# Examples:
rotate_largest = RuleTemplate(
    name="rotate_largest",
    selectors=[select_largest],
    operations=[rotate],
    parameters={"angle": [90, 180, 270]}
)

recolor_by_size = RuleTemplate(
    name="recolor_by_size",
    selectors=[select_all],
    operations=[recolor],
    parameters={"size->color": mapping}
)
```

**Rule Categories:**
1. Single-object transformations (rotate, scale, translate)
2. Multi-object operations (align, group, connect)
3. Grid-level operations (tile, crop, extend)
4. Conditional operations (if X then Y)
5. Compositional rules (do A then B)

**Success Criteria:**
- [ ] 50+ rule templates defined
- [ ] Templates cover 70%+ of training tasks
- [ ] Each template has parameter space defined

**Files to Create:**
- `src/rules/templates.py`
- `src/rules/rule_library.py`
- `tests/test_templates.py`

---

### 2.2 Hypothesis Proposer
**Deliverable:** Generate candidate rule programs from observations

**Approach 1: Heuristic Proposer (Phase 2)**
```python
class HeuristicProposer:
    def propose(self, input_grid, output_grid, objects, relations):
        """Generate rule hypotheses by:
        1. Comparing input vs output
        2. Detecting changes (color, position, count)
        3. Matching to templates
        4. Instantiating parameters
        """
        hypotheses = []

        # Detect what changed
        if num_objects_changed:
            hypotheses.extend(object_addition_rules())
        if colors_changed:
            hypotheses.extend(recoloring_rules())
        if positions_changed:
            hypotheses.extend(spatial_rules())

        return hypotheses
```

**Approach 2: Neural Proposer (Phase 4)**
- Transformer that learns to propose rules
- Trained on (grid_pair, rule) examples

**Success Criteria:**
- [ ] Generate 10-100 hypotheses per task
- [ ] Correct rule in top-10 for 50%+ of simple tasks
- [ ] Proposals cover diverse rule types

**Files to Create:**
- `src/proposer/heuristic_proposer.py`
- `src/proposer/hypothesis.py`
- `tests/test_proposer.py`

---

## Phase 3: Evaluator & Executor (Week 7)

### 3.1 Program Executor
**Deliverable:** Execute rule programs and produce output grids

```python
class ProgramExecutor:
    def execute(self, program: Program, input_grid: Grid) -> Grid:
        """Execute DSL program step-by-step"""

    def execute_differentiable(self, program, input_grid):
        """Soft execution for gradient-based learning"""
        # Use neural executor or relaxed operations
```

**Success Criteria:**
- [ ] Execute all DSL primitives correctly
- [ ] Handle composition of operations
- [ ] Catch execution errors gracefully
- [ ] Produce valid output grids

**Files to Create:**
- `src/executor/program_executor.py`
- `src/executor/differentiable_executor.py`
- `tests/test_executor.py`

---

### 3.2 Hypothesis Evaluator
**Deliverable:** Score hypotheses on training pairs

**Scoring Function:**
```python
def score_hypothesis(hypothesis, task):
    score = 0.0

    # 1. Correctness on training pairs
    for input_grid, output_grid in task.train:
        predicted = execute(hypothesis, input_grid)
        if predicted == output_grid:
            score += 10.0  # exact match
        else:
            score += pixel_accuracy(predicted, output_grid)

    # 2. Simplicity (Occam's razor)
    score -= program_length(hypothesis) * 0.1

    # 3. Generalization heuristics
    if uses_absolute_coordinates(hypothesis):
        score -= 2.0  # penalize overfitting
    if has_magic_numbers(hypothesis):
        score -= 1.0

    # 4. Consistency across pairs
    if works_on_all_or_none(hypothesis):
        score += 5.0  # prefer consistent rules

    return score
```

**Success Criteria:**
- [ ] Correctly rank hypotheses
- [ ] Ground-truth rule ranks in top-5 for 60%+ tasks
- [ ] Scoring completes in <0.1s per hypothesis

**Files to Create:**
- `src/evaluator/scorer.py`
- `src/evaluator/metrics.py`
- `tests/test_evaluator.py`

---

## Phase 4: Global Workspace Controller (Week 8-9)

### 4.1 Workspace State Representation
**Deliverable:** State representation for the workspace

```python
class WorkspaceState:
    # Input
    task: ARCTask
    objects: List[GridObject]
    relations: RelationGraph

    # Current hypotheses
    admitted_hypotheses: List[Hypothesis]  # Top-k in workspace
    rejected_hypotheses: List[Hypothesis]  # Previously tried

    # Progress tracking
    best_score: float
    best_program: Program
    iteration: int

    # Memory
    episodic_memory: List[WorkspaceState]  # past states
    learned_priors: Dict[str, float]  # rule probabilities
```

**Files to Create:**
- `src/workspace/state.py`

---

### 4.2 Attention-Based Controller
**Deliverable:** Select and broadcast hypotheses

**Phase 4a: Heuristic Controller**
```python
class HeuristicController:
    def select(self, hypotheses: List[Hypothesis], state: WorkspaceState, k: int):
        """Select top-k hypotheses for workspace"""

        # Score each hypothesis
        scores = []
        for h in hypotheses:
            score = self.evaluator.score(h, state.task)

            # Diversity bonus
            if not similar_to_admitted(h, state.admitted_hypotheses):
                score += 2.0

            # Novelty bonus
            if not_recently_tried(h, state.rejected_hypotheses):
                score += 1.0

            scores.append(score)

        # Select top-k with diversity
        top_k = top_k_diverse(hypotheses, scores, k)
        return top_k

    def broadcast(self, hypotheses: List[Hypothesis], state: WorkspaceState):
        """Make hypotheses globally available to all modules"""
        state.admitted_hypotheses = hypotheses
        # Modules can now condition on these
```

**Phase 4b: Neural Controller (Phase 6)**
```python
class NeuralController(nn.Module):
    """Transformer-based controller with learned attention"""

    def forward(self, hypotheses, state):
        # Encode hypotheses and state
        h_embed = self.hypothesis_encoder(hypotheses)
        s_embed = self.state_encoder(state)

        # Attention-based selection
        queries = self.query_net(s_embed)
        attention = torch.matmul(queries, h_embed.T)
        top_k = torch.topk(attention, k)

        return top_k.indices
```

**Success Criteria:**
- [ ] Select diverse, high-quality hypotheses
- [ ] Maintain workspace size k=3-10
- [ ] Improve search efficiency vs random selection

**Files to Create:**
- `src/workspace/controller.py`
- `src/workspace/heuristic_controller.py`
- `src/workspace/neural_controller.py` (Phase 6)
- `tests/test_controller.py`

---

### 4.3 Reasoning Loop
**Deliverable:** Main search loop with workspace dynamics

```python
class CognitiveWorkspace:
    def solve(self, task: ARCTask, max_iterations: int = 50):
        # Initialize
        state = WorkspaceState(task)
        state.objects = self.perception.extract_objects(task)
        state.relations = self.perception.extract_relations(state.objects)

        for t in range(max_iterations):
            # 1. PROPOSE: Generate hypotheses
            hypotheses = self.proposer.propose(state)

            # 2. SELECT: Controller picks top-k
            admitted = self.controller.select(hypotheses, state, k=5)

            # 3. BROADCAST: Make admitted hypotheses global
            self.controller.broadcast(admitted, state)

            # 4. EVALUATE: Score admitted hypotheses
            for h in admitted:
                score = self.evaluator.score(h, task)
                if score > state.best_score:
                    state.best_score = score
                    state.best_program = h

            # 5. REVISE: Modules condition on workspace
            # (proposer will see admitted hypotheses next round)

            # 6. CHECK STOPPING
            if state.best_score >= SOLVED_THRESHOLD:
                break
            if no_progress_for_n_iterations(state, n=10):
                break

        return state.best_program
```

**Success Criteria:**
- [ ] Complete reasoning loop runs end-to-end
- [ ] Solves 10+ simple training tasks
- [ ] Iterative refinement improves solutions
- [ ] Early stopping works correctly

**Files to Create:**
- `src/workspace/cognitive_workspace.py`
- `tests/test_reasoning_loop.py`

---

## Phase 5: Integration & Baseline Evaluation (Week 10)

### 5.1 End-to-End System
**Deliverable:** Complete pipeline from task to solution

```python
# Example usage
workspace = CognitiveWorkspace(
    perception=ObjectPerception(),
    proposer=HeuristicProposer(),
    controller=HeuristicController(),
    evaluator=HypothesisEvaluator(),
    executor=ProgramExecutor()
)

solution = workspace.solve(task)
print(f"Solution: {solution}")
```

**Files to Create:**
- `src/main.py`
- `scripts/run_solver.py`
- `scripts/evaluate.py`

---

### 5.2 Evaluation & Metrics
**Deliverable:** Benchmark on ARC dataset

**Metrics:**
1. **Accuracy:** % of tasks solved correctly
2. **Coverage:** % of tasks where correct rule proposed
3. **Efficiency:** Average iterations to solution
4. **Interpretability:** Rule complexity/readability

**Evaluation Sets:**
- Easy tasks (10-20 handpicked simple tasks)
- Training set (400 tasks)
- Evaluation set (400 tasks)

**Target Metrics (Phase 5):**
- [ ] Solve 20+ easy tasks (50%+)
- [ ] Solve 40+ training tasks (10%+)
- [ ] Average search depth < 100 iterations

**Files to Create:**
- `evaluation/benchmark.py`
- `evaluation/metrics.py`
- `results/baseline_results.json`

---

## Phase 6: Advanced Features (Week 11-14)

### 6.1 Neural Proposer
**Deliverable:** Learn to propose better hypotheses

**Architecture:**
```python
class NeuralProposer(nn.Module):
    def __init__(self):
        self.grid_encoder = CNNEncoder()  # or ViT
        self.program_decoder = TransformerDecoder()

    def forward(self, input_grid, output_grid, objects, relations):
        # Encode input/output pair
        io_embed = self.grid_encoder(input_grid, output_grid)
        obj_embed = self.object_encoder(objects)

        # Generate program tokens
        program_logits = self.program_decoder(io_embed, obj_embed)

        # Sample top-k programs
        programs = self.sample_programs(program_logits, k=10)
        return programs
```

**Training:**
- Supervised: train on (grid_pair, program) from solved tasks
- Reinforcement: reward = execution correctness
- Imitation: distill from heuristic proposer's successful proposals

**Success Criteria:**
- [ ] Outperform heuristic proposer on coverage
- [ ] Generate correct rule in top-10 for 70%+ of tasks
- [ ] Fast inference (<0.5s per proposal batch)

**Files to Create:**
- `src/proposer/neural_proposer.py`
- `src/proposer/training.py`
- `training/train_proposer.py`

---

### 6.2 Curriculum Learning & Meta-Learning
**Deliverable:** Learn to learn across tasks

**Curriculum:**
```python
# Sort tasks by difficulty
easy_tasks = tasks with:
    - Few objects (< 5)
    - Simple transformations (single primitive)
    - Small grids (< 10x10)

medium_tasks = tasks with:
    - Multiple objects
    - Composition (2-3 primitives)
    - Medium grids

hard_tasks = remaining tasks

# Train in stages
train_on(easy_tasks, epochs=10)
train_on(medium_tasks, epochs=10)
train_on(hard_tasks, epochs=20)
```

**Meta-Learning:**
- Train controller to adapt quickly to new tasks
- Few-shot learning: given 1-2 solved tasks, solve similar ones
- Transfer learned priors across tasks

**Success Criteria:**
- [ ] Training converges faster with curriculum
- [ ] Controller learns task similarity
- [ ] Transfer improves solve rate on similar tasks

**Files to Create:**
- `src/training/curriculum.py`
- `src/training/meta_learning.py`

---

### 6.3 Critic & Early Stopping
**Deliverable:** Learn when to stop search

**Critic Network:**
```python
class Critic(nn.Module):
    """Predicts expected return from current state"""

    def value(self, state: WorkspaceState) -> float:
        """Estimate: will we solve this task?"""
        features = extract_state_features(state)
        return self.net(features)

    def stopping_confidence(self, state: WorkspaceState) -> float:
        """Should we commit to best program and stop?"""
        if state.best_score >= THRESHOLD:
            return 1.0

        # Predict diminishing returns
        future_value = self.value(state)
        if future_value < state.best_score + epsilon:
            return 0.8  # unlikely to improve

        return 0.0
```

**Training:**
- Supervised: from trajectories of solved tasks
- RL: reward for stopping at optimal time

**Success Criteria:**
- [ ] Reduce wasted search on unsolvable tasks
- [ ] Stop when sufficient confidence reached
- [ ] Improve overall efficiency by 2x

**Files to Create:**
- `src/workspace/critic.py`
- `src/training/train_critic.py`

---

## Phase 7: Advanced Perception (Week 15-16)

### 7.1 Neural Object Detection
**Deliverable:** Learned object detector for complex scenes

**When Needed:**
- Tasks with overlapping objects
- Ambiguous segmentation boundaries
- Abstract "objects" (patterns, regions)

**Architecture:**
- Slot Attention for object-centric learning
- Train on synthetic ARC-style scenes
- Fine-tune on real ARC tasks

**Success Criteria:**
- [ ] Handle complex overlapping scenes
- [ ] Improve perception accuracy to 95%+
- [ ] Detect abstract objects (symmetry groups, etc.)

**Files to Create:**
- `src/perception/neural_detector.py`
- `src/perception/slot_attention.py`

---

### 7.2 Learned Relations & Abstractions
**Deliverable:** Discover useful relations from data

**Approach:**
- Graph Neural Network on object graphs
- Learn to predict useful relations for solving
- Discover abstract properties (symmetry, periodicity)

**Files to Create:**
- `src/perception/relation_learner.py`
- `src/perception/gnn.py`

---

## Phase 8: Optimization & Scaling (Week 17-18)

### 8.1 Efficiency Improvements

**Optimizations:**
1. **Caching:**
   - Cache execution results
   - Cache object segmentations
   - Cache hypothesis scores

2. **Parallelization:**
   - Parallel hypothesis evaluation
   - Batch execution on GPU
   - Multi-process search

3. **Pruning:**
   - Early rejection of bad hypotheses
   - Prune redundant rules
   - Adaptive search depth

**Target:** 10x speedup

**Files to Create:**
- `src/optimization/caching.py`
- `src/optimization/parallel.py`

---

### 8.2 Hyperparameter Tuning

**Key Hyperparameters:**
- Workspace size k (3-10)
- Max iterations (50-200)
- Diversity penalty (0.5-2.0)
- Scoring weights
- Temperature for selection

**Method:**
- Grid search on validation set
- Bayesian optimization
- Task-adaptive hyperparameters

---

## Phase 9: Advanced Modules (Week 19-20)

### 9.1 Memory & Context Module

**Episodic Memory:**
```python
class EpisodicMemory:
    """Store and retrieve past solving experiences"""

    def store(self, task, solution, trajectory):
        """Store successful solution"""

    def retrieve_similar(self, task, k=5):
        """Find similar solved tasks"""
        # Use task embedding similarity

    def extract_priors(self):
        """Learn rule priors from memory"""
        # Frequent rules get higher prior
```

**Success Criteria:**
- [ ] Reuse solutions for similar tasks
- [ ] Learn common patterns
- [ ] Improve few-shot performance

**Files to Create:**
- `src/memory/episodic_memory.py`
- `src/memory/retrieval.py`

---

### 9.2 Verification & Explanation Module

**Verification:**
```python
class Verifier:
    """Verify solution correctness and explain reasoning"""

    def verify(self, program, task):
        """Check if program solves all training pairs"""

    def explain(self, program, task):
        """Generate human-readable explanation"""
        # "The rule rotates the largest blue object 90 degrees
        #  and copies it to each corner"
```

**Success Criteria:**
- [ ] Catch incorrect solutions before test submission
- [ ] Generate interpretable explanations
- [ ] Build trust with users

**Files to Create:**
- `src/verification/verifier.py`
- `src/explanation/explainer.py`

---

## Phase 10: Final Evaluation & Refinement (Week 21-22)

### 10.1 Comprehensive Evaluation

**Full Benchmark:**
- [ ] Training set: 400 tasks
- [ ] Evaluation set: 400 tasks
- [ ] Test set: 100 tasks (submit predictions)

**Target Goals:**
- [ ] Solve 100+ training tasks (25%+)
- [ ] Solve 50+ evaluation tasks (12%+)
- [ ] Competitive with baseline solvers
- [ ] Demonstrate systematic generalization

---

### 10.2 Analysis & Insights

**Analyses:**
1. **Error Analysis:** What types of tasks fail?
2. **Ablation Studies:** Which components are critical?
3. **Workspace Dynamics:** How does attention evolve?
4. **Biological Correspondence:** Workspace "ignition" patterns
5. **Interpretability:** Rule quality and readability

**Deliverables:**
- Comprehensive results paper
- Visualizations of workspace dynamics
- Comparison with baselines
- Open-source release

---

## Technology Stack

### Core Libraries
```python
# Data & Computation
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Utilities
tqdm>=4.65.0
loguru>=0.7.0
pyyaml>=6.0
hydra-core>=1.3.0  # config management

# Development
pytest>=7.3.0
black>=23.3.0
mypy>=1.3.0
```

### Project Structure
```
ARC_explorations/
├── Cognitive_Workspace/
│   ├── README.md
│   ├── Cognitive_Workspace.md
│   ├── IMPLEMENTATION_PLAN.md (this file)
│   ├── src/
│   │   ├── data/
│   │   │   ├── arc_loader.py
│   │   │   └── grid_utils.py
│   │   ├── dsl/
│   │   │   ├── primitives.py
│   │   │   ├── operations.py
│   │   │   └── executor.py
│   │   ├── perception/
│   │   │   ├── object_detector.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── relations.py
│   │   │   └── neural_detector.py
│   │   ├── rules/
│   │   │   ├── templates.py
│   │   │   └── rule_library.py
│   │   ├── proposer/
│   │   │   ├── heuristic_proposer.py
│   │   │   ├── neural_proposer.py
│   │   │   └── hypothesis.py
│   │   ├── executor/
│   │   │   ├── program_executor.py
│   │   │   └── differentiable_executor.py
│   │   ├── evaluator/
│   │   │   ├── scorer.py
│   │   │   └── metrics.py
│   │   ├── workspace/
│   │   │   ├── state.py
│   │   │   ├── controller.py
│   │   │   ├── cognitive_workspace.py
│   │   │   └── critic.py
│   │   ├── memory/
│   │   │   ├── episodic_memory.py
│   │   │   └── retrieval.py
│   │   ├── verification/
│   │   │   └── verifier.py
│   │   └── main.py
│   ├── tests/
│   │   └── test_*.py
│   ├── scripts/
│   │   ├── run_solver.py
│   │   ├── evaluate.py
│   │   └── visualize_workspace.py
│   ├── training/
│   │   ├── train_proposer.py
│   │   ├── train_controller.py
│   │   └── train_critic.py
│   ├── configs/
│   │   └── default.yaml
│   ├── data/
│   │   ├── training/
│   │   └── evaluation/
│   └── results/
│       ├── logs/
│       └── checkpoints/
```

---

## Development Principles

### 1. Start Simple, Iterate
- Build heuristic versions first (Phase 1-5)
- Add neural components once basics work (Phase 6+)
- Validate each component independently

### 2. Test-Driven Development
- Write tests for each module
- Maintain >80% code coverage
- Integration tests for full pipeline

### 3. Interpretability First
- Log workspace states at each iteration
- Visualize attention patterns
- Generate human-readable explanations
- Make debugging easy

### 4. Measure Everything
- Track solve rates by task difficulty
- Monitor search efficiency
- Profile compute bottlenecks
- A/B test design choices

### 5. Fail Fast, Learn Faster
- If a task is unsolvable, stop early
- Analyze failures to improve components
- Use failures to grow rule library

---

## Success Metrics by Phase

| Phase | Deliverable | Success Metric |
|-------|-------------|----------------|
| 0 | Data + DSL | Load all tasks; express 10+ solutions |
| 1 | Perception | 80%+ object detection accuracy |
| 2 | Proposer | Correct rule in top-10 for 50%+ tasks |
| 3 | Evaluator | Ground-truth ranks top-5 in 60%+ tasks |
| 4 | Workspace | Solve 10+ simple tasks end-to-end |
| 5 | Baseline | 10%+ training set accuracy |
| 6 | Neural | 20%+ training set accuracy |
| 7 | Advanced | 25%+ training set, 12%+ eval set |
| 8 | Optimized | 2x faster, same accuracy |
| 9 | Complete | All modules integrated |
| 10 | Final | Production-ready, published |

---

## Risk Mitigation

### High-Risk Areas

1. **DSL Coverage:**
   - Risk: DSL can't express complex transformations
   - Mitigation: Iteratively expand based on failure analysis
   - Fallback: Hybrid symbolic-neural execution

2. **Search Explosion:**
   - Risk: Too many hypotheses to evaluate
   - Mitigation: Strong priors, early pruning, learned critics
   - Fallback: Beam search with aggressive pruning

3. **Perception Failures:**
   - Risk: Can't segment complex scenes
   - Mitigation: Hybrid rule-based + neural approach
   - Fallback: Manual object annotations for debugging

4. **Controller Convergence:**
   - Risk: Controller doesn't learn effective policies
   - Mitigation: Start with heuristics, gradually add learning
   - Fallback: Keep heuristic controller as baseline

### Medium-Risk Areas

1. **Training Data Scarcity:** ARC has limited training data
   - Mitigation: Synthetic data generation, meta-learning

2. **Compute Requirements:** Neural components may be slow
   - Mitigation: Caching, parallelization, model distillation

3. **Evaluation Overfitting:** May overfit to training set
   - Mitigation: Hold-out validation, test on diverse tasks

---

## Timeline Summary

- **Weeks 1-2:** Foundation (data, DSL)
- **Weeks 3-4:** Perception
- **Weeks 5-6:** Rules & Proposer
- **Weeks 7:** Evaluator & Executor
- **Weeks 8-9:** Workspace & Controller
- **Weeks 10:** Integration & Baseline
- **Weeks 11-14:** Neural Components
- **Weeks 15-16:** Advanced Perception
- **Weeks 17-18:** Optimization
- **Weeks 19-20:** Advanced Modules
- **Weeks 21-22:** Final Evaluation

**Total: ~5 months for complete implementation**

**Minimum Viable Product (MVP): Weeks 1-10 (~2.5 months)**

---

## Next Steps

To get started immediately:

1. **Set up environment:**
   ```bash
   cd /home/user/ARC_explorations/Cognitive_Workspace
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download ARC dataset:**
   ```bash
   mkdir -p data
   cd data
   git clone https://github.com/fchollet/ARC-AGI.git
   ```

3. **Start with Phase 0.1:**
   - Create `src/data/arc_loader.py`
   - Load and visualize first 10 tasks
   - Validate data format

4. **Daily rhythm:**
   - Morning: Implement 1-2 components
   - Afternoon: Write tests
   - Evening: Run experiments, analyze results

---

## Questions to Answer During Implementation

1. What workspace size k works best? (3, 5, 10?)
2. How many iterations needed on average?
3. Which rule types are most useful?
4. Does neural proposer beat heuristic?
5. Can we learn task similarity for transfer?
6. What's the failure mode distribution?
7. Do workspace dynamics match biological signatures?

---

## Conclusion

This plan transforms the theoretical Cognitive Workspace into a concrete, implementable system. By building incrementally with clear milestones, we can:

1. **Validate the core idea** (Phases 1-5)
2. **Scale with neural components** (Phases 6-7)
3. **Optimize for production** (Phases 8-9)
4. **Achieve competitive performance** (Phase 10)

The key insight: start with interpretable heuristics, validate the architecture works, then gradually add learning where it helps most.

Let's build it! 🚀
