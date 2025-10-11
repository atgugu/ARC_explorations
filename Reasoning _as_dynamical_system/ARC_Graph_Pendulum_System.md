# ARC “Graph Pendulum” System — Overview & Design

**Subtitle:** A Stability-Aware, Dynamical-Systems Approach to ARC Reasoning on a Skill Graph

**Keywords:** ARC, neurosymbolic, dynamical systems, stability, Lyapunov, beam/MCTS, program synthesis, ICECUBER, causality, interpretability

---

## Abstract

We present the **ARC Graph Pendulum System**, a solver architecture that behaves like a **controlled dynamical system** on a **skill graph**. Early nodes extract structured facts from the grids; later nodes compose hypotheses/programs, execute, critique, and repair. By monitoring how **small perturbations** in early nodes amplify (or dampen) downstream behavior, the controller learns where the system is **stable**, reuses **stable subgraphs**, and avoids **chaotic dead ends**. This yields improved **sample efficiency** and **solve rates** on ARC’s tiny training sets.

Core contributions include: (1) a typed **Node/Edge/Trajectory/Basin** abstraction with cacheable contracts; (2) a **graph geometry** over nodes via **behavior vectors** and **angles/distances**; (3) a **stability meter** with Lyapunov-like indicators and sensitivity sweeps; (4) a **stability-aware controller** combining beam/MCTS and repair loops; and (5) a **landscape index** for basin discovery and reuse. We detail a minimal viable system, evaluation metrics, guardrails, and learning dynamics over time.

---

## 1. Purpose

**Goal.** Build an ARC solver that acts like a **controlled dynamical system** on a **skill graph**. The system discovers and exploits **stable basins** of node trajectories to guide hypothesis composition and execution, while avoiding chaotic regions where upstream noise overwhelms signal.

**Why now.** ARC’s scarcity of examples makes brute-force exploration brittle. A stability-aware controller can **squeeze more signal** from few pairs, **reuse** robust subgraphs, and provide **diagnosable** traces for automated repair.

---

## 2. Core Abstractions

### 2.1 Node (skill/specialist)
A typed function:
\[
f: x \;\mapsto\; (y,\ \text{artifacts},\ \text{telemetry})
\]
with a predictable **contract** and **hashable inputs** for caching. Nodes are **pure** by default; LLM-driven nodes can run in **deterministic** mode for analysis passes.

### 2.2 Edge (compatibility)
A directed relation indicating **meaningful composition**. Each edge carries:
- **Geometry**: angle and distance (defined in §4).  
- **Utility**: causal credit accumulated online.

### 2.3 Trajectory
A **path of activated nodes** from perception → hypothesis → execution → critique, optionally **closing local repair loops**.

### 2.4 Basin (stable region)
A **cluster of trajectories** with **low sensitivity** to small upstream perturbations and **consistent** outcome patterns.

---

## 3. Node Taxonomy

- **Feature extractors:** color histogrammer, connected-components & objectifier, symmetry/periodicity detector, shape-primitive detector, object tracker across train pairs, affine + morphology probes, rule miners (color/position maps).
- **Reasoners/composers:** hypothesis generator (DSL templates), constraint aggregator, program synthesizer, counterexample finder, repairer, explainer.
- **Executors:** fast grid VM for the DSL (e.g., **ICECUBER**), train→test unit-test runner.
- **Critics:** IoU/Hamming/object-wise correspondence metrics, failure profiler (**what changed? which objects diverged?**).

---

## 4. Data Artifacts

- **Facts:** objects, primitives, symmetries, correspondences, color maps.  
- **Programs:** DSL candidates + **provenance** (which nodes proposed/edited them).  
- **Reports:** critic summaries, failure taxonomies, ablations.  
- **Telemetry:** node I/O signatures, behavior vectors, edge utilities, stability indicators.

---

## 5. Graph Geometry & Learning

### 5.1 Behavior vectors
For each node \(i\), build a vector by concatenating:
1. **I/O signatures** on a probe bank of mini-cases,  
2. **Latent descriptors** (primitive/rule histograms),  
3. **Downstream enablement deltas** (what the node enables for later stages).

### 5.2 Angle and distance
Define **angle** between nodes \(i,j\) as:
\[
\theta(i,j) = \arccos\big(\operatorname{cos\_sim}(b_i, b_j)\big)
\]
**Distance** can be **path length** with edge costs \(c(i,j)=1-\operatorname{cos\_sim}(b_i, b_j)\) or \(1/\operatorname{cos\_sim}\) when appropriate.

### 5.3 Utility updates (causal credit)
Edges \(i \to j\) accrue **causal credit** via counterfactuals:
\[
\Delta u_{i\to j} \;=\; \Delta \text{success}\big(\text{using } i\!\to\!j \text{ vs. skipping } j\big) \;-\; \lambda \cdot \text{variance},
\]
tempered by variance penalties to avoid credit from lucky runs.

---

## 6. Dynamics & Control Loop (High-Level)

1. **Perception path** activates extractor nodes → **structured facts**.  
2. **Composer nodes** propose hypotheses/programs guided by facts and historical edge utilities.  
3. **Executor** runs candidates; **Critics** compute scores and failure taxonomies.  
4. **Local repair loops** attempt minimal edits (translate/scale/remap) for localized failures.  
5. **Stability meter** probes sensitivity via small \(\varepsilon\)-perturbations; controller prefers **low-variance basins**, explores alternatives when **chaos** is detected.  
6. **Logging** streams all trajectory summaries to a **landscape index** for basin discovery and reuse.

---

## 7. Measuring Chaos vs. Stability

### 7.1 Local stability probes (Lyapunov-like)
Duplicate run \(K\) times with \(\varepsilon\)-perturbations in early nodes; track divergence of final metrics (IoU, object mapping accuracy). A positive empirical “exponent” indicates **chaotic** region.

### 7.2 Sensitivity sweep
Finite-difference sensitivity of final score to early-node knobs (thresholds, seed). **Low sensitivity ⇒ stable basin.**

### 7.3 Trajectory entropy
Entropy of **node visitation distributions** across repeats. **Low entropy clusters** behave like **attractors**.

### 7.4 Bifurcation map
Vary one upstream parameter; plot **regime shifts** in success pattern (e.g., selected rule family). Identify **critical points**.

---

## 8. Landscape Construction

Represent each trajectory by a fixed-length summary vector:
\[
[\ \text{visited-nodes histogram} \ |\ \overline{\theta}\ \text{along path}\ |\ \text{cumulative utility}\ |\ \text{final scores}\ |\ \text{failure taxonomy}\ ].
\]
Reduce with **UMAP/t-SNE**; **cluster** to find stable regions (tight clusters, low intra-variance, consistent success) vs. **chaotic spray**. Maintain **region descriptors**: recurring node motifs, successful rule families, typical failures.

---

## 9. Controller: How to Move in the Landscape

### 9.1 Base controller
- Start **greedy in stable basins**: prefer edges with **high utility** and **low sensitivity**.  
- Search with **beam** or **MCTS** over node paths, pruning using critic feedback.

### 9.2 Exploration policy
- **Multi-armed bandit** over **basin IDs**; within-basin, beam/MCTS over paths with **stability-aware scoring**.

### 9.3 Stability-aware exploration
If in a **chaotic patch** (high sensitivity, high trajectory entropy), **backtrack** to nearest **low-curvature junction** (edges historically not amplifying variance) and try an **orthogonal node motif**.

### 9.4 Repair loops (local micro-cycles)
Critic pinpoints failure type:
- **Placement offset** → translation search.  
- **Scale mismatch** → scale adjuster.  
- **Color mismatch** → color remap fitter.  
Prefer **minimal-delta** repairers over regeneration.

---

## 10. Concrete Benefits

- **More reliable search:** Stability filtering kills rabbit holes where randomness swamps signal.  
- **Compositional reuse:** Angles cluster semantically similar skills → graph **self-organizes** into feature/reasoning families, improving transfer.  
- **Causal credit assignment:** Edge utilities quantify “**A→B was useful**,” not mere correlation.  
- **Sample efficiency:** Prefer **low-variance chains**, extracting maximal information from few pairs.  
- **Diagnosability:** Chaotic regions + critic traces explain **why** a hypothesis failed, enabling automated repair and human insight.

---

## 11. Pitfalls & Guardrails

- **Angle definition leakage:** If behavior vectors overuse downstream success, early nodes look similar because later nodes fixed errors.  
  - **Fix:** separate **intrinsic** (I/O signatures) vs. **extrinsic** (utility) descriptors; regularize both.

- **LLM variance:** Temperature/noise can fake chaos.  
  - **Fix:** **deterministic decoding** for analysis passes; confine randomness to **controlled perturbations**.

- **Cost blow-up:** Graph runs can explode combinatorially.  
  - **Fix:** **caching/memoization** per node+input; **canonicalize artifacts**; **early-exit** via cheap critics.

- **Overfitting basins:** Stable ≠ universally correct.  
  - **Fix:** **basin generalization tests** on held-out micro-probes; penalize non-transferring basins.

---

## 12. Software Architecture (Minimal Viable)

- **Node runtime:** typed registry, caching layer, deterministic/stochastic modes, **provenance**.  
- **Controller:** **beam/MCTS** over node paths with **stability-aware** scoring.  
- **Executor/VM:** optimized DSL interpreter (e.g., **ICECUBER**) + oracle tests.  
- **Critic suite:** metrics + failure taxonomy.  
- **Analytics:** trajectory logger, UMAP projection, **basin clustering**, dashboards.

---

## 13. Steps (V0 Tight Loop)

1. **Probe bank:** 50–100 tiny synthetic ARC-like snippets covering primitives (mirror, flood fill, object copy/move, color remap, crop, tiling).  
2. **Node library (minimal):** segmentation/objects (connected components), symmetry detector, periodicity, bbox grouper; transformation proposers (translate/rotate/scale, paint-by-rule, color map); program synthesizer over DSL (ICECUBER) with ~20 templates; critic (IoU, object alignment, failure taxonomy).  
3. **Angles:** run each node on the probe bank; create **256-d behavior vectors** (primitive counts, success flags, output descriptors). Cosine for \(\theta\); costs \(= 1 - \cos\).  
4. **Controller:** best-first over paths of length \(\le L\) with **utility = predicted success − sensitivity penalty**; keep top-\(k\) trajectories (beam).  
5. **Stability meter:** for current best path, run **5 \(\varepsilon\)-perturbations** up front; if variance \(>\tau\), **demote basin** and branch to nearest **low-variance junction**.  
6. **Repair loop:** route by failure report (placement → translation; color → remap; etc.).  
7. **Analytics:** log every trajectory vector; project with UMAP; mark clusters as basins; persist edge utilities.

---

## 14. Evaluation & Safety

**Primary metrics:** success rate, steps per solve, executions per solve, **stability variance**, **basin transfer** across tasks.  
**Guards:** deterministic decoding for analysis, strict caching/memoization, early-exit via cheap critics, reproducible seeds.

---

## 15. Learning Over Time

- **Edge utility updates:**  
  \[
  \Delta w(A\!\to\!B) \leftarrow \alpha \cdot \big(\text{final\_score\_gain} - \text{expected\_gain}\big)
  \]
- **Meta-parameters:** learn per-node **trust** to adapt exploration budgets.  
- **Option policies:** collapse successful **subpaths into macro-nodes** (“detect-objects → align → copy”) with their own angle/utility signatures.  
- **Stability regularizer:** add a **preference term** in the controller objective for **low-sensitivity** subgraphs (acts like a **Lyapunov prior**).

---

## 16. Minimal Pseudocode

```text
initialize NodeRegistry, Controller, Executor, Critics, Analytics
build ProbeBank

# Precompute behavior vectors and angles
for node in NodeRegistry:
  b[node] = behavior_vector(node, ProbeBank)

while tasks remain:
  τ = next_task()
  # facts from perception chain
  facts = run_nodes(["components","symmetry","periodicity","bbox"], τ, cache=True)

  # search stable basins first
  candidates = Controller.search_paths(start=facts, max_len=L, beam=k,
                score = predicted_success - λ * sensitivity_penalty)

  best = select_best(candidates)
  if unstable(best): best = Controller.branch_to_low_variance_junction(best)

  # execute + critique
  program = compose_program(best)
  result, report = Executor.run(program, τ), Critics.assess(program, τ)

  if report.fail and report.localizable:
      best = repair_loop(best, report)
      program = compose_program(best)
      result, report = Executor.run(program, τ), Critics.assess(program, τ)

  # update utilities, log trajectory
  update_edge_utilities(best, report)
  Analytics.log(trajectory_vec(best, report))
```

---

## 17. Conclusion

The **Graph Pendulum** reformulates ARC solving as **controlled dynamics on a skill graph**. By explicitly measuring **stability vs. chaos**, assigning **causal credit** to edges, and **preferring low-variance basins**, the controller navigates toward reliable, reusable reasoning motifs. The result is a solver that is **sample-efficient**, **interpretable**, and **diagnosable**, aligning with ARC’s core challenge: learning to **generalize** from scarce evidence.

---

### Acknowledgments
This concept builds on neurosymbolic program synthesis, dynamical-systems analysis, and practical engineering lessons from stability-aware controllers. Any errors are our own.
