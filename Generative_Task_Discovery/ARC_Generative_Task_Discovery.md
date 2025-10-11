# Generative Task Discovery via Learned Priors for ARC: From Solver to Scientist

**Keywords:** ARC, program synthesis, neurosymbolic AI, self-curriculum, generative modeling, meta-reasoning, GNNs, transformers

---

## Abstract

We propose a neurosymbolic framework that **learns priors over transformations and object rules from ARC** (Abstraction and Reasoning Corpus) and then **generates new solvable ARC-like tasks** to train better inductive biases via **self-curriculum learning**. Unlike solvers that passively adapt to a fixed dataset, our system acts as a **task scientist**: it infers a **probabilistic grammar of transformations**, samples new tasks from this grammar, **verifies solvability** via a differentiable symbolic executor, and schedules them to the solver with **adaptive difficulty**. The key contributions are: (1) a **Typed Rule Grammar (TRG)** capturing ARC abstractions; (2) a **Generative Prior Model (GPM)** parameterized by a transformer/GNN that models distributions over transformation programs and object layouts; (3) a **Solvability Verifier (SV)** coupling a learned proof search with a differentiable executor; and (4) a **Self-Curriculum Engine (SCE)** that estimates task difficulty and information gain to optimize training. We provide a full algorithm, training objectives, ablations, and evaluation protocols. The approach yields a closed learning loop in which **task generation → solving → posterior updates** converges toward increasingly general reasoning competence.

---

## 1. Introduction

ARC is designed to evaluate broad generalization by requiring systems to infer abstract, causal-like rules from very few examples. Progress is hindered by the dataset’s small size and heterogeneity. We hypothesize that **learning a generative prior over ARC-style rules** enables a solver to **produce new, valid tasks**, thereby exposing itself to a **structured curriculum** that complements scarce supervision and encourages **systematic generalization**.

We develop an **ARC task generator** that internalizes regularities across tasks—object segmentation, symmetry, color remapping, set/relational operations, topological edits—and then **samples task programs** which are rendered into image grids, accompanied by **training pairs**. A **neurosymbolic verifier** ensures that generated tasks are non-trivial, solvable by a bounded search, and diagnostic of particular reasoning skills. The solver and generator are trained **in tandem**: as the solver improves, the generator **adapts difficulty** and **diversifies** distributions to reduce overfitting and probe new corners of the hypothesis space.

---

## 2. Background and Motivation

ARC tasks map small integer-valued grids (≤30×30) to outputs via underlying rules: e.g., copy-with-offset, reflect-and-filter, grow/shrink, connect components, recolor based on roles, etc. Humans tend to explain solutions with **symbolic programs** over **objects and relations**. Current systems either search in symbolic program spaces or learn neural heuristics for guidance. However, the closed-world training regime limits systematicity. **We convert ARC from a static benchmark to a generative research domain** by learning a **prior over rule programs** and **object layouts**, then **sampling** tasks with guarantees of solvability and novelty.

---

## 3. Problem Formulation

Let ![X](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathcal{X}) denote the space of ARC grids and ![T](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathcal{T}) the space of **tasks** ![tau](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\tau), each a set of training pairs ![training pairs](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\{(x_i,y_i)\}_{i=1}^k) plus a test input ![x star](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}x^\star). A **task program** ![p in P](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p\in\mathcal{P}) maps inputs to outputs: ![y equals f_p(x)](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}y%20=%20f_p(x)). Our aim is to learn a **prior** ![p_theta](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p_\theta(p,%20L)) over programs ![p](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p) and object layouts ![L](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}L) (scene graphs), from observed tasks ![D](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathcal{D}), such that sampling ![sampling](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}(p,L)\sim%20p_\theta) and rendering yields **valid ARC-like tasks**. A solver ![S_phi](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}S_\phi) maps ![tau to y hat](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\tau%20\mapsto%20\hat{y}^\star). We seek a **closed loop**:

![Closed Loop](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\text{learn%20}%20p_\theta%20\%20\Rightarrow%20\%20\text{sample%20tasks%20}%20\tilde{\tau}%20\%20\Rightarrow%20\%20\text{train%20}%20S_\phi%20\%20\Rightarrow%20\%20\text{update%20}%20p_\theta%20\text{%20using%20solvability/novelty%20feedback})

---

## 4. Typed Rule Grammar (TRG)

We specify a **typed, factorized program space**:

- **Types.**
  - **Grid** ![G](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}G), **Object** ![O](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}O) (connected components), **Mask** ![M](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}M), **Color** ![C](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}C), **Relation** ![R](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}R), **Set** ![S](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}S).
- **Primitives.**  
  - **Perceptual:** `components(G)→S[O]`, `bbox(O)`, `centroid(O)`, `shape(O)`, `area(O)`.  
  - **Geometric:** `rotate(G,k)`, `reflect(G,axis)`, `translate(G,dx,dy)`, `scale(O,s)`.  
  - **Morphological:** `dilate(M)`, `erode(M)`, `fill(M,C)`.  
  - **Relational:** `match(Oi,Oj,by=shape|size|color)`, `group(S[O],criterion)`.  
  - **Logical/Set:** `map`, `filter`, `compose`, `union`, `difference`.  
  - **Color rules:** `remap(C→C')`, `palette(S[C])`.
- **Schemas (rule templates).**  
  - `copy_with_transform`: select ![S[O]](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}S[O]), apply ![T](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}T), place with constraint ![K](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}K).  
  - `symmetry_enforce`: reflect/rotate to enforce pattern.  
  - `role_based_recolor`: recolor object by role in a relation graph.  
  - `grow_connect`: extend objects to connect/complete shapes.  
  - `frame_and_fill`: construct borders and fill regions under constraints.

Each **schema** is a graph of primitives with typed edges; **parameters** ![psi](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\psi) (e.g., rotation ![k](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}k), axis, palette) and **selectors** ![sigma](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\sigma) (which objects) are learned as random variables.

---

## 5. Generative Prior Model (GPM)

We model ![p_theta(p,L)](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p_\theta(p,%20L)) as a product of factors:

![GPM Factorization](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p_\theta(p,L)%20=%20p_\theta(L)\%20p_\theta(p\mid%20L))

where ![L](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}L) is a **scene graph** of object prototypes, spatial layouts, and palettes.

### 5.1 Scene Graph Generator ![p_theta(L)](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p_\theta(L))
- **Object prototypes:** shape tokens (e.g., line, block, L-shape), size distributions, colors.  
- **Layout:** a **graph transformer** samples object counts, poses, and relations (adjacency, alignment, symmetry).  
- **Constraints:** keep grids within ARC size, avoid degenerate overlaps unless intended.

### 5.2 Program Generator ![p_theta(p|L)](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p_\theta(p\mid%20L))
- **Schema selection:** a **sequence model** (transformer) chooses schema nodes conditioned on $L$.  
- **Primitive wiring:** edges define dataflow; a **type-checker** enforces valid compositions.  
- **Parameterization:** distributions over discrete/continuous parameters (e.g., rotation ![k in set](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}k%20\in%20\{0,1,2,3\})).  
- **Selector policies:** GNN computes attention over ![L](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}L) to pick objects by role.

### 5.3 Training Objectives for GPM
Given observed tasks $\mathcal{D}$, we learn $\theta$ via **amortized maximum likelihood**:

$$
\max_\theta \sum_{\tau\in\mathcal{D}} \mathbb{E}_{q_\phi(p,L\mid \tau)} \big[\log p_\theta(\tau\mid p,L) + \log p_\theta(p,L) - \log q_\phi(p,L\mid \tau)\big],
$$

with an **inference network** $q_\phi$ that parses tasks to latent programs and layouts. The likelihood uses a **differentiable executor** (Sec. 6) to render $f_p(x_i)$ and compare to $y_i$.

---

## 6. Differentiable Executor and Solvability Verifier (SV)

### 6.1 Differentiable Executor
Each primitive is a small neural/symbolic module with **piecewise-differentiable** behavior. For non-differentiable steps (argmax, connected components), we use **straight-through estimators** or **continuous relaxations** (e.g., soft labeling of components). The executor composes modules according to $p$ to produce $\hat{y}_i = f_p(x_i)$ and a test prediction $\hat{y}^\star$.

**Loss:**  
$$
\mathcal{L}_{exe} = \sum_i \ell(\hat{y}_i, y_i) + \lambda\ \Omega(p),
$$
where $\ell$ includes Hamming/IoU and $\Omega$ regularizes program length/complexity.

### 6.2 Solvability Verifier
A task is **admitted** if:
1. **Consistency:** the same program $p$ solves all training pairs.  
2. **Generalization sanity:** $p$ is **deterministic** and type-safe on $x^\star$.  
3. **Non-triviality:** ablations of critical edges in $p$ worsen $\ell$ beyond a margin.  
4. **Bounded search proof:** a neural proof-search attempts $K$-step repairs; if found, store the **witness** $p^\dagger$.

SV returns `(is_solvable, p^\dagger, diagnostics)`; unsolvable samples are rejected or recycled via **guided resampling** of weak factors.

---

## 7. Self-Curriculum Engine (SCE)

The SCE controls **which generated tasks** reach the solver and when.

- **Difficulty estimator $d(\tau)$:** predicts solver loss, minimal program length, branching factor, symmetry order, etc.  
- **Information gain $IG(\tau) = \Delta H(\Phi)$:** estimated reduction in parameter entropy for solver modules or rule posteriors.  
- **Scheduling:** sample a batch maximizing a **utility** $U(\tau) = \alpha\ d(\tau) + \beta\ IG(\tau) - \gamma\ \text{redundancy}(\tau, \mathcal{B})$.  
- **Anti-overfitting:** penalize tasks too similar to recent batches; enforce **coverage** over schema families.

---

## 8. Solver Architecture $S_\phi$

A **neurosymbolic solver** with:
1. **Perception**: multi-expert feature bank (CNN/ViT/GNN/LSTM) producing object candidates and relations.  
2. **Program induction**: search over TRG with a policy network guiding expansions; critic scores partial programs.  
3. **Executor**: same differentiable modules as SV.  
4. **Learning**: mixed supervised (from parsed programs/witnesses) and RL (success reward + sparsity penalty).

---

## 9. Closed-Loop Training

**Algorithm 1: Self-Generating ARC Curriculum**

1. **Initialize** $p_\theta, q_\phi, S_\psi$.  
2. **Parse** $\mathcal{D}$ with $q_\phi$ to warm-start $p_\theta$.  
3. **Repeat**:  
   a. Sample $(p,L) \sim p_\theta$; render task $\tilde{\tau}$.  
   b. **Verify** via SV → keep $(\tilde{\tau}, p^\dagger)$ if solvable & non-trivial; else resample.  
   c. **Score** difficulty and IG; schedule batch $\mathcal{B}$.  
   d. **Train solver** $S_\psi$ on $\mathcal{B}$ (supervised on $p^\dagger$ where available + RL on held-out test).  
   e. **Update prior** $p_\theta$ with feedback: increase probability mass around fruitful but underexplored schemas; reduce collapsed modes.  
   f. Periodically **evaluate** on held-out ARC and **diversity metrics**.

---

## 10. Objectives and Regularization

- **Prior learning:** ELBO (Sec. 5.3) + **minimum description length** penalty on programs.  
- **Executor training:** $\mathcal{L}_{exe}$ with **consistency** and **equivariance** constraints (e.g., rotation-equivariant when appropriate).  
- **Solver training:**  
  - **Supervised**: $\mathcal{L}_{prog} = \text{XE}(\hat{p}, p^\dagger) + \eta\ \text{XE}(\hat{\sigma}, \sigma^\dagger)$.  
  - **RL**: reward $r = \mathbb{1}[\hat{y}^\star=y^\star] - \lambda |p|$.  
- **Curriculum utility:** optimize scheduler parameters to maximize downstream validation accuracy subject to **diversity** constraints.

---

## 11. Evaluation Protocol

### 11.1 Benchmarks
- **ARC public/hidden splits** (strictly avoid contamination).  
- **Generated diagnostic suites**: each anchored to a schema family (symmetry, recolor-by-role, growth/connectivity, compositional effects).

### 11.2 Metrics
- **Generalization accuracy** on ARC test.  
- **Sample efficiency**: performance vs. number of supervised ARC tasks.  
- **Transfer**: performance on synthetic-but-novel distributions generated by **held-out** parts of TRG.  
- **Diversity**: coverage across schema families; N-gram diversity of primitive sequences; structural edit distance between programs.  
- **Solvability rate** and **non-triviality rate** of generated tasks.  
- **Ablations**: remove SV, remove curriculum, remove differentiable executor, collapse TRG to flat primitives, etc.

### 11.3 Baselines
- Symbolic program search with neural guidance (no generation).  
- End-to-end neural solvers trained on ARC only.  
- Data augmentation with naive perturbations (no TRG priors).  
- Program synthesis with fixed hand-crafted DSL (no learned prior).

---

## 12. Implementation Details

- **Grids:** 30×30 with values in $\{0,\dots,9\}$ (colors).  
- **Objects:** connected components under 4-connectivity; also maintain soft component maps for differentiability.  
- **Scene graphs:** nodes = object prototypes; edges = adjacency, alignment, symmetry relations.  
- **Executor library:** compiled set of ~40 primitives with typed signatures and gradient surrogates.  
- **Parsing network $q_\phi$:** encoder-decoder over pairs $(x_i,y_i)$; predicts schema skeleton, then fills parameters.  
- **GPM backbone:** transformer decoder over TRG tokens with cross-attention to scene graph embeddings (from a GNN).  
- **SV search:** beam width $B$, max steps $K$, learned priority queue.  
- **SCE:** difficulty predictors trained from solver logs; IG approximated with Bayesian linear probes on solver internal representations.

---

## 13. Theoretical Notes

- **Identifiability up to symmetries.** Different programs may be functionally equivalent; we quotient by a **congruence** over TRG graphs when computing diversity and MDL.  
- **Generalization via priors.** The learned $p_\theta(p)$ is a **simplicity-biased prior**: shorter, compositional programs with reusable substructures have higher probability, encouraging **Occam-consistent** solutions.  
- **Curriculum as active learning.** SCE approximates **Bayesian experimental design**: select tasks maximizing posterior contraction of solver parameters.

---

## 14. Limitations and Risks

- **Mode collapse** in GPM: mitigated by diversity regularization and coverage constraints.  
- **Overfitting to executor quirks:** addressed via randomized primitive implementations and cross-checks with a discrete reference executor.  
- **Verification cost:** SV can be expensive; we cap resources and recycle failures via learned proposal distributions.  
- **Measuring “human-like” reasoning:** competence may improve without matching human explanations; add evaluation with **explanatory alignment** (program readability, compositional reuse).

---

## 15. Broader Impact

A system that invents tasks to expand its own reasoning can influence education (adaptive problem generation), scientific discovery (hypothesis proposal), and automated testing (robustness diagnostics). The same capability must be governed to avoid spurious or biased curricula; we recommend open diagnostics and **data sheets for generated tasks**.

---

## 16. Reproducibility Checklist

- Open-source TRG spec and executor library with unit tests.  
- Seeded generation pipelines; fixed splits for eval.  
- Logged witness programs for all accepted tasks.  
- Full ablation suite with config files.  
- Compute budget and wall-time reporting per component.

---

## 17. Related Directions (brief)

- Program synthesis and neurosymbolic learning; differentiable interpreters; meta-learning & MAML for structure learning; Bayesian experimental design for curricula; graph generative models and scene graphs. (Citations to be inserted in a camera-ready version.)

---

## 18. Pseudocode (Core Loop)

```text
Input: ARC data D, TRG, executor E
Initialize GPM pθ, Parser qφ, Solver Sψ, Scheduler Π

# Warm-start GPM by parsing ARC
for τ in D:
    (p̂, L̂) ← qφ(τ)                      # parse to program & layout
    θ ← argmaxθ ELBO(τ; p̂, L̂, θ)        # train pθ via amortized ELBO

while not converged:
    Π.refresh()                           # update difficulty, IG predictors
    B ← ∅
    while |B| < batch_size:
        (p, L) ~ pθ                       # sample program & layout
        τ ← render(p, L)                  # generate (x_i, y_i), x*
        solvable, p†, diag ← SV(E, τ)     # verify
        if solvable and non_trivial(diag):
            score ← Π(τ, Sψ)              # difficulty × IG × novelty
            B ← B ∪ {(τ, p†, score)}
    B ← top_k(B, by=score)

    # Train solver with mixed supervision
    Sψ ← update_solver(Sψ; B, E)

    # Update generator prior using feedback signals
    θ ← update_prior(θ; successes=B, failures=diag)

    # Evaluate on held-out ARC & diagnostics
    report_metrics(Sψ, pθ, SV)
```

---

## 19. Expected Outcomes

- **Higher ARC generalization** at equal or lower sample budgets.  
- **Improved compositionality** (shorter witness programs, greater substructure reuse).  
- **Robust transfer** to synthetic-but-systematic distributions unseen in original ARC.  
- **Measurable curriculum effects**: earlier mastery of core schemas; later emergence of rare compositions.

---

## 20. Conclusion

We present a principled route to turn an ARC solver into an **ARC scientist** by **learning and sampling from priors over rule programs and scenes**, verifying solvability with a **differentiable symbolic executor**, and **actively scheduling** generated tasks to maximize learning. This **closed-loop system** pushes beyond static evaluation toward **self-expanding reasoning competence**, a step toward general-purpose abstraction and meta-reasoning.

---

### Appendix A: Concrete TRG Snippet (Illustrative)

- `schema: role_based_recolor`
  - inputs: $G$  
  - steps:  
    1. $S[O] = \text{components}(G)$  
    2. $R = \text{group}(S[O], \text{by=shape})$  
    3. $\pi = \text{role\_assign}(R, \text{by=size→rank})$  
    4. $\text{map}(O \in S[O], \text{fill}(O, \text{palette}[\pi(O)]))$

- `schema: copy_with_transform`
  - select: `selector` over $S[O]$ (e.g., largest; unique color)  
  - transform: `T ∈ {rotate k, reflect axis, translate dx,dy}`  
  - place: avoid overlaps or define compositing rule (overwrite/maximum).

---

### Appendix B: Metrics Details

- **Program diversity:** average pairwise graph edit distance; unique schema n-grams.  
- **Difficulty calibration:** Spearman correlation of predicted vs. realized solver loss.  
- **Executor fidelity:** percent agreement with discrete executor; gradient norm stability.
