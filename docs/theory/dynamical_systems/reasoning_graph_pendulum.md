ARC “Graph Pendulum” System — Overview & Design
System Description
Purpose

Build an ARC solver that behaves like a controlled dynamical system on a skill graph: early nodes extract structured facts from the grids; later nodes compose hypotheses/programs, execute, critique, and repair. By monitoring how small perturbations in early nodes amplify (or dampen) downstream behavior, we learn where the system is stable, reuse stable subgraphs, and avoid chaotic dead ends—improving sample-efficiency and success on tiny ARC training sets.

Core Abstractions

Node (skill/specialist): A typed function x → (y, artifacts, telemetry) with a predictable contract and hashable inputs for caching. Nodes are pure by default; LLM nodes run in deterministic mode for analysis passes.

Edge (compatibility): A directed relation indicating meaningful composition. Edges carry geometry (angle, distance) and utility (causal credit).

Trajectory: A path of activated nodes from perception → hypothesis → execution → critique, optionally closing local repair loops.

Basin (stable region): A cluster of trajectories with low sensitivity to small upstream perturbations and consistent outcome patterns.

Node Taxonomy

Feature extractors: color histogrammer, connected-components & objectifier, symmetry/periodicity detector, shape-primitive detector, object tracker across train pairs, affine + morphology probes, rule miners (color/position maps).

Reasoners/composers: hypothesis generator (DSL templates), constraint aggregator, program synthesizer, counterexample finder, repairer, explainer.

Executors: fast grid VM for the DSL (e.g., ICECUBER), train→test unit-test runner.

Critics: IoU/Hamming/object-wise correspondence metrics, failure profiler (what changed? which objects diverged?).

Data Artifacts

Facts: objects, primitives, symmetries, correspondences, color maps.

Programs: DSL candidates + provenance (which nodes proposed/edited them).

Reports: critic summaries, failure taxonomies, ablations.

Telemetry: node I/O signatures, behavior vectors, edge utilities, stability indicators.

Graph Geometry & Learning

Behavior vector per node: concatenation of (i) I/O signatures on probe mini-cases, (ii) latent descriptors (primitive/rule histograms), (iii) downstream enablement deltas.

Angle (θ): arccos(cosine_similarity(behavior_i, behavior_j))—semantic proximity.

Distance (d): path length with edge cost 1 − similarity.

Utility updates: edges accrue causal credit via counterfactuals (Δ success using A→B vs. skipping B), tempered by variance penalties.

Dynamics & Control Loop (High-Level)

Perception path activates extractor nodes → structured facts.

Composer nodes propose hypotheses/programs guided by facts and past utilities.

Executor runs candidates; Critics compute scores and failure taxonomies.

Local repair loops attempt minimal edits (translate/scale/remap) when failures are localized.

Stability meter probes sensitivity via small ε-perturbations upstream; controller prefers low-variance basins, explores alternatives when chaos detected.

Logging streams all trajectory summaries to a landscape index for basin discovery and reuse.

Evaluation & Safety

Primary metrics: success rate, steps per solve, executions per solve, stability variance, transfer of basins across tasks.

Guards: deterministic decoding for analysis, strict caching/memoization, early-exit via cheap critics, reproducible seeds.

Software Architecture (Minimal Viable)

Node runtime: typed registry, caching layer, deterministic/ stochastic modes, provenance.

Controller: beam/MCTS over node paths with stability-aware scoring.

Executor/VM: optimized DSL interpreter + oracle tests.

Critic suite: metrics + failure taxonomy.

Analytics: trajectory logger, UMAP projection, basin clustering, dashboards.

Steps
1) Make the “graph pendulum” precise
Nodes (skills / specialists)

Feature extractors: color histogrammer, connected-component segmenter, symmetry/periodicity detector, shape-primitive detector, object tracker (across train pairs), affine+morphology probes, rule miners (mapping colors/positions).
Reasoners / composers: hypothesis generator (DSL templates), constraint aggregator, program synthesizer, counterexample finder, repairer, explainer.
Executors: fast grid VM for your DSL (or ICECUBER), unit-test runner from train→test.
Critics: metric computers (IoU, Hamming, object-wise correspondence score), failure profiler (what changed? which objects diverged?).

Edges & geometry

Angle (θ) between two nodes = similarity of behavior vectors. Build a vector per node from:
(i) I/O signatures on a probe set of ARC mini-cases,
(ii) latent descriptors (e.g., histogram of detected primitives / rule types),
(iii) “what they enable” (downstream success deltas).
Use cosine similarity ⇒ θ = arccos(sim).

Distance (d) between nodes = graph path length with edge costs = 1/sim or (1−sim).

Over time, edge weights update via observed causal utility: Δ success when traversing A→B vs skipping B.

Dynamics

A trajectory is a path from sensory nodes → hypothesis/program → execution → critique → (optional) local repair loop.

Small perturbations = minor prompt tweaks, random seeds, or deliberately injected micro-variations in early feature nodes (e.g., different segmentation thresholds).

The system is a discrete-time dynamical system on state = (current hypothesis, artifact set, node activations) with a controller that chooses next node.

2) Measuring chaos vs stability
Local stability probes

Empirical Lyapunov-like indicator: duplicate run K times with ε-perturbations in early nodes; track divergence of final metrics (IoU, object mapping accuracy). Positive “exponent” ⇒ chaotic region.

Sensitivity sweep: finite-difference sensitivity of final score to early node knobs (thresholds, seed). Low sensitivity ⇒ stable basin.

Trajectory entropy: entropy of node visitation distributions across repeats; low entropy clusters = attractors.

Bifurcation map: vary one upstream parameter and plot regime shifts in success pattern type (e.g., rule family selected).

Landscape construction

Represent each trajectory by a fixed-length summary vector:
[visited nodes histogram | average θ along path | cumulative utility | final scores | failure taxonomy].

Reduce with UMAP or t-SNE; cluster to find stable regions (tight clusters with low intra-variance, consistent success modes) vs chaotic spray.

Maintain region descriptors: which node motifs recur, which rule families succeed, typical failure modes.

3) Controller: how to move in the landscape
Base controller

Start greedy in stable basins: prefer edges with high historical utility and low measured sensitivity.

Exploration policy: multi-armed bandit over basin IDs; within a basin use MCTS or beam search over node paths with pruning by critic feedback.

Stability-aware exploration

If in a chaotic patch (high sensitivity, high trajectory entropy), backtrack to nearest “low-θ curvature” junction (edges whose addition didn’t amplify variance historically) and try an orthogonal node motif.

Repair loops (local micro-cycles)

Critic pinpoints failure type (e.g., color remap correct, object placement off by translation). Route to the minimal-delta repairer node (translation/scale/tiling adjuster) instead of regenerating from scratch.

4) Concrete benefits

More reliable search: Stability filtering kills rabbit holes where LLM randomness swamps the signal. You’ll reuse motifs that behave consistently on ARC’s tiny datasets.

Compositional reuse: Angles cluster semantically similar skills—your graph self-organizes into “feature families” and “reasoning families,” improving transfer across tasks.

Causal credit assignment: Edge utilities quantify “A→B was actually useful,” not just correlation, so over time the topology reflects causal pipelines that work.

Sample efficiency: ARC gives few pairs; a stability-biased controller squeezes more signal from them by preferring low-variance chains.

Diagnosability: Chaotic regions + critic traces explain why a hypothesis failed (good for automated repair and for human insight).

5) Pitfalls & guardrails

Angle definition leakage: If behavior vectors use downstream success too heavily, early nodes look similar just because later nodes fixed things. Fix: separate intrinsic descriptors (I/O signatures) from extrinsic utility; use both and regularize.

LLM variance: Temperature/noise can fake chaos. Fix: deterministic decoding for analysis passes; confine randomness to controlled perturbations.

Cost blow-up: Graph runs can explode. Fix: caching/memoization per node+input; canonicalize artifacts; early-stopping via cheap critics.

Overfitting basins: Stable ≠ correct universally. Fix: basin generalization test—hold-out micro-probes; penalize basins that don’t transfer.

6) How to implement v0 (tight loop)

Probe bank: 50–100 tiny synthetic ARC-like snippets covering primitives (mirror, flood fill, object copy/move, color remap, crop, tiling).

Node library (start minimal):
Segmentation/objects (connected components), symmetry detector, periodicity, bounding-box grouper.
Transformation proposers: translate/rotate/scale, paint-by-rule, color map.
Program synthesizer over your DSL (ICECUBER) with ~20 templates.
Critic: IoU, object alignment report, failure taxonomy.

Angles: run each node on the probe bank; create 256-d behavior vectors (counts of primitives found, success flags, output descriptors). Cosine for θ; edge costs = 1−cos.

Controller: best-first over paths of length ≤ L with utility = predicted success − sensitivity penalty. Keep top-k trajectories (beam).

Stability meter: For current best path, run 5 ε-perturbations up front; if variance > τ, demote basin and branch to the nearest low-variance junction.

Repair loop: If failure report says “placement offset”, route to translation search; if “color mismatch”, route to color remap fitter; etc.

Analytics: Log every trajectory vector; project with UMAP; mark clusters as basins. Persist edge utilities.

7) Learning over time

Edge utility updates: Δw(A→B) ← α · (final_score_gain − expected_gain).

Meta-parameters: learn per-node trust to adapt exploration budgets.

Option policies: collapse successful subpaths into macro-nodes (“detect-objects → align → copy”) with their own angle/utility signatures.

Stability regularizer: preference term in the controller objective for low-sensitivity subgraphs (acts like a Lyapunov prior).
