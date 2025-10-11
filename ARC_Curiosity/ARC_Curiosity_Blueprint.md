# Curiosity as a First-Class Control Signal in ARC-AGI
**Operationalizing Learnable Novelty across Generator, Workspace, and Navigator**

**Keywords:** ARC, curiosity, information gain, learning progress, epistemic uncertainty, empowerment, global workspace, stability-aware search

---

## Abstract

We formalize **human-like curiosity** as a first-class decision signal in a neurosymbolic ARC system. Curiosity is not random exploration; it is **expected learning progress from acquiring new evidence**. We operationalize it at three levels: (i) the **Generator** chooses *what to practice*, (ii) the **Workspace** decides *which hypothesis to think about next*, and (iii) the **Navigator** selects *where search is stable vs. interesting*. We give concrete math and implementable scores using **Bayesian surprise**, **epistemic uncertainty**, **learning progress (LP)**, **empowerment**, and **information gain (IG)**. We add a **curiosity budget & ledger** to keep exploration efficient and safe, and provide ARC-specific mechanics and evaluation plans.

---

## 1) Curiosity: The Core Principle

Humans aren’t maximally novelty-seeking; we’re drawn to **learnable novelty**—situations that promise progress. Define curiosity as
$$
\mathbf{Curiosity}(\cdot) \;=\; \underbrace{\text{Novelty}}_{\text{we haven't seen this}} \times \underbrace{\text{Learnability}}_{\text{we can improve here}} \times \underbrace{\text{Usefulness}}_{\text{it helps future tasks}}.
$$

We implement this with measurable signals: **Bayesian surprise**, **epistemic uncertainty**, **learning progress (LP)**, **empowerment** (control over outcomes), and **task informativeness (IG)**.

---

## 2) Curiosity Signals (Concrete, Computable)

### A. Bayesian Surprise (any model or module $M$)
Given prior $p(\theta)$, posterior $p(\theta\mid \mathcal{D}\cup\{e\})$ after evidence $e$:
$$
\mathrm{Surprise}_M(e) \;=\; \mathrm{KL}\!\left[p(\theta\mid \mathcal{D}\cup\{e\}) \,\|\, p(\theta\mid \mathcal{D})\right].
$$
**Use:** Generator (schema priors), Workspace (rule proposer), Navigator (edge utilities).

### B. Epistemic Uncertainty (predictive variance)
Approximate with ensembles / MC-dropout / SWAG:
$$
\mathrm{Var}_{\text{epistemic}}\!\left[\mathrm{score}(h)\right].
$$
High variance flags promising knowledge gaps.

### C. Learning Progress (LP)
For any metric $m$ (accuracy, solve rate, time-to-solve, stability variance):
$$
\mathrm{LP}(t) \;=\; m(t) - m(t-\Delta).
$$
Prioritize contexts where LP has been positive recently (the **Goldilocks** zone).

### D. Information Gain (IG) about policies or priors)
$$
\mathrm{IG} \;=\; \mathbb{E}_{\text{outcome}}\!\left[\mathrm{KL}\big(p(\phi \mid \text{outcome}) \,\|\, p(\phi)\big)\right],
$$
where $\phi$ are parameters of proposer/critic/executor.

### E. Empowerment (controllability)
Maximize mutual information between actions $A$ and reachable states $S$:
$$
\mathrm{Empower}(s) \;\approx\; I(A; S' \mid S=s).
$$
Prefer branches where small, interpretable actions produce reliable, diverse outcomes.

---

## 3) Where Curiosity Lives (by Subsystem)

### 3.1 Generator (the Scientist): *What should we practice next?*

**Task curiosity score**
$$
C_{\text{task}}(\tau) \;=\; \alpha\,\mathrm{IG}_{\text{solver}}(\tau) \;+\; \beta\,\mathrm{Surprise}_{\text{prior}}(\tau) \;+\; \gamma\,\mathrm{LP}_{\text{forecast}}(\tau) \;-\; \delta\,\mathrm{Redundancy}(\tau).
$$

**Use:** Rank sampled tasks; select a Pareto-frontier of *(difficulty, curiosity, stability)* for the batch.

**Mechanic:** UCB-style scheduler per schema family:
$$
\mathrm{UCB}_k \;=\; \hat{\mu}_k \;+\; c\sqrt{\tfrac{\ln N}{n_k}},
$$
where arm $k$ is a schema bucket, $\hat{\mu}_k$ = recent LP, $n_k$ pulls, $N$ total pulls.

---

### 3.2 Workspace (the Conductor): *Which hypothesis is worth thinking about next?*

For each candidate hypothesis $h$ (program edit / parameter set):
$$
\mathrm{Score}(h) \;=\; \underbrace{\mathrm{Fit}(h)}_{\text{critic}} \;-\; \lambda \underbrace{\mathrm{Instability}(h)}_{\text{navigator}} \;+\; \eta \underbrace{\mathrm{Curiosity}(h)}_{\text{below}}.
$$
with
$$
\mathrm{Curiosity}(h) \;=\; \alpha\,\mathrm{Var}_{\text{epistemic}}[\mathrm{Fit}(h)] \;+\; \beta\,\mathrm{IG}(h) \;+\; \rho\,\mathrm{Empower}(h).
$$

**Behavior:** hypotheses that are *plausible* but *uncertain and informative* win workspace admission—“I can learn something here.”

---

### 3.3 Graph Pendulum (the Navigator): *Where is interesting but not chaotic?*

**Basin curiosity** combines stability with novelty:
$$
C_{\text{basin}}(b) \;=\; \underbrace{\exp\!\big(-\mathrm{Var}_{\text{stab}}(b)\big)}_{\text{stable}} \cdot \underbrace{\mathrm{Novelty}(b)}_{\text{rare motif}} \cdot \underbrace{\mathrm{LP}(b)}_{\text{recent gains}}.
$$

**Edge curiosity** for trying new transitions $i\!\to\!j$:
$$
C_{i\to j} \;=\; \omega_1\,\mathrm{IG}_{i\to j} \;+\; \omega_2\,\mathrm{Surprise}_{i\to j} \;+\; \omega_3\,\mathrm{CoverageGap}_{\mathrm{schema}(j)}.
$$

**Policy:** prefer **stable-but-uncertain** regions; downweight **chaotic** (high sensitivity, entropy) unless expected IG justifies a **bounded probe**.

---

## 4) Unifying Objective (per Decision)

At any choice point (task selection, hypothesis selection, path expansion), maximize
$$
U \;=\; \underbrace{\mathbb{E}[\mathrm{SolveGain}]}_{\text{exploitation}} \;-\; \lambda \underbrace{\mathrm{Compute}}_{\text{budget}} \;-\; \mu \underbrace{\mathrm{Instability}}_{\text{navigator}} \;+\; \kappa \underbrace{\mathrm{Curiosity}}_{\text{IG/LP/Surprise}}.
$$

Tune $(\lambda,\mu,\kappa)$ via meta-optimization; or adapt online with a **curiosity budget**.

---

## 5) The Curiosity Budget & Ledger (Practical Control)

- **Budget:** allocate a fraction $B\in[0,1]$ of steps/evals to curiosity-driven choices. Start higher, anneal as confidence grows.  
- **Ledger:** per module/basin/edge, track *curiosity spend* vs. *learning gain*. Reassign budget toward components with best **LP-per-cost**.

```text
if LP_per_cost(component) < threshold for T windows:
    shrink curiosity quota for that component
else:
    expand curiosity quota
```

---

## 6) Concrete Mechanics (ARC-Specific)

**A. Curiosity for color/shape rule mining**  
- *Novelty:* low cosine similarity of detected palettes/shape-grams to memory bank.  
- *IG:* estimated reduction in error of rule prior (e.g., mapping colors→roles).  
- *LP:* recent improvement when such rules were explored on similar tasks.

**B. Curiosity for program edits**  
- *Uncertainty:* ensemble variance of executor-fit after a proposed edit (e.g., “reflect+translate” with unknown offset).  
- *Empowerment:* edit families that historically yield diverse controllable outcomes (offset tweaks reliably fix placement).

**C. Curiosity for generator curricula**  
- Target **schema holes** (e.g., symmetry + recolor-by-role).  
- Prefer **nearby** novel compositions that share objects but alter relations (learnable novelty).

---

## 7) Minimal Code Sketch (Pseudo)

```python
def curiosity_score_task(tau, solver_posterior, prior):
    IG = expected_KL_posterior_contraction(tau, solver_posterior)
    surprise = KL(prior_after(tau), prior)
    LP_forecast = meta.predict_learning_progress(tau)
    redundancy = similarity_to_recent(tau)
    return alpha*IG + beta*surprise + gamma*LP_forecast - delta*redundancy

def curiosity_score_hypothesis(h):
    fit_mu, fit_var = evaluator.predict_fit(h, epistemic=True)
    IG = expected_policy_info_gain(h)
    empower = empowerment_estimator(h)
    return alpha*fit_var + beta*IG + rho*empower

def navigator_expand(path):
    for edge in admissible_edges(path.tail):
        instability = stability_meter.variance(edge)
        curiosity = edge_IG(edge) + edge_surprise(edge) + coverage_gap(edge)
        utility = predict_solve_gain(edge) - mu*instability + kappa*curiosity
        push(edge, utility)
```

---

## 8) Guardrails (to Keep Curiosity Healthy)

- **Stability gate:** curiosity expansions only if predicted instability < $\tau$, or run as **bounded probes** (tiny budget, hard stop).  
- **De-leak novelty:** compute novelty on **intrinsic** descriptors (I/O signatures, grammar stats), not post-hoc solved outputs.  
- **Reproducibility:** deterministic decoding in analysis passes; inject noise only via controlled $\varepsilon$-perturbations.  
- **Ethical/compute limits:** curiosity budget cap per batch; prefer low-cost IG estimators (linear probes, small ensembles).

---

## 9) Evaluation Plan for Curiosity

- **Learning curves:** with vs. without curiosity (same compute).  
- **LP concentration:** proportion of gains attributable to curiosity-chosen actions.  
- **Data efficiency:** solves per training pair; executions per solve.  
- **Generalization:** improvement on held-out schema combos targeted by curiosity.  
- **Stability impact:** variance & trajectory entropy under curiosity schedules.  
- **Ablations:** remove IG, remove LP, remove empowerment, swap UCB with $\varepsilon$-greedy.

---

## 10) How This Mirrors Human Curiosity

- **Goldilocks zone (LP):** we seek tasks neither trivial nor impossible.  
- **Surprise & uncertainty:** attention spikes when a pattern conflicts with priors but is still graspable.  
- **Empowerment:** we prefer environments where our actions causally matter.  
- **Broad-to-narrow:** the Generator explores families (broad), the Workspace focuses hypotheses (narrow), and the Navigator ensures the terrain is learnable.

---

## TL;DR

Make curiosity a scalar **decision bonus** grounded in **information gain**, **learning progress**, **uncertainty**, and **empowerment**.  
Deploy it **hierarchically**: the **Generator** chooses *what to practice*, the **Workspace** selects *which hypothesis to think about*, and the **Navigator** ensures we do it in *stable, learnable basins*.  
This is human-like curiosity—formalized and plugged into our ARC-AGI.
