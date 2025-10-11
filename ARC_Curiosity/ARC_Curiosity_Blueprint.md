# Curiosity as a First-Class Control Signal in ARC-AGI
**Operationalizing Learnable Novelty across Generator, Workspace, and Navigator**

**Keywords:** ARC, curiosity, information gain, learning progress, epistemic uncertainty, empowerment, global workspace, stability-aware search

---

## Abstract

We formalize **human-like curiosity** as a first-class decision signal in a neurosymbolic ARC system. Curiosity is not random exploration; it is **expected learning progress from acquiring new evidence**. We operationalize it at three levels: (i) the **Generator** chooses *what to practice*, (ii) the **Workspace** decides *which hypothesis to think about next*, and (iii) the **Navigator** selects *where search is stable vs. interesting*. We give concrete math and implementable scores using **Bayesian surprise**, **epistemic uncertainty**, **learning progress (LP)**, **empowerment**, and **information gain (IG)**. We add a **curiosity budget & ledger** to keep exploration efficient and safe, and provide ARC-specific mechanics and evaluation plans.

---

## 1) Curiosity: The Core Principle

Humans aren’t maximally novelty-seeking; we’re drawn to **learnable novelty**—situations that promise progress. Define curiosity as
![Curiosity Definition](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathbf{Curiosity}(\cdot)%20\;=\;%20\underbrace{\text{Novelty}}_{\text{we%20haven't%20seen%20this}}%20\times%20\underbrace{\text{Learnability}}_{\text{we%20can%20improve%20here}}%20\times%20\underbrace{\text{Usefulness}}_{\text{it%20helps%20future%20tasks}})

We implement this with measurable signals: **Bayesian surprise**, **epistemic uncertainty**, **learning progress (LP)**, **empowerment** (control over outcomes), and **task informativeness (IG)**.

---

## 2) Curiosity Signals (Concrete, Computable)

### A. Bayesian Surprise (any model or module ![M](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}M))
Given prior ![p(theta)](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p(\theta)), posterior ![p(theta|D cup e)](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}p(\theta\mid%20\mathcal{D}\cup\{e\})) after evidence ![e](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}e):
![Bayesian Surprise](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathrm{Surprise}_M(e)%20\;=\;%20\mathrm{KL}\!\left[p(\theta\mid%20\mathcal{D}\cup\{e\})%20\,\|\,%20p(\theta\mid%20\mathcal{D})\right])
**Use:** Generator (schema priors), Workspace (rule proposer), Navigator (edge utilities).

### B. Epistemic Uncertainty (predictive variance)
Approximate with ensembles / MC-dropout / SWAG:
![Epistemic Uncertainty](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathrm{Var}_{\text{epistemic}}\!\left[\mathrm{score}(h)\right])
High variance flags promising knowledge gaps.

### C. Learning Progress (LP)
For any metric ![m](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}m) (accuracy, solve rate, time-to-solve, stability variance):
![Learning Progress](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathrm{LP}(t)%20\;=\;%20m(t)%20-%20m(t-\Delta))
Prioritize contexts where LP has been positive recently (the **Goldilocks** zone).

### D. Information Gain (IG) about policies or priors)
![Information Gain](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathrm{IG}%20\;=\;%20\mathbb{E}_{\text{outcome}}\!\left[\mathrm{KL}\big(p(\phi%20\mid%20\text{outcome})%20\,\|\,%20p(\phi)\big)\right])

where ![phi](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\phi) are parameters of proposer/critic/executor.

### E. Empowerment (controllability)
Maximize mutual information between actions ![A](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}A) and reachable states ![S](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}S):
![Empowerment](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathrm{Empower}(s)%20\;\approx\;%20I(A;%20S'%20\mid%20S=s))
Prefer branches where small, interpretable actions produce reliable, diverse outcomes.

---

## 3) Where Curiosity Lives (by Subsystem)

### 3.1 Generator (the Scientist): *What should we practice next?*

**Task curiosity score**

![Task Curiosity Score](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}C_{\text{task}}(\tau)%20\;=\;%20\alpha\,\mathrm{IG}_{\text{solver}}(\tau)%20\;+\;%20\beta\,\mathrm{Surprise}_{\text{prior}}(\tau)%20\;+\;%20\gamma\,\mathrm{LP}_{\text{forecast}}(\tau)%20\;-\;%20\delta\,\mathrm{Redundancy}(\tau))

**Use:** Rank sampled tasks; select a Pareto-frontier of *(difficulty, curiosity, stability)* for the batch.

**Mechanic:** UCB-style scheduler per schema family:

![UCB Formula](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathrm{UCB}_k%20\;=\;%20\hat{\mu}_k%20\;+\;%20c\sqrt{\frac{\ln%20N}{n_k}})

where arm ![k](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}k) is a schema bucket, ![mu_k](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\hat{\mu}_k) = recent LP, ![n_k](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}n_k) pulls, ![N](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}N) total pulls.

---

### 3.2 Workspace (the Conductor): *Which hypothesis is worth thinking about next?*

For each candidate hypothesis ![h](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}h) (program edit / parameter set):
![Hypothesis Score](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathrm{Score}(h)%20\;=\;%20\underbrace{\mathrm{Fit}(h)}_{\text{critic}}%20\;-\;%20\lambda%20\underbrace{\mathrm{Instability}(h)}_{\text{navigator}}%20\;+\;%20\eta%20\underbrace{\mathrm{Curiosity}(h)}_{\text{below}})
with

![Hypothesis Curiosity](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathrm{Curiosity}(h)%20\;=\;%20\alpha\,\mathrm{Var}_{\text{epistemic}}[\mathrm{Fit}(h)]%20\;+\;%20\beta\,\mathrm{IG}(h)%20\;+\;%20\rho\,\mathrm{Empower}(h))

**Behavior:** hypotheses that are *plausible* but *uncertain and informative* win workspace admission—“I can learn something here.”

---

### 3.3 Graph Pendulum (the Navigator): *Where is interesting but not chaotic?*

**Basin curiosity** combines stability with novelty:
![Basin Curiosity](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}C_{\text{basin}}(b)%20\;=\;%20\underbrace{\exp\!\big(-\mathrm{Var}_{\text{stab}}(b)\big)}_{\text{stable}}%20\cdot%20\underbrace{\mathrm{Novelty}(b)}_{\text{rare%20motif}}%20\cdot%20\underbrace{\mathrm{LP}(b)}_{\text{recent%20gains}})

**Edge curiosity** for trying new transitions ![i to j](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}i\!\to\!j):
![Edge Curiosity](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}C_{i\to%20j}%20\;=\;%20\omega_1\,\mathrm{IG}_{i\to%20j}%20\;+\;%20\omega_2\,\mathrm{Surprise}_{i\to%20j}%20\;+\;%20\omega_3\,\mathrm{CoverageGap}_{\mathrm{schema}(j)})

**Policy:** prefer **stable-but-uncertain** regions; downweight **chaotic** (high sensitivity, entropy) unless expected IG justifies a **bounded probe**.

---

## 4) Unifying Objective (per Decision)

At any choice point (task selection, hypothesis selection, path expansion), maximize

![Unifying Objective](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}U%20\;=\;%20\underbrace{\mathbb{E}[\mathrm{SolveGain}]}_{\text{exploitation}}%20\;-\;%20\lambda%20\underbrace{\mathrm{Compute}}_{\text{budget}}%20\;-\;%20\mu%20\underbrace{\mathrm{Instability}}_{\text{navigator}}%20\;+\;%20\kappa%20\underbrace{\mathrm{Curiosity}}_{\text{IG/LP/Surprise}})

Tune ![lambda mu kappa](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}(\lambda,\mu,\kappa)) via meta-optimization; or adapt online with a **curiosity budget**.

---

## 5) The Curiosity Budget & Ledger (Practical Control)

- **Budget:** allocate a fraction ![B in 0,1](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}B\in[0,1]) of steps/evals to curiosity-driven choices. Start higher, anneal as confidence grows.  
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

- **Stability gate:** curiosity expansions only if predicted instability < ![tau](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\tau), or run as **bounded probes** (tiny budget, hard stop).  
- **De-leak novelty:** compute novelty on **intrinsic** descriptors (I/O signatures, grammar stats), not post-hoc solved outputs.  
- **Reproducibility:** deterministic decoding in analysis passes; inject noise only via controlled ![epsilon](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\varepsilon)-perturbations.  
- **Ethical/compute limits:** curiosity budget cap per batch; prefer low-cost IG estimators (linear probes, small ensembles).

---

## 9) Evaluation Plan for Curiosity

- **Learning curves:** with vs. without curiosity (same compute).  
- **LP concentration:** proportion of gains attributable to curiosity-chosen actions.  
- **Data efficiency:** solves per training pair; executions per solve.  
- **Generalization:** improvement on held-out schema combos targeted by curiosity.  
- **Stability impact:** variance & trajectory entropy under curiosity schedules.  
- **Ablations:** remove IG, remove LP, remove empowerment, swap UCB with ![epsilon](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\varepsilon)-greedy.

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
