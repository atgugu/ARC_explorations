# Curiosity as a First-Class Control Signal in ARC-AGI
**Operationalizing Learnable Novelty across Generator, Workspace, and Navigator**

**Keywords:** ARC, curiosity, information gain, learning progress, epistemic uncertainty, empowerment, global workspace, stability-aware search

---

## Abstract

We formalize **human-like curiosity** as a first-class decision signal in a neurosymbolic ARC system. Curiosity is not random exploration; it is **expected learning progress from acquiring new evidence**. We operationalize it at three levels: (i) the **Generator** chooses *what to practice*, (ii) the **Workspace** decides *which hypothesis to think about next*, and (iii) the **Navigator** selects *where search is stable vs. interesting*. We give concrete math and implementable scores using **Bayesian surprise**, **epistemic uncertainty**, **learning progress (LP)**, **empowerment**, and **information gain (IG)**. We add a **curiosity budget & ledger** to keep exploration efficient and safe, and provide ARC-specific mechanics and evaluation plans.

---

## 1) Curiosity: The Core Principle

Humans aren't maximally novelty-seeking; we're drawn to **learnable novelty**—situations that promise progress. Define curiosity as

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BCuriosity%7D%28%5Ccdot%29%20%5C%3B%3D%5C%3B%20%5Cunderbrace%7B%5Ctext%7BNovelty%7D%7D_%7B%5Ctext%7Bwe%20haven%27t%20seen%20this%7D%7D%20%5Ctimes%20%5Cunderbrace%7B%5Ctext%7BLearnability%7D%7D_%7B%5Ctext%7Bwe%20can%20improve%20here%7D%7D%20%5Ctimes%20%5Cunderbrace%7B%5Ctext%7BUsefulness%7D%7D_%7B%5Ctext%7Bit%20helps%20future%20tasks%7D%7D."></p>


We implement this with measurable signals: **Bayesian surprise**, **epistemic uncertainty**, **learning progress (LP)**, **empowerment** (control over outcomes), and **task informativeness (IG)**.

---

## 2) Curiosity Signals (Concrete, Computable)

### A. Bayesian Surprise (any model or module <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=M" style="vertical-align: middle;">)
Given prior <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=p%28%5Ctheta%29" style="vertical-align: middle;">, posterior <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=p%28%5Ctheta%5Cmid%20%5Cmathcal%7BD%7D%5Ccup%5C%7Be%5C%7D%29" style="vertical-align: middle;"> after evidence <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=e" style="vertical-align: middle;">:

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BSurprise%7D_M%28e%29%20%5C%3B%3D%5C%3B%20%5Cmathrm%7BKL%7D%5C%21%5Cleft%5Bp%28%5Ctheta%5Cmid%20%5Cmathcal%7BD%7D%5Ccup%5C%7Be%5C%7D%29%20%5C%2C%5C%7C%5C%2C%20p%28%5Ctheta%5Cmid%20%5Cmathcal%7BD%7D%29%5Cright%5D."></p>

**Use:** Generator (schema priors), Workspace (rule proposer), Navigator (edge utilities).

### B. Epistemic Uncertainty (predictive variance)
Approximate with ensembles / MC-dropout / SWAG:

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BVar%7D_%7B%5Ctext%7Bepistemic%7D%7D%5C%21%5Cleft%5B%5Cmathrm%7Bscore%7D%28h%29%5Cright%5D."></p>

High variance flags promising knowledge gaps.

### C. Learning Progress (LP)
For any metric <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=m" style="vertical-align: middle;"> (accuracy, solve rate, time-to-solve, stability variance):

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BLP%7D%28t%29%20%5C%3B%3D%5C%3B%20m%28t%29%20-%20m%28t-%5CDelta%29."></p>

Prioritize contexts where LP has been positive recently (the **Goldilocks** zone).

### D. Information Gain (IG) about policies or priors)

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BIG%7D%20%5C%3B%3D%5C%3B%20%5Cmathbb%7BE%7D_%7B%5Ctext%7Boutcome%7D%7D%5C%21%5Cleft%5B%5Cmathrm%7BKL%7D%5Cbig%28p%28%5Cphi%20%5Cmid%20%5Ctext%7Boutcome%7D%29%20%5C%2C%5C%7C%5C%2C%20p%28%5Cphi%29%5Cbig%29%5Cright%5D%2C"></p>

where <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=%5Cphi" style="vertical-align: middle;"> are parameters of proposer/critic/executor.

### E. Empowerment (controllability)
Maximize mutual information between actions <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=A" style="vertical-align: middle;"> and reachable states <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=S" style="vertical-align: middle;">:

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BEmpower%7D%28s%29%20%5C%3B%5Capprox%5C%3B%20I%28A%3B%20S%27%20%5Cmid%20S%3Ds%29."></p>

Prefer branches where small, interpretable actions produce reliable, diverse outcomes.

---

## 3) Where Curiosity Lives (by Subsystem)

### 3.1 Generator (the Scientist): *What should we practice next?*

**Task curiosity score**

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=C_%7B%5Ctext%7Btask%7D%7D%28%5Ctau%29%20%5C%3B%3D%5C%3B%20%5Calpha%5C%2C%5Cmathrm%7BIG%7D_%7B%5Ctext%7Bsolver%7D%7D%28%5Ctau%29%20%5C%3B%2B%5C%3B%20%5Cbeta%5C%2C%5Cmathrm%7BSurprise%7D_%7B%5Ctext%7Bprior%7D%7D%28%5Ctau%29%20%5C%3B%2B%5C%3B%20%5Cgamma%5C%2C%5Cmathrm%7BLP%7D_%7B%5Ctext%7Bforecast%7D%7D%28%5Ctau%29%20%5C%3B-%5C%3B%20%5Cdelta%5C%2C%5Cmathrm%7BRedundancy%7D%28%5Ctau%29."></p>


**Use:** Rank sampled tasks; select a Pareto-frontier of *(difficulty, curiosity, stability)* for the batch.

**Mechanic:** UCB-style scheduler per schema family:

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BUCB%7D_k%20%5C%3B%3D%5C%3B%20%5Chat%7B%5Cmu%7D_k%20%5C%3B%2B%5C%3B%20c%5Csqrt%7B%5Ctfrac%7B%5Cln%20N%7D%7Bn_k%7D%7D%2C"></p>

where arm <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=k" style="vertical-align: middle;"> is a schema bucket, <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cmu%7D_k" style="vertical-align: middle;"> = recent LP, <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=n_k" style="vertical-align: middle;"> pulls, <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=N" style="vertical-align: middle;"> total pulls.

---

### 3.2 Workspace (the Conductor): *Which hypothesis is worth thinking about next?*

For each candidate hypothesis <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=h" style="vertical-align: middle;"> (program edit / parameter set):

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BScore%7D%28h%29%20%5C%3B%3D%5C%3B%20%5Cunderbrace%7B%5Cmathrm%7BFit%7D%28h%29%7D_%7B%5Ctext%7Bcritic%7D%7D%20%5C%3B-%5C%3B%20%5Clambda%20%5Cunderbrace%7B%5Cmathrm%7BInstability%7D%28h%29%7D_%7B%5Ctext%7Bnavigator%7D%7D%20%5C%3B%2B%5C%3B%20%5Ceta%20%5Cunderbrace%7B%5Cmathrm%7BCuriosity%7D%28h%29%7D_%7B%5Ctext%7Bbelow%7D%7D."></p>

with

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BCuriosity%7D%28h%29%20%5C%3B%3D%5C%3B%20%5Calpha%5C%2C%5Cmathrm%7BVar%7D_%7B%5Ctext%7Bepistemic%7D%7D%5B%5Cmathrm%7BFit%7D%28h%29%5D%20%5C%3B%2B%5C%3B%20%5Cbeta%5C%2C%5Cmathrm%7BIG%7D%28h%29%20%5C%3B%2B%5C%3B%20%5Crho%5C%2C%5Cmathrm%7BEmpower%7D%28h%29."></p>


**Behavior:** hypotheses that are *plausible* but *uncertain and informative* win workspace admission—"I can learn something here."

---

### 3.3 Graph Pendulum (the Navigator): *Where is interesting but not chaotic?*

**Basin curiosity** combines stability with novelty:

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=C_%7B%5Ctext%7Bbasin%7D%7D%28b%29%20%5C%3B%3D%5C%3B%20%5Cunderbrace%7B%5Cexp%5C%21%5Cbig%28-%5Cmathrm%7BVar%7D_%7B%5Ctext%7Bstab%7D%7D%28b%29%5Cbig%29%7D_%7B%5Ctext%7Bstable%7D%7D%20%5Ccdot%20%5Cunderbrace%7B%5Cmathrm%7BNovelty%7D%28b%29%7D_%7B%5Ctext%7Brare%20motif%7D%7D%20%5Ccdot%20%5Cunderbrace%7B%5Cmathrm%7BLP%7D%28b%29%7D_%7B%5Ctext%7Brecent%20gains%7D%7D."></p>


**Edge curiosity** for trying new transitions <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=i%5C%21%5Cto%5C%21j" style="vertical-align: middle;">:

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=C_%7Bi%5Cto%20j%7D%20%5C%3B%3D%5C%3B%20%5Comega_1%5C%2C%5Cmathrm%7BIG%7D_%7Bi%5Cto%20j%7D%20%5C%3B%2B%5C%3B%20%5Comega_2%5C%2C%5Cmathrm%7BSurprise%7D_%7Bi%5Cto%20j%7D%20%5C%3B%2B%5C%3B%20%5Comega_3%5C%2C%5Cmathrm%7BCoverageGap%7D_%7B%5Cmathrm%7Bschema%7D%28j%29%7D."></p>


**Policy:** prefer **stable-but-uncertain** regions; downweight **chaotic** (high sensitivity, entropy) unless expected IG justifies a **bounded probe**.

---

## 4) Unifying Objective (per Decision)

At any choice point (task selection, hypothesis selection, path expansion), maximize

<p align="center"><img alt="equation" src="https://render.githubusercontent.com/render/math?math=U%20%5C%3B%3D%5C%3B%20%5Cunderbrace%7B%5Cmathbb%7BE%7D%5B%5Cmathrm%7BSolveGain%7D%5D%7D_%7B%5Ctext%7Bexploitation%7D%7D%20%5C%3B-%5C%3B%20%5Clambda%20%5Cunderbrace%7B%5Cmathrm%7BCompute%7D%7D_%7B%5Ctext%7Bbudget%7D%7D%20%5C%3B-%5C%3B%20%5Cmu%20%5Cunderbrace%7B%5Cmathrm%7BInstability%7D%7D_%7B%5Ctext%7Bnavigator%7D%7D%20%5C%3B%2B%5C%3B%20%5Ckappa%20%5Cunderbrace%7B%5Cmathrm%7BCuriosity%7D%7D_%7B%5Ctext%7BIG%2FLP%2FSurprise%7D%7D."></p>


Tune <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=%28%5Clambda%2C%5Cmu%2C%5Ckappa%29" style="vertical-align: middle;"> via meta-optimization; or adapt online with a **curiosity budget**.

---

## 5) The Curiosity Budget & Ledger (Practical Control)

- **Budget:** allocate a fraction <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=B%5Cin%5B0%2C1%5D" style="vertical-align: middle;"> of steps/evals to curiosity-driven choices. Start higher, anneal as confidence grows.  
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
- *Uncertainty:* ensemble variance of executor-fit after a proposed edit (e.g., "reflect+translate" with unknown offset).  
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

- **Stability gate:** curiosity expansions only if predicted instability < <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=%5Ctau" style="vertical-align: middle;">, or run as **bounded probes** (tiny budget, hard stop).  
- **De-leak novelty:** compute novelty on **intrinsic** descriptors (I/O signatures, grammar stats), not post-hoc solved outputs.  
- **Reproducibility:** deterministic decoding in analysis passes; inject noise only via controlled <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=%5Cvarepsilon" style="vertical-align: middle;">-perturbations.  
- **Ethical/compute limits:** curiosity budget cap per batch; prefer low-cost IG estimators (linear probes, small ensembles).

---

## 9) Evaluation Plan for Curiosity

- **Learning curves:** with vs. without curiosity (same compute).  
- **LP concentration:** proportion of gains attributable to curiosity-chosen actions.  
- **Data efficiency:** solves per training pair; executions per solve.  
- **Generalization:** improvement on held-out schema combos targeted by curiosity.  
- **Stability impact:** variance & trajectory entropy under curiosity schedules.  
- **Ablations:** remove IG, remove LP, remove empowerment, swap UCB with <img alt="inline-eq" src="https://render.githubusercontent.com/render/math?math=%5Cvarepsilon" style="vertical-align: middle;">-greedy.

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
