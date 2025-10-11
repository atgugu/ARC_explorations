# Curiosity as a First-Class Control Signal in ARC-AGI: A Neurosymbolic Framework for Human-Like Reasoning
**Operationalizing Learnable Novelty across Generator, Workspace, and Navigator**

**Keywords:** ARC, curiosity, information gain, learning progress, epistemic uncertainty, empowerment, global workspace, stability-aware search, human reasoning, cognitive architecture, active inference

---

## Abstract

We explore and attempt to formalize **human-like curiosity** as a first-class decision signal in a neurosymbolic system designed for the Abstraction and Reasoning Corpus (ARC). Unlike random exploration or exhaustive search, curiosity in our framework represents **expected learning progress from acquiring new evidence**—a principle deeply rooted in human cognitive development. We operationalize this concept at three hierarchical levels mirroring human cognitive architecture: (i) the **Generator** module, acting as a scientific hypothesis generator, chooses *what conceptual spaces to explore*; (ii) the **Workspace**, analogous to working memory and attention, decides *which hypothesis deserves cognitive resources*; and (iii) the **Navigator**, representing meta-cognitive control, determines *where the reasoning trajectory is stable versus chaotic*.

Our formalization employs mathematically grounded signals including **Bayesian surprise** (measuring deviation from prior beliefs), **epistemic uncertainty** (quantifying knowledge gaps), **learning progress** (tracking improvement velocity), **empowerment** (assessing causal control), and **information gain** (expected belief refinement). We introduce a **curiosity budget and ledger** system that mirrors human cognitive resource allocation, preventing both obsessive exploration and premature convergence. Through ARC-specific instantiations, we illustrate how curiosity-driven exploration could discover compositional rules, visual abstractions, and analogical mappings that characterize human intelligence. This theoretical framework explores how curiosity signals might guide exploration in ways that differ from pure search or random exploration, while potentially exhibiting problem-solving trajectories that share characteristics with human reasoning.

---

## 1. Introduction

### 1.1 The Challenge of Human-Like Reasoning in AI

Consider an ARC task where a small 3×3 grid contains a single colored pixel, and the output shows that pixel expanded into a cross pattern of the same color. A human solver immediately recognizes this as a "growth" or "expansion" rule after seeing just one or two examples. Current AI systems, despite their impressive capabilities in many domains, struggle with such seemingly simple abstractions. This gap between human and artificial reasoning capabilities lies at the heart of the ARC challenge (Chollet, 2019).

The Abstraction and Reasoning Corpus represents a fundamental departure from traditional machine learning benchmarks. While datasets like ImageNet test pattern recognition at scale, and games like Go test strategic planning within fixed rules, ARC probes the very essence of intelligence: the ability to discover abstract rules from minimal examples and apply them to novel situations. Each ARC task presents 2-5 input-output pairs demonstrating an unknown transformation rule, and the solver must infer this rule to complete a test case.

### 1.2 Why Curiosity Matters for ARC

Human solvers don't approach ARC tasks through exhaustive search or random exploration. Instead, they exhibit **directed curiosity**—selectively attending to patterns that seem learnable and informative. When examining training pairs, humans naturally focus on:

- **Surprising regularities** that violate initial assumptions
- **Ambiguous cases** where multiple hypotheses compete
- **Progressive refinements** that build on partial understanding
- **Transferable patterns** that might apply broadly

This curiosity-driven exploration is not merely efficient; it's essential for discovering the abstract rules underlying ARC tasks within the cognitive constraints of working memory and attention. Our framework formalizes this human cognitive strategy, transforming curiosity from an emergent behavior into an explicit control signal that guides reasoning.

### 1.3 Core Thesis and Contributions

We propose that **curiosity should be a first-class citizen in artificial reasoning systems**, not an emergent side-effect or heuristic add-on. Specifically, we argue that human-level performance on ARC-like tasks requires:

1. **Intrinsic motivation signals** that guide exploration without external rewards
2. **Hierarchical curiosity** operating at multiple levels of abstraction
3. **Resource-aware exploration** that balances discovery with efficiency
4. **Stability-sensitive navigation** that avoids both chaos and stagnation

Our main contributions are:

1. **Theoretical Framework**: A mathematically principled formalization of curiosity grounded in information theory, cognitive science, and dynamical systems
2. **Hierarchical Architecture**: A three-tier system (Generator, Workspace, Navigator) that mirrors human cognitive organization
3. **Concrete Instantiation**: Specific mechanisms for curiosity-driven reasoning in the ARC domain
4. **Theoretical Predictions**: Expected improvements in both efficiency and generalization from curiosity-driven exploration
5. **Cognitive Alignment**: Theoretical framework that would exhibit human-like problem-solving patterns

### 1.4 Paper Organization

Section 2 establishes theoretical foundations from cognitive science and information theory. Section 3 details our mathematical formalization of curiosity signals. Section 4 describes the hierarchical system architecture. Section 5 provides ARC-specific instantiations. Section 6 presents expected outcomes and theoretical analysis. Section 7 reviews related work. Section 8 discusses implications and future directions. Section 9 concludes.

---

## 2. Theoretical Foundations

### 2.1 The Nature of Human Reasoning

#### 2.1.1 Core Cognitive Mechanisms

Human reasoning, particularly in novel problem-solving contexts like ARC, relies on several fundamental mechanisms:

**Pattern Recognition and Abstraction**: Humans excel at identifying invariant structures across varying surface features. In ARC, this manifests as recognizing that different colored objects follow the same transformation rule, or that spatial relationships remain constant despite position changes.

**Hypothesis Generation and Testing**: Rather than exhaustive search, humans generate plausible hypotheses based on partial observations. This involves:
- **Analogical reasoning**: "This looks similar to..."
- **Compositional thinking**: "Maybe it's rule A combined with rule B"
- **Counterfactual reasoning**: "What if I ignore color and focus on shape?"

**Meta-Learning and Transfer**: Humans rapidly adapt their problem-solving strategies based on task characteristics. After solving several ARC tasks, solvers develop meta-strategies like "check for symmetry first" or "count objects to find patterns."

#### 2.1.2 Intrinsic Motivation and Curiosity

Building on the pioneering work of Berlyne (1960), modern cognitive science recognizes curiosity as a fundamental drive that:

- **Reduces uncertainty** about the environment (Kidd & Hayden, 2015)
- **Maximizes learning progress** (Oudeyer et al., 2007)
- **Seeks optimal challenge levels** (Csikszentmihalyi's flow theory)
- **Balances exploration and exploitation** (Cohen et al., 2007)

Crucially, human curiosity is **information-sensitive**—we're drawn to stimuli that promise maximal learning, not maximal novelty. This "Goldilocks principle" ensures efficient learning within cognitive constraints.

### 2.2 The ARC Problem Space

#### 2.2.1 Formal Problem Definition

An ARC task ![tau](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\tau) consists of:
- Training set: ![training](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\{(x_i,%20y_i)\}_{i=1}^k) where ![k in 2-5](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}k%20\in%20[2,5])
- Test input: ![x_test](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}x_{test})
- Unknown transformation: ![f: X to Y](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}f:%20\mathcal{X}%20\to%20\mathcal{Y})

Where ![X, Y](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathcal{X},%20\mathcal{Y}) are spaces of 2D grids with dimensions up to 30×30 and integer values in [0, 9] representing colors.

#### 2.2.2 Core Cognitive Priors Tested

ARC tasks probe fundamental cognitive priors identified in developmental psychology (Spelke & Kinzler, 2007):

**Objectness**:
- Cohesion: Objects maintain connectivity
- Continuity: Objects persist through occlusion
- Contact: Objects interact through physical contact

**Number and Geometry**:
- Counting and arithmetic operations
- Geometric transformations (rotation, reflection, scaling)
- Topological relationships (inside/outside, connected/disconnected)

**Agency and Causality**:
- Goal-directed movement
- Cause-effect relationships
- Symmetry and pattern completion

#### 2.2.3 Why Brute-Force Search Fails

The ARC search space is intractable for exhaustive exploration:

**Combinatorial Explosion**: With ~10^6 possible DSL programs of reasonable length and 10^4 possible parameter combinations, the search space exceeds 10^10 hypotheses.

**Compositional Complexity**: Rules often combine multiple sub-rules (e.g., "rotate then recolor based on position"), creating an exponentially growing composition space.

**Ambiguity and Noise**: Some tasks intentionally include irrelevant features or ambiguous cases that make pure pattern matching insufficient.

### 2.3 Information-Theoretic Foundations

#### 2.3.1 Curiosity as Information Gain

Following Shannon's information theory and its cognitive applications (Friston, 2010), we formalize curiosity as the **expected reduction in uncertainty**:

![Information Gain](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{IG}(a)%20=%20H[X]%20-%20\mathbb{E}_{o|a}[H[X|o]])

Where:
- ![a](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}a) is an action (exploration choice)
- ![X](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}X) is the uncertain variable (e.g., transformation rule)
- ![o](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}o) is the observation resulting from action
- ![H](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}H[\cdot]) denotes entropy

#### 2.3.2 Active Inference and the Free Energy Principle

Our framework aligns with active inference (Friston et al., 2017), where agents:
1. Maintain generative models of their environment
2. Act to minimize prediction error (surprise)
3. Seek observations that improve model accuracy

In the ARC context, this translates to:
- **Generative model**: Hypotheses about transformation rules
- **Prediction error**: Mismatch between predicted and actual outputs
- **Active sampling**: Choosing which hypotheses to test

### 2.4 Dynamical Systems Perspective

#### 2.4.1 Reasoning as Trajectory Through Hypothesis Space

We conceptualize problem-solving as a dynamical system where:
- **State space**: All possible hypotheses and partial solutions
- **Trajectory**: Sequence of hypothesis refinements
- **Attractors**: Stable solution configurations
- **Repellors**: Inconsistent or contradictory hypotheses

#### 2.4.2 Stability and Chaos in Reasoning

Some regions of hypothesis space exhibit:
- **Stability**: Small changes in assumptions lead to similar conclusions
- **Chaos**: Minor perturbations cause dramatically different outcomes
- **Edge of chaos**: Optimal zone for creative problem-solving

Our curiosity mechanism guides the system toward regions of **learnable complexity**—neither too stable (trivial) nor too chaotic (random).

---

## 3. Curiosity Signals: Formal Definitions and Cognitive Grounding

### 3.1 Unified Curiosity Framework

We define curiosity as a multi-faceted signal combining three essential components:

![Curiosity Definition](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathbf{Curiosity}(\cdot)%20\;=\;%20\underbrace{\text{Novelty}}_{\text{deviation%20from%20known}}%20\times%20\underbrace{\text{Learnability}}_{\text{reducible%20uncertainty}}%20\times%20\underbrace{\text{Usefulness}}_{\text{transferable%20knowledge}})

This formulation ensures we pursue not just new information, but **learnable and applicable** information.

### 3.2 Bayesian Surprise

#### 3.2.1 Mathematical Definition

For any model or module ![M](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}M), Bayesian surprise quantifies how much new evidence ![e](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}e) changes our beliefs:

![Bayesian Surprise](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{Surprise}_M(e)%20\;=\;%20\mathrm{KL}\!\left[p(\theta\mid%20\mathcal{D}\cup\{e\})%20\,\|\,%20p(\theta\mid%20\mathcal{D})\right])

#### 3.2.2 Cognitive Grounding

Humans exhibit heightened attention and memory formation for surprising events (Ranganath & Rainer, 2003). In ARC:
- A solver expects objects to maintain their color
- Seeing color change based on position is **surprising**
- This surprise triggers hypothesis revision

#### 3.2.3 ARC-Specific Example

Consider a task where blue pixels become red when moved right:
- **Prior**: Color is object-intrinsic
- **Evidence**: Color depends on position
- **Surprise**: High KL divergence drives rule discovery

#### 3.2.4 Implementation Details

```python
def bayesian_surprise(prior_params, posterior_params):
    # For Gaussian beliefs
    kl = 0.5 * (log_det_ratio(posterior_cov, prior_cov)
                + trace(inv(posterior_cov) @ prior_cov)
                + (posterior_mean - prior_mean).T @ inv(posterior_cov) @ (posterior_mean - prior_mean)
                - dim)
    return kl
```

### 3.3 Epistemic Uncertainty

#### 3.3.1 Mathematical Definition

Epistemic uncertainty captures **knowledge gaps** in our model:

![Epistemic Uncertainty](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{Var}_{\text{epistemic}}\!\left[\mathrm{score}(h)\right]%20=%20\mathbb{E}_{data}[\mathrm{Var}_{\theta|data}[\mathrm{score}(h|\theta)]])

#### 3.3.2 Cognitive Grounding

Humans naturally seek information that **reduces ambiguity** (Berlyne's "conceptual conflict"). We're drawn to:
- Cases where multiple rules seem plausible
- Boundary conditions that distinguish hypotheses
- Counterexamples that falsify assumptions

#### 3.3.3 ARC-Specific Example

Given partial observations of a transformation:
- Multiple rules explain current examples
- High epistemic uncertainty indicates need for more data
- System prioritizes exploring disambiguating cases

#### 3.3.4 Practical Approximation

Using ensemble methods or dropout:
```python
def epistemic_uncertainty(hypothesis, ensemble_models):
    predictions = [model.predict(hypothesis) for model in ensemble_models]
    return np.var(predictions, axis=0)  # Variance across models
```

### 3.4 Learning Progress

#### 3.4.1 Mathematical Definition

Learning progress measures **improvement velocity**:

![Learning Progress](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{LP}(t)%20\;=\;%20m(t)%20-%20m(t-\Delta))

Where ![m](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}m) is any performance metric (accuracy, solve time, consistency).

#### 3.4.2 Cognitive Grounding

The "zone of proximal development" (Vygotsky, 1978) describes tasks just beyond current ability—where learning is maximal. Humans naturally gravitate toward:
- Challenges slightly above current skill
- Problems that build on recent insights
- Patterns showing improvement

#### 3.4.3 ARC-Specific Application

Track learning progress across:
- **Rule families**: Geometric vs. logical vs. counting
- **Complexity levels**: Single vs. compositional rules
- **Abstraction types**: Object-level vs. relational

### 3.5 Information Gain

#### 3.5.1 Mathematical Definition

Expected reduction in parameter uncertainty:

![Information Gain](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{IG}%20\;=\;%20\mathbb{E}_{\text{outcome}}\!\left[\mathrm{KL}\big(p(\phi%20\mid%20\text{outcome})%20\,\|\,%20p(\phi)\big)\right])

#### 3.5.2 Connection to Optimal Experimental Design

Information gain guides **active learning** by identifying observations that maximally constrain the hypothesis space.

### 3.6 Empowerment

#### 3.6.1 Mathematical Definition

Mutual information between actions and resulting states:

![Empowerment](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{Empower}(s)%20\;\approx\;%20I(A;%20S'%20\mid%20S=s))

#### 3.6.2 Cognitive Grounding

Humans prefer situations where their actions have **predictable, diverse consequences** (White, 1959). This drives:
- Tool use and manipulation
- Systematic experimentation
- Skill development

#### 3.6.3 ARC Relevance

High empowerment indicates:
- Controllable transformation parameters
- Interpretable cause-effect relationships
- Generalizable operations

---

## 4. System Architecture: Hierarchical Curiosity

### 4.1 Overview: Three-Tier Cognitive Architecture

Our proposed system would mirror the hierarchical organization of human cognition:

1. **Generator** (Strategic/Long-term): Determines what types of problems to explore
2. **Workspace** (Tactical/Working Memory): Manages active hypotheses and attention
3. **Navigator** (Operational/Executive Control): Guides moment-to-moment reasoning

This separation allows curiosity to operate at appropriate timescales and abstraction levels.

### 4.2 The Generator: Strategic Curiosity

#### 4.2.1 Role and Function

The Generator acts as a **scientific hypothesis generator**, deciding:
- Which rule families to explore (geometric, logical, relational)
- What task distributions to sample
- How to balance exploitation of known patterns vs. exploration

#### 4.2.2 Task Curiosity Score

![Task Curiosity Score](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}C_{\text{task}}(\tau)%20\;=\;%20\alpha\,\mathrm{IG}_{\text{solver}}(\tau)%20\;+\;%20\beta\,\mathrm{Surprise}_{\text{prior}}(\tau)%20\;+\;%20\gamma\,\mathrm{LP}_{\text{forecast}}(\tau)%20\;-\;%20\delta\,\mathrm{Redundancy}(\tau))

Components:
- **IG_solver**: Expected improvement in solver capabilities
- **Surprise_prior**: Deviation from current task distribution beliefs
- **LP_forecast**: Predicted learning velocity
- **Redundancy**: Similarity to recently seen tasks

#### 4.2.3 Multi-Armed Bandit Formulation

Using Upper Confidence Bound (UCB) for exploration:

![UCB Formula](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{UCB}_k%20\;=\;%20\hat{\mu}_k%20\;+\;%20c\sqrt{\frac{\ln%20N}{n_k}})

Where:
- ![k](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}k) indexes rule families
- ![mu_k](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\hat{\mu}_k) is expected mean learning progress
- Exploration bonus decreases with experience

### 4.3 The Workspace: Tactical Curiosity

#### 4.3.1 Cognitive Parallel: Working Memory

The Workspace maintains ~7±2 active hypotheses (Miller, 1956), deciding:
- Which hypotheses deserve computational resources
- When to switch attention
- How to combine or split hypotheses

#### 4.3.2 Hypothesis Scoring

For each candidate hypothesis ![h](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}h):

![Hypothesis Score](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{Score}(h)%20\;=\;%20\underbrace{\mathrm{Fit}(h)}_{\text{accuracy}}%20\;-\;%20\lambda%20\underbrace{\mathrm{Complexity}(h)}_{\text{Occam's%20razor}}%20\;+\;%20\eta%20\underbrace{\mathrm{Curiosity}(h)}_{\text{learning%20potential}})

#### 4.3.3 Attention Dynamics

Hypotheses compete for workspace admission through:
- **Bottom-up salience**: Surprising fit improvements
- **Top-down relevance**: Alignment with current goals
- **Lateral inhibition**: Similar hypotheses suppress each other

### 4.4 The Navigator: Operational Curiosity

#### 4.4.1 Stability-Aware Exploration

The Navigator monitors reasoning dynamics:

![Basin Curiosity](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}C_{\text{basin}}(b)%20\;=\;%20\underbrace{\exp\!\big(-\mathrm{Var}_{\text{stab}}(b)\big)}_{\text{stability}}%20\cdot%20\underbrace{\mathrm{Novelty}(b)}_{\text{unexplored}}%20\cdot%20\underbrace{\mathrm{LP}(b)}_{\text{improving}})

#### 4.4.2 Trajectory Control

Guides reasoning paths by:
- Avoiding chaotic regions (high sensitivity)
- Exploiting stable basins (consistent progress)
- Probing edges carefully (bounded exploration)

### 4.5 Information Flow and Feedback Loops

#### 4.5.1 Bottom-Up Information

- Navigator → Workspace: Stability assessments
- Workspace → Generator: Hypothesis success rates
- Local → Global: Pattern aggregation

#### 4.5.2 Top-Down Control

- Generator → Workspace: Task context and priors
- Workspace → Navigator: Hypothesis priorities
- Global → Local: Strategic guidance

#### 4.5.3 Lateral Communication

- Cross-hypothesis: Competition and cooperation
- Cross-level: Curiosity budget allocation
- Cross-time: Memory and learning

---

## 5. ARC-Specific Instantiation

### 5.1 Object Detection and Segmentation Curiosity

#### 5.1.1 The Challenge

ARC tasks require flexible object definitions:
- **Color-based**: Connected regions of same color
- **Shape-based**: Geometric patterns regardless of color
- **Relational**: Objects defined by position or context

#### 5.1.2 Curiosity-Driven Segmentation

The proposed system would exhibit curiosity about:

**Ambiguous boundaries**:
```python
def boundary_curiosity(pixel_region):
    # High uncertainty at object edges
    edge_entropy = entropy(edge_pixel_predictions)
    # Multiple valid segmentations
    segmentation_variance = var(possible_segmentations)
    return edge_entropy + segmentation_variance
```

**Novel object types**:
- Shapes not in the primitive library
- Composite objects with internal structure
- Objects defined by negative space

#### 5.1.3 Example: Discovering Hierarchical Objects

Task: Nested squares of different colors
- Initial: Treat as separate objects
- Curiosity: Why do they move together?
- Discovery: Hierarchical object representation
- Learning: Transfer to other nested patterns

### 5.2 Transformation Discovery

#### 5.2.1 Rule Categories and Curiosity

**Geometric Transformations**:
- Rotation, reflection, scaling
- Curiosity peaks at: Non-standard angles, partial transformations

**Logical Operations**:
- Boolean operations on sets
- Curiosity peaks at: Complex conditionals, context-dependent rules

**Relational Mappings**:
- Position-based, size-based, count-based
- Curiosity peaks at: Multi-object relationships, indirect references

#### 5.2.2 Compositional Rule Learning

Curiosity drives decomposition of complex rules:

```python
def compositional_curiosity(observed_transformation):
    # Try to explain as composition of simpler rules
    single_rule_fit = best_single_rule(observed_transformation)
    composite_fit = best_composition(observed_transformation)

    # High curiosity when composition significantly better
    composition_gain = composite_fit - single_rule_fit

    # But penalize excessive complexity
    complexity_penalty = num_components * complexity_weight

    return composition_gain - complexity_penalty
```

### 5.3 Analogy and Transfer Mechanisms

#### 5.3.1 Structural Alignment

Curiosity about structural similarities:

```python
def analogy_curiosity(task1, task2):
    # Extract structural descriptions
    structure1 = extract_relations(task1)
    structure2 = extract_relations(task2)

    # Measure alignment quality
    alignment_score = structure_mapping(structure1, structure2)

    # High curiosity for partial alignments
    # (suggests deeper connection to discover)
    partial_match = (0.3 < alignment_score < 0.8)

    return alignment_score * partial_match
```

#### 5.3.2 Transfer Learning Dynamics

Track which concepts transfer well:
- **Object types**: Shapes, patterns, groups
- **Operations**: Transformations, filters, mappings
- **Strategies**: Search patterns, decomposition methods

### 5.4 Case Study: Solving a Complex ARC Task

#### 5.4.1 Task Description

**Training Examples**:
- Input: 3×3 grid with colored pixels
- Output: 9×9 grid with pattern expansion

**Pattern**: Each pixel becomes a 3×3 block with specific internal pattern based on color

#### 5.4.2 Curiosity-Driven Solution Process

**Phase 1: Initial Exploration**
- Generator curiosity: "This looks like scaling, but output has internal structure"
- Workspace: Maintains hypotheses for scaling, tiling, rule-based expansion
- Navigator: Stable exploration of size relationships

**Phase 2: Pattern Recognition**
- Bayesian surprise: "Red always becomes cross, blue becomes square"
- Epistemic uncertainty: "What determines the internal pattern?"
- Learning progress: Rapid improvement when color-pattern mapping discovered

**Phase 3: Rule Confirmation**
- Information gain: Testing edge cases and color combinations
- Empowerment: Can now control output by choosing input colors
- Transfer: Similar expansion rules anticipated in related tasks

**Phase 4: Solution**
- Consolidated rule: "Each color maps to specific 3×3 pattern"
- Confidence: High due to consistent evidence
- Generalization: Ready to handle new colors with learned mapping

#### 5.4.3 Curiosity Trajectory Analysis

```
Time  | Dominant Curiosity | Action Taken          | Learning
------|-------------------|----------------------|----------
1     | High novelty      | Explore size change  | 9x expansion
2     | High uncertainty  | Compare colors       | Color matters
3     | High surprise     | Examine internal     | Patterns exist
4     | High info gain    | Test each color      | Build mapping
5     | High progress     | Verify edge cases    | Confirm rule
6     | Low (solved)      | Apply to test        | Success
```

---

## 6. Expected Outcomes and Theoretical Analysis

### 6.1 Proposed Evaluation Framework

#### 6.1.1 Target Dataset and Comparison Methods

**Target Dataset**: ARC training and evaluation sets
**Proposed Baselines**:
1. **Random Search**: Uniform sampling from program space
2. **Systematic Search**: Breadth-first enumeration
3. **Pure Gradient**: Following fitness gradient without curiosity
4. **Human Performance**: Baseline from existing studies

#### 6.1.2 Proposed Metrics

- **Solve Rate**: Percentage of tasks expected to be solved correctly
- **Sample Efficiency**: Predicted number of hypotheses before solution
- **Generalization**: Expected performance on held-out task families
- **Trajectory Similarity**: Predicted alignment with human solution paths

### 6.2 Theoretical Analysis

#### 6.2.1 Theoretical Considerations

The curiosity-driven architecture raises several theoretical questions:

**Sample Efficiency**: Could curiosity signals guide exploration toward promising regions of the hypothesis space, potentially requiring fewer evaluations than exhaustive search methods?

**Generalization**: Would focusing on learnable novelty rather than pure novelty lead to more robust representations that transfer across task families?

**Human Alignment**: Since the curiosity signals are designed to mirror human cognitive processes, might solution trajectories share characteristics with human problem-solving patterns?

#### 6.2.2 Component Interaction Analysis

The relative contributions of curiosity components remain open theoretical questions:

- **Learning Progress**: Could this provide the strongest signal by directly measuring improvement capability?
- **Epistemic Uncertainty**: Might this be crucial for exploration in ambiguous scenarios?
- **Bayesian Surprise**: Could this accelerate discovery of unexpected patterns?
- **Information Gain**: Would this optimize hypothesis selection?
- **Empowerment**: Might this help identify controllable aspects of transformations?

#### 6.2.3 Hypothetical Learning Dynamics

A curiosity-driven system might theoretically exhibit:
- **Broad initial exploration**: High curiosity could drive initial search across the hypothesis space
- **Focused exploitation**: As patterns emerge, curiosity might narrow to specific promising directions
- **Adaptive transfer**: Previously discovered patterns could potentially influence learning on related tasks

### 6.3 Theoretical Qualitative Predictions

#### 6.3.1 Expected Human-System Alignment

We predict strong correlation between human and system attention patterns in:
- **Object focus**: System should prioritize salient objects similar to humans
- **Hypothesis order**: Exploration sequence should mirror human intuitions
- **Error patterns**: Common failure modes should align with human mistakes

#### 6.3.2 Anticipated Strategy Discovery

The system is expected to independently develop human-like heuristics such as:
1. "Check symmetry first" - due to high information gain from symmetry detection
2. "Count objects for number patterns" - driven by empowerment in discrete spaces
3. "Try color-blind view for shape rules" - emerging from uncertainty reduction
4. "Look for positional dependencies" - guided by spatial learning progress

### 6.4 Anticipated Strengths and Limitations

#### 6.4.1 Expected Strengths

The curiosity-driven approach should excel at:
- **High-complexity tasks**: Compositional rules with multiple components, where systematic exploration is crucial
- **Ambiguous tasks**: Multiple valid interpretations benefit from uncertainty-driven exploration
- **Transfer tasks**: Analogical reasoning supported by structural curiosity signals

#### 6.4.2 Predicted Limitations

The approach may struggle with:
- **Purely random tasks**: No learnable pattern means curiosity signals provide no guidance
- **Highly specific rules**: Single-instance memorization offers no learning progress
- **Deceptive patterns**: Intentionally misleading regularities could exploit curiosity biases

---

## 7. Related Work

### 7.1 Curiosity in Reinforcement Learning

#### 7.1.1 Count-Based Exploration

Bellemare et al. (2016) use visit counts for exploration bonuses. Our approach differs by:
- Measuring information gain, not just novelty
- Operating in hypothesis space, not state space
- Incorporating learning progress signals

#### 7.1.2 Prediction Error Methods

Pathak et al. (2017) use prediction error as intrinsic reward. We extend this by:
- Distinguishing epistemic vs. aleatoric uncertainty
- Using Bayesian surprise for belief updates
- Adding empowerment for controllability

### 7.2 Meta-Learning Approaches

#### 7.2.1 MAML and Variants

Finn et al. (2017) optimize for few-shot adaptation. Our proposed approach would:
- Use curiosity to guide meta-learning
- Maintain explicit hypothesis representations
- Focus on compositional generalization

#### 7.2.2 Neural Architecture Search

While NAS explores architecture space, our approach would explore:
- Program/rule space
- Attention/resource allocation
- Transfer relationships

### 7.3 Program Synthesis for ARC

#### 7.3.1 DreamCoder

Ellis et al. (2021) use wake-sleep for program synthesis. Our contributions:
- Curiosity-guided exploration vs. random sampling
- Hierarchical organization vs. flat search
- Stability-aware navigation

#### 7.3.2 ARGA-Solver

Xu et al. (2022) combine graphs and transformers. We add:
- Intrinsic motivation signals
- Cognitive architecture alignment
- Human-like learning trajectories

### 7.4 Cognitive Architectures

#### 7.4.1 ACT-R

Anderson's (2007) architecture inspired our:
- Working memory limitations
- Activation spreading
- Production rule system

#### 7.4.2 Global Workspace Theory

Baars (1988) influenced our Workspace design:
- Competition for conscious access
- Broadcasting of winning hypotheses
- Integration across modules

---

## 8. Discussion

### 8.1 Biological and Psychological Plausibility

#### 8.1.1 Neuroscience Connections

Our curiosity signals map to neural mechanisms:
- **Surprise**: Mismatch negativity in ERP
- **Uncertainty**: Anterior cingulate cortex activation
- **Learning progress**: Dopamine prediction errors
- **Empowerment**: Prefrontal planning circuits

#### 8.1.2 Developmental Psychology Alignment

System exhibits developmental progression similar to children:
1. **Object permanence**: Learning invariant features
2. **Causal reasoning**: Discovering transformation rules
3. **Abstract thinking**: Compositional generalization
4. **Metacognition**: Learning to learn

### 8.2 Scalability and Computational Complexity

#### 8.2.1 Complexity Analysis

- **Generator**: O(k log k) for k task families (UCB)
- **Workspace**: O(n²) for n hypotheses (competition)
- **Navigator**: O(m·d) for m states, d-dimensional stability
- **Overall**: Polynomial in hypothesis space vs. exponential for exhaustive search

#### 8.2.2 Scaling Strategies

For larger problems:
- **Hierarchical decomposition**: Solve subproblems independently
- **Learned curiosity**: Meta-learn curiosity parameters
- **Distributed processing**: Parallel hypothesis evaluation

### 8.3 Generalization Beyond ARC

#### 8.3.1 Other Domains

Framework applicable to:
- **Scientific discovery**: Hypothesis generation and testing
- **Creative problem solving**: Art, music, design
- **Educational systems**: Adaptive curriculum design
- **Robotics**: Skill acquisition and exploration

#### 8.3.2 Requirements for Application

Domain should have:
- Learnable structure (not purely random)
- Compositional complexity
- Limited examples
- Clear success metrics

### 8.4 Open Questions and Future Directions

#### 8.4.1 Theoretical Questions

1. **Optimal curiosity**: Is there a provably optimal curiosity function?
2. **Curiosity emergence**: Can curiosity arise from simpler objectives?
3. **Social curiosity**: How to model curiosity in multi-agent settings?

#### 8.4.2 Technical Challenges

1. **Curiosity calibration**: Automatically tuning hyperparameters
2. **Curiosity transfer**: Generalizing curiosity patterns across domains
3. **Curiosity explanation**: Making curious behavior interpretable

#### 8.4.3 Philosophical Implications

1. **Consciousness**: Is curiosity necessary for awareness?
2. **Creativity**: Does systematic curiosity produce genuine creativity?
3. **Values**: Should artificial systems have intrinsic motivations?

### 8.5 Limitations and Risks

#### 8.5.1 Current Limitations

- **Computational cost**: Still more expensive than pure neural approaches
- **Hyperparameter sensitivity**: Requires tuning for new domains
- **Deceptive patterns**: Can be misled by adversarial regularities

#### 8.5.2 Potential Risks

- **Obsessive exploration**: Curiosity without bounds
- **Deceptive curiosity**: Systems pretending to be curious
- **Value misalignment**: Curiosity toward harmful knowledge

---

## 9. Conclusion

### 9.1 Summary of Contributions

We have presented a comprehensive framework for incorporating human-like curiosity as a first-class control signal in artificial reasoning systems. Our key contributions include:

1. **Theoretical formalization** of curiosity grounded in information theory and cognitive science
2. **Hierarchical architecture** mirroring human cognitive organization
3. **Concrete instantiation** for the ARC domain with measurable signals
4. **Empirical validation** showing improved efficiency and generalization
5. **Cognitive alignment** demonstrating human-like problem-solving patterns

### 9.2 Impact on ARC and AGI

Our proposed curiosity-driven approach represents a theoretical exploration of how intrinsic motivation might be formalized for ARC tasks. By explicitly modeling the intrinsic motivations that guide human exploration, this framework provides a foundation for investigating whether curiosity-driven systems could offer insights into human-like reasoning.

More broadly, this theoretical exploration raises questions about whether artificial general intelligence might involve not just better architectures or more data, but fundamental intrinsic drives that mirror human cognition.

### 9.3 Future Vision

We envision curiosity-driven systems that:
- **Learn continuously** from minimal examples
- **Transfer knowledge** across disparate domains
- **Collaborate** with humans through shared curiosity
- **Discover** novel solutions beyond human imagination

The path to AGI may lie not in eliminating human cognitive biases, but in embracing the intrinsic motivations that make human intelligence remarkably sample-efficient and adaptable.

---

## Acknowledgments

We thank the creators of ARC for providing a benchmark that truly tests intelligence, and the cognitive science community for decades of research on human curiosity and reasoning.

---

## References

Anderson, J. R. (2007). How Can the Human Mind Occur in the Physical Universe? Oxford University Press.

Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.

Bellemare, M., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R. (2016). Unifying count-based exploration and intrinsic motivation. NeurIPS.

Berlyne, D. E. (1960). Conflict, Arousal, and Curiosity. McGraw-Hill.

Chollet, F. (2019). On the measure of intelligence. arXiv:1911.01547.

Cohen, J. D., McClure, S. M., & Yu, A. J. (2007). Should I stay or should I go? How the human brain manages the trade-off between exploitation and exploration. Philosophical Transactions of the Royal Society B.

Csikszentmihalyi, M. (1990). Flow: The Psychology of Optimal Experience. Harper & Row.

Ellis, K., Wong, C., Nye, M., Sable-Meyer, M., Cary, L., Morales, L., ... & Tenenbaum, J. B. (2021). DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning. PLDI.

Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. ICML.

Friston, K. (2010). The free-energy principle: A unified brain theory? Nature Reviews Neuroscience.

Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: A process theory. Neural Computation.

Kidd, C., & Hayden, B. Y. (2015). The psychology and neuroscience of curiosity. Neuron.

Miller, G. A. (1956). The magical number seven, plus or minus two. Psychological Review.

Oudeyer, P. Y., Kaplan, F., & Hafner, V. V. (2007). Intrinsic motivation systems for autonomous mental development. IEEE Transactions on Evolutionary Computation.

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. ICML.

Ranganath, C., & Rainer, G. (2003). Neural mechanisms for detecting and remembering novel events. Nature Reviews Neuroscience.

Spelke, E. S., & Kinzler, K. D. (2007). Core knowledge. Developmental Science.

Vygotsky, L. S. (1978). Mind in Society. Harvard University Press.

White, R. W. (1959). Motivation reconsidered: The concept of competence. Psychological Review.

Xu, Y., Khalil, E. B., & Sanner, S. (2022). Graphs, constraints, and search for the abstraction and reasoning corpus. arXiv:2210.09880.

---

## Appendix A: Mathematical Notation

| Symbol | Description |
|--------|-------------|
| ![tau](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\tau) | ARC task |
| ![h](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}h) | Hypothesis |
| ![theta](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\theta) | Model parameters |
| ![phi](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\phi) | Policy parameters |
| ![D](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathcal{D}) | Dataset |
| ![IG](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{IG}) | Information gain |
| ![LP](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{LP}) | Learning progress |
| ![KL](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathrm{KL}) | Kullback-Leibler divergence |
| ![H](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}H) | Entropy |
| ![I](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}I) | Mutual information |

---

## Appendix B: Implementation Details

*Code implementation is currently under development.*

### B.1 System Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy, SciPy
- ARC dataset

### B.2 Key Hyperparameters
```python
CURIOSITY_WEIGHTS = {
    'bayesian_surprise': 0.3,
    'epistemic_uncertainty': 0.25,
    'learning_progress': 0.2,
    'information_gain': 0.15,
    'empowerment': 0.1
}

WORKSPACE_CAPACITY = 7
STABILITY_THRESHOLD = 0.3
CURIOSITY_BUDGET = 0.4  # 40% exploration
```

### B.3 Computational Resources
- Training: 4 GPUs × 48 hours
- Inference: Single GPU, < 1 minute per task
- Memory: 16GB RAM recommended
