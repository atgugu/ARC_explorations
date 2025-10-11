# Math Rendering Test for GitHub

This file tests GitHub's native math rendering capabilities.

## Inline Math Examples

Here's some inline math: $x = y + z$ and more complex: $\alpha + \beta = \gamma$.

Variables with subscripts: $x_1, x_2, \ldots, x_n$ and superscripts: $x^2 + y^2 = z^2$.

Greek letters: $\alpha, \beta, \gamma, \delta, \theta, \lambda, \mu, \kappa, \omega$.

Functions: $\sin(x), \cos(x), \log(x), \exp(x)$.

## Display Math Examples

### Simple Equations

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

$$E = mc^2$$

### Complex Expressions (from ARC documents)

$$\mathbf{Curiosity}(\cdot) = \underbrace{\text{Novelty}}_{\text{we haven't seen this}} \times \underbrace{\text{Learnability}}_{\text{we can improve here}} \times \underbrace{\text{Usefulness}}_{\text{it helps future tasks}}$$

$$\mathrm{Surprise}_M(e) = \mathrm{KL}\left[p(\theta\mid \mathcal{D}\cup\{e\}) \,\|\, p(\theta\mid \mathcal{D})\right]$$

$$C_{\text{task}}(\tau) = \alpha\,\mathrm{IG}_{\text{solver}}(\tau) + \beta\,\mathrm{Surprise}_{\text{prior}}(\tau) + \gamma\,\mathrm{LP}_{\text{forecast}}(\tau) - \delta\,\mathrm{Redundancy}(\tau)$$

$$\mathrm{UCB}_k = \hat{\mu}_k + c\sqrt{\frac{\ln N}{n_k}}$$

### Set Operations and Logic

$$\mathbb{E}_{\text{outcome}}\left[\mathrm{KL}\big(p(\phi \mid \text{outcome}) \,\|\, p(\phi)\big)\right]$$

$$U = \underbrace{\mathbb{E}[\mathrm{SolveGain}]}_{\text{exploitation}} - \lambda \underbrace{\mathrm{Compute}}_{\text{budget}} - \mu \underbrace{\mathrm{Instability}}_{\text{navigator}} + \kappa \underbrace{\mathrm{Curiosity}}_{\text{IG/LP/Surprise}}$$

### Functions and Mappings

$$f: x \mapsto (y,\ \text{artifacts},\ \text{telemetry})$$

$$B \in [0,1]$$

### Fractions and Complex Notation

$$\mathrm{LP}(t) = m(t) - m(t-\Delta)$$

$$\mathrm{Var}_{\text{epistemic}}\left[\mathrm{score}(h)\right]$$

## Test Results

If all the above math expressions render correctly on GitHub, then the native syntax is working properly and we can proceed with converting all the ARC documents.

Expected behavior:
- Inline math should render within the text flow
- Display math should be centered and larger
- All symbols, subscripts, superscripts should display correctly
- Complex expressions with braces, fractions, etc. should render properly