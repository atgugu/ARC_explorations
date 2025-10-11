# CodeCogs SVG Math Rendering Test

This file tests CodeCogs SVG rendering for various equation types that will be used in ARC documents.

## Basic Inline Math Tests

Simple variables: ![x](https://latex.codecogs.com/svg.latex?\bg_white&space;x) equals ![y](https://latex.codecogs.com/svg.latex?\bg_white&space;y) plus ![z](https://latex.codecogs.com/svg.latex?\bg_white&space;z)

Greek letters: ![alpha](https://latex.codecogs.com/svg.latex?\bg_white&space;\alpha), ![beta](https://latex.codecogs.com/svg.latex?\bg_white&space;\beta), ![gamma](https://latex.codecogs.com/svg.latex?\bg_white&space;\gamma), ![lambda](https://latex.codecogs.com/svg.latex?\bg_white&space;\lambda), ![mu](https://latex.codecogs.com/svg.latex?\bg_white&space;\mu), ![kappa](https://latex.codecogs.com/svg.latex?\bg_white&space;\kappa)

Subscripts and superscripts: ![x_k](https://latex.codecogs.com/svg.latex?\bg_white&space;x_k) and ![n_k](https://latex.codecogs.com/svg.latex?\bg_white&space;n_k) and ![x^2](https://latex.codecogs.com/svg.latex?\bg_white&space;x^2)

## Display Math Tests

### Simple Equations

Quadratic formula:
![quadratic](https://latex.codecogs.com/svg.latex?\bg_white&space;x%20=%20\frac{-b%20\pm%20\sqrt{b^2%20-%204ac}}{2a})

Einstein's equation:
![einstein](https://latex.codecogs.com/svg.latex?\bg_white&space;E%20=%20mc^2)

### Complex ARC Document Equations

Curiosity definition:
![curiosity](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathbf{Curiosity}(\cdot)%20\;=\;%20\underbrace{\text{Novelty}}_{\text{we%20haven't%20seen%20this}}%20\times%20\underbrace{\text{Learnability}}_{\text{we%20can%20improve%20here}}%20\times%20\underbrace{\text{Usefulness}}_{\text{it%20helps%20future%20tasks}})

Bayesian surprise:
![surprise](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathrm{Surprise}_M(e)%20\;=\;%20\mathrm{KL}\!\left[p(\theta\mid%20\mathcal{D}\cup\{e\})%20\,\|\,%20p(\theta\mid%20\mathcal{D})\right])

Learning progress:
![lp](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathrm{LP}(t)%20\;=\;%20m(t)%20-%20m(t-\Delta))

Task curiosity score:
![task_curiosity](https://latex.codecogs.com/svg.latex?\bg_white&space;C_{\text{task}}(\tau)%20\;=\;%20\alpha\,\mathrm{IG}_{\text{solver}}(\tau)%20\;+\;%20\beta\,\mathrm{Surprise}_{\text{prior}}(\tau)%20\;+\;%20\gamma\,\mathrm{LP}_{\text{forecast}}(\tau)%20\;-\;%20\delta\,\mathrm{Redundancy}(\tau))

UCB formula:
![ucb](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathrm{UCB}_k%20\;=\;%20\hat{\mu}_k%20\;+\;%20c\sqrt{\frac{\ln%20N}{n_k}})

### Complex Expressions with Underbraces

Hypothesis scoring:
![hypothesis](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathrm{Score}(h)%20\;=\;%20\underbrace{\mathrm{Fit}(h)}_{\text{critic}}%20\;-\;%20\lambda%20\underbrace{\mathrm{Instability}(h)}_{\text{navigator}}%20\;+\;%20\eta%20\underbrace{\mathrm{Curiosity}(h)}_{\text{below}})

Unifying objective:
![unifying](https://latex.codecogs.com/svg.latex?\bg_white&space;U%20\;=\;%20\underbrace{\mathbb{E}[\mathrm{SolveGain}]}_{\text{exploitation}}%20\;-\;%20\lambda%20\underbrace{\mathrm{Compute}}_{\text{budget}}%20\;-\;%20\mu%20\underbrace{\mathrm{Instability}}_{\text{navigator}}%20\;+\;%20\kappa%20\underbrace{\mathrm{Curiosity}}_{\text{IG/LP/Surprise}})

### Function Mappings and Set Notation

Function mapping:
![mapping](https://latex.codecogs.com/svg.latex?\bg_white&space;f:%20x%20\mapsto%20(y,\%20\text{artifacts},\%20\text{telemetry}))

Information gain:
![ig](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathrm{IG}%20\;=\;%20\mathbb{E}_{\text{outcome}}\!\left[\mathrm{KL}\big(p(\phi%20\mid%20\text{outcome})%20\,\|\,%20p(\phi)\big)\right])

Empowerment:
![empower](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathrm{Empower}(s)%20\;\approx\;%20I(A;%20S'%20\mid%20S=s))

Budget constraint:
![budget](https://latex.codecogs.com/svg.latex?\bg_white&space;B%20\in%20[0,1])

### Variance and Complex Notation

Epistemic uncertainty:
![var](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathrm{Var}_{\text{epistemic}}\!\left[\mathrm{score}(h)\right])

Basin curiosity:
![basin](https://latex.codecogs.com/svg.latex?\bg_white&space;C_{\text{basin}}(b)%20\;=\;%20\underbrace{\exp\!\big(-\mathrm{Var}_{\text{stab}}(b)\big)}_{\text{stable}}%20\cdot%20\underbrace{\mathrm{Novelty}(b)}_{\text{rare%20motif}}%20\cdot%20\underbrace{\mathrm{LP}(b)}_{\text{recent%20gains}})

## Dark Mode Test (without bg_white)

For comparison, here's the same equation without the white background:
![no_bg](https://latex.codecogs.com/svg.latex?\mathrm{LP}(t)%20\;=\;%20m(t)%20-%20m(t-\Delta))

And with white background:
![with_bg](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathrm{LP}(t)%20\;=\;%20m(t)%20-%20m(t-\Delta))

## Test Results Expected

If CodeCogs SVG rendering works properly:
- ✅ All equations should display correctly
- ✅ White background versions should work in both light and dark modes
- ✅ Complex expressions with underbraces should render properly
- ✅ Greek letters, subscripts, and superscripts should be clear
- ✅ Function mappings and set notation should be readable

## URL Encoding Reference

For developers: URLs are encoded as follows:
- Spaces: `%20` or `&space;`
- Background: `\bg_white&space;` at start
- Special chars: `\mid` → `\mid`, `\{` → `\{`, etc.