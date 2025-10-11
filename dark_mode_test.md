# Dark Mode Visibility Solutions Test

Testing various approaches to solve the black text on black background issue in GitHub dark mode.

## Current Approach (Not Working)
![Current](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathbf{Curiosity}(\cdot)%20=%20\text{Novelty}%20\times%20\text{Learnability})

## Enhanced CodeCogs Parameters

### Option 1: Different Background Syntax
![bg syntax](https://latex.codecogs.com/svg.latex?\bg{white}&space;\color{black}\mathbf{Curiosity}(\cdot)%20=%20\text{Novelty}%20\times%20\text{Learnability})

### Option 2: Pagecolor Command
![pagecolor](https://latex.codecogs.com/svg.latex?\pagecolor{white}\color{black}\mathbf{Curiosity}(\cdot)%20=%20\text{Novelty}%20\times%20\text{Learnability})

### Option 3: Boxed Equation for Visibility
![boxed](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\boxed{\mathbf{Curiosity}(\cdot)%20=%20\text{Novelty}%20\times%20\text{Learnability}})

### Option 4: Frame with Border
![framed](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\fbox{$\mathbf{Curiosity}(\cdot)%20=%20\text{Novelty}%20\times%20\text{Learnability}$})

### Option 5: High Contrast with Different Colors
![high contrast](https://latex.codecogs.com/svg.latex?\bg{white}&space;\color{black}\large\mathbf{Curiosity}(\cdot)%20=%20\text{Novelty}%20\times%20\text{Learnability})

## GitHub Native MathJax Test

### Display Math:
$$\mathbf{Curiosity}(\cdot) = \text{Novelty} \times \text{Learnability}$$

### Inline Math:
The parameter $\alpha$ controls the weight.

### Complex Expression:
$$C_{\text{task}}(\tau) = \alpha\,\mathrm{IG}_{\text{solver}}(\tau) + \beta\,\mathrm{Surprise}_{\text{prior}}(\tau)$$

## Test Instructions

1. **View this file on GitHub in light mode**
2. **Switch to dark mode and compare visibility**
3. **Identify which approach provides best contrast**

## Expected Results

- **If GitHub native MathJax works**: Use that for all equations (simplest solution)
- **If enhanced CodeCogs parameters work**: Apply best option to all 47 equations
- **If none work well**: Consider alternative approaches

## Visibility Checklist

For each approach, check:
- ✅ Visible in GitHub light mode
- ✅ Visible in GitHub dark mode
- ✅ Good contrast ratio
- ✅ Professional appearance
- ✅ Maintains mathematical formatting