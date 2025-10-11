# Math Visibility Test

Testing different approaches to ensure equation visibility in both light and dark modes.

## Current Approach (Potentially Invisible)
![Current Test](https://latex.codecogs.com/svg.latex?\bg_white&space;\mathbf{Curiosity}(\cdot)%20=%20\text{Novelty}%20\times%20\text{Learnability})

## Fixed Approach (Should Be Visible)
![Fixed Test](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\mathbf{Curiosity}(\cdot)%20=%20\text{Novelty}%20\times%20\text{Learnability})

## Complex Expression Test
### Current:
![Complex Current](https://latex.codecogs.com/svg.latex?\bg_white&space;C_{\text{task}}(\tau)%20=%20\alpha\,\mathrm{IG}_{\text{solver}}(\tau)%20+%20\beta\,\mathrm{Surprise}_{\text{prior}}(\tau))

### Fixed:
![Complex Fixed](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}C_{\text{task}}(\tau)%20=%20\alpha\,\mathrm{IG}_{\text{solver}}(\tau)%20+%20\beta\,\mathrm{Surprise}_{\text{prior}}(\tau))

## Inline Math Test
Current: ![inline](https://latex.codecogs.com/svg.latex?\bg_white&space;\alpha)
Fixed: ![inline fixed](https://latex.codecogs.com/svg.latex?\bg_white&space;\color{black}\alpha)

## Test Results
If the "Fixed" versions are consistently visible in both light and dark modes while "Current" versions have visibility issues, then the `\color{black}` parameter is the solution.

## Expected Behavior
- **Fixed equations**: Should show black text on white background in all modes
- **Current equations**: May show black text on black/dark background in some contexts