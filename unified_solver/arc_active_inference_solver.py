"""
ARC Active Inference Solver (AAIS)
===================================

A unified system blending:
- Active Inference: Bayesian belief updating during inference
- Curiosity-Driven Search: Information gain and epistemic uncertainty
- Stability-Aware Selection: Prefer robust, low-variance hypotheses
- Global Workspace: Limited capacity attention over hypotheses
- Program Synthesis: DSL-based compositional transformations

Always produces top-2 predictions and learns during inference.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import copy


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class Grid:
    """Represents an ARC grid"""
    data: np.ndarray

    def __init__(self, data):
        self.data = np.array(data, dtype=np.int32)

    @property
    def shape(self):
        return self.data.shape

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def copy(self):
        return Grid(self.data.copy())


@dataclass
class ARCTask:
    """An ARC task with training pairs and test input"""
    train_pairs: List[Tuple[Grid, Grid]]
    test_input: Grid

    def __len__(self):
        return len(self.train_pairs)


@dataclass
class Hypothesis:
    """A transformation hypothesis (program)"""
    program: Callable  # The transformation function
    name: str  # Human-readable name
    complexity: float  # Program complexity (for MDL)
    parameters: Dict = field(default_factory=dict)

    def apply(self, grid: Grid) -> Grid:
        """Apply transformation to a grid"""
        try:
            return self.program(grid, **self.parameters)
        except Exception as e:
            # Return input unchanged if transformation fails
            return grid.copy()

    def __hash__(self):
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other):
        return self.name == other.name and self.parameters == other.parameters


@dataclass
class BeliefState:
    """
    Belief distribution over hypotheses (Active Inference)

    Represents P(h | D_t) where h is a hypothesis and D_t is observed data at time t
    """
    probabilities: Dict[Hypothesis, float] = field(default_factory=dict)

    # Curiosity signals
    epistemic_uncertainty: float = 1.0  # Entropy of belief distribution
    learning_progress: float = 0.0  # Change in uncertainty
    information_gain: Dict[Hypothesis, float] = field(default_factory=dict)

    # Stability tracking
    stability_scores: Dict[Hypothesis, float] = field(default_factory=dict)
    observation_count: int = 0

    def normalize(self):
        """Normalize probabilities to sum to 1"""
        total = sum(self.probabilities.values())
        if total > 0:
            for h in self.probabilities:
                self.probabilities[h] /= total

    def entropy(self) -> float:
        """Compute Shannon entropy H[P(h)]"""
        return -sum(p * np.log(p + 1e-10) for p in self.probabilities.values() if p > 0)

    def top_k(self, k: int) -> List[Tuple[Hypothesis, float]]:
        """Get top-k hypotheses by probability"""
        sorted_hyps = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_hyps[:k]


# =============================================================================
# Perception Module: Extract patterns and features from grids
# =============================================================================

class PerceptionModule:
    """
    Extract structured features from ARC grids

    Implements ideas from: Curiosity Framework, Graph Pendulum
    """

    def perceive(self, grid: Grid) -> Dict:
        """Extract all perceptual features"""
        return {
            'objects': self.detect_objects(grid),
            'colors': self.extract_colors(grid),
            'symmetries': self.detect_symmetries(grid),
            'patterns': self.detect_patterns(grid),
            'shape': grid.shape,
        }

    def detect_objects(self, grid: Grid) -> List[Dict]:
        """Detect connected components as objects"""
        objects = []
        visited = np.zeros_like(grid.data, dtype=bool)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid.data[i, j] != 0:
                    obj = self._flood_fill(grid.data, i, j, visited)
                    if obj['pixels']:
                        objects.append(obj)

        return objects

    def _flood_fill(self, data, i, j, visited):
        """Flood fill to find connected component"""
        color = data[i, j]
        pixels = []
        stack = [(i, j)]

        while stack:
            ci, cj = stack.pop()
            if (ci < 0 or ci >= data.shape[0] or
                cj < 0 or cj >= data.shape[1] or
                visited[ci, cj] or data[ci, cj] != color):
                continue

            visited[ci, cj] = True
            pixels.append((ci, cj))

            # 4-connectivity
            stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])

        return {
            'color': int(color),
            'pixels': pixels,
            'size': len(pixels),
            'bbox': self._compute_bbox(pixels) if pixels else None
        }

    def _compute_bbox(self, pixels):
        """Compute bounding box of pixels"""
        if not pixels:
            return None
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        return (min(rows), min(cols), max(rows), max(cols))

    def extract_colors(self, grid: Grid) -> Dict:
        """Extract color statistics"""
        unique, counts = np.unique(grid.data, return_counts=True)
        return {
            'unique_colors': set(int(c) for c in unique if c != 0),
            'color_counts': dict(zip([int(c) for c in unique], [int(cnt) for cnt in counts])),
            'dominant_color': int(unique[np.argmax(counts)]) if len(unique) > 0 else 0
        }

    def detect_symmetries(self, grid: Grid) -> Dict:
        """Detect symmetries in the grid"""
        return {
            'horizontal': self._is_symmetric_horizontal(grid),
            'vertical': self._is_symmetric_vertical(grid),
            'diagonal': self._is_symmetric_diagonal(grid),
        }

    def _is_symmetric_horizontal(self, grid: Grid) -> bool:
        return np.array_equal(grid.data, np.flip(grid.data, axis=0))

    def _is_symmetric_vertical(self, grid: Grid) -> bool:
        return np.array_equal(grid.data, np.flip(grid.data, axis=1))

    def _is_symmetric_diagonal(self, grid: Grid) -> bool:
        if grid.shape[0] != grid.shape[1]:
            return False
        return np.array_equal(grid.data, grid.data.T)

    def detect_patterns(self, grid: Grid) -> Dict:
        """Detect repeating patterns"""
        return {
            'is_tiled': self._detect_tiling(grid),
            'is_sparse': np.count_nonzero(grid.data) < grid.data.size * 0.3,
            'is_dense': np.count_nonzero(grid.data) > grid.data.size * 0.7,
        }

    def _detect_tiling(self, grid: Grid) -> bool:
        """Check if grid is a tiled pattern"""
        # Simple heuristic: check for repeating rows or columns
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return False

        # Check row repetition
        for period in range(1, grid.shape[0] // 2 + 1):
            if all(np.array_equal(grid.data[i], grid.data[i % period])
                   for i in range(grid.shape[0])):
                return True

        return False


# =============================================================================
# Hypothesis Generator: Create transformation candidates using DSL
# =============================================================================

class HypothesisGenerator:
    """
    Generate transformation hypotheses using a typed DSL

    Implements ideas from: Generative Task Discovery, Probabilistic Program Spaces
    """

    def __init__(self, perception: PerceptionModule):
        self.perception = perception
        self.primitive_library = self._build_primitive_library()

    def generate_hypotheses(self, task: ARCTask, features: Dict) -> List[Hypothesis]:
        """
        Generate candidate transformation hypotheses

        Uses perception features to guide hypothesis generation
        """
        hypotheses = []

        # Identity transformation (baseline)
        hypotheses.append(self._create_identity())

        # Geometric transformations
        hypotheses.extend(self._generate_geometric_transforms(features))

        # Color transformations
        hypotheses.extend(self._generate_color_transforms(features))

        # Object-based transformations
        hypotheses.extend(self._generate_object_transforms(features))

        # Spatial transformations
        hypotheses.extend(self._generate_spatial_transforms(features))

        # Compositional transformations
        hypotheses.extend(self._generate_compositional_transforms(features))

        return hypotheses

    def _build_primitive_library(self) -> Dict[str, Callable]:
        """Build library of primitive transformations"""
        return {
            # Geometric primitives
            'rotate_90': lambda g, **kw: Grid(np.rot90(g.data, k=1)),
            'rotate_180': lambda g, **kw: Grid(np.rot90(g.data, k=2)),
            'rotate_270': lambda g, **kw: Grid(np.rot90(g.data, k=3)),
            'flip_horizontal': lambda g, **kw: Grid(np.flip(g.data, axis=0)),
            'flip_vertical': lambda g, **kw: Grid(np.flip(g.data, axis=1)),
            'transpose': lambda g, **kw: Grid(g.data.T),

            # Color primitives
            'invert_colors': self._invert_colors,
            'replace_color': self._replace_color,
            'swap_colors': self._swap_colors,

            # Morphological
            'dilate': self._dilate,
            'erode': self._erode,
            'fill_background': self._fill_background,

            # Spatial
            'crop': self._crop,
            'extend': self._extend,
            'tile': self._tile,
            'zoom': self._zoom,
        }

    def _create_identity(self) -> Hypothesis:
        """Identity transformation (no change)"""
        return Hypothesis(
            program=lambda g, **kw: g.copy(),
            name="identity",
            complexity=0.0
        )

    def _generate_geometric_transforms(self, features: Dict) -> List[Hypothesis]:
        """Generate geometric transformation hypotheses"""
        hypotheses = []

        # Rotations
        for k, name in [(1, "rotate_90"), (2, "rotate_180"), (3, "rotate_270")]:
            hypotheses.append(Hypothesis(
                program=lambda g, k=k, **kw: Grid(np.rot90(g.data, k=k)),
                name=name,
                complexity=1.0,
                parameters={'k': k}
            ))

        # Flips
        hypotheses.append(Hypothesis(
            program=lambda g, **kw: Grid(np.flip(g.data, axis=0)),
            name="flip_horizontal",
            complexity=1.0
        ))
        hypotheses.append(Hypothesis(
            program=lambda g, **kw: Grid(np.flip(g.data, axis=1)),
            name="flip_vertical",
            complexity=1.0
        ))

        # Transpose (for square grids or if symmetry detected)
        if features.get('shape', (0, 0))[0] == features.get('shape', (0, 0))[1]:
            hypotheses.append(Hypothesis(
                program=lambda g, **kw: Grid(g.data.T),
                name="transpose",
                complexity=1.0
            ))

        return hypotheses

    def _generate_color_transforms(self, features: Dict) -> List[Hypothesis]:
        """Generate color transformation hypotheses"""
        hypotheses = []

        colors = features.get('colors', {}).get('unique_colors', set())

        if len(colors) > 0:
            # Color replacement hypotheses
            for old_color in colors:
                for new_color in range(10):  # ARC uses colors 0-9
                    if new_color != old_color:
                        hypotheses.append(Hypothesis(
                            program=self._replace_color,
                            name=f"replace_{old_color}_with_{new_color}",
                            complexity=1.5,
                            parameters={'old_color': old_color, 'new_color': new_color}
                        ))

        # Color inversion
        hypotheses.append(Hypothesis(
            program=self._invert_colors,
            name="invert_colors",
            complexity=1.0
        ))

        return hypotheses

    def _generate_object_transforms(self, features: Dict) -> List[Hypothesis]:
        """Generate object-based transformation hypotheses"""
        hypotheses = []

        objects = features.get('objects', [])

        if len(objects) > 0:
            # Filter objects by property
            hypotheses.append(Hypothesis(
                program=self._filter_largest_object,
                name="keep_largest_object",
                complexity=2.0
            ))

            hypotheses.append(Hypothesis(
                program=self._filter_smallest_object,
                name="keep_smallest_object",
                complexity=2.0
            ))

        return hypotheses

    def _generate_spatial_transforms(self, features: Dict) -> List[Hypothesis]:
        """Generate spatial transformation hypotheses"""
        hypotheses = []

        # Scaling
        for scale in [2, 3]:
            hypotheses.append(Hypothesis(
                program=self._zoom,
                name=f"zoom_{scale}x",
                complexity=1.5,
                parameters={'scale': scale}
            ))

        # Tiling
        if features.get('patterns', {}).get('is_sparse', False):
            for nx, ny in [(2, 1), (1, 2), (2, 2)]:
                hypotheses.append(Hypothesis(
                    program=self._tile,
                    name=f"tile_{nx}x{ny}",
                    complexity=2.0,
                    parameters={'nx': nx, 'ny': ny}
                ))

        return hypotheses

    def _generate_compositional_transforms(self, features: Dict) -> List[Hypothesis]:
        """Generate compositional transformations (combine primitives)"""
        hypotheses = []

        # Common compositions
        # Rotate then flip
        hypotheses.append(Hypothesis(
            program=lambda g, **kw: Grid(np.flip(np.rot90(g.data, k=1), axis=0)),
            name="rotate_90_then_flip_h",
            complexity=2.0
        ))

        # Flip then transpose
        hypotheses.append(Hypothesis(
            program=lambda g, **kw: Grid(np.flip(g.data, axis=1).T),
            name="flip_v_then_transpose",
            complexity=2.0
        ))

        return hypotheses

    # Primitive implementations

    def _invert_colors(self, grid: Grid, **kwargs) -> Grid:
        """Invert non-zero colors"""
        result = grid.data.copy()
        mask = result != 0
        if mask.any():
            result[mask] = 9 - result[mask]
        return Grid(result)

    def _replace_color(self, grid: Grid, old_color: int, new_color: int, **kwargs) -> Grid:
        """Replace all instances of old_color with new_color"""
        result = grid.data.copy()
        result[result == old_color] = new_color
        return Grid(result)

    def _swap_colors(self, grid: Grid, color1: int, color2: int, **kwargs) -> Grid:
        """Swap two colors"""
        result = grid.data.copy()
        mask1 = result == color1
        mask2 = result == color2
        result[mask1] = color2
        result[mask2] = color1
        return Grid(result)

    def _dilate(self, grid: Grid, **kwargs) -> Grid:
        """Dilate non-zero pixels"""
        result = grid.data.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid.data[i, j] != 0:
                    color = grid.data[i, j]
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                            if result[ni, nj] == 0:
                                result[ni, nj] = color
        return Grid(result)

    def _erode(self, grid: Grid, **kwargs) -> Grid:
        """Erode non-zero pixels"""
        result = grid.data.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid.data[i, j] != 0:
                    # Check if any neighbor is background
                    has_bg_neighbor = False
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if (ni < 0 or ni >= grid.shape[0] or
                            nj < 0 or nj >= grid.shape[1] or
                            grid.data[ni, nj] == 0):
                            has_bg_neighbor = True
                            break
                    if has_bg_neighbor:
                        result[i, j] = 0
        return Grid(result)

    def _fill_background(self, grid: Grid, color: int = 1, **kwargs) -> Grid:
        """Fill background with color"""
        result = grid.data.copy()
        result[result == 0] = color
        return Grid(result)

    def _crop(self, grid: Grid, **kwargs) -> Grid:
        """Crop to bounding box of non-zero pixels"""
        nonzero = np.argwhere(grid.data != 0)
        if len(nonzero) == 0:
            return grid.copy()

        min_row, min_col = nonzero.min(axis=0)
        max_row, max_col = nonzero.max(axis=0)
        return Grid(grid.data[min_row:max_row+1, min_col:max_col+1])

    def _extend(self, grid: Grid, padding: int = 1, **kwargs) -> Grid:
        """Extend grid with padding"""
        result = np.zeros((grid.shape[0] + 2*padding, grid.shape[1] + 2*padding), dtype=np.int32)
        result[padding:-padding, padding:-padding] = grid.data
        return Grid(result)

    def _tile(self, grid: Grid, nx: int = 2, ny: int = 2, **kwargs) -> Grid:
        """Tile the grid nx by ny times"""
        return Grid(np.tile(grid.data, (nx, ny)))

    def _zoom(self, grid: Grid, scale: int = 2, **kwargs) -> Grid:
        """Zoom grid by repeating each pixel"""
        return Grid(np.repeat(np.repeat(grid.data, scale, axis=0), scale, axis=1))

    def _filter_largest_object(self, grid: Grid, **kwargs) -> Grid:
        """Keep only the largest object"""
        features = self.perception.perceive(grid)
        objects = features.get('objects', [])

        if not objects:
            return grid.copy()

        largest = max(objects, key=lambda o: o['size'])
        result = np.zeros_like(grid.data)
        for i, j in largest['pixels']:
            result[i, j] = largest['color']

        return Grid(result)

    def _filter_smallest_object(self, grid: Grid, **kwargs) -> Grid:
        """Keep only the smallest object"""
        features = self.perception.perceive(grid)
        objects = features.get('objects', [])

        if not objects:
            return grid.copy()

        smallest = min(objects, key=lambda o: o['size'])
        result = np.zeros_like(grid.data)
        for i, j in smallest['pixels']:
            result[i, j] = smallest['color']

        return Grid(result)


# =============================================================================
# Active Inference Engine: Update beliefs based on observations
# =============================================================================

class ActiveInferenceEngine:
    """
    Bayesian belief updating with curiosity signals

    Implements ideas from: Curiosity Framework, Probabilistic Program Spaces,
    Active Inference, Information Theory
    """

    def __init__(self, prior_weight: float = 1.0):
        self.prior_weight = prior_weight

    def initialize_beliefs(self, hypotheses: List[Hypothesis]) -> BeliefState:
        """Initialize uniform prior over hypotheses"""
        belief = BeliefState()

        # Uniform prior with MDL bias (prefer simpler programs)
        for h in hypotheses:
            # P(h) ∝ exp(-complexity)
            prior_prob = np.exp(-h.complexity * 0.1)
            belief.probabilities[h] = prior_prob

        belief.normalize()
        belief.epistemic_uncertainty = belief.entropy()

        # Initialize stability scores to None (not yet computed)
        for h in hypotheses:
            belief.stability_scores[h] = None

        return belief

    def update_beliefs(self,
                      belief: BeliefState,
                      observation: Tuple[Grid, Grid],
                      hypotheses: List[Hypothesis]) -> BeliefState:
        """
        Active Inference: Update beliefs based on new observation

        Implements Bayesian update: P(h | D_new) ∝ P(D_new | h) * P(h | D_old)
        """
        input_grid, output_grid = observation
        prev_entropy = belief.entropy()

        # Compute likelihood for each hypothesis
        likelihoods = {}
        for h in hypotheses:
            if h not in belief.probabilities:
                continue

            # Apply hypothesis to input
            predicted = h.apply(input_grid)

            # Compute likelihood: how well does prediction match observation?
            likelihood = self._compute_likelihood(predicted, output_grid)
            likelihoods[h] = likelihood

        # Bayesian update: posterior ∝ likelihood × prior
        new_belief = BeliefState()
        for h in hypotheses:
            if h in belief.probabilities:
                prior = belief.probabilities[h]
                likelihood = likelihoods.get(h, 1e-10)
                new_belief.probabilities[h] = likelihood * prior

        new_belief.normalize()

        # Update curiosity signals
        new_belief.epistemic_uncertainty = new_belief.entropy()
        new_belief.learning_progress = prev_entropy - new_belief.epistemic_uncertainty
        new_belief.observation_count = belief.observation_count + 1

        # Compute information gain for each hypothesis
        for h in hypotheses:
            if h in new_belief.probabilities and h in belief.probabilities:
                # KL divergence measures information gain
                new_p = new_belief.probabilities[h]
                old_p = belief.probabilities[h]
                if old_p > 0 and new_p > 0:
                    ig = new_p * np.log(new_p / old_p)
                    new_belief.information_gain[h] = ig

        # Copy stability scores
        new_belief.stability_scores = belief.stability_scores.copy()

        return new_belief

    def _compute_likelihood(self, predicted: Grid, observed: Grid) -> float:
        """
        Compute P(observation | hypothesis)

        Uses pixel-wise accuracy with smoothing
        """
        if predicted.shape != observed.shape:
            # Shape mismatch → very low likelihood
            return 1e-10

        # Compute pixel accuracy
        matches = np.sum(predicted.data == observed.data)
        total = predicted.data.size
        accuracy = matches / total

        # Convert to likelihood with temperature
        # Higher temperature = more forgiving
        temperature = 0.1
        likelihood = np.exp(accuracy / temperature)

        return likelihood

    def compute_curiosity_score(self, hypothesis: Hypothesis, belief: BeliefState) -> float:
        """
        Compute curiosity score for a hypothesis

        Combines information gain and epistemic uncertainty
        """
        # Information gain component
        ig = belief.information_gain.get(hypothesis, 0.0)

        # Epistemic uncertainty component (prefer uncertain hypotheses to explore)
        prob = belief.probabilities.get(hypothesis, 0.0)
        eu = -prob * np.log(prob + 1e-10)  # Contribution to entropy

        # Combined curiosity score
        curiosity = ig + 0.5 * eu

        return curiosity


# =============================================================================
# Stability Filter: Test hypothesis robustness
# =============================================================================

class StabilityFilter:
    """
    Assess stability of hypotheses using perturbation analysis

    Implements ideas from: Graph Pendulum, Dynamical Systems
    """

    def __init__(self, n_perturbations: int = 5):
        self.n_perturbations = n_perturbations

    def assess_stability(self,
                        hypothesis: Hypothesis,
                        task: ARCTask,
                        belief: BeliefState) -> float:
        """
        Compute stability score for a hypothesis

        Stable hypotheses have low variance across small perturbations
        """
        # Check if already computed (not None)
        if hypothesis in belief.stability_scores and belief.stability_scores[hypothesis] is not None:
            return belief.stability_scores[hypothesis]

        # Test hypothesis on all training examples
        accuracies = []
        for input_grid, output_grid in task.train_pairs:
            predicted = hypothesis.apply(input_grid)
            acc = self._compute_accuracy(predicted, output_grid)
            accuracies.append(acc)

        if not accuracies:
            belief.stability_scores[hypothesis] = 0.0
            return 0.0

        # Stability = consistency across examples
        # High mean + low variance = stable
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        # Stability score: high mean, low variance
        stability = mean_acc * np.exp(-std_acc)

        belief.stability_scores[hypothesis] = stability
        return stability

    def _compute_accuracy(self, predicted: Grid, target: Grid) -> float:
        """Compute pixel-wise accuracy"""
        if predicted.shape != target.shape:
            return 0.0

        matches = np.sum(predicted.data == target.data)
        total = predicted.data.size
        return matches / total

    def filter_chaotic(self,
                      hypotheses: List[Hypothesis],
                      task: ARCTask,
                      belief: BeliefState,
                      threshold: float = 0.1) -> List[Hypothesis]:
        """Filter out chaotic (unstable) hypotheses"""
        stable_hypotheses = []

        for h in hypotheses:
            stability = self.assess_stability(h, task, belief)
            if stability >= threshold:
                stable_hypotheses.append(h)

        return stable_hypotheses if stable_hypotheses else hypotheses


# =============================================================================
# Workspace Controller: Limited capacity attention over hypotheses
# =============================================================================

class WorkspaceController:
    """
    Global Workspace with limited capacity attention

    Implements ideas from: Global Workspace, Cognitive Workspace
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity

    def select_hypotheses(self,
                         hypotheses: List[Hypothesis],
                         belief: BeliefState,
                         curiosity_weight: float = 0.3,
                         stability_weight: float = 0.3) -> List[Hypothesis]:
        """
        Select top-k hypotheses for workspace

        Combines probability, curiosity, and stability
        """
        if len(hypotheses) <= self.capacity:
            return hypotheses

        # Score each hypothesis
        scores = {}
        for h in hypotheses:
            prob = belief.probabilities.get(h, 0.0)
            stability = belief.stability_scores.get(h, None)
            if stability is None:
                stability = 0.0
            curiosity = belief.information_gain.get(h, 0.0)

            # Combined score
            score = (
                (1.0 - curiosity_weight - stability_weight) * prob +
                curiosity_weight * curiosity +
                stability_weight * stability
            )
            scores[h] = score

        # Select top-k
        ranked = sorted(hypotheses, key=lambda h: scores.get(h, 0.0), reverse=True)
        return ranked[:self.capacity]


# =============================================================================
# Unified Solver: The main system
# =============================================================================

class ARCActiveInferenceSolver:
    """
    ARC Active Inference Solver - Unified System

    Blends:
    - Active Inference: Bayesian belief updating
    - Curiosity-Driven Search: Information gain
    - Stability-Aware Selection: Robust hypotheses
    - Global Workspace: Limited capacity attention
    - Program Synthesis: DSL-based transformations

    Always produces top-2 predictions and learns during inference.
    """

    def __init__(self,
                 workspace_capacity: int = 20,
                 n_perturbations: int = 5):

        self.perception = PerceptionModule()
        self.generator = HypothesisGenerator(self.perception)
        self.active_inference = ActiveInferenceEngine()
        self.stability_filter = StabilityFilter(n_perturbations)
        self.workspace = WorkspaceController(workspace_capacity)

    def solve(self, task: ARCTask, verbose: bool = False) -> List[Grid]:
        """
        Solve an ARC task and return top-2 predictions

        Process:
        1. Perceive patterns in training examples
        2. Generate initial hypotheses
        3. Initialize belief distribution
        4. For each training example (Active Inference):
           - Update beliefs based on observation
           - Compute curiosity signals
           - Filter unstable hypotheses
           - Select workspace hypotheses
        5. Rank hypotheses by posterior × stability
        6. Return top-2 predictions
        """

        if verbose:
            print(f"\n{'='*60}")
            print(f"ARC Active Inference Solver")
            print(f"{'='*60}")
            print(f"Training examples: {len(task.train_pairs)}")
            print(f"Test input shape: {task.test_input.shape}")

        # Step 1: Perceive features from first training example
        if len(task.train_pairs) == 0:
            # No training data, return input unchanged
            return [task.test_input.copy(), task.test_input.copy()]

        first_input, first_output = task.train_pairs[0]
        features = self.perception.perceive(first_input)

        if verbose:
            print(f"\nPerceived features:")
            print(f"  Objects: {len(features.get('objects', []))}")
            print(f"  Colors: {features.get('colors', {}).get('unique_colors', set())}")
            print(f"  Symmetries: {features.get('symmetries', {})}")

        # Step 2: Generate hypotheses
        hypotheses = self.generator.generate_hypotheses(task, features)

        if verbose:
            print(f"\nGenerated {len(hypotheses)} initial hypotheses")

        # Step 3: Initialize beliefs
        belief = self.active_inference.initialize_beliefs(hypotheses)

        if verbose:
            print(f"Initial entropy: {belief.epistemic_uncertainty:.3f}")

        # Step 4: Active Inference - learn from each training example
        for idx, (input_grid, output_grid) in enumerate(task.train_pairs):
            if verbose:
                print(f"\n--- Training Example {idx + 1} ---")

            # Update beliefs (Active Inference)
            belief = self.active_inference.update_beliefs(belief, (input_grid, output_grid), hypotheses)

            if verbose:
                print(f"Entropy after observation: {belief.epistemic_uncertainty:.3f}")
                print(f"Learning progress: {belief.learning_progress:.3f}")

            # Assess stability of hypotheses
            for h in hypotheses:
                self.stability_filter.assess_stability(h, task, belief)

            # Filter chaotic hypotheses
            stable_hypotheses = self.stability_filter.filter_chaotic(hypotheses, task, belief)

            if verbose:
                print(f"Stable hypotheses: {len(stable_hypotheses)}/{len(hypotheses)}")

            # Workspace selection (limited capacity attention)
            workspace_hypotheses = self.workspace.select_hypotheses(
                stable_hypotheses, belief,
                curiosity_weight=0.2,
                stability_weight=0.3
            )

            if verbose:
                print(f"Workspace hypotheses: {len(workspace_hypotheses)}")
                top_5 = belief.top_k(5)
                print(f"\nTop 5 hypotheses:")
                for h, p in top_5:
                    stability = belief.stability_scores.get(h, None)
                    if stability is None:
                        stability = 0.0
                    print(f"  {h.name}: p={p:.4f}, stability={stability:.4f}")

        # Step 5: Rank hypotheses by posterior probability × stability
        final_scores = {}
        for h in hypotheses:
            posterior = belief.probabilities.get(h, 0.0)
            stability = belief.stability_scores.get(h, None)
            if stability is None:
                stability = 0.0

            # Combined score
            final_scores[h] = posterior * stability

        # Get top-2 hypotheses with output diversity enforcement
        top_2_hypotheses = self._select_diverse_top_2(
            hypotheses, final_scores, task.test_input, verbose
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Final Top-2 Hypotheses:")
            print(f"{'='*60}")
            for i, h in enumerate(top_2_hypotheses, 1):
                print(f"{i}. {h.name}")
                print(f"   Posterior: {belief.probabilities.get(h, 0.0):.4f}")
                stability = belief.stability_scores.get(h, None)
                if stability is None:
                    stability = 0.0
                print(f"   Stability: {stability:.4f}")
                print(f"   Final Score: {final_scores.get(h, 0.0):.4f}")

        # Step 6: Apply top-2 hypotheses to test input
        predictions = []
        for h in top_2_hypotheses:
            prediction = h.apply(task.test_input)
            predictions.append(prediction)

        # Ensure we always return exactly 2 predictions
        while len(predictions) < 2:
            predictions.append(task.test_input.copy())

        return predictions[:2]

    def _select_diverse_top_2(self,
                             hypotheses: List[Hypothesis],
                             scores: Dict[Hypothesis, float],
                             test_input: Grid,
                             verbose: bool = False) -> List[Hypothesis]:
        """
        Select top-2 hypotheses ensuring different outputs

        Strategy:
        1. Select best hypothesis by score
        2. Find best hypothesis that produces different output
        3. Fallback to second-best if all produce same output (edge case)

        Args:
            hypotheses: List of hypotheses to select from
            scores: Final scores for each hypothesis
            test_input: Test input to apply hypotheses to
            verbose: Print debug information

        Returns:
            List of exactly 2 hypotheses (or duplicates if needed)
        """
        if len(hypotheses) == 0:
            return []
        if len(hypotheses) == 1:
            return [hypotheses[0], hypotheses[0]]

        # Get top-1 by score
        ranked = sorted(hypotheses, key=lambda h: scores.get(h, 0.0), reverse=True)
        top_1 = ranked[0]

        try:
            output_1 = top_1.apply(test_input)
        except Exception as e:
            if verbose:
                print(f"\n  Warning: Top-1 hypothesis failed to apply: {e}")
            # Fallback to simple top-2
            return ranked[:2] if len(ranked) >= 2 else [ranked[0], ranked[0]]

        # Find best hypothesis with different output
        best_different = None
        best_different_score = -1.0

        for h in ranked[1:]:
            try:
                output_h = h.apply(test_input)

                # Check if output is different
                if not np.array_equal(output_h.data, output_1.data):
                    score_h = scores.get(h, 0.0)
                    if score_h > best_different_score:
                        best_different = h
                        best_different_score = score_h

            except Exception as e:
                # Skip hypotheses that fail to apply
                if verbose:
                    print(f"\n  Warning: Hypothesis {h.name} failed: {e}")
                continue

        # Select top-2
        if best_different is not None:
            top_2 = best_different
            if verbose:
                print(f"\n  ✓ Diversity enforced: Selected {top_2.name} (different output)")
                print(f"    Top-1: {top_1.name} (score: {scores.get(top_1, 0.0):.9f})")
                print(f"    Top-2: {top_2.name} (score: {scores.get(top_2, 0.0):.9f})")
        else:
            # All hypotheses produce same output (edge case)
            top_2 = ranked[1] if len(ranked) > 1 else ranked[0]
            if verbose:
                print(f"\n  Note: All hypotheses produce same output (edge case)")
                print(f"    Using second-best by score: {top_2.name}")

        return [top_1, top_2]


# =============================================================================
# Utility Functions
# =============================================================================

def visualize_grid(grid: Grid, title: str = ""):
    """Simple text visualization of a grid"""
    print(f"\n{title}")
    print("-" * (grid.shape[1] * 2 + 1))
    for row in grid.data:
        print("|" + "".join(f"{c}" + " " for c in row) + "|")
    print("-" * (grid.shape[1] * 2 + 1))


def create_example_task() -> ARCTask:
    """Create a simple example task for testing"""
    # Task: Flip vertically
    train_pairs = [
        (
            Grid([[1, 2], [3, 4]]),
            Grid([[2, 1], [4, 3]])
        ),
        (
            Grid([[5, 6], [7, 8]]),
            Grid([[6, 5], [8, 7]])
        ),
    ]
    test_input = Grid([[1, 2], [3, 4]])

    return ARCTask(train_pairs, test_input)


if __name__ == "__main__":
    # Example usage
    print("ARC Active Inference Solver - Example")

    # Create example task
    task = create_example_task()

    # Create solver
    solver = ARCActiveInferenceSolver(workspace_capacity=20)

    # Solve task
    predictions = solver.solve(task, verbose=True)

    # Display results
    print(f"\n{'='*60}")
    print("PREDICTIONS")
    print(f"{'='*60}")

    visualize_grid(task.test_input, "Test Input:")
    visualize_grid(predictions[0], "Prediction 1:")
    visualize_grid(predictions[1], "Prediction 2:")
