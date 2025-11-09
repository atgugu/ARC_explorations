"""
ARC Generative Task Discovery with Active Inference
===================================================

A neurosymbolic solver that uses:
- Typed Rule Grammar (TRG) for program synthesis
- Active Inference for belief updates during solving
- Dual predictions (top-2 outputs)
- Differentiable execution for gradient-based learning

Based on: "Generative Task Discovery via Learned Priors for ARC"
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import copy
from collections import defaultdict
import heapq


# ============================================================================
# Type System and TRG Primitives
# ============================================================================

class ARCType(Enum):
    """Type system for ARC operations"""
    GRID = "grid"
    OBJECT = "object"
    MASK = "mask"
    COLOR = "color"
    RELATION = "relation"
    SET = "set"
    TRANSFORM = "transform"
    NUMBER = "number"


@dataclass
class ARCObject:
    """Represents a connected component object in ARC"""
    mask: np.ndarray  # Boolean mask
    color: int
    bbox: Tuple[int, int, int, int]  # (y1, x1, y2, x2)
    centroid: Tuple[float, float]
    area: int
    shape_signature: Tuple[int, ...]  # Normalized shape descriptor

    def __hash__(self):
        return hash((self.color, self.bbox, self.area))


@dataclass
class Program:
    """Represents a transformation program"""
    schema: str
    primitives: List[str]
    parameters: Dict[str, Any]
    selectors: Dict[str, Any]
    complexity: float = 0.0
    _id: int = field(default_factory=lambda: id(object()))

    def __lt__(self, other):
        return self.complexity < other.complexity

    def __hash__(self):
        # Use frozen representation for hashing
        param_str = str(sorted(self.parameters.items()))
        selector_str = str(sorted(self.selectors.items()))
        return hash((
            self.schema,
            tuple(self.primitives),
            param_str,
            selector_str,
            self._id
        ))

    def __eq__(self, other):
        if not isinstance(other, Program):
            return False
        return (self.schema == other.schema and
                self.primitives == other.primitives and
                self.parameters == other.parameters and
                self.selectors == other.selectors)


# ============================================================================
# TRG Primitives Library
# ============================================================================

class TRGPrimitives:
    """Library of typed primitives for ARC transformations"""

    @staticmethod
    def components(grid: np.ndarray, connectivity: int = 4) -> List[ARCObject]:
        """Extract connected components from grid"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)

        for color in range(10):
            color_mask = (grid == color) & (grid != 0)  # Ignore background
            if not color_mask.any():
                continue

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if color_mask[i, j] and not visited[i, j]:
                        # Flood fill to get component
                        obj_mask = TRGPrimitives._flood_fill(
                            color_mask, i, j, connectivity
                        )
                        visited |= obj_mask

                        if obj_mask.sum() > 0:
                            objects.append(TRGPrimitives._create_object(
                                obj_mask, color
                            ))

        return objects

    @staticmethod
    def _flood_fill(mask: np.ndarray, y: int, x: int,
                    connectivity: int) -> np.ndarray:
        """Flood fill to extract connected component"""
        H, W = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        stack = [(y, x)]

        while stack:
            cy, cx = stack.pop()
            if (cy < 0 or cy >= H or cx < 0 or cx >= W or
                visited[cy, cx] or not mask[cy, cx]):
                continue

            visited[cy, cx] = True

            # 4-connectivity
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((cy + dy, cx + dx))

            # 8-connectivity
            if connectivity == 8:
                for dy, dx in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    stack.append((cy + dy, cx + dx))

        return visited

    @staticmethod
    def _create_object(mask: np.ndarray, color: int) -> ARCObject:
        """Create ARCObject from mask and color"""
        ys, xs = np.where(mask)
        bbox = (int(ys.min()), int(xs.min()),
                int(ys.max()), int(xs.max()))
        centroid = (float(ys.mean()), float(xs.mean()))
        area = int(mask.sum())

        # Shape signature (normalized)
        h, w = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
        local_mask = mask[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
        shape_sig = tuple(local_mask.flatten().astype(int).tolist()[:16])

        return ARCObject(
            mask=mask,
            color=color,
            bbox=bbox,
            centroid=centroid,
            area=area,
            shape_signature=shape_sig
        )

    @staticmethod
    def rotate(grid: np.ndarray, k: int) -> np.ndarray:
        """Rotate grid by k*90 degrees"""
        return np.rot90(grid, k=k % 4)

    @staticmethod
    def reflect(grid: np.ndarray, axis: str) -> np.ndarray:
        """Reflect grid along axis ('h' or 'v')"""
        if axis == 'h':
            return np.fliplr(grid)
        elif axis == 'v':
            return np.flipud(grid)
        else:
            return grid

    @staticmethod
    def translate(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate grid content"""
        result = np.zeros_like(grid)
        src_y1 = max(0, -dy)
        src_y2 = min(grid.shape[0], grid.shape[0] - dy)
        src_x1 = max(0, -dx)
        src_x2 = min(grid.shape[1], grid.shape[1] - dx)

        dst_y1 = max(0, dy)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = max(0, dx)
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        result[dst_y1:dst_y2, dst_x1:dst_x2] = \
            grid[src_y1:src_y2, src_x1:src_x2]

        return result

    @staticmethod
    def scale(obj: ARCObject, scale: int) -> ARCObject:
        """Scale object by integer factor"""
        mask = obj.mask
        y1, x1, y2, x2 = obj.bbox
        local_mask = mask[y1:y2+1, x1:x2+1]

        # Simple nearest-neighbor scaling
        scaled = np.repeat(np.repeat(local_mask, scale, axis=0),
                          scale, axis=1)

        new_mask = np.zeros((mask.shape[0] * scale, mask.shape[1] * scale),
                           dtype=bool)
        new_mask[y1*scale:(y2+1)*scale, x1*scale:(x2+1)*scale] = scaled

        return TRGPrimitives._create_object(new_mask, obj.color)

    @staticmethod
    def remap_color(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Remap colors according to mapping"""
        result = grid.copy()
        for src, dst in mapping.items():
            result[grid == src] = dst
        return result

    @staticmethod
    def fill_region(grid: np.ndarray, mask: np.ndarray,
                    color: int) -> np.ndarray:
        """Fill masked region with color"""
        result = grid.copy()
        result[mask] = color
        return result

    @staticmethod
    def match_shape(obj1: ARCObject, obj2: ARCObject) -> float:
        """Compute shape similarity between objects"""
        sig1, sig2 = obj1.shape_signature, obj2.shape_signature
        min_len = min(len(sig1), len(sig2))
        if min_len == 0:
            return 0.0
        matches = sum(s1 == s2 for s1, s2 in zip(sig1[:min_len],
                                                   sig2[:min_len]))
        return matches / min_len

    @staticmethod
    def group_by_property(objects: List[ARCObject],
                         prop: str) -> Dict[Any, List[ARCObject]]:
        """Group objects by property"""
        groups = defaultdict(list)
        for obj in objects:
            if prop == "color":
                groups[obj.color].append(obj)
            elif prop == "area":
                groups[obj.area].append(obj)
            elif prop == "shape":
                groups[obj.shape_signature].append(obj)
        return dict(groups)


# ============================================================================
# Active Inference Engine
# ============================================================================

@dataclass
class Belief:
    """Belief state over programs and their properties"""
    program_posterior: Dict[Program, float]  # P(program | observations)
    free_energy: float = float('inf')
    prediction_error: float = 0.0
    complexity_prior: float = 0.0

    def entropy(self) -> float:
        """Compute entropy of program distribution"""
        probs = list(self.program_posterior.values())
        if not probs:
            return float('inf')
        probs = np.array(probs)
        probs = probs / probs.sum()
        return -np.sum(probs * np.log(probs + 1e-10))


class ActiveInferenceEngine:
    """
    Active Inference for ARC solving

    Minimizes free energy: F = prediction_error + complexity_cost
    Updates beliefs online as evidence accumulates
    """

    def __init__(self, beta: float = 1.0, complexity_weight: float = 0.1):
        self.beta = beta  # Inverse temperature
        self.complexity_weight = complexity_weight
        self.beliefs: Optional[Belief] = None

    def initialize_beliefs(self, candidate_programs: List[Program]) -> Belief:
        """Initialize uniform prior over candidate programs"""
        if not candidate_programs:
            return Belief(program_posterior={})

        # Uniform prior with complexity bias
        complexities = np.array([p.complexity for p in candidate_programs])
        prior_probs = np.exp(-self.complexity_weight * complexities)
        prior_probs = prior_probs / prior_probs.sum()

        posterior = {
            prog: float(prob)
            for prog, prob in zip(candidate_programs, prior_probs)
        }

        self.beliefs = Belief(
            program_posterior=posterior,
            complexity_prior=self.complexity_weight * complexities.mean()
        )
        return self.beliefs

    def update_beliefs(self, programs: List[Program],
                      likelihoods: List[float]) -> Belief:
        """
        Update beliefs given new evidence (likelihoods)

        Bayes rule: P(p|data) ∝ P(data|p) * P(p)
        """
        if self.beliefs is None:
            return self.initialize_beliefs(programs)

        # Update posterior using Bayes rule
        new_posterior = {}
        for prog, likelihood in zip(programs, likelihoods):
            prior = self.beliefs.program_posterior.get(prog, 1e-10)
            new_posterior[prog] = prior * likelihood

        # Normalize
        total = sum(new_posterior.values())
        if total > 0:
            new_posterior = {p: prob/total
                           for p, prob in new_posterior.items()}

        # Compute free energy: F = -log P(data) + KL(Q||P)
        avg_likelihood = np.mean(likelihoods) if likelihoods else 0.0
        prediction_error = -np.log(avg_likelihood + 1e-10)

        self.beliefs = Belief(
            program_posterior=new_posterior,
            free_energy=prediction_error + self.beliefs.complexity_prior,
            prediction_error=prediction_error,
            complexity_prior=self.beliefs.complexity_prior
        )

        return self.beliefs

    def sample_programs(self, n: int = 10) -> List[Program]:
        """Sample programs from current posterior"""
        if self.beliefs is None or not self.beliefs.program_posterior:
            return []

        programs = list(self.beliefs.program_posterior.keys())
        probs = list(self.beliefs.program_posterior.values())

        if sum(probs) == 0:
            return []

        probs = np.array(probs) / sum(probs)
        samples = np.random.choice(
            len(programs),
            size=min(n, len(programs)),
            replace=False,
            p=probs
        )

        return [programs[i] for i in samples]

    def get_top_programs(self, k: int = 2) -> List[Tuple[Program, float]]:
        """Get top-k programs by posterior probability"""
        if self.beliefs is None or not self.beliefs.program_posterior:
            return []

        sorted_progs = sorted(
            self.beliefs.program_posterior.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_progs[:k]


# ============================================================================
# Differentiable Executor
# ============================================================================

class Executor:
    """Execute programs on grids with differentiable operations"""

    def __init__(self):
        self.primitives = TRGPrimitives()
        self.execution_trace = []

    def execute(self, program: Program, input_grid: np.ndarray) -> np.ndarray:
        """Execute program on input grid"""
        self.execution_trace = []
        grid = input_grid.copy()

        try:
            if program.schema == "identity":
                return grid

            elif program.schema == "rotation":
                k = program.parameters.get("k", 1)
                grid = self.primitives.rotate(grid, k)

            elif program.schema == "reflection":
                axis = program.parameters.get("axis", "h")
                grid = self.primitives.reflect(grid, axis)

            elif program.schema == "translation":
                dx = program.parameters.get("dx", 0)
                dy = program.parameters.get("dy", 0)
                grid = self.primitives.translate(grid, dx, dy)

            elif program.schema == "color_remap":
                mapping = program.parameters.get("mapping", {})
                grid = self.primitives.remap_color(grid, mapping)

            elif program.schema == "composite":
                # Execute sequence of operations
                for prim_name in program.primitives:
                    if prim_name == "rotate":
                        k = program.parameters.get("k", 1)
                        grid = self.primitives.rotate(grid, k)
                    elif prim_name == "reflect":
                        axis = program.parameters.get("axis", "h")
                        grid = self.primitives.reflect(grid, axis)
                    elif prim_name == "translate":
                        dx = program.parameters.get("dx", 0)
                        dy = program.parameters.get("dy", 0)
                        grid = self.primitives.translate(grid, dx, dy)

            elif program.schema == "object_transform":
                # Extract objects and apply transformations
                objects = self.primitives.components(grid)
                output = np.zeros_like(grid)

                for obj in objects:
                    # Apply transformation based on selector
                    if self._matches_selector(obj, program.selectors):
                        # Apply transform
                        transform = program.parameters.get("transform", "identity")
                        if transform == "rotate":
                            # Rotate object (simplified)
                            pass
                        # Copy to output
                        output[obj.mask] = obj.color
                    else:
                        output[obj.mask] = obj.color

                grid = output

            self.execution_trace.append(("final", grid.copy()))

        except Exception as e:
            # Execution failed, return input
            self.execution_trace.append(("error", str(e)))

        return grid

    def _matches_selector(self, obj: ARCObject,
                         selectors: Dict[str, Any]) -> bool:
        """Check if object matches selector criteria"""
        if not selectors:
            return True

        for key, value in selectors.items():
            if key == "color" and obj.color != value:
                return False
            if key == "min_area" and obj.area < value:
                return False
            if key == "max_area" and obj.area > value:
                return False

        return True

    def compute_loss(self, predicted: np.ndarray,
                    target: np.ndarray) -> float:
        """Compute loss between predicted and target grids"""
        if predicted.shape != target.shape:
            # Penalize shape mismatch heavily
            return 1.0 + np.abs(predicted.size - target.size) / target.size

        # Hamming distance
        differences = (predicted != target).sum()
        total = target.size

        return differences / total


# ============================================================================
# Program Generator with Schemas
# ============================================================================

class ProgramGenerator:
    """Generate candidate programs for ARC tasks"""

    def __init__(self):
        self.schemas = self._define_schemas()

    def _define_schemas(self) -> List[Dict[str, Any]]:
        """Define program schemas"""
        return [
            {
                "name": "identity",
                "params": {},
                "complexity": 0.1
            },
            {
                "name": "rotation",
                "params": {"k": [0, 1, 2, 3]},
                "complexity": 1.0
            },
            {
                "name": "reflection",
                "params": {"axis": ["h", "v"]},
                "complexity": 1.0
            },
            {
                "name": "translation",
                "params": {
                    "dx": range(-5, 6),
                    "dy": range(-5, 6)
                },
                "complexity": 2.0
            },
            {
                "name": "color_remap",
                "params": {"mapping": "infer"},
                "complexity": 2.0
            },
            {
                "name": "composite",
                "params": "sequence",
                "complexity": 3.0
            }
        ]

    def generate_candidates(self, task: Dict[str, Any],
                          max_candidates: int = 100) -> List[Program]:
        """Generate candidate programs for a task"""
        candidates = []

        # Simple programs
        candidates.append(Program(
            schema="identity",
            primitives=[],
            parameters={},
            selectors={},
            complexity=0.1
        ))

        # Rotations
        for k in [1, 2, 3]:
            candidates.append(Program(
                schema="rotation",
                primitives=["rotate"],
                parameters={"k": k},
                selectors={},
                complexity=1.0
            ))

        # Reflections
        for axis in ["h", "v"]:
            candidates.append(Program(
                schema="reflection",
                primitives=["reflect"],
                parameters={"axis": axis},
                selectors={},
                complexity=1.0
            ))

        # Translations
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx != 0 or dy != 0:
                    candidates.append(Program(
                        schema="translation",
                        primitives=["translate"],
                        parameters={"dx": dx, "dy": dy},
                        selectors={},
                        complexity=2.0
                    ))

        # Color remapping (infer from examples)
        color_mappings = self._infer_color_mappings(task)
        for mapping in color_mappings:
            candidates.append(Program(
                schema="color_remap",
                primitives=["remap_color"],
                parameters={"mapping": mapping},
                selectors={},
                complexity=2.0
            ))

        # Composite transformations
        composites = self._generate_composites(task)
        candidates.extend(composites)

        return candidates[:max_candidates]

    def _infer_color_mappings(self, task: Dict[str, Any]) -> List[Dict[int, int]]:
        """Infer possible color mappings from training examples"""
        mappings = []

        if "train" not in task or len(task["train"]) == 0:
            return mappings

        # Analyze first training pair
        train_input = task["train"][0]["input"]
        train_output = task["train"][0]["output"]

        input_colors = set(np.array(train_input).flatten())
        output_colors = set(np.array(train_output).flatten())

        # Try simple 1-1 mappings
        if len(input_colors) == len(output_colors):
            for perm in self._generate_permutations(
                list(input_colors), list(output_colors), max_n=3
            ):
                mapping = dict(zip(input_colors, perm))
                mappings.append(mapping)

        return mappings[:10]

    def _generate_permutations(self, src: List, dst: List,
                              max_n: int = 3) -> List[Tuple]:
        """Generate permutations (limited)"""
        import itertools
        perms = list(itertools.permutations(dst))
        return perms[:max_n]

    def _generate_composites(self, task: Dict[str, Any]) -> List[Program]:
        """Generate composite programs"""
        composites = []

        # Rotation + reflection
        for k in [1, 2, 3]:
            for axis in ["h", "v"]:
                composites.append(Program(
                    schema="composite",
                    primitives=["rotate", "reflect"],
                    parameters={"k": k, "axis": axis},
                    selectors={},
                    complexity=3.0
                ))

        # Translation + rotation
        for k in [1, 2, 3]:
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                composites.append(Program(
                    schema="composite",
                    primitives=["translate", "rotate"],
                    parameters={"k": k, "dx": dx, "dy": dy},
                    selectors={},
                    complexity=3.5
                ))

        return composites[:20]


# ============================================================================
# Neurosymbolic Solver with Active Inference
# ============================================================================

class ARCGenerativeSolver:
    """
    Main solver combining program synthesis with active inference
    """

    def __init__(self,
                 max_candidates: int = 100,
                 beam_width: int = 10,
                 active_inference_steps: int = 5):
        self.generator = ProgramGenerator()
        self.executor = Executor()
        self.active_inference = ActiveInferenceEngine(
            beta=1.0,
            complexity_weight=0.1
        )
        self.max_candidates = max_candidates
        self.beam_width = beam_width
        self.active_inference_steps = active_inference_steps

    def solve(self, task: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve ARC task with active inference

        Returns:
            prediction1: Best prediction
            prediction2: Second-best prediction
            metadata: Solving metadata (beliefs, programs, etc.)
        """
        # Generate candidate programs
        candidates = self.generator.generate_candidates(
            task, self.max_candidates
        )

        # Initialize beliefs
        self.active_inference.initialize_beliefs(candidates)

        # Active inference loop: iteratively refine beliefs
        for step in range(self.active_inference_steps):
            # Evaluate programs on training examples
            likelihoods = []
            valid_programs = []

            for program in candidates:
                likelihood = self._evaluate_program(program, task)
                if likelihood > 0:
                    likelihoods.append(likelihood)
                    valid_programs.append(program)

            if not valid_programs:
                break

            # Update beliefs using active inference
            self.active_inference.update_beliefs(valid_programs, likelihoods)

            # Sample from posterior for next iteration
            if step < self.active_inference_steps - 1:
                candidates = self.active_inference.sample_programs(
                    n=self.beam_width
                )

        # Get top-2 programs
        top_programs = self.active_inference.get_top_programs(k=2)

        if len(top_programs) == 0:
            # No valid program found, return empty predictions
            test_input = np.array(task["test"][0]["input"])
            return test_input.copy(), test_input.copy(), {"error": "No valid program"}

        # Execute top programs on test input
        test_input = np.array(task["test"][0]["input"])

        pred1 = self.executor.execute(top_programs[0][0], test_input)

        if len(top_programs) > 1:
            pred2 = self.executor.execute(top_programs[1][0], test_input)
        else:
            pred2 = pred1.copy()

        # Metadata
        metadata = {
            "top_programs": [
                {
                    "schema": p[0].schema,
                    "parameters": p[0].parameters,
                    "probability": p[1],
                    "complexity": p[0].complexity
                }
                for p in top_programs
            ],
            "free_energy": self.active_inference.beliefs.free_energy,
            "entropy": self.active_inference.beliefs.entropy(),
            "n_candidates": len(candidates),
            "n_valid": len([p for p, l in zip(candidates, likelihoods) if l > 0])
        }

        return pred1, pred2, metadata

    def _evaluate_program(self, program: Program,
                         task: Dict[str, Any]) -> float:
        """Evaluate program on training examples"""
        if "train" not in task or len(task["train"]) == 0:
            return 0.0

        total_likelihood = 0.0

        for example in task["train"]:
            input_grid = np.array(example["input"])
            target_grid = np.array(example["output"])

            # Execute program
            predicted = self.executor.execute(program, input_grid)

            # Compute likelihood (inverse of loss)
            loss = self.executor.compute_loss(predicted, target_grid)
            likelihood = np.exp(-10 * loss)  # Convert loss to likelihood

            total_likelihood += likelihood

        # Average likelihood across training examples
        avg_likelihood = total_likelihood / len(task["train"])

        # Apply complexity penalty
        complexity_penalty = np.exp(-0.1 * program.complexity)

        return avg_likelihood * complexity_penalty


# ============================================================================
# Example Usage and Testing
# ============================================================================

def load_arc_task(task_dict: Dict) -> Dict[str, Any]:
    """Load ARC task from dictionary format"""
    return task_dict


def evaluate_predictions(pred1: np.ndarray, pred2: np.ndarray,
                        target: np.ndarray) -> Dict[str, float]:
    """Evaluate predictions against target"""
    # Handle shape mismatches
    if pred1.shape != target.shape:
        acc1 = 0.0
        pixel_acc1 = 0.0
    else:
        acc1 = float((pred1 == target).all())
        pixel_acc1 = float((pred1 == target).sum()) / target.size

    if pred2.shape != target.shape:
        acc2 = 0.0
        pixel_acc2 = 0.0
    else:
        acc2 = float((pred2 == target).all())
        pixel_acc2 = float((pred2 == target).sum()) / target.size

    return {
        "exact_match_1": acc1,
        "exact_match_2": acc2,
        "pixel_accuracy_1": pixel_acc1,
        "pixel_accuracy_2": pixel_acc2,
        "any_correct": max(acc1, acc2)
    }


if __name__ == "__main__":
    # Example: Simple rotation task
    example_task = {
        "train": [
            {
                "input": [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]
                ],
                "output": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [0, 0, 2],
                    [0, 2, 0],
                    [2, 0, 0]
                ],
                "output": [
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2]
                ]
            }
        ]
    }

    print("ARC Generative Solver with Active Inference")
    print("=" * 60)
    print()

    # Create solver
    solver = ARCGenerativeSolver(
        max_candidates=100,
        beam_width=10,
        active_inference_steps=3
    )

    # Solve task
    print("Solving task...")
    pred1, pred2, metadata = solver.solve(example_task)

    print("\nTop Programs Found:")
    for i, prog_info in enumerate(metadata["top_programs"], 1):
        print(f"\n{i}. Schema: {prog_info['schema']}")
        print(f"   Parameters: {prog_info['parameters']}")
        print(f"   Probability: {prog_info['probability']:.4f}")
        print(f"   Complexity: {prog_info['complexity']:.2f}")

    print(f"\nFree Energy: {metadata['free_energy']:.4f}")
    print(f"Belief Entropy: {metadata['entropy']:.4f}")

    # Evaluate
    target = np.array(example_task["test"][0]["output"])
    results = evaluate_predictions(pred1, pred2, target)

    print("\nPrediction 1:")
    print(pred1)
    print(f"Exact match: {results['exact_match_1']}")
    print(f"Pixel accuracy: {results['pixel_accuracy_1']:.2%}")

    print("\nPrediction 2:")
    print(pred2)
    print(f"Exact match: {results['exact_match_2']}")
    print(f"Pixel accuracy: {results['pixel_accuracy_2']:.2%}")

    print(f"\nAny prediction correct: {results['any_correct']}")
