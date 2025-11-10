"""
Parameter Inference Module

Analyzes training examples to infer correct parameters for primitives.
This addresses the critical gap: solver knows WHAT to do but not specific parameters.

Key insight from compositional analysis:
- 66% of tasks use compositional programs
- 50 near-misses (70-95% accuracy)
- Problem: Pattern correct, parameters wrong
- Solution: Learn parameters from training examples
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass


@dataclass
class InferredParameters:
    """Container for inferred parameters"""
    color_mappings: List[Dict[int, int]]  # List of color mapping dicts
    translations: List[Tuple[int, int]]  # List of (dx, dy) tuples
    rotations: List[int]  # List of rotation angles (1, 2, 3 for 90°, 180°, 270°)
    scale_factors: List[int]  # List of scaling factors
    morphology_iterations: List[int]  # List of dilation/erosion iterations
    reflections: List[str]  # List of reflection axes ('h', 'v')

    def __repr__(self):
        parts = []
        if self.color_mappings:
            parts.append(f"colors:{len(self.color_mappings)}")
        if self.translations:
            parts.append(f"trans:{len(self.translations)}")
        if self.rotations:
            parts.append(f"rot:{len(self.rotations)}")
        if self.scale_factors:
            parts.append(f"scale:{len(self.scale_factors)}")
        return f"InferredParams({', '.join(parts)})"


class ParameterInference:
    """
    Infers parameters from training examples

    For each training example pair (input, output), analyzes:
    - Which colors change to which
    - How objects are spatially shifted
    - Size changes (scaling)
    - Shape changes (morphology, rotation)
    """

    @staticmethod
    def infer_all_parameters(task: Dict[str, Any]) -> InferredParameters:
        """
        Infer all parameters from task training examples

        Args:
            task: ARC task with 'train' examples

        Returns:
            InferredParameters with lists of possible parameter values
        """
        if 'train' not in task or len(task['train']) == 0:
            return InferredParameters(
                color_mappings=[],
                translations=[],
                rotations=[],
                scale_factors=[],
                morphology_iterations=[],
                reflections=[]
            )

        # Collect parameters from all training examples
        all_color_maps = []
        all_translations = []
        all_rotations = []
        all_scales = []
        all_morphology = []
        all_reflections = []

        for train_ex in task['train']:
            input_grid = np.array(train_ex['input'])
            output_grid = np.array(train_ex['output'])

            # Infer color mappings
            color_maps = ParameterInference.infer_color_mapping(input_grid, output_grid)
            all_color_maps.extend(color_maps)

            # Infer translations
            translations = ParameterInference.infer_translation(input_grid, output_grid)
            all_translations.extend(translations)

            # Infer rotations
            rotations = ParameterInference.infer_rotation(input_grid, output_grid)
            all_rotations.extend(rotations)

            # Infer scaling
            scales = ParameterInference.infer_scaling(input_grid, output_grid)
            all_scales.extend(scales)

            # Infer morphology iterations
            morphology = ParameterInference.infer_morphology_iterations(input_grid, output_grid)
            all_morphology.extend(morphology)

            # Infer reflections
            reflections = ParameterInference.infer_reflection(input_grid, output_grid)
            all_reflections.extend(reflections)

        # Deduplicate and return most common
        return InferredParameters(
            color_mappings=ParameterInference._deduplicate_color_maps(all_color_maps),
            translations=list(set(all_translations)),
            rotations=list(set(all_rotations)),
            scale_factors=list(set(all_scales)),
            morphology_iterations=list(set(all_morphology)),
            reflections=list(set(all_reflections))
        )

    @staticmethod
    def infer_color_mapping(input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict[int, int]]:
        """
        Infer color mapping from input to output

        Strategy:
        1. If shapes match, compare pixel-by-pixel to find color changes
        2. Build mapping {input_color: output_color}
        3. Validate mapping is consistent

        Returns:
            List of possible color mappings (usually just one if consistent)
        """
        # Only works if shapes match
        if input_grid.shape != output_grid.shape:
            return []

        # Build mapping from pixel comparisons
        color_map = {}

        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = int(input_grid[i, j])
                out_color = int(output_grid[i, j])

                if in_color in color_map:
                    # Check consistency
                    if color_map[in_color] != out_color:
                        # Inconsistent mapping - color depends on position
                        # This is not a simple global color remap
                        return []
                else:
                    color_map[in_color] = out_color

        # Filter out identity mappings (color doesn't change)
        filtered_map = {k: v for k, v in color_map.items() if k != v}

        if not filtered_map:
            return []  # No color changes

        return [filtered_map]

    @staticmethod
    def infer_translation(input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Infer translation (dx, dy) from input to output

        Strategy:
        1. Extract non-zero patterns from input
        2. Find matching patterns in output
        3. Calculate offset

        Returns:
            List of possible translations
        """
        # Only works if shapes match
        if input_grid.shape != output_grid.shape:
            return []

        # Simple approach: try small offsets and check correlation
        translations = []

        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue

                # Check if translation matches
                matches = ParameterInference._check_translation(input_grid, output_grid, dx, dy)

                if matches > 0.8:  # High match threshold
                    translations.append((dx, dy))

        return translations[:3]  # Return top 3

    @staticmethod
    def _check_translation(input_grid: np.ndarray, output_grid: np.ndarray,
                          dx: int, dy: int) -> float:
        """Check how well translation (dx, dy) matches input to output"""
        h, w = input_grid.shape

        # Calculate overlap region
        y_start_in = max(0, -dy)
        y_end_in = min(h, h - dy)
        x_start_in = max(0, -dx)
        x_end_in = min(w, w - dx)

        y_start_out = max(0, dy)
        x_start_out = max(0, dx)

        if y_end_in <= y_start_in or x_end_in <= x_start_in:
            return 0.0

        # Extract regions
        in_region = input_grid[y_start_in:y_end_in, x_start_in:x_end_in]
        out_region = output_grid[y_start_out:y_start_out+(y_end_in-y_start_in),
                                 x_start_out:x_start_out+(x_end_in-x_start_in)]

        # Calculate match (non-zero pixels)
        non_zero_mask = in_region != 0
        if not non_zero_mask.any():
            return 0.0

        matches = np.sum((in_region == out_region) & non_zero_mask)
        total = np.sum(non_zero_mask)

        return matches / total if total > 0 else 0.0

    @staticmethod
    def infer_rotation(input_grid: np.ndarray, output_grid: np.ndarray) -> List[int]:
        """
        Infer rotation from input to output

        Returns:
            List of rotation values (1=90°, 2=180°, 3=270°)
        """
        rotations = []

        for k in [1, 2, 3]:
            rotated = np.rot90(input_grid, k)
            if rotated.shape == output_grid.shape:
                accuracy = np.mean(rotated == output_grid)
                if accuracy > 0.8:
                    rotations.append(k)

        return rotations

    @staticmethod
    def infer_scaling(input_grid: np.ndarray, output_grid: np.ndarray) -> List[int]:
        """
        Infer scaling factor from input to output

        Returns:
            List of scaling factors (upscaling if >1, downscaling if output smaller)
        """
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        factors = []

        # Check upscaling
        if out_h > in_h and out_w > in_w:
            h_factor = out_h / in_h
            w_factor = out_w / in_w

            if h_factor == w_factor and h_factor == int(h_factor):
                factors.append(int(h_factor))

        # Check downscaling
        elif out_h < in_h and out_w < in_w:
            h_factor = in_h / out_h
            w_factor = in_w / out_w

            if h_factor == w_factor and h_factor == int(h_factor):
                factors.append(-int(h_factor))  # Negative for downscaling

        return factors

    @staticmethod
    def infer_morphology_iterations(input_grid: np.ndarray, output_grid: np.ndarray) -> List[int]:
        """
        Infer number of dilation/erosion iterations

        Strategy:
        1. Count non-zero pixels in input vs output
        2. If output has more, likely dilation
        3. If output has fewer, likely erosion
        4. Estimate iterations based on pixel count change

        Returns:
            List of iteration counts (1, 2, 3)
        """
        if input_grid.shape != output_grid.shape:
            return []

        in_count = np.sum(input_grid != 0)
        out_count = np.sum(output_grid != 0)

        if in_count == 0 or out_count == 0:
            return []

        # Ratio of non-zero pixels
        ratio = out_count / in_count

        iterations = []

        # Dilation typically increases pixels by ~30-50% per iteration
        if ratio > 1.2:
            if 1.2 <= ratio < 1.6:
                iterations.append(1)
            elif 1.6 <= ratio < 2.2:
                iterations.append(2)
            else:
                iterations.append(3)

        # Erosion typically decreases pixels
        elif ratio < 0.8:
            if 0.5 <= ratio < 0.8:
                iterations.append(1)
            elif 0.3 <= ratio < 0.5:
                iterations.append(2)
            else:
                iterations.append(3)

        return iterations

    @staticmethod
    def infer_reflection(input_grid: np.ndarray, output_grid: np.ndarray) -> List[str]:
        """
        Infer reflection axis

        Returns:
            List of axes ('h' for horizontal, 'v' for vertical)
        """
        if input_grid.shape != output_grid.shape:
            return []

        axes = []

        # Check horizontal flip
        flipped_h = np.fliplr(input_grid)
        if np.mean(flipped_h == output_grid) > 0.8:
            axes.append('h')

        # Check vertical flip
        flipped_v = np.flipud(input_grid)
        if np.mean(flipped_v == output_grid) > 0.8:
            axes.append('v')

        return axes

    @staticmethod
    def _deduplicate_color_maps(color_maps: List[Dict[int, int]]) -> List[Dict[int, int]]:
        """
        Deduplicate color mappings and return most common

        Strategy:
        1. Convert dicts to frozensets for hashing
        2. Count occurrences
        3. Return most common ones
        """
        if not color_maps:
            return []

        # Convert to frozensets for counting
        map_counts = Counter()
        for cm in color_maps:
            frozen = frozenset(cm.items())
            map_counts[frozen] += 1

        # Get most common (up to 5)
        most_common = map_counts.most_common(5)

        # Convert back to dicts
        return [dict(frozen) for frozen, count in most_common]


def test_parameter_inference():
    """Test parameter inference on sample transformations"""

    print("="*70)
    print("PARAMETER INFERENCE - TESTING")
    print("="*70)

    # Test 1: Color mapping
    print("\n1. Color Mapping Inference")
    print("-"*70)

    input_grid = np.array([
        [1, 2, 0],
        [2, 1, 0],
        [0, 0, 3]
    ])

    # Swap colors: 1→2, 2→1, 3→5
    output_grid = np.array([
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 5]
    ])

    color_maps = ParameterInference.infer_color_mapping(input_grid, output_grid)
    print(f"Input:\n{input_grid}")
    print(f"\nOutput:\n{output_grid}")
    print(f"\nInferred color mappings: {color_maps}")
    print(f"Expected: {{1: 2, 2: 1, 3: 5}}")

    # Test 2: Translation
    print("\n2. Translation Inference")
    print("-"*70)

    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 3, 4, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    # Shifted right by 2
    output_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2],
        [0, 0, 0, 3, 4],
        [0, 0, 0, 0, 0]
    ])

    translations = ParameterInference.infer_translation(input_grid, output_grid)
    print(f"Inferred translations: {translations}")
    print(f"Expected: (2, 0) for right shift")

    # Test 3: Rotation
    print("\n3. Rotation Inference")
    print("-"*70)

    input_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Rotated 90° clockwise
    output_grid = np.array([
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3]
    ])

    rotations = ParameterInference.infer_rotation(input_grid, output_grid)
    print(f"Inferred rotations: {rotations}")
    print(f"Expected: [1] for 90° CW")

    # Test 4: Scaling
    print("\n4. Scaling Inference")
    print("-"*70)

    input_grid = np.array([
        [1, 2],
        [3, 4]
    ])

    # Upscaled 2x
    output_grid = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ])

    scales = ParameterInference.infer_scaling(input_grid, output_grid)
    print(f"Inferred scales: {scales}")
    print(f"Expected: [2] for 2x upscaling")

    # Test 5: Full inference on task
    print("\n5. Full Parameter Inference on Task")
    print("-"*70)

    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[2, 1], [4, 3]]
            }
        ]
    }

    params = ParameterInference.infer_all_parameters(task)
    print(f"\nInferred parameters: {params}")
    print(f"Color mappings: {params.color_mappings}")
    print(f"Translations: {params.translations}")
    print(f"Rotations: {params.rotations}")

    print("\n" + "="*70)
    print("✓ Parameter inference tests complete!")
    print("="*70)


if __name__ == "__main__":
    test_parameter_inference()
