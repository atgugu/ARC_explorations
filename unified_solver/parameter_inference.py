"""
Parameter Inference for ARC Program Synthesis
============================================

Learn parameters from training examples instead of enumerating fixed values.
Key innovation of Phase 3: Task-specific parameters, not generic operations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from arc_active_inference_solver import Grid, ARCTask


class ParameterInference:
    """Infer program parameters from training examples"""

    def infer_all_parameters(self, train_pairs: List[Tuple[Grid, Grid]], verbose: bool = False) -> Dict:
        """
        Analyze training pairs and extract all learnable parameters

        Returns dict with:
        - color_map: Dict[int, int] or None
        - scale_factor: int or None
        - tile_factor: Tuple[int, int] or None
        - movement_vector: Tuple[int, int] or None
        - rotation: int or None
        """
        params = {}

        # Color mapping
        color_map = self.infer_color_mapping(train_pairs)
        if color_map:
            params['color_map'] = color_map
            if verbose:
                print(f"  Inferred color map: {color_map}")

        # Scale factor
        scale = self.infer_scale_factor(train_pairs)
        if scale:
            params['scale_factor'] = scale
            if verbose:
                print(f"  Inferred scale factor: {scale}x")

        # Tile factor
        tile = self.infer_tile_factor(train_pairs)
        if tile:
            params['tile_factor'] = tile
            if verbose:
                print(f"  Inferred tile factor: {tile[0]}x{tile[1]}")

        # Rotation
        rotation = self.infer_rotation(train_pairs)
        if rotation is not None:
            params['rotation'] = rotation
            if verbose:
                rot_names = {90: "90°", 180: "180°", 270: "270°"}
                print(f"  Inferred rotation: {rot_names.get(rotation, str(rotation))}")

        # Flip
        flip = self.infer_flip(train_pairs)
        if flip:
            params['flip'] = flip
            if verbose:
                print(f"  Inferred flip: {flip}")

        return params

    def infer_color_mapping(self, train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict[int, int]]:
        """
        Infer consistent color mapping from training examples

        Strategy:
        1. For each training pair, track color correspondences
        2. Build frequency map: input_color → {output_colors: counts}
        3. Return most frequent mapping for each input color

        Returns:
            Dict[int, int]: input_color → output_color mapping
            None if no consistent mapping found
        """
        color_freq = defaultdict(lambda: defaultdict(int))
        total_pixels = 0

        for input_grid, output_grid in train_pairs:
            # Only works if same size
            if input_grid.shape != output_grid.shape:
                continue

            # Track color correspondences
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    in_color = input_grid.data[i, j]
                    out_color = output_grid.data[i, j]
                    color_freq[in_color][out_color] += 1
                    total_pixels += 1

        if total_pixels == 0:
            return None

        # Extract most frequent mapping for each color
        color_map = {}
        for in_color, out_colors in color_freq.items():
            if not out_colors:
                continue

            # Get most frequent output color for this input color
            best_out_color = max(out_colors.items(), key=lambda x: x[1])[0]

            # Only include if mapping is consistent (>70% of pixels)
            total = sum(out_colors.values())
            consistency = out_colors[best_out_color] / total

            if consistency > 0.7:
                # Only include if mapping actually changes the color
                # (or if it's a minority color that stays the same)
                if in_color != best_out_color or consistency < 0.95:
                    color_map[in_color] = best_out_color

        # Return None if mapping is trivial (identity on all colors)
        if not color_map:
            return None

        # Return None if mapping is just identity
        if all(k == v for k, v in color_map.items()):
            return None

        return color_map

    def infer_scale_factor(self, train_pairs: List[Tuple[Grid, Grid]]) -> Optional[int]:
        """
        Infer zoom/scale factor if output is scaled version of input

        Returns:
            int: scale factor (2, 3, 4, etc.)
            None if no consistent scaling
        """
        scale_factors = []

        for input_grid, output_grid in train_pairs:
            if input_grid.shape[0] == 0 or input_grid.shape[1] == 0:
                continue

            # Check if output is integer multiple of input
            if output_grid.shape[0] % input_grid.shape[0] != 0:
                continue
            if output_grid.shape[1] % input_grid.shape[1] != 0:
                continue

            scale_h = output_grid.shape[0] // input_grid.shape[0]
            scale_w = output_grid.shape[1] // input_grid.shape[1]

            # Check for uniform scaling
            if scale_h == scale_w and scale_h > 1:
                # Verify it's actually a zoom (pixels are repeated)
                is_zoom = True
                for i in range(input_grid.shape[0]):
                    for j in range(input_grid.shape[1]):
                        expected_color = input_grid.data[i, j]
                        # Check if all corresponding output pixels match
                        for di in range(scale_h):
                            for dj in range(scale_w):
                                out_i = i * scale_h + di
                                out_j = j * scale_w + dj
                                if output_grid.data[out_i, out_j] != expected_color:
                                    is_zoom = False
                                    break
                            if not is_zoom:
                                break
                        if not is_zoom:
                            break
                    if not is_zoom:
                        break

                if is_zoom:
                    scale_factors.append(scale_h)

        # Return if all consistent
        if scale_factors and all(s == scale_factors[0] for s in scale_factors):
            return scale_factors[0]

        return None

    def infer_tile_factor(self, train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[int, int]]:
        """
        Infer tiling factor (NxM repetition of input)

        Returns:
            (n, m): tile n times vertically, m times horizontally
            None if no consistent tiling
        """
        tile_factors = []

        for input_grid, output_grid in train_pairs:
            if input_grid.shape[0] == 0 or input_grid.shape[1] == 0:
                continue

            # Check if output is integer multiple of input
            if output_grid.shape[0] % input_grid.shape[0] != 0:
                continue
            if output_grid.shape[1] % input_grid.shape[1] != 0:
                continue

            n = output_grid.shape[0] // input_grid.shape[0]
            m = output_grid.shape[1] // input_grid.shape[1]

            if n == 1 and m == 1:
                continue  # Not a tiling, just same size

            # Verify it's actually a tiling (pattern repeats)
            is_tiling = True
            for i in range(output_grid.shape[0]):
                for j in range(output_grid.shape[1]):
                    src_i = i % input_grid.shape[0]
                    src_j = j % input_grid.shape[1]
                    if output_grid.data[i, j] != input_grid.data[src_i, src_j]:
                        is_tiling = False
                        break
                if not is_tiling:
                    break

            if is_tiling:
                tile_factors.append((n, m))

        # Return if all consistent
        if tile_factors and all(t == tile_factors[0] for t in tile_factors):
            return tile_factors[0]

        return None

    def infer_rotation(self, train_pairs: List[Tuple[Grid, Grid]]) -> Optional[int]:
        """
        Infer rotation angle (90, 180, 270 degrees)

        Returns:
            int: rotation in degrees (90, 180, 270)
            None if no consistent rotation
        """
        rotations = []

        for input_grid, output_grid in train_pairs:
            # Check 90° rotation
            rotated_90 = np.rot90(input_grid.data, k=1)
            if np.array_equal(rotated_90, output_grid.data):
                rotations.append(90)
                continue

            # Check 180° rotation
            rotated_180 = np.rot90(input_grid.data, k=2)
            if np.array_equal(rotated_180, output_grid.data):
                rotations.append(180)
                continue

            # Check 270° rotation
            rotated_270 = np.rot90(input_grid.data, k=3)
            if np.array_equal(rotated_270, output_grid.data):
                rotations.append(270)
                continue

            # No rotation detected for this pair
            return None

        # Return if all consistent and non-empty
        if rotations and all(r == rotations[0] for r in rotations):
            return rotations[0]

        return None

    def infer_flip(self, train_pairs: List[Tuple[Grid, Grid]]) -> Optional[str]:
        """
        Infer flip direction (horizontal or vertical)

        Returns:
            str: "horizontal", "vertical", or None
        """
        flips = []

        for input_grid, output_grid in train_pairs:
            if input_grid.shape != output_grid.shape:
                return None

            # Check horizontal flip
            flipped_h = np.fliplr(input_grid.data)
            if np.array_equal(flipped_h, output_grid.data):
                flips.append("horizontal")
                continue

            # Check vertical flip
            flipped_v = np.flipud(input_grid.data)
            if np.array_equal(flipped_v, output_grid.data):
                flips.append("vertical")
                continue

            # No flip detected for this pair
            return None

        # Return if all consistent and non-empty
        if flips and all(f == flips[0] for f in flips):
            return flips[0]

        return None


def apply_color_mapping(grid: Grid, color_map: Dict[int, int]) -> Grid:
    """Apply learned color mapping to grid"""
    result = grid.copy()
    for old_color, new_color in color_map.items():
        result.data[grid.data == old_color] = new_color
    return result


if __name__ == "__main__":
    # Test parameter inference
    print("Testing Parameter Inference")
    print("=" * 60)

    # Test 1: Color mapping
    print("\nTest 1: Color Mapping")
    print("-" * 40)
    train = [
        (Grid([[1, 2], [3, 1]]), Grid([[5, 6], [7, 5]])),
        (Grid([[2, 1], [1, 3]]), Grid([[6, 5], [5, 7]])),
    ]

    inferrer = ParameterInference()
    params = inferrer.infer_all_parameters(train, verbose=True)
    print(f"Result: {params}")

    # Test 2: Scale factor
    print("\n\nTest 2: Scale Factor (2x zoom)")
    print("-" * 40)
    train = [
        (Grid([[1, 2]]), Grid([[1, 1, 2, 2]])),
        (Grid([[3], [4]]), Grid([[3, 3], [3, 3], [4, 4], [4, 4]])),
    ]

    params = inferrer.infer_all_parameters(train, verbose=True)
    print(f"Result: {params}")

    # Test 3: Rotation
    print("\n\nTest 3: Rotation (90°)")
    print("-" * 40)
    train = [
        (Grid([[1, 2], [3, 4]]), Grid([[2, 4], [1, 3]])),
        (Grid([[5, 6], [7, 8]]), Grid([[6, 8], [5, 7]])),
    ]

    params = inferrer.infer_all_parameters(train, verbose=True)
    print(f"Result: {params}")

    print("\n" + "=" * 60)
    print("✓ Parameter inference tests complete")
