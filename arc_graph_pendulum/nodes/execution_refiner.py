"""
Execution Refiner - Fixes common precision errors in program execution.

Addresses:
1. Color-to-background substitution errors (X→0)
2. Edge/boundary handling issues
3. Incomplete pixel transformations
4. Small color mapping errors
"""

import numpy as np
from typing import Callable, Any, Dict, List, Tuple
from scipy.ndimage import label


class ExecutionRefiner:
    """Refines program execution to fix common precision errors."""

    def __init__(self):
        self.refinement_strategies = [
            self._fix_incomplete_fills,
            self._fix_edge_boundaries,
            self._fix_color_leakage,
            self._fix_object_boundaries,
        ]

    def refine_output(self, output: np.ndarray, input_grid: np.ndarray,
                     train_examples: List[Tuple[np.ndarray, np.ndarray]],
                     original_output: np.ndarray = None) -> np.ndarray:
        """
        Refine program output to fix common execution errors.

        Args:
            output: The program's output
            input_grid: The test input
            train_examples: Training examples for pattern learning
            original_output: Optional original unrefined output

        Returns:
            Refined output grid
        """
        refined = output.copy()

        # Apply each refinement strategy
        for strategy in self.refinement_strategies:
            refined = strategy(refined, input_grid, train_examples)

        return refined

    def _fix_incomplete_fills(self, output: np.ndarray, input_grid: np.ndarray,
                             train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Fix incomplete fill operations where some pixels remain as background.

        Common error: Object extraction or color fill leaves some pixels as 0.
        """
        refined = output.copy()

        # Check if output has unexpected background pixels
        # Look for small isolated background regions that should be filled

        # Find connected background components
        background_mask = (refined == 0)
        if not np.any(background_mask):
            return refined

        labeled, num_components = label(background_mask)

        # For each background region, check if it should be filled
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled == comp_id)
            comp_size = np.sum(comp_mask)

            # If it's a very small background region (1-5 pixels) surrounded by color
            if comp_size <= 5:
                # Find surrounding colors
                dilated = self._dilate_mask(comp_mask, 1)
                surrounding = refined[dilated & ~comp_mask]
                non_zero = surrounding[surrounding != 0]

                if len(non_zero) > 0:
                    # Fill with most common surrounding color
                    fill_color = np.bincount(non_zero).argmax()
                    refined[comp_mask] = fill_color

        return refined

    def _fix_edge_boundaries(self, output: np.ndarray, input_grid: np.ndarray,
                            train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Fix edge and boundary handling errors.

        Common error: Edge pixels not properly transformed.
        """
        refined = output.copy()

        # Check if training examples show edge preservation or transformation
        edge_should_match_input = self._check_edge_pattern(train_examples)

        if edge_should_match_input:
            # Restore edge pixels from input if they were incorrectly modified
            # Top row
            if refined.shape[0] == input_grid.shape[0]:
                if np.any(refined[0, :] == 0) and np.any(input_grid[0, :] != 0):
                    refined[0, :] = input_grid[0, :]

            # Bottom row
            if refined.shape[0] == input_grid.shape[0]:
                if np.any(refined[-1, :] == 0) and np.any(input_grid[-1, :] != 0):
                    refined[-1, :] = input_grid[-1, :]

            # Left column
            if refined.shape[1] == input_grid.shape[1]:
                if np.any(refined[:, 0] == 0) and np.any(input_grid[:, 0] != 0):
                    refined[:, 0] = input_grid[:, 0]

            # Right column
            if refined.shape[1] == input_grid.shape[1]:
                if np.any(refined[:, -1] == 0) and np.any(input_grid[:, -1] != 0):
                    refined[:, -1] = input_grid[:, -1]

        return refined

    def _fix_color_leakage(self, output: np.ndarray, input_grid: np.ndarray,
                          train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Fix color leakage where wrong colors appear in small amounts.

        Common error: Wrong color substitution (e.g., 5→2 for 4 pixels).
        """
        refined = output.copy()

        # Find colors that appear in very small quantities
        unique_colors, counts = np.unique(refined, return_counts=True)

        # Exclude background
        non_zero_mask = unique_colors != 0
        unique_colors = unique_colors[non_zero_mask]
        counts = counts[non_zero_mask]

        if len(counts) == 0:
            return refined

        total_non_zero = np.sum(counts)

        # Identify colors that appear in < 2% of non-background pixels
        rare_colors = []
        for color, count in zip(unique_colors, counts):
            if count / total_non_zero < 0.02 and count <= 10:
                rare_colors.append(color)

        # For each rare color, check if it should be replaced
        for rare_color in rare_colors:
            positions = np.argwhere(refined == rare_color)

            # Check surrounding colors
            for pos in positions:
                neighbors = self._get_neighbors(refined, pos[0], pos[1])
                neighbor_colors = [c for c in neighbors if c != 0 and c != rare_color]

                if neighbor_colors:
                    # Replace with most common neighbor
                    replacement = max(set(neighbor_colors), key=neighbor_colors.count)
                    refined[pos[0], pos[1]] = replacement

        return refined

    def _fix_object_boundaries(self, output: np.ndarray, input_grid: np.ndarray,
                               train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Fix object boundary errors where object edges are incomplete.

        Common error: Object extraction misses some boundary pixels.
        """
        refined = output.copy()

        # Find objects in output
        for color in range(1, 10):
            if not np.any(refined == color):
                continue

            color_mask = (refined == color)
            labeled, num_objects = label(color_mask)

            # For each object, check if boundary is clean
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)

                # Check for single-pixel holes or gaps
                obj_size = np.sum(obj_mask)
                if obj_size < 5:  # Skip very small objects
                    continue

                # Find pixels adjacent to object that are background
                dilated = self._dilate_mask(obj_mask, 1)
                boundary = dilated & ~obj_mask
                boundary_bg = boundary & (refined == 0)

                if np.any(boundary_bg):
                    # Check if these look like they should be part of object
                    bg_positions = np.argwhere(boundary_bg)

                    for pos in bg_positions:
                        # Count object-colored neighbors
                        neighbors = self._get_neighbors(refined, pos[0], pos[1])
                        obj_neighbor_count = sum(1 for c in neighbors if c == color)

                        # If >= 3 neighbors are the object color, include this pixel
                        if obj_neighbor_count >= 3:
                            refined[pos[0], pos[1]] = color

        return refined

    def _check_edge_pattern(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if training examples show edge preservation."""
        edge_matches = 0
        total = 0

        for input_grid, output_grid in train_examples:
            if input_grid.shape != output_grid.shape:
                continue

            # Check if edges match
            if np.array_equal(input_grid[0, :], output_grid[0, :]):
                edge_matches += 1
            total += 1

            if np.array_equal(input_grid[-1, :], output_grid[-1, :]):
                edge_matches += 1
            total += 1

        if total == 0:
            return False

        return edge_matches / total > 0.5

    def _dilate_mask(self, mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """Dilate a binary mask."""
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, iterations=radius)

    def _get_neighbors(self, grid: np.ndarray, row: int, col: int) -> List[int]:
        """Get 4-connected neighbors."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                neighbors.append(int(grid[r, c]))
        return neighbors

    def create_refining_wrapper(self, program_func: Callable,
                                train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Callable:
        """
        Wrap a program function to apply refinement to its output.

        Args:
            program_func: Original program function
            train_examples: Training examples for pattern learning

        Returns:
            Wrapped function that applies refinement
        """
        def refined_func(input_grid: np.ndarray) -> np.ndarray:
            # Execute original program
            output = program_func(input_grid)

            # Apply refinement
            refined_output = self.refine_output(output, input_grid, train_examples)

            return refined_output

        return refined_func
